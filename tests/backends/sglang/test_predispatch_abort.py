# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Execute actual patched SGLang methods without importing CUDA dependencies.

Set SGLANG_SOURCE_DIR to the checkout with docker/patch/sglang/v0.5.20.patch
applied. Events control the abort windows; no inference hardware is needed.
"""

import ast
import asyncio
import copy
import dataclasses
import os
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest


class Request(SimpleNamespace):
    def normalize_batch_and_arguments(self):
        self.max_thinking_tokens = None
        self.routed_dp_rank = None

    def __getitem__(self, index):
        return self.children[index]

    def regenerate_rid(self):
        self.rid += "-child"
        return self.rid


@pytest.fixture
def manager():
    source = os.environ.get("SGLANG_SOURCE_DIR")
    if not source:
        pytest.skip("Set SGLANG_SOURCE_DIR to the patched SGLang 0.5.20 checkout")
    path = Path(source) / "python/sglang/srt/managers/tokenizer_manager.py"
    tree = ast.parse(path.read_text())
    classes = {n.name: n for n in tree.body if isinstance(n, ast.ClassDef)}
    methods = {
        "generate_request",
        "_wait_for_request_unpaused",
        "_raise_if_request_aborted",
        "_send_one_request",
        "_send_batch_request",
        "_mark_state_dispatched",
        "abort_request",
        "pause_generation",
        "continue_generation",
        "_release_req_states_on_failure",
        "_handle_batch_request",
    }
    manager_class = ast.ClassDef(
        name="Manager",
        bases=[],
        keywords=[],
        decorator_list=[],
        body=[n for n in classes["TokenizerManager"].body if getattr(n, "name", None) in methods],
    )
    module = ast.Module(
        body=[
            ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
            classes["RequestAbortedError"],
            classes["ReqState"],
            manager_class,
        ],
        type_ignores=[],
    )
    ns = dict(
        __name__=__name__,
        asyncio=asyncio,
        dataclasses=dataclasses,
        copy=copy,
        GenerateReqInput=Request,
        TokenizedGenerateReqInput=Request,
        AbortReq=SimpleNamespace,
        BatchTokenizedGenerateReqInput=SimpleNamespace,
        BatchTokenizedEmbeddingReqInput=SimpleNamespace,
        get_serving=lambda: SimpleNamespace(tokenizer_worker_num=1),
        get_disagg=lambda: SimpleNamespace(language_only=False),
        logger=SimpleNamespace(warning=lambda *a: None, exception=lambda *a: None),
        wrap_shm_features=lambda obj: obj,
        set_time_batch=lambda *a: None,
        nullcontext=nullcontext,
        get_bool_env_var=lambda *a: False,
    )
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), ns)
    tm = ns["Manager"]()
    tm.error = ns["RequestAbortedError"]
    tm.rid_to_state = {}
    tm.encoder_dispatch_ready = {}
    tm.is_pause = False
    tm.is_pause_cond = asyncio.Condition()
    tm.enable_metrics = False
    tm.auto_create_handle_loop = lambda: None
    tm._set_default_priority = lambda obj: None
    tm.request_logger = SimpleNamespace(log_received_request=lambda *a: None)
    tm.tokenizer = None
    tm.model_update_lock = SimpleNamespace(reader_lock=asyncio.Lock(), is_locked=AsyncMock(return_value=False))
    tm._validate_and_resolve_lora = AsyncMock()
    tm.sent = []
    tm._dispatch_to_scheduler = tm.sent.append
    tm._async_dispatch_to_scheduler = AsyncMock()
    tm.prepared = []
    tm.cancelled = []

    async def prepare(items):
        result = list(items)
        tm.prepared.append(result)
        return result

    tm.cuda_vmm_feature_transport = SimpleNamespace(
        prepare_for_dispatch_async=prepare,
        cancel_for_dispatch=tm.cancelled.append,
    )

    def init(obj, request=None):
        for child in [obj] if obj.is_single else obj.children:
            tm.rid_to_state[child.rid] = ns["ReqState"](
                [], False, asyncio.Event(), child, SimpleNamespace(set_finished_time=lambda: None)
            )

    tm._init_req_state = init

    def tokenized(obj):
        return Request(
            rid=obj.rid,
            mm_inputs=None,
            input_ids=[1],
            sampling_params=SimpleNamespace(),
            stream=False,
            wrap_pickle_fields=lambda: None,
            time_stats=SimpleNamespace(
                set_api_server_dispatch_time=lambda: None, set_api_server_dispatch_finish_time=lambda: None
            ),
        )

    tm._tokenize_one_request = AsyncMock(side_effect=tokenized)
    tm.tokenized = tokenized

    async def response(obj, request):
        yield {"rid": obj.rid}

    tm._wait_one_response = response
    tm._should_use_batch_tokenization = lambda *a: True
    tm._batch_tokenize_and_process = AsyncMock(side_effect=lambda size, obj: [tokenized(c) for c in obj.children])
    return tm


def request(rid="r"):
    return Request(
        rid=rid, is_single=True, max_thinking_tokens=None, routed_dp_rank=None, return_prompt_token_ids=False
    )


async def started_generation(tm, obj):
    # A turn through the event loop runs the task up to its controlled wait.
    task = asyncio.create_task(anext(tm.generate_request(obj)))
    await asyncio.sleep(0)
    assert not task.done()
    return task


@pytest.mark.parametrize("abort_all", [False, True])
async def test_paused_abort_returns_without_continue_and_cleans_state(manager, abort_all):
    tm = manager
    tm.is_pause = True
    task = await started_generation(tm, request())
    # Allow the Condition waiter to acquire its lock before aborting.
    await asyncio.sleep(0)
    tm.abort_request(rid="r", abort_all=abort_all)
    with pytest.raises(tm.error) as exc:
        await asyncio.wait_for(task, 1)
    assert exc.value.status_code == 499
    assert tm.is_pause
    assert not tm.rid_to_state
    assert not tm._tokenize_one_request.called
    assert len(tm.sent) == 1  # only the scheduler AbortReq


async def test_pause_abort_drain_does_not_need_continue(manager):
    tm = manager
    tm.is_pause = True
    task = await started_generation(tm, request())
    await asyncio.sleep(0)
    entered, release = asyncio.Event(), asyncio.Event()

    async def locked():
        entered.set()
        await release.wait()
        return False

    tm.model_update_lock.is_locked = locked
    pause = asyncio.create_task(tm.pause_generation(SimpleNamespace(mode="abort")))
    await entered.wait()
    assert tm.rid_to_state["r"].abort_requested
    release.set()
    await asyncio.wait_for(pause, 1)
    with pytest.raises(tm.error):
        await asyncio.wait_for(task, 1)
    assert tm.is_pause and not tm.rid_to_state


@pytest.mark.parametrize("window", ["tokenize", "prepare", "reader"])
@pytest.mark.parametrize("batch", [False, True, "sequential"])
async def test_abort_before_dispatch_cleans_resources(manager, window, batch):
    tm = manager
    obj = request()
    if batch:
        obj = Request(
            rid=["a", "b"], is_single=False, children=[request("a"), request("b")], batch_size=2, parallel_sample_num=1
        )
    if batch == "sequential":
        tm._should_use_batch_tokenization = lambda *a: False
    entered, release = asyncio.Event(), asyncio.Event()
    if window == "reader":
        await tm.model_update_lock.reader_lock.acquire()
    else:
        owner, name = (
            (tm.cuda_vmm_feature_transport, "prepare_for_dispatch_async")
            if window == "prepare"
            else (tm, "_batch_tokenize_and_process" if batch is True else "_tokenize_one_request")
        )
        original = getattr(owner, name)

        async def block(*args):
            entered.set()
            await release.wait()
            return await original(*args)

        setattr(owner, name, block)
    task = await started_generation(tm, obj)
    if window != "reader":
        await entered.wait()
    tm.abort_request(abort_all=True)
    if window == "reader":
        tm.model_update_lock.reader_lock.release()
    else:
        release.set()
    with pytest.raises(tm.error):
        await asyncio.wait_for(task, 1)
    assert not tm.rid_to_state
    assert len(tm.sent) == 1
    assert tm.cancelled == tm.prepared
    assert len(tm.cancelled) == (0 if window == "reader" else 1)


async def test_dispatched_request_keeps_scheduler_abort_path(manager):
    tm = manager
    obj = request()
    tm._init_req_state(obj)
    await tm._send_one_request(tm.tokenized(obj))
    tm.abort_request(abort_all=True)
    state = tm.rid_to_state["r"]
    assert state.dispatched and not state.abort_requested and not state.event.is_set()
    assert len(tm.sent) == 2 and not tm.cancelled


async def test_resume_and_abort_race_prefers_abort(manager):
    tm = manager
    tm.is_pause = True
    task = await started_generation(tm, request())
    await tm.continue_generation(SimpleNamespace())
    tm.abort_request(abort_all=True)
    with pytest.raises(tm.error):
        await asyncio.wait_for(task, 1)
    assert not tm.rid_to_state and not tm._tokenize_one_request.called


async def test_cancel_paused_request_leaves_no_wait_tasks(manager):
    tm = manager
    tm.is_pause = True
    before = asyncio.all_tasks()
    task = await started_generation(tm, request())
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert not tm.rid_to_state
    assert asyncio.all_tasks() == before


@pytest.mark.parametrize("window", ["tokenize", "prefix", "child_prepare"])
async def test_parallel_sampling_does_not_spawn_children_after_abort(manager, window):
    tm = manager
    obj = Request(rid=["r"], is_single=False, children=[request()], batch_size=1, parallel_sample_num=2)
    entered, release = asyncio.Event(), asyncio.Event()
    if window == "tokenize":
        original = tm._tokenize_one_request

        async def tokenize(child):
            entered.set()
            await release.wait()
            return await original(child)

        tm._tokenize_one_request = tokenize
    else:

        async def prefix_response(child, request):
            # Model the scheduler finishing the temporary prefix request.
            tm.rid_to_state.pop(child.rid)
            if window == "prefix":
                entered.set()
                await release.wait()
            yield {}

        tm._wait_one_response = prefix_response
        if window == "child_prepare":
            original = tm.cuda_vmm_feature_transport.prepare_for_dispatch_async
            calls = 0

            async def prepare(items):
                nonlocal calls
                calls += 1
                if calls == 2:
                    entered.set()
                    await release.wait()
                return await original(items)

            tm.cuda_vmm_feature_transport.prepare_for_dispatch_async = prepare
    task = await started_generation(tm, obj)
    await entered.wait()
    tm.abort_request(abort_all=True)
    release.set()
    with pytest.raises(tm.error):
        await asyncio.wait_for(task, 1)
    assert not tm.rid_to_state
    # At most the prefix request was sent, never an expanded sample.
    generations = [o for o in tm.sent if isinstance(o, Request)]
    assert len(generations) == (0 if window == "tokenize" else 1)


async def test_condition_wait_error_is_not_swallowed(manager):
    tm = manager
    tm.is_pause = True
    tm.is_pause_cond.wait_for = AsyncMock(side_effect=RuntimeError("condition failed"))
    with pytest.raises(RuntimeError, match="condition failed"):
        await asyncio.wait_for(anext(tm.generate_request(request())), 1)
    assert not tm.rid_to_state


async def test_continue_without_abort_dispatches_normally(manager):
    tm = manager
    tm.is_pause = True
    obj = request()
    task = await started_generation(tm, obj)
    await tm.continue_generation(SimpleNamespace())
    assert await asyncio.wait_for(task, 1) == {"rid": "r"}
    assert any(isinstance(o, Request) for o in tm.sent)
    assert not tm.cancelled


async def test_nonstream_generate_returns_http_499(manager):
    from http import HTTPStatus

    from fastapi.responses import ORJSONResponse

    path = Path(os.environ["SGLANG_SOURCE_DIR"]) / "python/sglang/srt/entrypoints/http_server.py"
    tree = ast.parse(path.read_text())
    nodes = [
        n
        for n in tree.body
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
        and n.name in {"generate_request", "_create_error_response"}
    ]
    for node in nodes:
        node.decorator_list = []
    module = ast.Module(
        body=[
            ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
            *nodes,
        ],
        type_ignores=[],
    )
    ns = dict(
        HTTPStatus=HTTPStatus,
        ORJSONResponse=ORJSONResponse,
        envs=SimpleNamespace(SGLANG_ENABLE_REQUEST_HEADER_OVERRIDES=SimpleNamespace(get=lambda: False)),
        _global_state=SimpleNamespace(tokenizer_manager=manager),
        logger=SimpleNamespace(error=lambda *a: None),
    )
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), ns)
    manager.is_pause = True
    obj = request()
    obj.stream = False
    task = asyncio.create_task(ns["generate_request"](obj, SimpleNamespace()))
    await asyncio.sleep(0)
    manager.abort_request(abort_all=True)
    response = await asyncio.wait_for(task, 1)
    assert response.status_code == 499
    assert not manager.rid_to_state
