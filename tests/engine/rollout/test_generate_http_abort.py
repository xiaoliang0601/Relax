# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Exercise rollout abort handling without loading the inference stack."""

import ast
import asyncio
import contextvars
import copy
import inspect
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

from relax.engine.rollout.request_permit import GenerationAborted
from relax.utils.http_utils import _post, is_expected_sglang_499, is_router_no_available_workers


def load_dispatch(post):
    path = Path(__file__).resolve().parents[3] / "relax/engine/rollout/sglang_rollout.py"
    tree = ast.parse(path.read_text())
    dispatch = next(n for n in tree.body if isinstance(n, ast.AsyncFunctionDef) and n.name == "_dispatch_generate")
    post_generation = next(
        n for n in tree.body if isinstance(n, ast.AsyncFunctionDef) and n.name == "_post_generation"
    )
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "GenerateState")
    post_generate = next(n for n in cls.body if isinstance(n, ast.AsyncFunctionDef) and n.name == "post_generate")
    ns = dict(
        inspect=inspect,
        GenerationAborted=GenerationAborted,
        httpx=httpx,
        is_expected_sglang_499=is_expected_sglang_499,
        is_router_no_available_workers=is_router_no_available_workers,
        Sample=SimpleNamespace(Status=SimpleNamespace(ABORTED="aborted")),
        post=post,
        _holding_session_lock=contextvars.ContextVar("test_holding_session_lock", default=False),
    )
    module = ast.Module(
        body=[
            ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
            dispatch,
            post_generation,
            post_generate,
        ],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), ns)
    return ns


@pytest.mark.parametrize("status", [499, 400, 200])
@pytest.mark.parametrize("custom", [False, True])
async def test_generate_http_abort_preserves_partial_sample_and_releases_permit(status, custom):
    calls = []

    def handle(request):
        calls.append(request)
        return httpx.Response(status, json={"error": {"message": "Request test was aborted"}} if status != 200 else {})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:

        async def post(url, payload, headers=None, **kwargs):
            return await _post(client, url, payload, headers=headers, **kwargs)

        ns = load_dispatch(post)

        async def generate(args, sample, sampling_params, evaluation=False):
            await ns["_post_generation"](args, state, "http://test/generate", {}, None, evaluation)
            return sample

        generate.manages_inference_permit = custom
        ns["generate"] = generate
        ns["load_function"] = lambda path: generate
        state = SimpleNamespace(aborted=False, semaphore=asyncio.Semaphore(1), dp_rank_context=nullcontext)
        args = SimpleNamespace(
            custom_generate_function_path="custom" if custom else None, fully_async=False, use_slime_router=False
        )
        sample = SimpleNamespace(
            generate_function_path=None,
            status="pending",
            response="existing partial",
            tokens=[1, 2],
            response_length=1,
            rollout_log_probs=[-0.2],
            loss_mask=[0],
        )
        previous = copy.deepcopy(vars(sample))
        if status == 400:
            with pytest.raises(httpx.HTTPStatusError):
                await ns["_dispatch_generate"](state, args, sample, {})
        else:
            assert await ns["_dispatch_generate"](state, args, sample, {}) is sample
            assert sample.status == ("aborted" if status == 499 else "pending")
        assert {k: v for k, v in vars(sample).items() if k != "status"} == {
            k: v for k, v in previous.items() if k != "status"
        }
        assert not state.semaphore.locked()
        assert not ns["_holding_session_lock"].get()
        assert len(calls) == 1


async def test_post_generate_releases_permit_on_http_error():
    async def post(*args, **kwargs):
        request = httpx.Request("POST", "http://test/generate")
        response = httpx.Response(499, request=request, json={"error": {"message": "Request test was aborted"}})
        raise httpx.HTTPStatusError("server abort", request=request, response=response)

    ns = load_dispatch(post)
    permit = asyncio.Semaphore(1)
    state = SimpleNamespace(inference_permit=lambda: permit)
    with pytest.raises(httpx.HTTPStatusError):
        await ns["post_generate"](state, "http://test/generate", {})
    assert not permit.locked()
