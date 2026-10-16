# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""CPU regression tests for the patched SGLang startup GC request."""

import ast
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import requests


@pytest.fixture
def freeze_gc():
    source = os.environ.get("SGLANG_SOURCE_DIR")
    if not source:
        pytest.skip("Set SGLANG_SOURCE_DIR to the patched SGLang 0.5.20 checkout")
    path = Path(source) / "python/sglang/srt/entrypoints/http_server.py"
    function = next(
        node
        for node in ast.parse(path.read_text()).body
        if isinstance(node, ast.FunctionDef) and node.name == "_freeze_gc_after_server_warmup"
    )
    clock = SimpleNamespace(now=0.0)

    def sleep(seconds: float) -> None:
        clock.now += seconds

    post = Mock(return_value=Mock())
    warning = Mock()
    serving = SimpleNamespace(admin_api_key="admin-key", api_key="user-key")
    namespace = {
        "ServerArgs": object,
        "get_serving": lambda: serving,
        "requests": SimpleNamespace(post=post, exceptions=requests.exceptions),
        "time": SimpleNamespace(monotonic=lambda: clock.now, sleep=sleep),
        "ssl_verify_of": lambda _args: "test-ca.pem",
        "logger": SimpleNamespace(warning=warning),
    }
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"), namespace)
    return SimpleNamespace(
        run=lambda: namespace[function.name](SimpleNamespace(url=lambda: "https://localhost:30000")),
        post=post,
        warning=warning,
        clock=clock,
        serving=serving,
    )


def test_freeze_gc_retries_until_listener_is_ready(freeze_gc):
    response = Mock()
    freeze_gc.post.side_effect = [requests.ConnectionError("refused"), requests.ConnectionError("refused"), response]
    freeze_gc.run()
    assert freeze_gc.post.call_count == 3
    response.raise_for_status.assert_called_once_with()
    freeze_gc.warning.assert_not_called()
    assert freeze_gc.clock.now == pytest.approx(0.2)
    assert freeze_gc.post.call_args.kwargs == {
        "headers": {"Authorization": "Bearer admin-key"},
        "timeout": 10,
        "verify": "test-ca.pem",
    }


def test_freeze_gc_unavailable_listener_has_bounded_wait_and_warning(freeze_gc):
    freeze_gc.post.side_effect = requests.ConnectionError("refused")
    freeze_gc.run()
    assert freeze_gc.clock.now == pytest.approx(30)
    assert 300 <= freeze_gc.post.call_count <= 302
    freeze_gc.warning.assert_called_once_with("post-warmup freeze_gc failed", exc_info=True)


@pytest.mark.parametrize("error", [requests.exceptions.SSLError, requests.ReadTimeout, requests.HTTPError])
def test_freeze_gc_reports_non_startup_errors_without_retry(freeze_gc, error):
    if error is requests.HTTPError:
        freeze_gc.post.return_value.raise_for_status.side_effect = error("failed")
    else:
        freeze_gc.post.side_effect = error("failed")
    freeze_gc.run()
    assert freeze_gc.post.call_count == 1
    assert freeze_gc.clock.now == 0
    freeze_gc.warning.assert_called_once_with("post-warmup freeze_gc failed", exc_info=True)


@pytest.mark.parametrize("api_key", ["user-key", None])
def test_freeze_gc_preserves_auth_fallback(freeze_gc, api_key):
    freeze_gc.serving.admin_api_key = None
    freeze_gc.serving.api_key = api_key
    freeze_gc.run()
    expected_headers = {"Authorization": "Bearer user-key"} if api_key else {}
    assert freeze_gc.post.call_args.kwargs["headers"] == expected_headers
    freeze_gc.warning.assert_not_called()
