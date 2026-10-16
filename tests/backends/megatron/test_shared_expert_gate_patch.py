# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""CPU checks for shared-gate publication and native compatibility."""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest


_SPEC = importlib.util.spec_from_file_location(
    "shared_expert_gate_under_test",
    Path(__file__).resolve().parents[3] / "relax/backends/megatron/megatron_patch/shared_expert_gate_patch.py",
)
module = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(module)


def _runtime(monkeypatch, *, readiness=True, version=None):
    events = []

    class SharedExpert:
        use_shared_expert_gate = True
        gate_weight = object()

        def wait_current_stream(self):
            events.append("wait")
            return "result"

    if version is not None:
        SharedExpert.shared_gate_param_readiness_version = version
    shared = ModuleType("megatron.core.transformer.moe.shared_experts")
    shared.SharedExpertMLP = SharedExpert
    utils = ModuleType("megatron.core.utils")
    if readiness:
        utils.ensure_params_ready = lambda params: events.append(("publish", params))
    monkeypatch.setitem(sys.modules, shared.__name__, shared)
    monkeypatch.setitem(sys.modules, utils.__name__, utils)
    return SharedExpert, events


@pytest.mark.parametrize("gate", [False, True])
def test_gate_publication_precedes_stream_wait_and_install_is_idempotent(monkeypatch, gate):
    shared, events = _runtime(monkeypatch)
    assert module.install()
    installed = shared.wait_current_stream
    assert module.install()
    assert shared.wait_current_stream is installed
    instance = shared()
    instance.use_shared_expert_gate = gate
    assert instance.wait_current_stream() == "result"
    assert events == ([("publish", [instance.gate_weight]), "wait"] if gate else ["wait"])
    assert shared.shared_gate_param_readiness_version == 1


@pytest.mark.parametrize("readiness", [False, True])
def test_native_readiness_is_preserved(monkeypatch, readiness):
    shared, _ = _runtime(monkeypatch, readiness=readiness, version=1)
    original = shared.wait_current_stream
    assert module.install()
    assert shared.wait_current_stream is original


def test_missing_readiness_does_not_advertise_capability(monkeypatch):
    shared, _ = _runtime(monkeypatch, readiness=False)
    original = shared.wait_current_stream
    assert not module.install()
    assert shared.wait_current_stream is original
    assert not hasattr(shared, "shared_gate_param_readiness_version")
