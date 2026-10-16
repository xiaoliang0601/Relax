# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""CPU checks of isolated HybridEP opt-in and collective preflight."""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest


ROOT = Path(__file__).resolve().parents[3]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def preflight(monkeypatch, tmp_path):
    env = load("hybrid_env", ROOT / "relax/utils/env.py")
    monkeypatch.setitem(sys.modules, "relax.utils.env", env)
    monkeypatch.setenv("QWEN38_HYBRID_EP_ROOT", str(tmp_path))
    monkeypatch.delenv("NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN", raising=False)
    expert, world = object(), object()
    dist = ModuleType("torch.distributed")
    dist.group = SimpleNamespace(WORLD=world)
    dist.get_world_size = lambda group: 32 if group is world else 8
    dist.get_rank = lambda group: 0
    dist.get_global_rank = lambda group, rank: rank

    def gather(output, value, group):
        assert group is expert
        output[:] = [dict(value, rank=i) if isinstance(value, dict) else value for i in range(8)]

    dist.all_gather_object = Mock(side_effect=gather)
    torch = ModuleType("torch")
    torch.distributed = dist
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "torch.distributed", dist)
    mpu = SimpleNamespace(get_expert_tensor_and_model_parallel_group=lambda: expert)
    for name, value in [
        ("tensor_model", 8),
        ("pipeline_model", 4),
        ("context", 1),
        ("expert_model", 8),
        ("expert_tensor", 1),
    ]:
        setattr(mpu, "get_" + name + "_parallel_world_size", lambda value=value: value)
    core = ModuleType("megatron.core")
    core.mpu = mpu
    megatron = ModuleType("megatron")
    megatron.core = core
    monkeypatch.setitem(sys.modules, "megatron", megatron)
    monkeypatch.setitem(sys.modules, "megatron.core", core)
    detect = Mock(return_value=8)
    for name in ("deep_ep", "deep_ep_cpp", "hybrid_ep_cpp"):
        module = ModuleType(name)
        module.__file__ = str(tmp_path / "isolated-import" / name / "module.so")
        module.ExtendedMemoryAllocator = lambda: SimpleNamespace(detect_accessible_ranks=detect)
        monkeypatch.setitem(sys.modules, name, module)
    logger = Mock()
    monkeypatch.setitem(sys.modules, "relax.utils.logging_utils", SimpleNamespace(get_logger=lambda name: logger))
    module = load("hybrid_preflight", ROOT / "relax/backends/megatron/hybridep_preflight.py")
    return module, mpu, dist, detect, expert, logger


@pytest.mark.parametrize("tp", [2, 8])
def test_valid_group_uses_real_peer_detection(preflight, tp):
    module, mpu, dist, detect, expert, logger = preflight
    mpu.get_tensor_model_parallel_world_size = lambda: tp
    module.validate_hybridep_worker_group()
    detect.assert_called_once_with(expert)
    assert dist.all_gather_object.call_count == 2
    logger.info.assert_called_once()


@pytest.mark.parametrize("tp", [2, 8])
def test_cross_node_group_is_rejected_before_peer_detection(preflight, tp):
    module, mpu, dist, detect, _, _ = preflight
    mpu.get_tensor_model_parallel_world_size = lambda: tp

    def gather(output, value, group):
        output[:] = [dict(value, rank=i, hostname=f"node-{i // 4}") for i in range(8)]

    dist.all_gather_object.side_effect = gather
    with pytest.raises(RuntimeError, match="intranode preflight rejected"):
        module.validate_hybridep_worker_group()
    detect.assert_not_called()


@pytest.mark.parametrize("failure", ["topology", "forced_peers", "import_path", "detected_peers"])
def test_bad_group_fails_collectively(preflight, monkeypatch, failure):
    module, mpu, dist, detect, _, _ = preflight
    if failure == "topology":
        mpu.get_context_parallel_world_size = lambda: 2
    elif failure == "forced_peers":
        monkeypatch.setenv("NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN", "8")
    elif failure == "import_path":
        monkeypatch.setattr(sys.modules["deep_ep"], "__file__", "/unrelated/deep_ep.py")
    else:
        detect.return_value = 7
    with pytest.raises(RuntimeError):
        module.validate_hybridep_worker_group()
    assert dist.all_gather_object.called
    if failure != "detected_peers":
        detect.assert_not_called()
