# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""CPU checks for retention guards and invocation-local checkpoint metadata."""

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest


ROOT = Path(__file__).resolve().parents[3]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def retention(monkeypatch):
    env = load("retention_env", ROOT / "relax/utils/env.py")
    monkeypatch.setitem(sys.modules, "relax.utils.env", env)
    return load("retention_patch", ROOT / "relax/backends/megatron/megatron_patch/qsa_attention_recompute_patch.py")


def arguments():
    return SimpleNamespace(
        recompute_granularity="full",
        recompute_method="uniform",
        recompute_num_layers=1,
        tensor_model_parallel_size=8,
        pipeline_model_parallel_size=4,
        context_parallel_size=1,
        expert_model_parallel_size=8,
        expert_tensor_parallel_size=1,
        bf16=True,
    )


def test_default_disabled_does_not_import_megatron(retention, monkeypatch):
    monkeypatch.delenv("RELAX_QWEN38_RETAIN_QSA_ATTENTION", raising=False)
    assert retention.install(None, None) == []


@pytest.mark.parametrize(
    "name,value",
    [
        ("context_parallel_size", 2),
        ("recompute_num_layers", 2),
        ("bf16", False),
        ("fp8", "hybrid"),
        ("cpu_offloading", True),
        ("distribute_saved_activations", True),
        ("virtual_pipeline_model_parallel_size", 2),
        ("cuda_graph_impl", "local"),
    ],
)
def test_rejects_unsupported_recompute_configuration(retention, name, value):
    args = arguments()
    retention.validate_args(args)
    setattr(args, name, value)
    with pytest.raises(RuntimeError):
        retention.validate_args(args)


def test_checkpoint_closures_keep_each_microbatch_metadata(retention):
    original = Mock(return_value="output")
    layer = SimpleNamespace(
        training=True, config=SimpleNamespace(distribute_saved_activations=False), _forward_mlp=original
    )
    pending = []

    def checkpoint(fn, distributed, hidden, padding):
        assert distributed is False
        pending.append((fn, hidden, padding))
        return fn(hidden, padding)

    retention.wrap_mlp(layer, checkpoint)
    first, second = object(), object()
    assert layer._forward_mlp("h1", padding_mask="p1", packed_seq_params=first) == "output"
    layer._forward_mlp("h2", padding_mask="p2", packed_seq_params=second)
    fn, hidden, padding = pending[0]
    fn(hidden, padding)
    assert original.call_args.args == ("h1",)
    assert original.call_args.kwargs == dict(inference_context=None, padding_mask="p1", packed_seq_params=first)
    assert layer._relax_qsa_attention_retained


def test_eval_bypasses_checkpoint_and_training_inference_is_rejected(retention):
    original, checkpoint = Mock(), Mock()
    layer = SimpleNamespace(training=False, _forward_mlp=original)
    retention.wrap_mlp(layer, checkpoint)
    layer._forward_mlp("h", "ctx", "mask", "packed")
    original.assert_called_once_with("h", "ctx", "mask", "packed")
    checkpoint.assert_not_called()
    layer.training = True
    with pytest.raises(RuntimeError, match="training-only"):
        layer._forward_mlp("h", inference_context="ctx")
