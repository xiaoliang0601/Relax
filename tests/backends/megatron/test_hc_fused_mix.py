# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""HC fusion parity, checkpoint, parameter-publication and fallback checks."""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch
from torch.utils.checkpoint import checkpoint


ROOT = Path(__file__).resolve().parents[3]
SPEC = importlib.util.spec_from_file_location(
    "hc_fused_mix_under_test", ROOT / "relax/backends/megatron/megatron_patch/hc_fused_mix_patch.py"
)
patch = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(patch)
gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="HC compilation requires CUDA")


def test_hc_fusion_disabled_does_not_import_megatron(monkeypatch):
    monkeypatch.setenv("RELAX_QWEN38_HC_FUSED_MIX", "0")
    monkeypatch.setitem(sys.modules, "megatron", None)
    assert not patch.install()


def test_hc_fusion_default_skips_missing_megatron(monkeypatch):
    monkeypatch.delenv("RELAX_QWEN38_HC_FUSED_MIX", raising=False)
    monkeypatch.setitem(sys.modules, "megatron", None)
    monkeypatch.setitem(sys.modules, "megatron.core.transformer.hyper_connection", None)
    assert not patch.install()


def test_hc_fusion_install_is_idempotent_and_cpu_falls_back(monkeypatch):
    monkeypatch.delenv("RELAX_QWEN38_HC_FUSED_MIX", raising=False)
    calls = []

    class HC:
        def _mix(self, x):
            calls.append(x)
            return x, None

    original = HC._mix
    core = ModuleType("megatron.core")
    core.utils = SimpleNamespace()
    hc_module = ModuleType("megatron.core.transformer.hyper_connection")
    hc_module.GatedResidualHyperConnection = HC
    monkeypatch.setitem(sys.modules, "megatron", ModuleType("megatron"))
    monkeypatch.setitem(sys.modules, "megatron.core", core)
    monkeypatch.setitem(sys.modules, hc_module.__name__, hc_module)
    compiled = []
    monkeypatch.setattr(torch, "compile", lambda fn, **kw: compiled.append(fn) or fn)
    assert patch.install()
    installed = HC._mix
    assert patch.install()
    assert HC._mix is installed and not compiled
    assert installed.__wrapped__ is original
    x = torch.ones(2, 1, 8)
    assert HC()._mix(x) == (x, None)
    assert calls == [x]
    assert not compiled


@pytest.mark.parametrize("module_exists", [False, True])
def test_hc_fusion_default_skips_mcore_without_gated_residual(monkeypatch, module_exists):
    monkeypatch.delenv("RELAX_QWEN38_HC_FUSED_MIX", raising=False)
    core = ModuleType("megatron.core")
    core.utils = SimpleNamespace()
    monkeypatch.setitem(sys.modules, "megatron", ModuleType("megatron"))
    monkeypatch.setitem(sys.modules, "megatron.core", core)
    name = "megatron.core.transformer.hyper_connection"
    monkeypatch.setitem(sys.modules, name, ModuleType(name) if module_exists else None)
    assert not patch.install()


@pytest.fixture
def runtime(monkeypatch):
    hc = pytest.importorskip("megatron.core.transformer.hyper_connection")
    from megatron.core.transformer.transformer_config import TransformerConfig

    cls = hc.GatedResidualHyperConnection
    original = cls._mix
    # Restore the class after each test, including when an assertion fails.
    monkeypatch.setattr(cls, "_mix", original)
    monkeypatch.setenv("RELAX_QWEN38_HC_FUSED_MIX", "1")
    assert patch.install()

    def make(combine=True, dtype=torch.bfloat16, hidden_size=128, rank=32):
        config = TransformerConfig(
            num_layers=2,
            hidden_size=hidden_size,
            num_attention_heads=4,
            params_dtype=dtype,
            sequence_parallel=True,
            tensor_model_parallel_size=2,
            mhc_num_residual_streams=4,
            mhc_gated_residual_rank=rank,
        )
        model = cls(config, use_combine=combine).cuda()
        with torch.no_grad():
            model.hc_norm.weight.normal_(0, 0.1)
        return model

    return make, original


def _close(actual, expected):
    assert torch.isfinite(actual).all()
    # BF16 reduction order may differ; check aggregate error as well as outliers.
    a, b = actual.float(), expected.float()
    relative = (a - b).norm() / b.norm().clamp_min(1e-12)
    assert relative < 0.005, relative
    torch.testing.assert_close(a, b, atol=0.008, rtol=0.04)


@gpu
@pytest.mark.parametrize("combine", [False, True])
@pytest.mark.parametrize("reentrant", [None, False, True])
def test_hc_fusion_forward_backward_and_outer_checkpoint(runtime, combine, reentrant):
    make, original = runtime
    torch.manual_seed(17)
    model = make(combine)
    params = list(model.parameters())
    for p in params:
        p.main_grad = torch.zeros_like(p, dtype=torch.float32)
    identity = [(id(p), p.data_ptr(), id(p.main_grad), p.sequence_parallel) for p in params]
    schema = {name: (p.shape, p.dtype) for name, p in model.state_dict().items()}
    # Multiple lengths, batch > 1, and noncontiguous inputs use the same wrapper.
    for rows in (17, 31):
        x = torch.randn(rows * 2, 2, 512, device="cuda", dtype=torch.bfloat16)[::2].requires_grad_()
        upstream = torch.randn(rows, 2, 128, device="cuda", dtype=torch.bfloat16)
        post_grad = torch.randn(rows, 2, 4, device="cuda")

        def loss(fn, value):
            block, post = fn(model, value)
            result = (block * upstream).sum()
            return result if post is None else result + (post * post_grad).sum()

        expected = original(model, x)
        reference = torch.autograd.grad(loss(original, x), [x, *params])
        actual = model._mix(x)
        for a, b in zip(actual, expected):
            if b is None:
                assert a is None
            else:
                _close(a, b)
        fn = lambda value: loss(type(model)._mix, value)
        result = fn(x) if reentrant is None else checkpoint(fn, x, use_reentrant=reentrant)
        result.backward()
        for value, expected_grad in zip([x, *params], reference):
            _close(value.grad, expected_grad)
            value.grad = None
    assert identity == [(id(p), p.data_ptr(), id(p.main_grad), p.sequence_parallel) for p in params]
    assert schema == {name: (p.shape, p.dtype) for name, p in model.state_dict().items()}


@gpu
def test_hc_fusion_publishes_params_and_preserves_hooks(runtime):
    make, original = runtime
    from megatron.core import utils

    if not hasattr(utils, "ensure_params_ready"):
        pytest.skip("MCore does not expose parameter publication")
    model = make()
    x = torch.randn(23, 1, 512, device="cuda", dtype=torch.bfloat16)
    calls = []

    def ready():
        calls.append("ready")

    for p in model.parameters():
        setattr(p, utils.PARAM_READY_CALLBACK_ATTR, ready)

    def ddp_hook(*args):
        raise AssertionError("Native DDP hook should be replaced by explicit publication")

    ddp_hook.__module__ = "megatron.core.distributed.distributed_data_parallel"
    ddp_hook.__qualname__ = "DistributedDataParallel._make_forward_pre_hook.<locals>.hook"
    handle = model.input_mix_weight_down.register_forward_pre_hook(ddp_hook)
    assert patch._supported(model, x, utils.ensure_params_ready)
    model._mix(x)
    assert calls == ["ready"]  # Same callback shared by all four parameters.
    handle.remove()

    def change_input(module, args):
        calls.append("custom")
        return (args[0] * 0.5,)

    handle = model.input_mix_weight_down.register_forward_pre_hook(change_input)
    assert not patch._supported(model, x, utils.ensure_params_ready)
    expected = original(model, x)
    actual = model._mix(x)
    assert calls == ["ready", "custom", "custom"]
    for a, b in zip(actual, expected):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    handle.remove()


@gpu
def test_hc_fusion_fp32_fallback_and_output_mixer(runtime):
    make, original = runtime
    model = make(dtype=torch.float32)
    x = torch.randn(7, 1, 512, device="cuda")
    assert not patch._supported(model, x, None)
    for actual, expected in zip(model._mix(x), original(model, x)):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    from megatron.core.transformer.hyper_connection import GatedResidualOutputMixer

    assert GatedResidualOutputMixer._mix is type(model)._mix


@gpu
@pytest.mark.parametrize("combine", [False, True])
def test_hc_fusion_production_width_dynamic_tokens(runtime, combine):
    make, original = runtime
    torch.manual_seed(42)
    model = make(combine, hidden_size=2560, rank=320)
    for rows in (11, 19):
        x = torch.randn(rows, 1, 10240, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        expected = tuple(t for t in original(model, x) if t is not None)
        actual = tuple(t for t in model._mix(x) if t is not None)
        upstream = tuple(torch.randn_like(t) * 0.01 for t in expected)
        expected_grads = torch.autograd.grad(expected, [x, *model.parameters()], grad_outputs=upstream)
        actual_grads = torch.autograd.grad(actual, [x, *model.parameters()], grad_outputs=upstream)
        for a, b in zip((*actual, *actual_grads), (*expected, *expected_grads)):
            _close(a, b)
