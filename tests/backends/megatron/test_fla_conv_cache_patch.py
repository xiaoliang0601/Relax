# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""CPU regression tests for preserving launches while eliminating redundant NB
keys."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest


_PATH = Path(__file__).resolve().parents[3] / "relax/backends/megatron/megatron_patch/fla_conv_cache_patch.py"
_SPEC = importlib.util.spec_from_file_location("fla_conv_cache_patch", _PATH)
_PATCH = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_PATCH)


def _kernel(body: str = "return T"):
    cache = set()
    calls = []

    def run(*args, **kwargs):
        bound = dict(zip(["T", "NB"], args)) | kwargs
        cache.add(bound["NB"])
        calls.append(bound)
        return bound["T"]

    tuner = SimpleNamespace(
        fn=SimpleNamespace(src=f"def kernel(T, NB):\n    {body}\n", arg_names=["T", "NB"]),
        keys=["D", "W", "NB"],
        run=run,
    )
    return SimpleNamespace(fn=tuner), calls, cache


@pytest.mark.parametrize("positional", [False, True])
def test_lengths_reuse_key_without_changing_tensor_arguments_or_launch(positional):
    kernel, calls, cache = _kernel()
    grid = lambda meta: (meta["BD"], 731, 1)
    assert _PATCH._normalize_unused_nb(kernel)
    for length in (4096, 16384, 65536, 131072):
        if positional:
            result = kernel.fn.run(length, length // 1024, grid=grid)
        else:
            result = kernel.fn.run(T=length, NB=length // 1024, grid=grid)
        assert result == length
        assert calls[-1]["grid"] is grid
    assert cache == {1}
    assert not _PATCH._normalize_unused_nb(kernel)


def test_refuses_nb_used_by_kernel():
    kernel, _, _ = _kernel("return T + NB")
    original = kernel.fn.run
    with pytest.raises(RuntimeError, match="no longer unused"):
        _PATCH._normalize_unused_nb(kernel)
    assert kernel.fn.run is original


def test_refuses_changed_autotune_contract():
    kernel, _, _ = _kernel()
    kernel.fn.keys = ["D", "W", "T"]
    with pytest.raises(RuntimeError, match="Unexpected"):
        _PATCH._normalize_unused_nb(kernel)


def test_disabled_patch_does_not_import_fla(monkeypatch):
    monkeypatch.setenv("RELAX_QWEN38_FLA_CONV_CACHE_REUSE", "0")
    _PATCH.apply_fla_conv_cache_patch()


@pytest.mark.parametrize("length", [7, 65, 129])
def test_installed_fla_interpreter_nb_preserves_forward_and_gradients(length):
    """Opt-in CPU interpreter test; never launches a CUDA kernel."""
    import os

    if os.environ.get("TRITON_INTERPRET") != "1":
        pytest.skip("Run with TRITON_INTERPRET=1 and installed FLA for CPU kernel verification")
    torch = pytest.importorskip("torch")
    kernels = pytest.importorskip("fla.modules.conv.triton.kernels")
    fwd = kernels.causal_conv1d_fwd_kernel.fn.fn
    bwd = kernels.causal_conv1d_bwd_kernel.fn.fn
    torch.manual_seed(1)
    dim, width, block = 16, 4, 32
    chunks = (length + block - 1) // block
    x = torch.randn(1, length, dim)
    weight, bias, dy = torch.randn(dim, width), torch.randn(dim), torch.randn_like(x)
    results = []
    for nb in (1, 125, 128):
        y = torch.empty_like(x)
        shared = dict(
            x=x,
            weight=weight,
            cu_seqlens=None,
            initial_state=None,
            chunk_indices=None,
            B=1,
            T=length,
            stride_x_n=length * dim,
            stride_x_t=dim,
            stride_x_d=1,
            D=dim,
            W=width,
            BT=block,
            BW=4,
            BD=16,
            NB=nb,
            HAS_WEIGHT=True,
            HAS_BIAS=True,
            USE_INITIAL_STATE=False,
            IS_VARLEN=False,
        )
        fwd[(1, chunks, 1)](y=y, bias=bias, residual=None, ACTIVATION=None, HAS_RESIDUAL=False, **shared)
        dx = torch.empty_like(x)
        dw, db = torch.empty(chunks, dim, width), torch.empty(chunks, dim)
        bwd[(1, chunks, 1)](
            y=y,
            dht=None,
            dy=dy,
            dx=dx,
            dw=dw,
            db=db,
            stride_dx_n=length * dim,
            stride_dx_t=dim,
            stride_dx_d=1,
            ACTIVATION=None,
            USE_FINAL_STATE=False,
            **shared,
        )
        results.append((y, dx, dw, db))
    for result in results[1:]:
        for reference, actual in zip(results[0], result, strict=True):
            torch.testing.assert_close(reference, actual, rtol=0, atol=0)
