# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Opt-in fixed-order FP64 pooling for frozen R4 packed QSA indexers."""

from __future__ import annotations

from typing import Any

import torch
import triton
import triton.language as tl

from relax.utils.env import Envs


@triton.jit
def _pool_r4(
    raw,
    offsets,
    output,
    TOTAL: tl.constexpr,
    COLUMNS: tl.constexpr,
    D: tl.constexpr,
    S0: tl.constexpr,
    S1: tl.constexpr,
    BLOCK: tl.constexpr,
):
    index = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    channel = index % D
    block = (index // D) % COLUMNS
    doc = index // (D * COLUMNS)
    active = index < TOTAL
    start = tl.load(offsets + doc, active, 0)
    end = tl.load(offsets + doc + 1, active, 0)
    complete = active & (4 * (block + 1) <= end - start)
    first = start + 4 * block
    a = tl.load(raw + first * S0 + channel * S1, complete, 0).to(tl.float64)
    b = tl.load(raw + (first + 1) * S0 + channel * S1, complete, 0).to(tl.float64)
    c = tl.load(raw + (first + 2) * S0 + channel * S1, complete, 0).to(tl.float64)
    d = tl.load(raw + (first + 3) * S0 + channel * S1, complete, 0).to(tl.float64)
    result = ((a + b) + c) + d
    tl.store(output + index, result * 0.25, active)


def fixed_pool(raw_keys: torch.Tensor, offsets: tuple[int, ...], max_doc_len: int) -> tuple:
    """Packed documents in increasing ID order; CPU offsets are the layout
    authority."""
    if (
        raw_keys.ndim != 2
        or raw_keys.dtype != torch.bfloat16
        or not raw_keys.is_cuda
        or raw_keys.shape[0] <= 0
        or raw_keys.shape[1] <= 0
        or any(s <= 0 for s in raw_keys.stride())
        or raw_keys.requires_grad
    ):
        raise ValueError("Fixed QSA pool requires a frozen CUDA BF16 matrix")
    if (
        not isinstance(offsets, tuple)
        or len(offsets) < 2
        or offsets[0] != 0
        or offsets[-1] != raw_keys.shape[0]
        or any(type(x) is not int for x in offsets)
        or any(a > b for a, b in zip(offsets, offsets[1:]))
        or max_doc_len != max(b - a for a, b in zip(offsets, offsets[1:]))
    ):
        raise ValueError("Invalid packed QSA offsets")
    # Match max(doc_ids)+1: trailing empty documents have no token IDs.
    while offsets[-2] == offsets[-1]:
        offsets = offsets[:-1]
    n_docs, columns, dim = len(offsets) - 1, max_doc_len // 4, raw_keys.shape[1]
    bounds = torch.tensor(offsets, device=raw_keys.device, dtype=torch.int64)
    positions = (torch.arange(columns, device=raw_keys.device, dtype=torch.long) * 4).unsqueeze(0).expand(n_docs, -1)
    valid = torch.arange(columns, device=raw_keys.device).unsqueeze(0) < ((bounds[1:] - bounds[:-1]) // 4).unsqueeze(1)
    pooled = torch.empty((n_docs, columns, dim), device=raw_keys.device, dtype=raw_keys.dtype)
    if pooled.numel():
        _pool_r4[(triton.cdiv(pooled.numel(), 256),)](
            raw_keys,
            bounds,
            pooled,
            pooled.numel(),
            columns,
            dim,
            raw_keys.stride(0),
            raw_keys.stride(1),
            256,
            num_warps=4,
            enable_fp_fusion=False,
        )
    return pooled, positions, valid


def install() -> None:
    """Install after the fused selector; unsupported opt-in configurations fail
    closed."""
    if not Envs.RELAX_QWEN38_QSA_FIXED_POOL:
        return
    if not Envs.RELAX_QWEN38_QSA_FUSED_SELECTOR:
        raise RuntimeError("QSA fixed pooling requires RELAX_QWEN38_QSA_FUSED_SELECTOR=1")

    import hashlib
    import inspect
    from pathlib import Path

    from megatron.core.transformer.experimental_attention_variant import qsa

    from relax.utils.logging_utils import get_logger

    from .qsa_select_patch import _layout

    original = qsa.QSAIndexer._pool_keys
    if getattr(original, "_qsa_fixed_pool_v1", False):
        return
    source = Path(inspect.getsourcefile(original))
    if (
        hashlib.sha256(source.read_bytes()).hexdigest()
        != "38e2cb7a6d3d61f3b96a38894605b16884d330944cb097d0e667eb3f131e9137"
    ):
        raise RuntimeError("QSA fixed pool requires the audited original implementation")

    def pool_keys(
        self: Any, raw_keys: torch.Tensor, doc_ids: torch.Tensor, positions: torch.Tensor, max_doc_len: int
    ) -> tuple:
        offsets = _layout.get()
        if (
            self.compress_ratio != 4
            or self.config.qsa_indexer_loss_coeff != 0
            or self.config.context_parallel_size != 1
            or any(p.requires_grad for p in self.parameters())
            or doc_ids.shape != (raw_keys.shape[0],)
            or positions.shape != doc_ids.shape
            or doc_ids.device != raw_keys.device
            or positions.device != raw_keys.device
        ):
            raise RuntimeError("Fixed QSA pool requires frozen packed R4 CP1 indexer without auxiliary loss")
        return fixed_pool(raw_keys, offsets, max_doc_len)

    pool_keys._qsa_fixed_pool_v1 = True
    qsa.QSAIndexer._pool_keys = pool_keys
    get_logger(__name__).info("QSA fixed-order FP64 R4 pooling enabled for frozen packed CP1 indexer")
