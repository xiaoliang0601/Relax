# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Opt-in QSA selector fusion preserving FP32 GEMM, TopK and route bits."""

from __future__ import annotations

import math
from contextvars import ContextVar
from typing import Any

import torch
import triton
import triton.language as tl
from megatron.core.transformer.experimental_attention_variant import qsa
from torch import Tensor

from relax.utils.env import Envs
from relax.utils.logging_utils import get_logger


logger = get_logger(__name__)

_layout: ContextVar[tuple[int, ...] | None] = ContextVar("qsa_select_layout", default=None)
# Megatron's 64 MiB budget sizes a tile from the widest document, which now
# over-reserves by the narrowing factor and cuts every document into several
# times more GEMM, TopK and Triton launches than the budget needs. Raising the
# budget reduces launch count while preserving route bytes. The tile is
# clamped per step.
_TILE_BYTES = 256 << 20
_original_select = qsa.QSAIndexer._select_blocks
_original_forward = qsa.QSAIndexer.forward


# Packed sample lengths vary within and across steps. Keep indexing sizes out of
# the compilation key; only tile geometry and algorithm constants specialize.
@triton.jit(do_not_specialize=["columns", "rows"])
def _score_kernel(
    scores,
    positions,
    output,
    columns,
    rows,
    ratio: tl.constexpr,
    scale: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    row = offsets // columns
    col = offsets % columns
    valid = row < rows
    base = row * 4 * columns + col
    a = tl.load(scores + base, valid, other=0)
    b = tl.load(scores + base + columns, valid, other=0)
    c = tl.load(scores + base + 2 * columns, valid, other=0)
    d = tl.load(scores + base + 3 * columns, valid, other=0)
    # ATen adds the reduction identity +0 to each head, normalizing -0.
    a = tl.where(a <= 0, 0.0, a)
    b = tl.where(b <= 0, 0.0, b)
    c = tl.where(c <= 0, 0.0, c)
    d = tl.where(d <= 0, 0.0, d)
    # Match ATen's sequential H=4 reduction and scalar-divide reciprocal multiply.
    value = ((a + b) + c) + d
    value = value * scale
    visible = (tl.load(positions + row, valid, other=0).to(tl.int64) + 1) // ratio
    value = tl.where(col < visible, value, float("-inf"))
    tl.store(output + offsets, value, valid)


def score_postprocess(scores: Tensor, positions: Tensor, ratio: int, dim: int) -> Tensor:
    rows, heads, columns = scores.shape
    assert heads == 4 and scores.dtype == torch.float32 and scores.is_contiguous()
    output = torch.empty((rows, columns), device=scores.device, dtype=torch.float32)
    _score_kernel[(triton.cdiv(rows * columns, 256),)](
        scores,
        positions,
        output,
        columns,
        rows,
        ratio,
        1.0 / math.sqrt(dim),
        256,
        enable_fp_fusion=False,
    )
    return output


@triton.jit(do_not_specialize=["columns", "nbytes"])
def _pack_all_candidates_kernel(
    scores, positions, output, columns, nbytes, ratio: tl.constexpr, scale: tl.constexpr, BLOCK: tl.constexpr
):
    row = tl.program_id(0).to(tl.int64)
    byte_start = tl.program_id(1) * BLOCK
    byte = byte_start + tl.arange(0, BLOCK)
    result = tl.full((BLOCK,), 0, tl.int32)
    # The output keeps the full document stride, including zero padding beyond
    # this tile's candidate range. Those CTAs need no score or position loads.
    if byte_start * 8 < columns:
        bit = tl.arange(0, 8)
        col = byte[:, None] * 8 + bit[None, :]
        valid = col < columns
        base = row * 4 * columns + col
        a = tl.load(scores + base, valid, other=0)
        b = tl.load(scores + base + columns, valid, other=0)
        c = tl.load(scores + base + 2 * columns, valid, other=0)
        d = tl.load(scores + base + 3 * columns, valid, other=0)
        # Keep exactly the same arithmetic as _score_kernel, including overflow
        # and NaN propagation. TopK(k=columns) followed by packing retains every
        # finite causal score, regardless of its rank.
        a = tl.where(a <= 0, 0.0, a)
        b = tl.where(b <= 0, 0.0, b)
        c = tl.where(c <= 0, 0.0, c)
        d = tl.where(d <= 0, 0.0, d)
        value = (((a + b) + c) + d) * scale
        visible = (tl.load(positions + row).to(tl.int64) + 1) // ratio
        finite = (tl.abs(value) != float("inf")) & (value == value)
        selected = valid & (col < visible) & finite
        result = tl.sum(selected.to(tl.int32) * (1 << bit[None, :]), 1)
    tl.store(output + row * nbytes + byte, result.to(tl.uint8), byte < nbytes)


def pack_all_candidates(scores: Tensor, positions: Tensor, ratio: int, dim: int, output: Tensor) -> None:
    """Fuse score processing and packing when TopK retains every candidate."""
    rows, heads, columns = scores.shape
    assert heads == 4 and scores.dtype == torch.float32 and scores.is_contiguous()
    nbytes = output.shape[1]
    if rows == 0 or nbytes == 0:
        return
    _pack_all_candidates_kernel[(rows, triton.cdiv(nbytes, 64))](
        scores, positions, output, columns, nbytes, ratio, 1.0 / math.sqrt(dim), 64, enable_fp_fusion=False
    )


@triton.jit
def _topk_mask(keys, valid, k: tl.constexpr):
    # Search ordered uint32 keys from the most significant bit. An exact count
    # can stop between adjacent scores: the chosen set already contains k IDs.
    threshold = tl.full((), 0, tl.uint32)
    shift = 31
    count = -1
    while (shift >= 0) & (count != k):
        proposed = threshold | (tl.full((), 1, tl.uint32) << shift)
        count = tl.sum((valid & (keys >= proposed)).to(tl.int32), 0)
        threshold = tl.where(count >= k, proposed, threshold)
        shift -= 1
    if count == k:
        chosen = valid & (keys >= threshold)
    else:
        # Match ATen CUDA radix TopK: fill cutoff ties in original column order.
        greater = valid & (keys > threshold)
        equal = valid & (keys == threshold)
        remaining = k - tl.sum(greater.to(tl.int32), 0)
        chosen = greater | (equal & (tl.cumsum(equal.to(tl.int32), 0) <= remaining))
    return chosen


@triton.jit(do_not_specialize=["columns", "nbytes"])
def _score_topk_pack_kernel(
    scores,
    positions,
    output,
    columns,
    nbytes,
    ratio: tl.constexpr,
    scale: tl.constexpr,
    k: tl.constexpr,
    BLOCK: tl.constexpr,
    BYTES: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    col = tl.arange(0, BLOCK)
    valid = col < columns
    base = row * 4 * columns + col
    a = tl.load(scores + base, valid, other=0)
    b = tl.load(scores + base + columns, valid, other=0)
    c = tl.load(scores + base + 2 * columns, valid, other=0)
    d = tl.load(scores + base + 3 * columns, valid, other=0)
    a = tl.where(a <= 0, 0.0, a)
    b = tl.where(b <= 0, 0.0, b)
    c = tl.where(c <= 0, 0.0, c)
    d = tl.where(d <= 0, 0.0, d)
    value = (((a + b) + c) + d) * scale
    visible = (tl.load(positions + row).to(tl.int64) + 1) // ratio
    value = tl.where(col < visible, value, float("-inf"))
    if visible > k:
        # ATen TopKTypeConfig<float>: NaNs rank above +Inf; signed zeros have
        # distinct keys. Nonfinite scores consume slots BEFORE finite filtering.
        raw = value.to(tl.uint32, bitcast=True)
        keys = raw ^ tl.where((raw & 0x80000000) != 0, 0xFFFFFFFF, 0x80000000).to(tl.uint32)
        keys = tl.where(value == value, keys, 0xFFFFFFFF).to(tl.uint32)
        keys = tl.where(valid, keys, 0)
        chosen = _topk_mask(keys, valid, k)
        selected = chosen & (keys > 0x007FFFFF) & (keys < 0xFF800000)
    else:
        # A row can see <=k blocks even in a wider tile. Keep its original GEMM
        # geometry, but avoid selection when all visible scores are retained.
        selected = valid & (col < visible) & (value == value) & (tl.abs(value) != float("inf"))
    grouped = tl.reshape(selected, (BLOCK // 8, 8))
    packed = tl.sum(grouped.to(tl.int32) * (1 << tl.arange(0, 8)[None, :]), 1)
    byte = tl.arange(0, BLOCK // 8)
    used = (columns + 7) // 8
    tl.store(output + row * nbytes + byte, packed.to(tl.uint8), byte < used)
    padding = tl.arange(0, BYTES)
    tl.store(output + row * nbytes + padding, 0, (padding >= used) & (padding < nbytes))


def score_topk_pack(scores: Tensor, positions: Tensor, ratio: int, dim: int, k: int, output: Tensor) -> None:
    """Fuse H4 score processing, exact CUDA radix selection and bitset
    packing."""
    rows, heads, columns = scores.shape
    assert heads == 4 and scores.dtype == torch.float32 and scores.is_contiguous()
    assert 0 <= k <= columns <= 32768
    if rows == 0:
        return
    if k == 0:
        output.zero_()
        return
    _score_topk_pack_kernel[(rows,)](
        scores,
        positions,
        output,
        columns,
        output.shape[1],
        ratio,
        1.0 / math.sqrt(dim),
        k,
        max(8, triton.next_power_of_2(columns)),
        triton.next_power_of_2(output.shape[1]),
        num_warps=4 if columns <= 8192 else (8 if columns <= 16384 else 16),
        enable_fp_fusion=False,
    )


@triton.jit(do_not_specialize=["k", "nbytes", "word_count"])
def _pack_kernel(
    ids, values, words, output, k, nbytes, word_count, BLOCK: tl.constexpr, WORDS: tl.constexpr, BYTES: tl.constexpr
):
    # Each row owns one CTA and disjoint scratch. Both barriers are needed:
    # initialize before any warp's atomic OR, then finish all ORs before reading.
    row = tl.program_id(0).to(tl.int64)
    word_offsets = tl.arange(0, WORDS)
    tl.store(words + row * word_count + word_offsets, 0, word_offsets < word_count)
    tl.debug_barrier()
    col = tl.arange(0, BLOCK)
    value = tl.load(values + row * k + col, col < k, other=float("-inf"))
    block = tl.load(ids + row * k + col, col < k, other=0)
    valid = (col < k) & (tl.abs(value) != float("inf")) & (value == value)
    bits = (1 << (block % 32)).to(tl.uint32)
    tl.atomic_or(words + row * word_count + block // 32, bits, mask=valid, sem="relaxed")
    tl.debug_barrier()
    byte = tl.arange(0, BYTES)
    word = tl.load(words + row * word_count + byte // 4, byte < nbytes, other=0).to(tl.uint32)
    value = (word >> (8 * (byte % 4))) & 255
    tl.store(output + row * nbytes + byte, value.to(tl.uint8), byte < nbytes)


def pack_topk(ids: Tensor, values: Tensor, output: Tensor) -> None:
    """Pack contiguous TopK results into the original, unpadded byte stride.

    Finite entries must have nonnegative block IDs within the output bitset.
    Scratch is private to each call, including calls on different CUDA streams.
    """
    rows, k = ids.shape
    nbytes = output.shape[1]
    if rows == 0 or nbytes == 0:
        return
    if k == 0:
        output.zero_()
        return
    word_count = triton.cdiv(nbytes, 4)
    words = torch.empty((rows, word_count), device=ids.device, dtype=torch.int32)
    _pack_kernel[(rows,)](
        ids,
        values,
        words,
        output,
        k,
        nbytes,
        word_count,
        triton.next_power_of_2(k),
        triton.next_power_of_2(word_count),
        triton.next_power_of_2(nbytes),
        num_ctas=1,
    )


@torch.no_grad()
def select_blocks(
    self: qsa.QSAIndexer,
    q: Tensor,
    pooled_keys: Tensor,
    block_valid: Tensor,
    doc_ids: Tensor,
    positions: Tensor,
    uniform_doc_len: int | None = None,
    output_format: str = "bits",
) -> tuple[Tensor, bool]:
    total, heads, dim = q.shape
    n_docs, columns, _ = pooled_keys.shape
    offsets = _layout.get()
    if n_docs == 1:
        offsets = (0, total)
    elif uniform_doc_len is not None:
        offsets = tuple(i * uniform_doc_len for i in range(n_docs + 1))
    if (
        total == 0
        or n_docs == 0
        or output_format != "bits"
        or heads != 4
        or not q.is_cuda
        or not positions.is_contiguous()
        or offsets is None
        or not isinstance(offsets, tuple)
        or any(type(v) is not int for v in offsets)
        or any(b < a for a, b in zip(offsets, offsets[1:]))
        or len(offsets) != n_docs + 1
        or offsets[0] != 0
        or offsets[-1] != total
        or columns <= 1
        or max(b - a for a, b in zip(offsets, offsets[1:])) // self.compress_ratio <= self.block_topk
    ):
        return _original_select(self, q, pooled_keys, block_valid, doc_ids, positions, uniform_doc_len, output_format)
    nbytes = (columns + 8) // 8
    k = min(self.block_topk, columns)
    gathered_bytes = (
        columns * dim * (pooled_keys.element_size() + 4)
        + columns * (heads * 4 + 8)
        + heads * dim * 4
        + nbytes * 4
        + k * 32
    )
    gathered_tile = max(1, min(total, qsa._QSA_SELECT_TILE_BYTES // gathered_bytes))
    if n_docs > 1 and gathered_tile >= min(total, 32) and total <= n_docs * gathered_tile:
        return _original_select(self, q, pooled_keys, block_valid, doc_ids, positions, uniform_doc_len, output_format)
    # Like the original selector, rely on _pool_keys producing a valid prefix
    # of floor(doc_length / R) complete blocks and document-relative positions.
    # Causal visibility never exceeds that prefix, so invalid pooled-key tails
    # cannot be selected. block_valid is not an arbitrary attention/padding mask.
    selected = torch.empty((total, nbytes), dtype=torch.uint8, device=q.device)
    # H800 and B300 use the same kernel; wider rows benefit from 16 warps
    # on B300. Retain ATen beyond each measured limit and on other devices.
    capability = torch.cuda.get_device_capability(q.device)
    fused_topk_limit = 32768 if capability == (10, 3) else (16384 if capability == (9, 0) else 0)
    # The tile only allocates as many columns as the tile can see, so sizing it
    # from the full ``columns`` bound leaves it far under budget and splits each
    # document into several times more GEMM, TopK and Triton launches than the
    # budget allows. Size from the widest narrow step instead; the loop below
    # clamps every tile to its own ``visible`` count anyway.
    widest = min(columns, max(1, max(b - a for a, b in zip(offsets, offsets[1:])) // self.compress_ratio))
    per_query = 4 * (widest * heads + heads * dim + nbytes)
    tile = max(1, min(total, _TILE_BYTES // max(1, per_query)))
    k = min(self.block_topk, columns)
    for start_doc, end_doc, doc in zip(offsets, offsets[1:], range(n_docs)):
        if start_doc == end_doc:
            continue
        key_t = pooled_keys[doc].float().T
        for start in range(start_doc, end_doc, tile):
            end = min(end_doc, start + tile)
            # Positions are document-relative, so the last row of the tile bounds
            # how many complete blocks any row can see. score_postprocess masks
            # every column past that bound to -inf, so narrowing the GEMM and the
            # TopK to it leaves every retained score and route bit unchanged.
            visible = min(columns, (end - start_doc) // self.compress_ratio)
            if visible == 0:
                selected[start:end].zero_()
                continue
            query = q[start:end].float().reshape(-1, dim)
            scores = (query @ key_t[:, :visible]).view(end - start, heads, visible)
            if visible <= k:
                pack_all_candidates(scores, positions[start:end], self.compress_ratio, dim, selected[start:end])
                continue
            if 0 < k and visible <= fused_topk_limit:
                score_topk_pack(scores, positions[start:end], self.compress_ratio, dim, k, selected[start:end])
                # Do not retain the H4 buffer while the next tile's GEMM allocates.
                del scores
                continue
            scores = score_postprocess(scores, positions[start:end], self.compress_ratio, dim)
            # Bits consume the selected set, not its rank order. CUDA TopK
            # gathers the same entries before the optional result sort.
            top = torch.topk(scores, k=min(k, visible), dim=-1, sorted=False)
            pack_topk(top.indices, top.values, selected[start:end])
    return selected, False


def indexer_forward(
    self: qsa.QSAIndexer,
    hidden_states: Tensor,
    rotary_pos_emb: Any,
    packed_seq_params: Any = None,
    output_format: str = "bits",
) -> qsa.QSASelection:
    offsets = getattr(packed_seq_params, "qsa_select_offsets_cpu", None)
    token = _layout.set(offsets)
    try:
        return _original_forward(self, hidden_states, rotary_pos_emb, packed_seq_params, output_format)
    finally:
        _layout.reset(token)


if Envs.RELAX_QWEN38_QSA_FUSED_SELECTOR:
    qsa.QSAIndexer._select_blocks = select_blocks
    qsa.QSAIndexer.forward = indexer_forward
    logger.info("QSA fused selector enabled (visible-range tiles, FP32 H4 scores, exact bitset packing)")
