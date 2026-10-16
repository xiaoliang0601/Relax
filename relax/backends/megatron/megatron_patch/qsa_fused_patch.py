# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Opt-in installation of the fused QSA attention kernel.

Replaces the FlexAttention call, not the routing: the frozen indexer, the R=4
bitset and the 128x128 classifier are reused unchanged, so a run can be moved
back to the validated FlexAttention path by clearing the flag alone.
"""

from __future__ import annotations

from typing import Any

from megatron.core.transformer.experimental_attention_variant import qsa
from torch import Tensor

from relax.utils.env import Envs
from relax.utils.logging_utils import get_logger

from . import qsa_cp_patch, qsa_fused_attention


logger = get_logger(__name__)

_original_flex_forward = qsa.QSACoreAttention._flex_forward
_backend = qsa_fused_attention


def _fused_flex_forward(
    self: qsa.QSACoreAttention,
    query: Tensor,
    key: Tensor,
    value: Tensor,
    selection: Any,
    is_thd: bool,
) -> Tensor:
    """Single-rank QSA attention; CP local queries go through qsa_cp_patch."""
    if not is_thd and query.shape[1] != 1:
        return _original_flex_forward(self, query, key, value, selection, is_thd)
    q, k, v = (x if is_thd else x.squeeze(1) for x in (query, key, value))
    if not _backend.supports(selection, q, k, self.sparse_backend):
        return _original_flex_forward(self, query, key, value, selection, is_thd)
    plan = qsa_cp_patch._cached("fused_plan", self, lambda: _backend.build_plan(selection, q.shape[0]))
    output = _backend.qsa_fused_attention(q, k, v, selection, plan, None, self.softmax_scale)
    sequence_length, heads, width = output.shape
    if is_thd:
        return output.reshape(sequence_length, heads * width)
    return output.reshape(sequence_length, 1, heads * width)


if Envs.RELAX_QWEN38_QSA_FUSED_ATTENTION:
    if Envs.RELAX_QWEN38_QSA_FUSED_ATTENTION == "triton_v2":
        from . import qsa_triton_v2

        _backend = qsa_triton_v2
    elif Envs.RELAX_QWEN38_QSA_FUSED_ATTENTION == "triton_v3":
        from . import qsa_triton_v3

        _backend = qsa_triton_v3
    qsa.QSACoreAttention._flex_forward = _fused_flex_forward
    qsa_cp_patch._fused_backend = _backend
    logger.info("Fused QSA attention enabled: %s (bitset-masked tiles, deterministic dK/dV)", _backend.__name__)
