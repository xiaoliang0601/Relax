# Copyright (c) 2026 Relax Authors. All Rights Reserved.
# Copyright 2025-2026 RadixArk, Inc. (miles).
# SPDX-License-Identifier: Apache-2.0
"""Opt-in fused HC write; native bias and dropout paths retain their
semantics."""

import importlib
from functools import wraps

import torch

from relax.utils.env import Envs
from relax.utils.logging_utils import get_logger


def install() -> bool:
    if not Envs.RELAX_QWEN38_HC_FUSED_COMBINE:
        return False
    name = "megatron.core.transformer.hyper_connection"
    try:
        module = importlib.import_module(name)
    except ModuleNotFoundError as exc:
        if exc.name not in ("megatron", "megatron.core", "megatron.core.transformer", name):
            raise
        return False
    cls = getattr(module, "GatedResidualHyperConnection", None)
    if cls is None:
        return False
    original = cls._inject
    if getattr(original, "_relax_fused_combine", False):
        return True

    @wraps(original)
    def inject(
        residual: torch.Tensor, gates: torch.Tensor, output: torch.Tensor, bias: torch.Tensor | None, n: int
    ) -> torch.Tensor:
        supported = (
            bias is None
            and residual.is_cuda
            and residual.dtype in (torch.bfloat16, torch.float16, torch.float32)
            and output.dtype == residual.dtype
            and gates.dtype == torch.float32
            and output.device == residual.device == gates.device
            and n in (1, 4)
            and residual.ndim == output.ndim == gates.ndim == 3
            and residual.shape[:2] == output.shape[:2] == gates.shape[:2]
            and residual.shape[-1] == n * output.shape[-1]
            and 0 < output.shape[-1] <= 4096
            and gates.shape[-1] == n
        )
        if not supported:
            return original(residual, gates, output, bias, n)
        from .hc_combine_ops import combine

        return combine(residual, output, gates, n)

    inject._relax_fused_combine = True
    cls._inject = staticmethod(inject)
    get_logger(__name__).info("Enabled fused Qwen3.8 HC residual combine (bias-free, native dropout fallback)")
    return True
