# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Reuse FLA convolution compilation/autotuning across packed token lengths."""

from __future__ import annotations

import ast
from functools import wraps
from typing import Any

from relax.utils.env import Envs


def _normalize_unused_nb(kernel: Any) -> bool:
    """Normalize only the unused NB argument, retaining the original launch
    grid.

    FLA passes ceil(B*T/1024) as an unused constexpr and autotune key. Every
    packed length then compiles and benchmarks 16 identical configurations.
    Patch the autotuner object in place so existing imports retain this fix.
    """
    tuner = kernel.fn
    if getattr(tuner, "_relax_normalized_nb", False):
        return False
    jit = tuner.fn
    function = ast.parse(jit.src).body[0]
    if "NB" not in jit.arg_names or any(
        isinstance(node, ast.Name) and node.id == "NB" for statement in function.body for node in ast.walk(statement)
    ):
        raise RuntimeError("FLA convolution NB is no longer unused; refusing cache normalization")
    if list(tuner.keys) != ["D", "W", "NB"]:
        raise RuntimeError("Unexpected FLA convolution autotune keys; refusing cache normalization")
    nb_position = jit.arg_names.index("NB")
    original_run = tuner.run

    @wraps(original_run)
    def run(*args: Any, **kwargs: Any) -> Any:
        if len(args) > nb_position:
            args = (*args[:nb_position], 1, *args[nb_position + 1 :])
        elif "NB" in kwargs:
            kwargs["NB"] = 1
        return original_run(*args, **kwargs)

    tuner.run = run
    tuner._relax_normalized_nb = True
    return True


def apply_fla_conv_cache_patch() -> None:
    """Opt in at worker startup, before the first FLA kernel launch."""
    if not Envs.RELAX_QWEN38_FLA_CONV_CACHE_REUSE:
        return
    from fla.modules.conv.triton import kernels

    from relax.utils.logging_utils import get_logger

    changed = sum(
        _normalize_unused_nb(getattr(kernels, name))
        for name in ("causal_conv1d_fwd_kernel", "causal_conv1d_bwd_kernel")
    )
    if changed:
        get_logger(__name__).info("FLA convolution cache reuse: normalized unused NB in %s kernels", changed)
