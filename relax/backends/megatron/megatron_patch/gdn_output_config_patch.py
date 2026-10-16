# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Opt-in audited FLA output configuration for Qwen38 on the B300
deployment."""

from __future__ import annotations

import hashlib
import importlib
from pathlib import Path

from relax.utils.env import Envs


def install() -> bool:
    if not Envs.RELAX_QWEN38_GDN_OUTPUT_SAFE_CONFIG:
        return False
    module = importlib.import_module("fla.ops.common.chunk_o")
    expected_sha = "f174d431ce67139204a0916f638b5af158dd5917e2cb37c0ca03f5ce735466a0"
    if hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest() != expected_sha:
        raise RuntimeError("FLA output implementation differs from audited source")
    kernel = module.chunk_fwd_kernel_o
    tuner = kernel.fn
    if getattr(kernel, "_relax_qwen38_gdn_output_config", False):
        if (
            len(tuner.configs) != 1
            or tuner.configs[0].kwargs != {"BK": 64, "BV": 64}
            or tuner.configs[0].num_warps != 4
            or tuner.configs[0].num_stages != 3
        ):
            raise RuntimeError("Audited GDN output configuration was changed after installation")
        return True
    if (
        type(kernel).__name__ != "Heuristics"
        or type(tuner).__name__ != "CachedAutotuner"
        or type(tuner.fn).__name__ != "JITFunction"
    ):
        raise RuntimeError("Unsupported GDN output kernel decorator chain")
    configs = tuner.configs
    if len(configs) != 3 or {(c.kwargs.get("BK"), c.kwargs.get("BV"), c.num_warps, c.num_stages) for c in configs} != {
        (128, 128, 8, 3),
        (64, 64, 4, 3),
        (32, 32, 2, 3),
    }:
        raise RuntimeError("Unrecognized GDN output kernel configurations")
    chosen = next(c for c in configs if c.kwargs == {"BK": 64, "BV": 64})
    # One original configuration bypasses autotuner selection and stale disk entries.
    # Keep its complete configuration object, the heuristics, and all launch arguments.
    tuner.configs = [chosen]
    tuner.cache = {}
    if hasattr(tuner, "best_config"):
        del tuner.best_config
    kernel._relax_qwen38_gdn_output_config = True
    from relax.utils.logging_utils import get_logger

    get_logger(__name__).info("Enabled audited Qwen38 GDN output BK64 BV64 warps4 stages3")
    return True
