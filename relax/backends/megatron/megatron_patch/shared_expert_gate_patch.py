# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Publish shared-expert gate parameters before the overlap stream waits."""

from __future__ import annotations

from functools import wraps
from typing import Any


def install() -> bool:
    """Install when MCore exposes parameter readiness; preserve native
    support."""
    try:
        from megatron.core.transformer.moe.shared_experts import SharedExpertMLP
    except ImportError:
        return False

    if getattr(SharedExpertMLP, "shared_gate_param_readiness_version", None) == 1:
        return True
    try:
        from megatron.core.utils import ensure_params_ready
    except ImportError:
        # Older MCore builds lack the readiness contract. The resident-overlap
        # verifier must still reject them rather than see a fabricated capability.
        return False

    if not callable(ensure_params_ready):
        return False
    original_wait = SharedExpertMLP.wait_current_stream

    @wraps(original_wait)
    def wait_current_stream(self: Any) -> Any:
        if self.use_shared_expert_gate:
            ensure_params_ready([self.gate_weight])
        return original_wait(self)

    SharedExpertMLP.wait_current_stream = wait_current_stream
    SharedExpertMLP.shared_gate_param_readiness_version = 1
    return True
