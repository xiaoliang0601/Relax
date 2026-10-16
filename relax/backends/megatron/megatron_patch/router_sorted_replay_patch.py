# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Opt-in Qwen38 FP32 top10 routing consistency across checkpoint grad
modes."""

from __future__ import annotations

import functools
from typing import Any, Callable

from relax.utils.env import Envs


def make_wrapper(original: Callable[..., Any]) -> Callable[..., Any]:
    """Use the already-supported precomputed-index path only for the audited
    case."""

    @functools.wraps(original)
    def wrapped(
        logits: Any,
        topk: int,
        use_pre_softmax: bool = False,
        num_groups: int | None = None,
        group_topk: int | None = None,
        scaling_factor: float | None = None,
        score_function: str = "softmax",
        expert_bias: Any = None,
        fused: bool = False,
        router_replay: Any = None,
        dense_output: bool = False,
        precomputed_indices: Any = None,
    ) -> Any:
        import torch

        if (
            not torch.is_grad_enabled()
            and precomputed_indices is None
            and router_replay is None
            and not fused
            and not dense_output
            and not use_pre_softmax
            and not num_groups
            and not group_topk
            and score_function == "softmax"
            and expert_bias is None
            and scaling_factor is None
            and topk == 10
            and logits.ndim == 2
            and logits.shape[1] == 512
            and logits.dtype == torch.float32
            and logits.is_cuda
        ):
            # The installed MCore path sorts in grad mode but not in no_grad.
            # Fixed selected-score order makes FP32 softmax reduction identical.
            precomputed_indices = torch.topk(logits, k=topk, dim=1, sorted=True).indices
        return original(
            logits=logits,
            topk=topk,
            use_pre_softmax=use_pre_softmax,
            num_groups=num_groups,
            group_topk=group_topk,
            scaling_factor=scaling_factor,
            score_function=score_function,
            expert_bias=expert_bias,
            fused=fused,
            router_replay=router_replay,
            dense_output=dense_output,
            precomputed_indices=precomputed_indices,
        )

    wrapped._relax_router_sorted_replay = True
    return wrapped


def install() -> bool:
    if not Envs.RELAX_QWEN38_ROUTER_SORTED_REPLAY:
        return False
    import hashlib
    import inspect

    from megatron.core.transformer.moe import moe_utils, router

    original = moe_utils.topk_routing_with_score_function
    if getattr(original, "_relax_router_sorted_replay", False):
        if router.topk_routing_with_score_function is not original:
            raise RuntimeError("Router sorted-replay aliases diverged")
        return True
    if router.topk_routing_with_score_function is not original:
        raise RuntimeError("Unrecognized router score-function override")
    expected = (
        "logits",
        "topk",
        "use_pre_softmax",
        "num_groups",
        "group_topk",
        "scaling_factor",
        "score_function",
        "expert_bias",
        "fused",
        "router_replay",
        "dense_output",
        "precomputed_indices",
    )
    if tuple(inspect.signature(original).parameters) != expected:
        raise RuntimeError("Unsupported installed router score-function signature")
    audited_sha = "6ece33ddea2596e64d6584a6d40ccbf9322f38a58ee8d04fd474b97d7ca78039"
    if hashlib.sha256(inspect.getsource(original).encode()).hexdigest() != audited_sha:
        raise RuntimeError("Installed router implementation differs from audited source")
    wrapped = make_wrapper(original)
    moe_utils.topk_routing_with_score_function = wrapped
    router.topk_routing_with_score_function = wrapped
    from relax.utils.logging_utils import get_logger

    get_logger(__name__).info("Enabled opt-in Qwen38 FP32 top10 sorted routing during checkpoint no_grad")
    return True
