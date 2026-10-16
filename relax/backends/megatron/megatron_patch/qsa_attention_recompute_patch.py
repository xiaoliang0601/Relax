# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Opt-in: retain QSA _forward_attention; checkpoint its original MLP.

Includes attention input norm/PLE/mHC/BDA in retained scope. Not core-attention-only.
"""

from __future__ import annotations

import hashlib
import inspect
import json
from functools import wraps
from types import MethodType
from typing import Any, Callable

from relax.utils.env import Envs


def validate_args(args: Any) -> None:
    expected = {
        "recompute_granularity": "full",
        "recompute_method": "uniform",
        "recompute_num_layers": 1,
        "tensor_model_parallel_size": 8,
        "pipeline_model_parallel_size": 4,
        "context_parallel_size": 1,
        "expert_model_parallel_size": 8,
        "expert_tensor_parallel_size": 1,
    }
    for name, value in expected.items():
        if getattr(args, name, None) != value:
            raise RuntimeError(f"QSA retained attention requires {name}={value!r}")
    if not getattr(args, "bf16", False) or getattr(args, "fp8", None) or getattr(args, "fp4", None):
        raise RuntimeError("QSA retained attention requires BF16 without FP8/FP4")
    if getattr(args, "cpu_offloading", False) or getattr(args, "distribute_saved_activations", False):
        raise RuntimeError("QSA retained attention excludes activation offload/distributed saved activations")
    if getattr(args, "virtual_pipeline_model_parallel_size", None):
        raise RuntimeError("QSA retained attention does not support virtual pipeline")
    if getattr(args, "cuda_graph_impl", "none") not in (None, "none"):
        raise RuntimeError("QSA retained attention does not support CUDA graphs")


def wrap_mlp(layer: Any, checkpoint: Callable) -> None:
    """Bind invocation-local non-tensor metadata; pass differentiable inputs
    explicitly."""
    original = layer._forward_mlp

    @wraps(original)
    def wrapped(hidden_states, inference_context=None, padding_mask=None, packed_seq_params=None):
        if not layer.training:
            return original(hidden_states, inference_context, padding_mask, packed_seq_params)
        if inference_context is not None:
            raise RuntimeError("QSA retained attention is training-only")

        def run(hidden, padding):
            return original(
                hidden,
                inference_context=None,
                padding_mask=padding,
                packed_seq_params=packed_seq_params,
            )

        return checkpoint(run, layer.config.distribute_saved_activations, hidden_states, padding_mask)

    layer._forward_mlp = wrapped
    layer._relax_qsa_attention_retained = True


def install(args: Any, chunks: Any) -> list[dict[str, Any]]:
    if not Envs.RELAX_QWEN38_RETAIN_QSA_ATTENTION:
        return []
    validate_args(args)
    from megatron.bridge.models.qwen_vl.modelling_qwen3_vl.transformer_block import Qwen3VLTransformerBlock
    from megatron.core import recompute, tensor_parallel
    from megatron.core.models.hybrid import hybrid_block
    from megatron.core.transformer.multi_token_prediction import MultiTokenPredictionLayer
    from megatron.core.transformer.transformer_layer import TransformerLayer

    from .qsa_bridge_recompute import qsa_bridge_checkpointed_forward
    from .qsa_split_recompute import qsa_split_checkpointed_forward

    # A vendored control-flow body must never silently drift from its inspected dependency.
    source_path = inspect.getsourcefile(recompute.checkpointed_forward)
    with open(source_path, "rb") as source:
        digest = hashlib.sha256(source.read()).hexdigest()
    if digest != EXPECTED_RECOMPUTE_SHA256:
        raise RuntimeError("MCore recompute.py changed; re-review QSA split checkpoint implementation")
    modules = [
        (f"chunk{i}.{name}", module) for i, chunk in enumerate(chunks) for name, module in chunk.named_modules()
    ]
    layers = []
    stacks = []
    bridge_stacks = []
    # Complete validation before changing any object or module binding.
    for path, module in modules:
        if isinstance(module, MultiTokenPredictionLayer) and module.mtp_layer_pattern is None:
            raise RuntimeError("Legacy MTP outer checkpoint is unsupported")
        if isinstance(module, (hybrid_block.HybridStack, Qwen3VLTransformerBlock)):
            config = module.config
            if (config.recompute_granularity, config.recompute_method, config.recompute_num_layers) != (
                "full",
                "uniform",
                1,
            ):
                raise RuntimeError("HybridStack effective recompute config differs from guarded args")
            if getattr(config, "cpu_offloading", False) or getattr(config, "distribute_saved_activations", False):
                raise RuntimeError("HybridStack offload/distributed saved activations are unsupported")
            if getattr(config, "fp8", None) or getattr(config, "fp4", None):
                raise RuntimeError("Retained attention excludes quantized stack configurations")
            if getattr(config, "cuda_graph_impl", "none") not in (None, "none"):
                raise RuntimeError("Retained attention excludes stack CUDA graphs")
            if isinstance(module, Qwen3VLTransformerBlock):
                if "_checkpointed_forward" in module.__dict__:
                    raise RuntimeError("Bridge instance checkpoint already overridden")
                if inspect.getsourcefile(type(module)._checkpointed_forward) != inspect.getsourcefile(
                    Qwen3VLTransformerBlock
                ):
                    raise RuntimeError("Unexpected Bridge checkpoint override")
                source = inspect.getsourcefile(Qwen3VLTransformerBlock)
                with open(source, "rb") as stream:
                    if hashlib.sha256(stream.read()).hexdigest() != EXPECTED_BRIDGE_SHA256:
                        raise RuntimeError("Bridge transformer_block.py changed; re-review checkpoint body")
                bridge_stacks.append(module)
            else:
                stacks.append(module)
        attention = getattr(module, "self_attention", None)
        if type(attention).__name__ != "QwenSparseSelfAttention":
            continue
        if not isinstance(module, TransformerLayer) or not callable(getattr(module, "_forward_mlp", None)):
            raise RuntimeError("Unsupported QSA owner")
        if getattr(module, "_relax_qsa_attention_retained", False):
            raise RuntimeError("QSA retained attention already installed")
        if attention.config.qsa_indexer_loss_coeff != 0 or any(
            p.requires_grad for p in attention.indexer.parameters()
        ):
            raise RuntimeError("Only frozen Stage-1 QSA indexers are supported")
        if any(
            getattr(module, flag, False)
            for flag in ("recompute_mlp", "recompute_input_layernorm", "recompute_pre_mlp_layernorm")
        ) or getattr(module.mlp, "moe_layer_recompute", False):
            raise RuntimeError("Unexpected inner MLP recomputation")
        layers.append((path, module))
    owned = {id(layer) for stack in stacks + bridge_stacks for layer in stack.layers}
    if not layers or any(id(layer) not in owned for _, layer in layers):
        raise RuntimeError("All QSA owners must be direct children of HybridStack or guarded Bridge block")
    if hybrid_block.checkpointed_forward is not recompute.checkpointed_forward:
        raise RuntimeError("HybridStack checkpoint entry already changed; review patch composition")
    manifest = [
        {
            "path": path,
            "layer_number": layer.layer_number,
            "attention": "retain _forward_attention including norm/PLE/mHC/BDA",
            "mlp": "checkpoint original _forward_mlp",
        }
        for path, layer in layers
    ]
    for _, layer in layers:
        wrap_mlp(layer, tensor_parallel.checkpoint)
    # The adapter routes every untouched stack to the original function.
    original = hybrid_block.checkpointed_forward

    @wraps(original)
    def dispatch(stack, *positional, **keywords):
        if any(getattr(layer, "_relax_qsa_attention_retained", False) for layer in stack.layers):
            return qsa_split_checkpointed_forward(stack, *positional, **keywords)
        return original(stack, *positional, **keywords)

    if stacks:
        hybrid_block.checkpointed_forward = dispatch
    for stack in bridge_stacks:
        if any(getattr(layer, "_relax_qsa_attention_retained", False) for layer in stack.layers):
            stack._checkpointed_forward = MethodType(qsa_bridge_checkpointed_forward, stack)
    from relax.utils.logging_utils import get_logger

    get_logger(__name__).info("QSA attention-retention manifest: %s", json.dumps(manifest))
    return manifest


EXPECTED_RECOMPUTE_SHA256 = "784f0d0d3fe15e4473ac1672f1a5f9a8e0c98a505e365061a3b3e57bbc293830"

EXPECTED_BRIDGE_SHA256 = "700a7abf8e372d1a2ccd496b5b224af68826d66a9b5a8a492e95b9de39556679"
