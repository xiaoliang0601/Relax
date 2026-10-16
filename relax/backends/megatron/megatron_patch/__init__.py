# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import os

# Runtime Megatron monkey-patches applied for their import side effects. Mirrors
# the slime ``megatron_utils/megatron_patch`` package: importing this subpackage
# swaps patched implementations into ``megatron.core`` before any training step.
from relax.utils.env import Envs

from . import (
    chunked_grad_coalesce_patch,  # noqa: F401
    chunked_mtp_loss_patch,  # noqa: F401
)


if Envs.RELAX_SFT_RESIDENT_PARAM_OVERLAP:
    from .shared_expert_gate_patch import install as install_shared_expert_gate_patch

    install_shared_expert_gate_patch()

# Install before selector/replay wrappers so their optimized paths remain authoritative.
if Envs.RELAX_QWEN38_QSA_CPU_LAYOUT:
    from . import qsa_cpu_layout_patch  # noqa: F401

if Envs.RELAX_QWEN38_PLE_STREAMED:
    from . import ple_stream_patch  # noqa: F401

if Envs.RELAX_QWEN38_QSA_SPARSE_BLOCKMASK:
    from . import qsa_block_mask_patch  # noqa: F401

# Replay must capture the image selector rather than the original native indexer.
if os.environ.get("RELAX_QWEN38_QSA_TP_SELECTOR", "0") == "1":
    if Envs.RELAX_QWEN38_QSA_COMMUNITY_BACKEND != "mixed":
        raise ValueError("TP QSA selector requires the mixed community attention backend")
    from .qsa_mixed_ops.selector import install_selector

    install_selector()

if Envs.RELAX_QWEN38_QSA_LOCAL_QUERY or Envs.RELAX_QWEN38_QSA_RECOMPUTE_REPLAY:
    from . import qsa_cp_patch  # noqa: F401

if Envs.RELAX_QWEN38_QSA_FUSED_SELECTOR:
    from . import qsa_select_patch  # noqa: F401

# Community kernels consume native selected IDs; Megatron retains indexer and CP handling.
if Envs.RELAX_QWEN38_QSA_COMMUNITY_BACKEND:
    from . import qsa_community_patch  # noqa: F401

# Installs after the CP patch so it can take over that path's attention call.
if Envs.RELAX_QWEN38_QSA_FUSED_ATTENTION:
    from . import qsa_fused_patch  # noqa: F401

if Envs.RELAX_QWEN38_FLA_CONV_CACHE_REUSE:
    from .fla_conv_cache_patch import apply_fla_conv_cache_patch

    apply_fla_conv_cache_patch()

if Envs.RELAX_QWEN38_QSA_ROUTED_QKV:
    from .qsa_routed_qkv_patch import install

    install()

if Envs.RELAX_QWEN38_ROUTER_SORTED_REPLAY:
    from .router_sorted_replay_patch import install as install_router_sorted_replay

    install_router_sorted_replay()

if Envs.RELAX_QWEN38_GDN_OUTPUT_SAFE_CONFIG:
    from .gdn_output_config_patch import install as install_gdn_output_config

    install_gdn_output_config()

# Install after the fused selector, which supplies the packed CPU boundaries.
if Envs.RELAX_QWEN38_QSA_FIXED_POOL:
    from .qsa_fixed_pool_patch import install as install_qsa_fixed_pool

    install_qsa_fixed_pool()

if Envs.RELAX_QWEN38_HC_FUSED_MIX:
    from .hc_fused_mix_patch import install as install_hc_fused_mix

    install_hc_fused_mix()

if Envs.RELAX_QWEN38_HC_FUSED_COMBINE:
    from .hc_fused_combine_patch import install as install_hc_fused_combine

    install_hc_fused_combine()

if Envs.RELAX_QWEN38_PLE_FUSED_CONV:
    from .ple_fused_conv_patch import install as install_ple_fused_conv

    install_ple_fused_conv()
