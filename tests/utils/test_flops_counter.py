# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import math
import sys
import types

import pytest

from relax.utils.training.flops_counter import (
    _DEVICE_FLOPS,
    FlopsCounter,
    _qwen38_indexer_pairs,
    _qwen38_selected_pairs,
    get_device_peak_flops,
)


# ---------------------------------------------------------------------------
# Helper: lightweight config object (mirrors verl test pattern)
# ---------------------------------------------------------------------------
class Config:
    def __init__(self, config_dict):
        for key, value in config_dict.items():
            if isinstance(value, dict):
                value = Config(value)
            setattr(self, key, value)


# ---------------------------------------------------------------------------
# GPU peak FLOPS tests
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "device_name, expected_raw_flops",
    [
        ("NVIDIA H100 80GB HBM3", 989e12),
        ("NVIDIA H800", 989e12),
        ("NVIDIA A100-SXM4-80GB", 312e12),
        ("NVIDIA A800-SXM4-80GB", 312e12),
        ("NVIDIA L40S", 362.05e12),
        ("NVIDIA H20", 148e12),
        ("Ascend910B3", 354e12),
    ],
)
def test_get_device_peak_flops_known_gpus(device_name, expected_raw_flops):
    result = get_device_peak_flops(unit="T", device_name=device_name)
    expected_tflops = expected_raw_flops / 1e12
    assert math.isclose(result, expected_tflops, rel_tol=1e-6), (
        f"Expected {expected_tflops} TFLOPS for {device_name}, got {result}"
    )


def test_get_device_peak_flops_unknown_gpu():
    result = get_device_peak_flops(unit="T", device_name="SomeUnknownGPU-XYZ")
    assert result == float("inf")


def test_get_device_peak_flops_defaults_to_cpu_when_device_properties_unavailable(monkeypatch):
    device_module = types.ModuleType("relax.utils.device")

    def get_device_properties():
        raise AttributeError("module 'torch.cpu' has no attribute 'get_device_properties'")

    device_module.device_module = types.SimpleNamespace(get_device_properties=get_device_properties)
    monkeypatch.setitem(sys.modules, "relax.utils.device", device_module)

    result = get_device_peak_flops(unit="T")

    assert math.isclose(result, _DEVICE_FLOPS["CPU"] / 1e12, rel_tol=1e-6)


def test_get_device_peak_flops_unit_conversion():
    tflops = get_device_peak_flops(unit="T", device_name="NVIDIA H100")
    gflops = get_device_peak_flops(unit="G", device_name="NVIDIA H100")
    assert math.isclose(tflops * 1000, gflops, rel_tol=1e-6)


def test_device_flops_table_has_common_gpus():
    common_keys = ["H100", "H800", "A100", "A800", "H20"]
    for key in common_keys:
        assert key in _DEVICE_FLOPS, f"Missing common GPU {key} in _DEVICE_FLOPS"


# ---------------------------------------------------------------------------
# FlopsCounter model FLOPS tests (expected values from verl test suite)
# ---------------------------------------------------------------------------
FLOPS_TEST_CONFIGS = {
    "qwen3_dense": {
        "config": {
            "model_type": "qwen3",
            "vocab_size": 151936,
            "hidden_size": 4096,
            "intermediate_size": 12288,
            "num_hidden_layers": 36,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "head_dim": 128,
        },
        "batch_seqlens_list": [[512, 1024, 2048], [4096, 4096, 4096]],
        "expected_tflops_list": [180997438046208 / 1e12, 648394032807936 / 1e12],
    },
    "qwen3_moe": {
        "config": {
            "model_type": "qwen3_moe",
            "hidden_size": 2048,
            "vocab_size": 151936,
            "num_hidden_layers": 48,
            "num_key_value_heads": 4,
            "num_attention_heads": 32,
            "head_dim": 128,
            "moe_intermediate_size": 768,
            "num_experts_per_tok": 8,
            "num_experts": 128,
        },
        "batch_seqlens_list": [[512, 1024, 2048], [4096, 4096, 4096]],
        "expected_tflops_list": [78593069678592 / 1e12, 306570470621184 / 1e12],
    },
    "deepseek_v3": {
        "config": {
            "model_type": "deepseek_v3",
            "hidden_size": 7168,
            "vocab_size": 129280,
            "moe_intermediate_size": 2048,
            "num_hidden_layers": 61,
            "first_k_dense_replace": 3,
            "num_attention_heads": 128,
            "n_routed_experts": 256,
            "num_experts_per_tok": 8,
            "n_shared_experts": 1,
            "kv_lora_rank": 512,
            "qk_rope_head_dim": 64,
            "v_head_dim": 128,
            "intermediate_size": 18432,
            "qk_nope_head_dim": 128,
            "q_lora_rank": 1536,
        },
        "batch_seqlens_list": [[512, 1024, 2048], [4096, 4096, 4096]],
        "expected_tflops_list": [848766538088448 / 1e12, 3145850406567936 / 1e12],
    },
}


FLOPS_TEST_VL_CONFIGS = {
    "qwen3_vl": {
        # Exact config of Qwen/Qwen3-VL-4B-Instruct. Note head_dim=128 is set
        # explicitly and num_attention_heads * head_dim (32 * 128 = 4096) !=
        # hidden_size (2560), so head_dim must be read from the config.
        "config": {
            "model_type": "qwen3_vl",
            "text_config": {
                "hidden_size": 2560,
                "vocab_size": 151936,
                "num_hidden_layers": 36,
                "num_key_value_heads": 8,
                "num_attention_heads": 32,
                "head_dim": 128,
                "intermediate_size": 9728,
            },
            "vision_config": {
                "num_heads": 16,
                "depth": 24,
                "hidden_size": 1024,
                "intermediate_size": 4096,
                "out_hidden_size": 2560,
                "spatial_merge_size": 2,
                "temporal_patch_size": 2,
                "in_channels": 3,
                "patch_size": 16,
                "deepstack_visual_indexes": [5, 11, 17],
            },
        },
        "batch_seqlens_list": [[512, 1024, 2048], [4096, 4096, 4096]],
        "images_seqlens_list": [[512, 1024, 2048], [4096, 4096, 4096]],
        "expected_tflops_list": [110225906466816 / 1e12, 415025911037952 / 1e12],
    },
    "qwen3_5": {
        "config": {
            "model_type": "qwen3_5",
            "text_config": {
                "vocab_size": 248320,
                "hidden_size": 4096,
                "intermediate_size": 12288,
                "num_hidden_layers": 32,
                "num_attention_heads": 16,
                "num_key_value_heads": 4,
                "head_dim": 256,
                "linear_conv_kernel_dim": 4,
                "linear_key_head_dim": 128,
                "linear_value_head_dim": 128,
                "linear_num_key_heads": 16,
                "linear_num_value_heads": 32,
                "layer_types": ["linear_attention" if bool((i + 1) % 4) else "full_attention" for i in range(32)],
            },
            "vision_config": {
                "num_heads": 16,
                "depth": 27,
                "hidden_size": 1152,
                "intermediate_size": 4304,
                "out_hidden_size": 4096,
                "spatial_merge_size": 2,
                "temporal_patch_size": 2,
                "in_channels": 3,
                "patch_size": 16,
            },
        },
        "batch_seqlens_list": [[512, 1024, 2048], [4096, 4096, 4096]],
        "images_seqlens_list": [[512, 1024, 2048], [4096, 4096, 4096]],
        "expected_tflops_list": [206090394402816 / 1e12, 724521757704192 / 1e12],
    },
    "qwen3_5_moe": {
        "config": {
            "model_type": "qwen3_5_moe",
            "text_config": {
                "vocab_size": 248320,
                "hidden_size": 2048,
                "num_hidden_layers": 40,
                "num_attention_heads": 16,
                "num_key_value_heads": 2,
                "head_dim": 256,
                "linear_conv_kernel_dim": 4,
                "linear_key_head_dim": 128,
                "linear_value_head_dim": 128,
                "linear_num_key_heads": 16,
                "linear_num_value_heads": 32,
                "moe_intermediate_size": 512,
                "shared_expert_intermediate_size": 512,
                "num_experts_per_tok": 8,
                "num_experts": 256,
                "layer_types": ["linear_attention" if bool((i + 1) % 4) else "full_attention" for i in range(40)],
            },
            "vision_config": {
                "num_heads": 16,
                "depth": 27,
                "hidden_size": 1152,
                "intermediate_size": 4304,
                "out_hidden_size": 2048,
                "spatial_merge_size": 2,
                "temporal_patch_size": 2,
                "in_channels": 3,
                "patch_size": 16,
            },
        },
        "batch_seqlens_list": [[512, 1024, 2048], [4096, 4096, 4096]],
        "images_seqlens_list": [[512, 1024, 2048], [4096, 4096, 4096]],
        "expected_tflops_list": [88082762170368 / 1e12, 321470349705216 / 1e12],
    },
}

FLOPS_TEST_OMNI_CONFIGS = {
    "qwen3_omni_moe": {
        "config": {
            "model_type": "qwen3_omni_moe",
            "thinker_config": {
                "text_config": {
                    "hidden_size": 2048,
                    "vocab_size": 3584,
                    "num_hidden_layers": 28,
                    "num_key_value_heads": 4,
                    "num_attention_heads": 28,
                    "head_dim": 128,
                    "moe_intermediate_size": 768,
                    "num_experts_per_tok": 8,
                    "num_experts": 128,
                },
                "vision_config": {
                    "num_heads": 16,
                    "depth": 27,
                    "hidden_size": 1152,
                    "intermediate_size": 4304,
                    "out_hidden_size": 3584,
                    "spatial_merge_size": 2,
                    "temporal_patch_size": 2,
                    "in_channels": 3,
                    "patch_size": 16,
                    "deepstack_visual_indexes": [8, 16, 24],
                },
                "audio_config": {
                    "d_model": 1280,
                    "num_hidden_layers": 32,
                    "encoder_attention_heads": 20,
                    "encoder_ffn_dim": 5120,
                    "num_mel_bins": 128,
                    "output_dim": 3584,
                    "n_window": 100,
                },
            },
        },
        "batch_seqlens": [512, 1024, 2048],
        "images_seqlens": [512, 1024],
        "audio_seqlens": [500, 1000],
        "expected_tflops": 43060677083136 / 1e12,
    },
}


@pytest.mark.parametrize("config_name", list(FLOPS_TEST_CONFIGS.keys()))
def test_flops_counter_model_estimation(config_name):
    test_data = FLOPS_TEST_CONFIGS[config_name]
    config = Config(test_data["config"])
    counter = FlopsCounter(config)

    for batch_seqlens, expected_tflops in zip(
        test_data["batch_seqlens_list"], test_data["expected_tflops_list"], strict=True
    ):
        # delta_time=1 so returned value = raw TFLOPS
        estimated_tflops, _ = counter.estimate(batch_seqlens, delta_time=1.0)
        assert math.isclose(estimated_tflops, expected_tflops, rel_tol=1e-6), (
            f"{config_name}: expected {expected_tflops:.2f} TFLOPS, got {estimated_tflops:.2f}"
        )


@pytest.mark.parametrize("config_name", list(FLOPS_TEST_VL_CONFIGS.keys()))
def test_flops_counter_vl_estimation(config_name):
    test_data = FLOPS_TEST_VL_CONFIGS[config_name]
    config = Config(test_data["config"])
    counter = FlopsCounter(config)

    for batch_seqlens, images_seqlens, expected_tflops in zip(
        test_data["batch_seqlens_list"],
        test_data["images_seqlens_list"],
        test_data["expected_tflops_list"],
        strict=True,
    ):
        estimated_tflops, _ = counter.estimate(batch_seqlens, delta_time=1.0, images_seqlens=images_seqlens)
        assert math.isclose(estimated_tflops, expected_tflops, rel_tol=1e-6), (
            f"{config_name}: expected {expected_tflops:.2f} TFLOPS, got {estimated_tflops:.2f}"
        )


@pytest.mark.parametrize("config_name", list(FLOPS_TEST_OMNI_CONFIGS.keys()))
def test_flops_counter_omni_estimation(config_name):
    test_data = FLOPS_TEST_OMNI_CONFIGS[config_name]
    config = Config(test_data["config"])
    counter = FlopsCounter(config)

    estimated_tflops, _ = counter.estimate(
        test_data["batch_seqlens"],
        delta_time=1.0,
        images_seqlens=test_data["images_seqlens"],
        audio_seqlens=test_data["audio_seqlens"],
    )
    assert math.isclose(estimated_tflops, test_data["expected_tflops"], rel_tol=1e-6), (
        f"{config_name}: expected {test_data['expected_tflops']:.2f} TFLOPS, got {estimated_tflops:.2f}"
    )


def test_mfu_calculation():
    peak_tflops = get_device_peak_flops(unit="T", device_name="NVIDIA H100 80GB HBM3")
    achieved_tflops = 200.0
    mfu = achieved_tflops / peak_tflops
    assert 0 < mfu < 1
    assert math.isclose(mfu, 200.0 / 989.0, rel_tol=1e-6)


def test_unknown_model_type_fallback():
    config = Config(
        {
            "model_type": "unknown_model_xyz",
            "hidden_size": 4096,
            "vocab_size": 151936,
            "intermediate_size": 12288,
            "num_hidden_layers": 36,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "head_dim": 128,
        }
    )
    counter = FlopsCounter(config)
    estimated, _ = counter.estimate([1024, 2048], delta_time=1.0)
    assert estimated > 0


def test_unknown_model_type_missing_fields_returns_zero():
    config = Config({"model_type": "unknown_model_xyz"})
    counter = FlopsCounter(config)
    estimated, _ = counter.estimate([1024, 2048], delta_time=1.0)
    assert estimated == 0


@pytest.fixture
def qwen38_config():
    # Published 48-layer text dimensions; vision uses the existing Qwen3.5 fixture.
    return Config(
        {
            "model_type": "qwen4_exp",
            "text_config": {
                "model_type": "qwen4_exp_text",
                "hidden_size": 2560,
                "vocab_size": 248320,
                "num_hidden_layers": 48,
                "layer_types": (["linear_attention"] * 3 + ["full_attention"]) * 12,
                "num_attention_heads": 24,
                "num_key_value_heads": 2,
                "head_dim": 256,
                "num_experts": 512,
                "num_experts_per_tok": 10,
                "moe_intermediate_size": 640,
                "shared_expert_intermediate_size": 640,
                "linear_num_key_heads": 16,
                "linear_key_head_dim": 128,
                "linear_num_value_heads": 48,
                "linear_value_head_dim": 128,
                "linear_conv_kernel_dim": 4,
                "indexer_budget": 2048,
                "indexer_compress_ratio": 4,
                "indexer_n_heads": 4,
                "indexer_kv_heads": 1,
                "indexer_head_dim": 128,
                "hc_count": 4,
                "hc_lowrank": 320,
                "ple_layer_ids": [2],
                "ple_embed_dim": 2560,
                "ple_conv_kernel_size": 4,
                "mtp": {"layer_types": ["full_attention"]},
            },
            "vision_config": {
                **FLOPS_TEST_VL_CONFIGS["qwen3_5"]["config"]["vision_config"],
                "out_hidden_size": 2560,
            },
        }
    )


_QWEN38_POLICY = {
    "freeze_params_name_list": [r"(^|\.)per_layer_embedding\.|self_attention\.indexer\."],
    "freeze_vision_model": True,
    "freeze_vision_projection": True,
}
_QWEN38_MTP = {"enable_mtp_training": True, "mtp_num_layers": 1, "mtp_detach_paths": ()}
_QWEN38_TABLE_ONLY_POLICY = {
    **_QWEN38_POLICY,
    "freeze_params_name_list": [
        r"self_attention\.indexer\.",
        r"(^|\.)per_layer_embedding\.ple_embedding\.ngram_embedding\.weight$",
    ],
}


@pytest.mark.parametrize("budget,ratio", [(4, 1), (4, 2), (8, 4)])
def test_qwen38_flops_sparse_pairs_match_causal_block_enumeration(budget, ratio):
    for length in range(4 * budget + 1):
        selected = indexer = 0
        for position in range(1, length + 1):
            full_blocks, tail = divmod(position, ratio)
            selected += min(full_blocks, budget // ratio) * ratio + tail
            if full_blocks > budget // ratio:
                indexer += full_blocks
        assert _qwen38_selected_pairs(length, budget, ratio) == selected
        assert _qwen38_indexer_pairs(length, budget, ratio) == indexer


@pytest.mark.parametrize("mtp_on,expected", [(False, 233.897266774016), (True, 261.24036079616)])
@pytest.mark.parametrize("text_only,sparse_label", [(False, False), (True, True)])
def test_flops_counter_qwen38_reference_and_mtp_golden(qwen38_config, mtp_on, expected, text_only, sparse_label):
    text = qwen38_config.text_config
    if sparse_label:
        text.layer_types = ["qwen_sparse_attention" if t == "full_attention" else t for t in text.layer_types]
        text.mtp = {"layer_types": ["qwen_sparse_attention"]}
    counter = FlopsCounter(text if text_only else qwen38_config)
    # Analytical reference: MTP1 adds 27.343094022144 TFLOPs for
    # one QSA/MoE block, three HC mixers, two HC injections, e/h projections,
    # and a full-token LM head. No vocabulary-sized embedding lookup GEMM.
    policy = {**_QWEN38_POLICY, **(_QWEN38_MTP if mtp_on else {})}
    result, _ = counter.estimate([512, 1024, 4096], delta_time=2.0, **policy)
    assert result == pytest.approx(expected / 2)
    doubled, _ = counter.estimate([512, 1024, 4096] * 2, delta_time=2.0, **policy)
    assert doubled == pytest.approx(2 * result)  # No attention across packed document boundaries.


def test_flops_counter_qwen38_frozen_vision_forward_cost(qwen38_config):
    counter = FlopsCounter(qwen38_config)
    baseline, _ = counter.estimate([1024], 1.0, **_QWEN38_POLICY)
    with_image, _ = counter.estimate([1024], 1.0, images_seqlens=[256], **_QWEN38_POLICY)
    # Frozen ViT: patch projection + 27 full-attention/GELU blocks + one merger.
    expected = (
        2 * 256 * 1152 * 3 * 2 * 16**2
        + 27 * (2 * 256 * (4 * 1152**2 + 2 * 1152 * 4304) + 4 * 256**2 * 1152)
        + 2 * 64 * ((4 * 1152) ** 2 + 4 * 1152 * 2560)
    ) / 1e12
    assert with_image - baseline == pytest.approx(expected)


@pytest.mark.parametrize("detach_embedding", [False, True])
@pytest.mark.parametrize("detach_backbone", [False, True])
@pytest.mark.parametrize("detach_head", [False, True])
@pytest.mark.parametrize("head_tokens", [None, 1024])
def test_flops_counter_qwen38_mtp_detach_paths(
    qwen38_config, detach_embedding, detach_backbone, detach_head, head_tokens
):
    paths = tuple(
        path
        for path, detach in zip(("embedding", "backbone", "lm-head"), (detach_embedding, detach_backbone, detach_head))
        if detach
    )
    policy = {**_QWEN38_POLICY, **_QWEN38_MTP, "mtp_detach_paths": paths}
    if head_tokens is not None:
        policy["lm_head_tokens"] = head_tokens
    result, _ = FlopsCounter(qwen38_config).estimate([512, 1024, 4096], 2.0, **policy)
    # Audited undetached golden; detach keeps e/h projection dInput because their
    # input norms are trainable. Only the MTP head's [248320,2560] dWeight drops.
    expected = 261.24036079616
    if detach_head:
        expected -= 2 * 2560 * 248320 * 5632 / 1e12
    if head_tokens is not None:
        # Main-head chunking never shortens the MTP projection's token count.
        expected -= 6 * 2560 * 248320 * (5632 - head_tokens) / 1e12
    assert result == pytest.approx(expected / 2)


@pytest.mark.parametrize("paths", [(), [], "none", ("none",), ["none"]])
def test_flops_counter_qwen38_mtp_no_detach_aliases(qwen38_config, paths):
    policy = {**_QWEN38_POLICY, **_QWEN38_MTP, "mtp_detach_paths": paths}
    result, _ = FlopsCounter(qwen38_config).estimate([512, 1024, 4096], 2.0, **policy)
    assert result == pytest.approx(261.24036079616 / 2)


def test_flops_counter_qwen38_mtp_default_detaches_all_paths(qwen38_config):
    policy = {**_QWEN38_POLICY, "enable_mtp_training": True, "mtp_num_layers": 1}
    counter = FlopsCounter(qwen38_config)
    default, _ = counter.estimate([1024], 1.0, **policy)
    explicit, _ = counter.estimate([1024], 1.0, mtp_detach_paths=["embedding", "backbone", "lm-head"], **policy)
    assert default > 0
    assert default == explicit


@pytest.mark.parametrize("deepstack_layers", [[], [8, 16]])
@pytest.mark.parametrize("explicit_trainable", [False, True])
def test_flops_counter_qwen38_trainable_vision_backward_cost(qwen38_config, deepstack_layers, explicit_trainable):
    qwen38_config.vision_config.deepstack_visual_indexes = deepstack_layers
    counter = FlopsCounter(qwen38_config)
    policy = {"freeze_params_name_list": _QWEN38_POLICY["freeze_params_name_list"]}
    if explicit_trainable:
        policy.update(freeze_vision_model=False, freeze_vision_projection=False)
    baseline, _ = counter.estimate([1024], 2.0, **policy)
    trained, _ = counter.estimate([1024], 2.0, images_seqlens=[256, 64], **policy)
    frozen, _ = counter.estimate([1024], 2.0, images_seqlens=[256, 64], **_QWEN38_POLICY)
    # Two independent images: patch Conv3d has fwd+dWeight (no pixel dInput),
    # 27 attention/GELU blocks and each merger have fwd+dInput+dWeight.
    patch = 320 * 1152 * 3 * 2 * 16**2
    blocks = 27 * 320 * (4 * 1152**2 + 2 * 1152 * 4304)
    attention = 27 * 1152 * (256**2 + 64**2)
    mergers = (1 + len(deepstack_layers)) * 80 * ((4 * 1152) ** 2 + 4 * 1152 * 2560)
    expected = (4 * patch + 6 * blocks + 12 * attention + 6 * mergers) / 2 / 1e12
    backward = (2 * patch + 4 * blocks + 8 * attention + 4 * mergers) / 2 / 1e12
    assert baseline > 0
    assert trained - baseline == pytest.approx(expected)
    assert trained - frozen == pytest.approx(backward)


@pytest.mark.parametrize("mtp_on", [False, True])
@pytest.mark.parametrize("ple_layers", [[2], [2, 6]])
def test_flops_counter_qwen38_table_only_freeze_adds_weight_gradients(qwen38_config, mtp_on, ple_layers):
    qwen38_config.text_config.ple_layer_ids = ple_layers
    counter = FlopsCounter(qwen38_config)
    mtp = _QWEN38_MTP if mtp_on else {}
    frozen, _ = counter.estimate([512, 1024, 4096], 2.0, **_QWEN38_POLICY, **mtp)
    table_only, _ = counter.estimate([512, 1024, 4096], 2.0, **_QWEN38_TABLE_ONLY_POLICY, **mtp)
    # Each PLE adds dWeight for key [10240,2560], value [2560,2560]
    # and depthwise conv [10240,1,4]. Lookup outputs are frozen, so no
    # projection dInput matmul; norms remain excluded elementwise operations.
    extra_flops_per_token = 2 * (10240 * 2560 + 2560 * 2560 + 10240 * 4)
    expected_delta = len(ple_layers) * (512 + 1024 + 4096) * extra_flops_per_token / 2 / 1e12
    assert frozen > 0
    assert table_only - frozen == pytest.approx(expected_delta)


@pytest.mark.parametrize("mtp_on", [False, True])
@pytest.mark.parametrize("unmatched_pattern", [[], [r"per_layer_embedding\.weight$"]])
def test_flops_counter_qwen38_unfrozen_table_adds_projection_input_gradients(qwen38_config, mtp_on, unmatched_pattern):
    counter = FlopsCounter(qwen38_config)
    mtp = _QWEN38_MTP if mtp_on else {}
    frozen_table, _ = counter.estimate([512, 1024, 4096], 2.0, **_QWEN38_TABLE_ONLY_POLICY, **mtp)
    policy = {**_QWEN38_POLICY, "freeze_params_name_list": [r"self_attention\.indexer\.", *unmatched_pattern]}
    unfrozen, _ = counter.estimate([512, 1024, 4096], 2.0, **policy, **mtp)
    # dInput for key/value projections; lookup scatter-add is not a dense matmul.
    expected_delta = (512 + 1024 + 4096) * 2 * (10240 * 2560 + 2560 * 2560) / 2 / 1e12
    assert frozen_table > 0
    assert unfrozen - frozen_table == pytest.approx(expected_delta)


@pytest.mark.parametrize(
    "extra_freeze",
    [
        r"per_layer_embedding\.key_proj\.weight$",
        r"per_layer_embedding\.value_proj\.weight$",
        r"per_layer_embedding\.norm_key\.weight$",
        r"per_layer_embedding\.norm_query\.weight$",
        r"per_layer_embedding\.norm_conv\.weight$",
        r"per_layer_embedding\.conv1d\.weight$",
        r"language_model\.decoder\.layers\.1\.per_layer_embedding\.key_proj\.weight$",
    ],
)
def test_flops_counter_qwen38_rejects_partial_ple_branch_freeze(qwen38_config, extra_freeze):
    policy = {
        **_QWEN38_TABLE_ONLY_POLICY,
        "freeze_params_name_list": [*_QWEN38_TABLE_ONLY_POLICY["freeze_params_name_list"], extra_freeze],
    }
    result, _ = FlopsCounter(qwen38_config).estimate([1024], 1.0, **policy)
    assert result == 0


@pytest.mark.parametrize(
    "overrides",
    [
        {"freeze_params_name_list": []},
        {"freeze_vision_model": False},
        {"freeze_vision_projection": False},
        {"mtp_num_layers": 2},
        {"mtp_detach_paths": ("embedding", "hidden")},
        {"mtp_detach_paths": ("none", "lm-head")},
        {"mtp_detach_paths": None},
        {"mtp_only_training": True},
        {"freeze_params_name_list": [r"per_layer_embedding\.", r"^decoder.*self_attention\.indexer\."]},
    ],
)
def test_flops_counter_qwen38_rejects_unaudited_policies(qwen38_config, overrides):
    policy = {**_QWEN38_POLICY, **_QWEN38_MTP, **overrides}
    result, _ = FlopsCounter(qwen38_config).estimate([1024], 1.0, images_seqlens=[256], **policy)
    assert result == 0


def test_flops_counter_qwen38_rejects_invalid_compression(qwen38_config):
    qwen38_config.text_config.indexer_budget = 7
    with pytest.raises(ValueError, match="divisible"):
        FlopsCounter(qwen38_config).estimate([1024], 1.0, **_QWEN38_POLICY)
