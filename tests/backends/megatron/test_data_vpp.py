import importlib
import sys
import types
from argparse import Namespace

import pytest
import torch

from relax.utils.data.data import get_first_fit_partitions


def _load_data_module(monkeypatch):
    # Importing a module also assigns attributes on its parent package. Restore
    # both caches so fake process groups cannot leak into later integration tests.
    parent = importlib.import_module("relax.backends.megatron")
    for leaf in ("data", "cp_utils"):
        name = f"relax.backends.megatron.{leaf}"
        monkeypatch.setitem(sys.modules, name, sys.modules.get(name))
        sys.modules.pop(name)
        monkeypatch.setattr(parent, leaf, None, raising=False)

    metric_parent = importlib.import_module("relax.utils.training")
    metric_name = "relax.utils.training.train_metric_utils"
    monkeypatch.setitem(sys.modules, metric_name, sys.modules.get(metric_name))
    sys.modules.pop(metric_name)
    monkeypatch.setattr(metric_parent, "train_metric_utils", None, raising=False)

    megatron = types.ModuleType("megatron")
    core = types.ModuleType("megatron.core")
    mpu = types.ModuleType("megatron.core.mpu")
    packed_seq_params = types.ModuleType("megatron.core.packed_seq_params")
    training = types.ModuleType("megatron.training")
    global_vars = types.ModuleType("megatron.training.global_vars")
    tracking_utils = types.ModuleType("relax.utils.tracking_utils")
    monkeypatch.setattr(importlib.import_module("relax.utils"), "tracking_utils", tracking_utils, raising=False)

    class _PackedSeqParams:
        def __init__(self, **kwargs):
            vars(self).update(kwargs)

    core.mpu = mpu
    packed_seq_params.PackedSeqParams = _PackedSeqParams
    global_vars.get_args = lambda: None

    modules = {
        "megatron": megatron,
        "megatron.core": core,
        "megatron.core.mpu": mpu,
        "megatron.core.packed_seq_params": packed_seq_params,
        "megatron.training": training,
        "megatron.training.global_vars": global_vars,
        "relax.utils.tracking_utils": tracking_utils,
    }
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)

    sys.modules.pop("relax.backends.megatron.data", None)
    return importlib.import_module("relax.backends.megatron.data")


@pytest.mark.parametrize("already_imported", [False, True])
def test_data_module_loader_restores_import_caches(monkeypatch, already_imported):
    parent = importlib.import_module("relax.backends.megatron")
    originals = {}
    for leaf in ("data", "cp_utils"):
        name = f"relax.backends.megatron.{leaf}"
        if already_imported:
            originals[leaf] = types.ModuleType(name)
            monkeypatch.setitem(sys.modules, name, originals[leaf])
            monkeypatch.setattr(parent, leaf, originals[leaf], raising=False)
        else:
            monkeypatch.delitem(sys.modules, name, raising=False)
            monkeypatch.delattr(parent, leaf, raising=False)

    with pytest.MonkeyPatch.context() as isolated:
        data_module = _load_data_module(isolated)
        assert data_module.mpu is sys.modules["megatron.core.mpu"]
        assert parent.cp_utils.mpu is data_module.mpu

    for leaf in ("data", "cp_utils"):
        name = f"relax.backends.megatron.{leaf}"
        if already_imported:
            assert sys.modules[name] is originals[leaf]
            assert getattr(parent, leaf) is originals[leaf]
        else:
            assert name not in sys.modules
            assert not hasattr(parent, leaf)


def test_vpp_microbatch_rounding_uses_ceil_multiple(monkeypatch):
    data_module = _load_data_module(monkeypatch)

    rounded = data_module._round_up_to_microbatch_group(torch.tensor([1, 2, 3, 5]), microbatch_group_size=4)

    assert rounded.tolist() == [4, 4, 4, 8]


@pytest.mark.parametrize(
    "seqlens, capacity, num_partitions, first_fit_count, expect_fallback",
    [
        pytest.param([2, 2, 3, 3, 5, 2], 6, 3, 3, True, id="kk-over-capacity"),
        pytest.param([6, 6, 10, 10, 7, 1] * 4, 20, 9, 8, True, id="split-fallback-to-target"),
        pytest.param([2, 2, 3, 3, 5, 2], 6, 4, 3, False, id="keep-valid-kk"),
    ],
)
def test_seqlen_partitions_respect_capacity(
    monkeypatch: pytest.MonkeyPatch,
    seqlens: list[int],
    capacity: int,
    num_partitions: int,
    first_fit_count: int,
    expect_fallback: bool,
) -> None:
    data_module = _load_data_module(monkeypatch)
    kk_partitions = data_module.get_seqlen_balanced_partitions(seqlens, num_partitions, equal_size=False)
    assert any(sum(seqlens[index] for index in partition) > capacity for partition in kk_partitions) == expect_fallback

    first_fit = get_first_fit_partitions(seqlens, capacity)
    assert len(first_fit) == first_fit_count
    assert data_module.get_minimum_num_micro_batch_size(seqlens, capacity) == first_fit_count

    partitions, dummy_offsets = data_module._get_seqlen_partitions_with_dummy_padding(
        seqlens, num_partitions, capacity
    )

    assert len(partitions) == num_partitions
    assert all(partitions)
    assert all(sum(seqlens[index] for index in partition) <= capacity for partition in partitions)
    assert sorted(index for partition in partitions for index in partition) == list(range(len(seqlens)))
    assert not dummy_offsets
    if not expect_fallback:
        assert partitions == kk_partitions


@pytest.mark.parametrize("cp_size", [1, 2])
@pytest.mark.parametrize("loss_coeff", [None, 0.0, 0.01])
def test_vl_packed_stage2_preserves_true_lengths_without_changing_cu(
    monkeypatch, cp_size: int, loss_coeff: float | None
):
    data_module = _load_data_module(monkeypatch)
    monkeypatch.setattr(data_module, "get_args", lambda: Namespace(qsa_indexer_loss_coeff=loss_coeff))
    monkeypatch.setattr(data_module.mpu, "get_context_parallel_world_size", lambda: cp_size, raising=False)
    monkeypatch.setattr(data_module.mpu, "get_context_parallel_rank", lambda: 0, raising=False)
    monkeypatch.setattr(data_module.mpu, "get_tensor_model_parallel_world_size", lambda: 2, raising=False)

    def slice_with_cp(tokens, pad_value, *_args, **_kwargs):
        if cp_size == 1:
            return tokens
        chunk = (tokens.size(0) + 3) // 4
        padded = torch.nn.functional.pad(tokens, (0, 4 * chunk - tokens.size(0)), value=pad_value)
        return torch.cat((padded[:chunk], padded[3 * chunk :]))

    monkeypatch.setattr(data_module, "slice_with_cp", slice_with_cp)
    batch = {
        "tokens": [torch.arange(5), torch.arange(2)],
        "loss_masks": [torch.ones(5, dtype=torch.int), torch.ones(2, dtype=torch.int)],
        "total_lengths": [5, 2],
        "response_lengths": [5, 2],
    }
    packed = data_module.get_batch(iter([(batch, None)]), ["tokens"], pad_multiplier=4, is_vl_model=True)
    params = packed["vlm_packed_seq_params"] if cp_size == 2 else packed["packed_seq_params"]
    assert packed.get("cp_output_is_zigzag", False) == (cp_size > 1)

    if cp_size == 2:
        assert params.cu_seqlens_q.tolist() == [0, 8, 16]
        assert params.cu_seqlens_q_padded.tolist() == [0, 8, 16]
        expected_lengths = [5, 2]
        expected_starts = (0, 8, 16)
    else:
        assert params.cu_seqlens_q.tolist() == [0, 5, 7, 8]
        expected_lengths = [5, 2, 0]
        expected_starts = (0, 5, 7, 8)
    assert params.qsa_select_offsets_cpu == expected_starts
    if loss_coeff is not None and loss_coeff > 0:
        assert params.qsa_stage2_valid_lengths.tolist() == expected_lengths
        assert params.qsa_stage2_layout_cpu == (expected_starts, tuple(expected_lengths))
    else:
        assert not hasattr(params, "qsa_stage2_valid_lengths")
        assert not hasattr(params, "qsa_stage2_layout_cpu")


def test_vl_cp_packing_keeps_two_image_samples_and_loss_masks_separate(monkeypatch):
    data_module = _load_data_module(monkeypatch)
    monkeypatch.setattr(data_module, "get_args", lambda: Namespace(qsa_indexer_loss_coeff=0.0))
    monkeypatch.setattr(data_module.mpu, "get_context_parallel_world_size", lambda: 2, raising=False)
    monkeypatch.setattr(data_module.mpu, "get_context_parallel_rank", lambda: 0, raising=False)
    monkeypatch.setattr(data_module.mpu, "get_tensor_model_parallel_world_size", lambda: 2, raising=False)

    def slice_with_cp(values, pad_value, *_args, **_kwargs):
        chunk = (values.size(0) + 3) // 4
        padded = torch.nn.functional.pad(values, (0, 4 * chunk - values.size(0)), value=pad_value)
        return torch.cat((padded[:chunk], padded[3 * chunk :]))

    monkeypatch.setattr(data_module, "slice_with_cp", slice_with_cp)
    lengths = [92, 99]
    tokens = [torch.full((length,), 10 + i, dtype=torch.long) for i, length in enumerate(lengths)]
    loss_masks = [torch.arange(length) >= length - (i + 2) for i, length in enumerate(lengths)]
    batch = {
        "tokens": tokens,
        "loss_masks": loss_masks,
        "total_lengths": lengths,
        "response_lengths": lengths,
        "multimodal_train_inputs": [
            {
                "pixel_values": torch.full((4, 3), 1.0),
                "image_grid_thw": torch.tensor([[1, 2, 2]]),
            },
            {
                "pixel_values": torch.full((9, 3), 2.0),
                "image_grid_thw": torch.tensor([[1, 3, 3]]),
            },
        ],
    }

    packed = data_module.get_batch(iter([(batch, None)]), ["tokens"], pad_multiplier=4, is_vl_model=True)

    assert packed["padded_total_lengths"] == [96, 104]
    assert packed["vlm_packed_seq_params"].cu_seqlens_q.tolist() == [0, 96, 200]
    assert packed["unsplit_attention_mask"].sum(dim=1).tolist() == lengths
    assert packed["multimodal_num_items"]["pixel_values"] == [4, 9]
    assert packed["multimodal_num_items"]["image_grid_thw"] == [1, 1]
    assert packed["multimodal_train_inputs"]["image_grid_thw"].tolist() == [[1, 2, 2], [1, 3, 3]]
    assert packed["multimodal_train_inputs"]["pixel_values"][:4].eq(1).all()
    assert packed["multimodal_train_inputs"]["pixel_values"][4:].eq(2).all()
    assert packed["unsplit_tokens"][0, : lengths[0]].eq(10).all()
    assert packed["unsplit_tokens"][1, : lengths[1]].eq(11).all()
    assert packed["full_loss_masks"].sum().item() == sum(i + 2 for i in range(2))


@pytest.mark.parametrize("cp_size", [1, 2])
@pytest.mark.parametrize("loss_coeff", [0.0, 0.01])
def test_allgather_packed_stage2_layout_keeps_last_document_padding(monkeypatch, cp_size: int, loss_coeff: float):
    data_module = _load_data_module(monkeypatch)
    monkeypatch.setattr(data_module, "get_args", lambda: Namespace(qsa_indexer_loss_coeff=loss_coeff))
    monkeypatch.setattr(data_module.mpu, "get_context_parallel_world_size", lambda: cp_size, raising=False)
    monkeypatch.setattr(data_module.mpu, "get_context_parallel_rank", lambda: 0, raising=False)
    monkeypatch.setattr(data_module.mpu, "get_tensor_model_parallel_world_size", lambda: 2, raising=False)
    batch = {
        "tokens": [torch.arange(5), torch.arange(2)],
        "loss_masks": [torch.ones(5, dtype=torch.int), torch.ones(2, dtype=torch.int)],
        "total_lengths": [5, 2],
        "response_lengths": [5, 2],
    }

    if loss_coeff > 0 and cp_size > 1:
        with pytest.raises(NotImplementedError, match="per-document zigzag"):
            data_module.get_batch(iter([(batch, None)]), ["tokens"], pad_multiplier=4, allgather_cp=True)
        return

    packed = data_module.get_batch(iter([(batch, None)]), ["tokens"], pad_multiplier=4, allgather_cp=True)
    params = packed["packed_seq_params"]
    assert params.cu_seqlens_q.tolist() == [0, 5, 7]
    assert params.cu_seqlens_q_padded.tolist() == [0, 5, 8 * cp_size]
    assert params.qsa_select_offsets_cpu == (0, 5, 8 * cp_size)
    assert params.cu_seqlens_q_cpu == [0, 5, 7]
    if loss_coeff > 0:
        assert params.qsa_stage2_layout_cpu == ((0, 5, 8), (5, 2))
    else:
        assert not hasattr(params, "qsa_stage2_layout_cpu")


def test_rollout_minibatch_plan_derives_from_global_batch(monkeypatch):
    data_module = _load_data_module(monkeypatch)
    args = Namespace(
        rollout_batch_size=8,
        n_samples_per_prompt=8,
        global_batch_size=32,
        num_steps_per_rollout=None,
    )

    plan = data_module.build_rollout_minibatch_plan(args, dp_size=2)

    assert plan.num_rollout_minis == 2
    assert plan.mini_rollout_batch_size == 4
    assert plan.mini_global_samples == 32
    assert plan.mini_local_sample_request == 16


def test_rollout_minibatch_plan_prefers_explicit_steps(monkeypatch):
    data_module = _load_data_module(monkeypatch)
    args = Namespace(
        rollout_batch_size=12,
        n_samples_per_prompt=8,
        global_batch_size=None,
        num_steps_per_rollout=3,
    )

    plan = data_module.build_rollout_minibatch_plan(args, dp_size=2)

    assert plan.num_rollout_minis == 3
    assert plan.mini_rollout_batch_size == 4
    assert plan.mini_global_samples == 32
    assert plan.mini_local_sample_request == 16


def test_rollout_minibatch_plan_rejects_non_divisible_prompt_groups(monkeypatch):
    data_module = _load_data_module(monkeypatch)
    args = Namespace(
        rollout_batch_size=9,
        n_samples_per_prompt=8,
        global_batch_size=None,
        num_steps_per_rollout=3,
    )

    with pytest.raises(ValueError, match="mini_rollout_batch_size must be divisible"):
        data_module.build_rollout_minibatch_plan(args, dp_size=2)


def test_log_rollout_data_creates_collective_stats_on_training_device(monkeypatch):
    data_module = _load_data_module(monkeypatch)
    loss_masks = [torch.tensor([0, 1]), torch.tensor([0, 1, 1])]
    requested_devices = []
    original_tensor = torch.tensor

    monkeypatch.setattr(data_module.mpu, "get_tensor_model_parallel_rank", lambda: 0, raising=False)
    monkeypatch.setattr(data_module.mpu, "is_pipeline_last_stage", lambda: True, raising=False)
    monkeypatch.setattr(data_module.mpu, "get_context_parallel_world_size", lambda: 1, raising=False)
    monkeypatch.setattr(
        data_module.mpu,
        "get_data_parallel_group",
        lambda with_context_parallel=True: object(),
        raising=False,
    )
    monkeypatch.setattr(data_module.device_utils, "make_current_torch_device", lambda: "training-device")

    def capture_tensor(*args, **kwargs):
        requested_devices.append(kwargs.get("device"))
        kwargs["device"] = "cpu"
        return original_tensor(*args, **kwargs)

    monkeypatch.setattr(data_module.torch, "tensor", capture_tensor)
    monkeypatch.setattr(data_module.dist, "all_reduce", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(data_module, "gather_log_data", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(data_module, "maybe_padded_total_lengths", lambda *_args, **_kwargs: None)

    args = Namespace(
        qkv_format="thd",
        is_vl_model=False,
        uses_unsplit_forward=False,
        dynamic_context_parallel=False,
        use_opd=False,
        rollout_batch_size=2,
        n_samples_per_prompt=1,
        ci_test=False,
        log_multi_turn=False,
        log_correct_samples=False,
    )
    rollout_data = {
        "total_lengths": [4, 6],
        "response_lengths": [2, 3],
        "loss_masks": loss_masks,
    }

    data_module.log_rollout_data(rollout_id=0, args=args, rollout_data=rollout_data)

    assert requested_devices == ["training-device"]


def test_concat_rollout_batches_preserves_order_and_scalar_metadata(monkeypatch):
    data_module = _load_data_module(monkeypatch)

    merged = data_module.concat_rollout_batches(
        [
            {
                "tokens": ["a", "b"],
                "total_lengths": [1, 2],
                "scores": torch.tensor([[1], [2]]),
                "weight_version": 7,
            },
            {
                "tokens": ["c"],
                "total_lengths": [3],
                "scores": torch.tensor([[3]]),
                "weight_version": 7,
            },
        ]
    )

    assert merged["tokens"] == ["a", "b", "c"]
    assert merged["total_lengths"] == [1, 2, 3]
    assert torch.equal(merged["scores"], torch.tensor([[1], [2], [3]]))
    assert merged["weight_version"] == 7


def test_get_data_iterator_uses_rollout_mini_boundaries_with_balance_data(monkeypatch):
    data_module = _load_data_module(monkeypatch)
    monkeypatch.setattr(
        data_module.mpu,
        "get_data_parallel_world_size",
        lambda with_context_parallel=False: 2,
        raising=False,
    )
    monkeypatch.setattr(data_module.mpu, "get_data_parallel_group", lambda: object(), raising=False)
    monkeypatch.setattr(data_module.mpu, "get_virtual_pipeline_model_parallel_world_size", lambda: None, raising=False)
    monkeypatch.setattr(data_module.mpu, "get_context_parallel_world_size", lambda: 1, raising=False)
    monkeypatch.setattr(data_module.device_utils, "make_current_torch_device", lambda: torch.device("cpu"))
    monkeypatch.setattr(data_module.dist, "all_reduce", lambda tensor, op=None, group=None: None)

    args = Namespace(
        balance_data=True,
        global_batch_size=32,
        micro_batch_size=4,
        use_dynamic_batch_size=False,
    )
    rollout_data = {
        "total_lengths": list(range(32)),
        data_module.ROLLOUT_MINI_LOCAL_SAMPLE_COUNTS_KEY: [16, 16],
    }

    data_iterators, num_microbatches = data_module.get_data_iterator(args, object(), rollout_data)

    assert num_microbatches == [4, 4]
    iterator = data_iterators[0]
    first_step = [iterator.get_next(["total_lengths"])["total_lengths"] for _ in range(4)]
    second_step = [iterator.get_next(["total_lengths"])["total_lengths"] for _ in range(4)]
    assert first_step[0] == [0, 1, 2, 3]
    assert first_step[-1] == [12, 13, 14, 15]
    assert second_step[0] == [16, 17, 18, 19]
    assert second_step[-1] == [28, 29, 30, 31]


def test_get_data_iterator_balance_data_without_boundaries_uses_regular_steps(monkeypatch):
    data_module = _load_data_module(monkeypatch)
    monkeypatch.setattr(
        data_module.mpu,
        "get_data_parallel_world_size",
        lambda with_context_parallel=False: 2,
        raising=False,
    )
    monkeypatch.setattr(data_module.mpu, "get_data_parallel_group", lambda: object(), raising=False)
    monkeypatch.setattr(data_module.mpu, "get_virtual_pipeline_model_parallel_world_size", lambda: None, raising=False)
    monkeypatch.setattr(data_module.mpu, "get_context_parallel_world_size", lambda: 1, raising=False)
    monkeypatch.setattr(data_module.device_utils, "make_current_torch_device", lambda: torch.device("cpu"))
    monkeypatch.setattr(data_module.dist, "all_reduce", lambda tensor, op=None, group=None: None)

    args = Namespace(
        balance_data=True,
        global_batch_size=16,
        micro_batch_size=4,
        use_dynamic_batch_size=False,
    )
    rollout_data = {"total_lengths": list(range(16))}

    _, num_microbatches = data_module.get_data_iterator(args, object(), rollout_data)

    assert num_microbatches == [2, 2]


@pytest.mark.parametrize("cp_size", [1, 8])
@pytest.mark.parametrize("text_only", [False, True])
def test_sft_multimodal_metrics_include_text_samples(monkeypatch, cp_size, text_only):
    data_module = _load_data_module(monkeypatch)
    monkeypatch.setattr(data_module.mpu, "get_tensor_model_parallel_rank", lambda: 0, raising=False)
    monkeypatch.setattr(data_module.mpu, "is_pipeline_last_stage", lambda: True, raising=False)
    monkeypatch.setattr(data_module.mpu, "get_context_parallel_world_size", lambda: cp_size, raising=False)
    monkeypatch.setattr(data_module.mpu, "get_data_parallel_group", lambda **kwargs: object(), raising=False)
    monkeypatch.setattr(data_module.device_utils, "make_current_torch_device", lambda: "cpu")
    monkeypatch.setattr(data_module.dist, "all_reduce", lambda *args, **kwargs: None)
    monkeypatch.setattr(data_module, "maybe_padded_total_lengths", lambda *args, **kwargs: None)
    captured = {}

    def capture(name, args, step, metrics, weights, reductions):
        captured.update(metrics=metrics, weights=weights, reductions=reductions)

    monkeypatch.setattr(data_module, "gather_log_data", capture)
    args = Namespace(
        loss_type="sft",
        qkv_format="thd",
        use_opd=False,
        rollout_batch_size=3,
        n_samples_per_prompt=1,
        ci_test=False,
        log_multi_turn=False,
    )
    batch = {
        "total_lengths": [10, 100, 80],
        "response_lengths": [10, 100, 80],
        "loss_masks": [],
        "learn_lengths": [4, 20, 60],
        "non_learn_lengths": [6, 80, 20],
    }
    if not text_only:
        batch["multimodal_train_inputs"] = [
            None,
            {"image_grid_thw": torch.tensor([[1, 14, 24]])},
            {
                "image_grid_thw": torch.tensor([[1, 8, 8], [1, 8, 8]]),
                "video_grid_thw": torch.tensor([[2, 4, 4]]),
                "audio_seqlens": torch.tensor([5]),
            },
        ]
    data_module.log_rollout_data(0, args, batch)

    assert "response_lengths" not in captured["metrics"]
    assert "total_lengths" not in captured["metrics"]
    for name, values in {
        "total_lengths": (10, 190 / 3, 100),
        "learn_lengths": (4, 28, 60),
        "non_learn_lengths": (6, 106 / 3, 80),
    }.items():
        for suffix, value in zip(("min", "mean", "max"), values, strict=True):
            assert captured["metrics"][f"{name}/{suffix}"] == pytest.approx(value)
        assert captured["weights"][f"{name}/mean"] == 3
    expected = {"image_count": (0, 1, 2), "multimodal_token_count": (0, 43, 84)}
    for name, values in expected.items():
        for suffix, value in zip(("min", "mean", "max"), values, strict=True):
            assert captured["metrics"][f"{name}/{suffix}"] == (0 if text_only else value)
        assert captured["weights"][f"{name}/mean"] == 3
        assert captured["reductions"][f"{name}/min"] == "min"
        assert captured["reductions"][f"{name}/max"] == "max"


@pytest.mark.parametrize("cp_size", [1, 8])
def test_gather_multimodal_metrics_weights_dp_samples_and_preserves_extrema(monkeypatch, cp_size):
    data_module = _load_data_module(monkeypatch)
    monkeypatch.setattr(data_module.mpu, "get_data_parallel_rank", lambda **kwargs: 0, raising=False)
    monkeypatch.setattr(data_module.mpu, "get_data_parallel_world_size", lambda **kwargs: 2 * cp_size, raising=False)
    monkeypatch.setattr(data_module.mpu, "get_data_parallel_src_rank", lambda **kwargs: 0, raising=False)
    monkeypatch.setattr(data_module.mpu, "get_data_parallel_group_gloo", lambda **kwargs: "dp_cp", raising=False)
    monkeypatch.setattr(data_module, "compute_rollout_step", lambda args, step: step)
    logged = []
    monkeypatch.setattr(
        data_module.tracking_utils, "log", lambda args, metrics, **kwargs: logged.append(metrics), raising=False
    )
    first = {"image_count/min": 0, "image_count/mean": 0.5, "image_count/max": 1}
    second = {"image_count/min": 3, "image_count/mean": 3, "image_count/max": 3}
    weights = {"image_count/mean": 2}

    def gather(local, output, *, dst, group):
        assert group == "dp_cp"
        output[:] = [(first, weights)] * cp_size + [(second, {"image_count/mean": 1})] * cp_size

    monkeypatch.setattr(data_module.dist, "gather_object", gather)
    result = data_module.gather_log_data(
        "rollout",
        Namespace(),
        2,
        first,
        weights,
        {"image_count/min": "min", "image_count/max": "max"},
    )
    assert result["rollout/image_count/min"] == 0
    assert result["rollout/image_count/mean"] == pytest.approx(4 / 3)
    assert result["rollout/image_count/max"] == 3
    assert logged == [result]


@pytest.mark.parametrize("strategy", ["balanced", "binpack"])
def test_vl_cp_dynamic_packing_budgets_per_document_padding(monkeypatch, strategy):
    data_module = _load_data_module(monkeypatch)
    monkeypatch.setattr(data_module.mpu, "get_data_parallel_world_size", lambda **_: 1, raising=False)
    monkeypatch.setattr(data_module.mpu, "get_data_parallel_group", lambda: object(), raising=False)
    monkeypatch.setattr(data_module.mpu, "get_virtual_pipeline_model_parallel_world_size", lambda: None, raising=False)
    monkeypatch.setattr(data_module.mpu, "get_context_parallel_world_size", lambda: 2, raising=False)
    monkeypatch.setattr(data_module.mpu, "get_context_parallel_rank", lambda: 0, raising=False)
    monkeypatch.setattr(data_module.mpu, "get_tensor_model_parallel_world_size", lambda: 2, raising=False)
    monkeypatch.setattr(data_module.device_utils, "make_current_torch_device", lambda: torch.device("cpu"))
    monkeypatch.setattr(data_module.dist, "all_reduce", lambda tensor, op=None, group=None: None)
    args = Namespace(
        sft_packing_strategy=strategy,
        balance_data=False,
        global_batch_size=3,
        micro_batch_size=1,
        use_dynamic_batch_size=True,
        max_tokens_per_gpu=8,
        allgather_cp=True,
        data_pad_size_multiplier=4,
        is_vl_model=True,
        qkv_format="thd",
        loss_type="sft",
    )
    monkeypatch.setattr(data_module, "get_args", lambda: args)
    # Raw 5+5+5 fits 16, while Bridge actually needs 8+8+8 tokens.
    rollout = {"total_lengths": [5, 5, 5]}
    iterators, counts = data_module.get_data_iterator(args, object(), rollout)
    assert counts == [2]
    assert sorted(idx for batch in iterators[0].micro_batch_indices for idx in batch) == [0, 1, 2]
    for batch in iterators[0].micro_batch_indices:
        packed = data_module.get_batch(
            iter(
                [
                    (
                        {
                            "tokens": [torch.arange(5) for _ in batch],
                            "total_lengths": [5] * len(batch),
                            "response_lengths": [5] * len(batch),
                            "loss_masks": [torch.ones(5, dtype=torch.int) for _ in batch],
                        },
                        None,
                    )
                ]
            ),
            ["tokens"],
            pad_multiplier=4,
            is_vl_model=True,
            allgather_cp=True,
        )
        assert packed["vlm_packed_seq_params"].cu_seqlens_q[-1] <= 16


@pytest.mark.parametrize("extra_slots", [1, 3])
def test_binpack_schedule_marks_extra_slots_as_dummy(monkeypatch, extra_slots):
    data_module = _load_data_module(monkeypatch)
    lengths = [6, 4, 3, 3]
    real = data_module.get_binpack_partitions(lengths, 10)
    partitions, dummy = data_module._get_seqlen_partitions_with_dummy_padding(
        lengths, len(real) + extra_slots, 10, packing_strategy="binpack"
    )
    assert partitions[: len(real)] == real
    assert dummy == set(range(len(real), len(partitions)))
    assert sorted(i for j, part in enumerate(partitions) if j not in dummy for i in part) == list(range(len(lengths)))
    assert all(sum(lengths[i] for i in part) <= 10 for part in partitions)


def test_binpack_rejects_schedule_smaller_than_assignment(monkeypatch):
    data_module = _load_data_module(monkeypatch)
    with pytest.raises(ValueError, match="synchronized count"):
        data_module._get_seqlen_partitions_with_dummy_padding([6, 6], 1, 10, packing_strategy="binpack")
