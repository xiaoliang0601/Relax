# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import math
from argparse import Namespace
from unittest.mock import Mock, call

import pytest

from relax.utils.training import train_metric_utils


class FakeTimer:
    def __init__(self):
        self.seq_lens = [100, 200]
        self.response_lens = [50, 100]

    def log_dict(self):
        return {
            "actor_train": 2.0,
            "log_probs": 1.0,
            "ref_log_probs": 1.0,
            "train_wait": 1.0,
            "train": 2.0,
        }

    def reset(self):
        pass


def _collect_perf_metrics(monkeypatch, peak_tflops):
    timer = FakeTimer()
    flops_counter = Mock()
    flops_counter.estimate.return_value = (300.0, peak_tflops)
    logged_metrics = {}

    monkeypatch.setattr(train_metric_utils, "Timer", lambda: timer)
    monkeypatch.setattr(
        train_metric_utils.tracking_utils,
        "log",
        lambda _args, metrics, step_key: logged_metrics.update(metrics),
    )

    args = Namespace(wandb_always_use_train_step=False)
    train_metric_utils.log_perf_data_raw(
        rollout_id=3,
        args=args,
        is_primary_rank=True,
        flops_counter=flops_counter,
        world_size=2,
    )

    return logged_metrics


def test_log_perf_data_raw_omits_non_finite_peak_metrics(monkeypatch):
    logged_metrics = _collect_perf_metrics(monkeypatch, float("inf"))

    assert "perf/actor_train_tflops" in logged_metrics
    assert "perf/device_peak_tflops" not in logged_metrics
    assert "perf/mfu/actor_train" not in logged_metrics
    assert all(math.isfinite(value) for value in logged_metrics.values() if isinstance(value, float))


def test_log_perf_data_raw_reports_mfu_for_known_peak(monkeypatch):
    logged_metrics = _collect_perf_metrics(monkeypatch, 100.0)

    assert logged_metrics["perf/device_peak_tflops"] == 100.0
    assert logged_metrics["perf/mfu/actor_train"] == 0.75


@pytest.mark.parametrize("model_type", ["qwen4_exp", "qwen3"])
def test_log_perf_data_raw_forwards_training_policy(monkeypatch, model_type):
    timer = FakeTimer()
    timer.images_seqlens = [256]
    counter = Mock(config=Namespace(model_type=model_type))
    counter.estimate.return_value = (25600.0, 989.0)
    logged = {}
    monkeypatch.setattr(train_metric_utils, "Timer", lambda: timer)
    monkeypatch.setattr(train_metric_utils.tracking_utils, "log", lambda _args, data, step_key: logged.update(data))
    args = Namespace(
        wandb_always_use_train_step=False,
        freeze_vision_model=True,
        freeze_vision_projection=True,
        freeze_params_name_list=[r"per_layer_embedding\.|self_attention\.indexer\."],
        enable_mtp_training=True,
        mtp_num_layers=1,
        mtp_detach_paths=(),
        mtp_only_training=False,
    )
    train_metric_utils.log_perf_data_raw(3, args, True, counter, world_size=128)
    estimate_kwargs = dict(
        batch_seqlens=[100, 200],
        delta_time=1.0,
        images_seqlens=[256],
        audio_seqlens=None,
        image_grid_thw=None,
        freeze_language_model=False,
        lora_rank=0,
        only_train_params_name_list=None,
        **{k: v for k, v in vars(args).items() if k != "wandb_always_use_train_step"},
    )
    counter.estimate.assert_has_calls([call(**estimate_kwargs), call(**estimate_kwargs, forward_only=True)])
    assert counter.estimate.call_count == 2
    assert logged["perf/actor_train_tflops"] == 100.0  # Divide by 128 ranks and 2s exactly once.
    assert math.isclose(logged["perf/mfu/actor_train"], 100.0 / 989.0)
