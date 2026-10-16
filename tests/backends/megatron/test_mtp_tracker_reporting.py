# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Acceptance reporting must not reduce or rescale current MCore losses
twice."""

import sys
import textwrap
import types
from pathlib import Path

import pytest
import torch


@pytest.mark.parametrize("total,expected", [(8.0, 0.75), (0.0, None)])
def test_current_tracker_is_reduced_once_and_loss_keeps_microbatch_average(monkeypatch, total, expected):
    calls = []

    class Helper:
        tracker = {
            "loss_values": torch.tensor([1.6]),
            "correct_values": torch.tensor([total * 0.75]),
            "total_values": torch.tensor([total]),
        }

        @staticmethod
        def reduce_metrics_in_tracker():
            calls.append("reduce")

        @staticmethod
        def clean_metrics_in_tracker():
            calls.append("clean")
            for value in Helper.tracker.values():
                value.zero_()

    module = types.ModuleType("megatron.core.transformer.multi_token_prediction")
    module.MTPLossLoggingHelper = Helper
    monkeypatch.setitem(sys.modules, module.__name__, module)
    path = Path(__file__).resolve().parents[3] / "relax/backends/megatron/model.py"
    source = path.read_text()
    start = source.index("        mtp_losses = None\n")
    end = source.index("        # per train step log.", start)
    scope = {
        "torch": torch,
        "num_microbatches": [2],
        "step_id": 0,
        "args": types.SimpleNamespace(
            enable_mtp_training=True, ci_test=False, calculate_per_token_loss=True, mtp_loss_scaling_factor=0.1
        ),
    }
    exec(compile(textwrap.dedent(source[start:end]), str(path), "exec"), scope)
    assert calls == ["reduce", "clean"]
    assert scope["mtp_losses"] == pytest.approx(0.8)
    assert scope["mtp_acceptance"] == expected
