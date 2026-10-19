# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Compile selector postprocessing without requiring Megatron or TP groups."""

import importlib.util
import sys
from pathlib import Path

import pytest
import torch


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Selector compilation requires CUDA BF16 matmul")
@pytest.mark.parametrize("ties", [False, True])
def test_compiled_selector_preserves_fp32_scores_causal_mask_and_topk(ties):
    path = Path(__file__).resolve().parents[3] / "relax/backends/megatron/megatron_patch/qsa_mixed_ops/selector.py"
    spec = importlib.util.spec_from_file_location("qsa_selector_compile_test", path)
    selector = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = selector
    spec.loader.exec_module(selector)
    torch.manual_seed(0)
    query = torch.randn(9, 4, 128, device="cuda", dtype=torch.bfloat16)
    keys = torch.randn(600, 128, device="cuda", dtype=torch.bfloat16)
    if ties:
        query.zero_()
    positions = torch.tensor([0, 1, 3, 4, 7, 2047, 2048, 2051, 2399], device="cuda")
    scores = selector.index_scores(query, keys)
    assert scores.dtype == torch.float32
    logits = scores.relu().sum(-1) / 128**0.5
    visible = torch.arange(len(keys), device="cuda")[None, :] < (positions[:, None] + 1) // 4
    values, expected = logits.masked_fill(~visible, -torch.inf).topk(512, dim=-1)
    expected = expected.masked_fill(values == -torch.inf, -1).to(torch.int32)
    actual = selector.compiled_score_topk(query, keys, positions, 512)
    assert actual.dtype == torch.int32
    assert torch.equal(actual, expected)
