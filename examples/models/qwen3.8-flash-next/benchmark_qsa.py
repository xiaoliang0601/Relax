# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Single-GPU microbenchmark for the public Qwen3.8 Grouped QSA implementation.

Loads source files directly: no Relax installation, Megatron or model weights.
Synthetic BF16 indexer inputs represent values AFTER projection/norm/pooling/RoPE.
Selector timing excludes those operations and TP communication. Attention timing
uses all query rows; selector timing models one configurable TP row shard.
The TP options do not change attention head counts or run collectives.
JIT/torch.compile warmup is excluded. This is not a full training benchmark.
It forces the Grouped backend; the mixed dispatcher may choose Flex or Flash
for the same sequence lengths during training.

Requires a CUDA GPU with BF16 support, PyTorch with CUDA BF16-to-FP32
torch.mm(out_dtype=...) support, and a compatible Triton installation.
The default synthetic attention shape is [tokens, 12 query heads, 256],
with one KV head. Indexer queries have four heads of dimension 128.
Each selected block contains four tokens; top-k selects up to 512 blocks.

Run from the repository root, using the eager selector first:

    python examples/models/qwen3.8-flash-next/benchmark_qsa.py --repo . --selector eager

To time the compiled selector or a packed pair of documents:

    python examples/models/qwen3.8-flash-next/benchmark_qsa.py --repo . --selector compile
    python examples/models/qwen3.8-flash-next/benchmark_qsa.py --repo . --lengths 8192,8192

Run on a single GPU. A downloaded copy can use --repo /path/to/Relax to
point at the checked-out codex/qwen38-public-code branch. JSON results
on stdout include GPU/software versions, tensor shapes/strides and timings.
Larger token counts require substantial GPU memory, including the dense
expanded route-ID buffer; begin with the default 16384-token workload.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import statistics
import sys
from pathlib import Path
from typing import Any


def load_source(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--lengths", default="16384", help="Comma-separated packed document lengths; each >2051")
    parser.add_argument("--selector", choices=("compile", "eager"), default="compile")
    parser.add_argument("--selector-tp-size", type=int, default=2)
    parser.add_argument("--selector-tp-rank", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=10)
    args = parser.parse_args()
    try:
        lengths = tuple(int(value) for value in args.lengths.split(","))
    except ValueError:
        parser.error("--lengths must contain comma-separated integer document lengths")
    if not lengths or min(lengths) <= 2051 or max(lengths) > 131072:
        parser.error("Use document lengths in [2052,131072]; this benchmark covers the Grouped sparse path")
    total = sum(lengths)
    if total > 131072:
        parser.error("This benchmark limits the complete packed buffer to 131072 tokens")
    if args.selector_tp_size < 1 or total % args.selector_tp_size:
        parser.error("Packed token count must be divisible by selector TP size")
    if not 0 <= args.selector_tp_rank < args.selector_tp_size or args.repeats < 1 or args.warmup < 1:
        parser.error("Invalid selector TP rank or repeat/warmup count")
    source = args.repo.expanduser().resolve() / "relax/backends/megatron/megatron_patch/qsa_mixed_ops"
    if not all((source / name).is_file() for name in ("grouped.py", "selector.py")):
        parser.error("--repo must point to a checkout containing qsa_mixed_ops/grouped.py and selector.py")

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU and compatible PyTorch/Triton are required")

    import triton

    os.environ.setdefault("RELAX_QWEN38_QSA_GROUPED_BACKWARD_WARPS", "4")
    os.environ.setdefault("RELAX_QWEN38_QSA_GROUPED_BACKWARD_STAGES", "3")
    os.environ.setdefault("RELAX_QWEN38_QSA_GROUPED_BACKWARD_BLOCK_K", "32")
    grouped = load_source(source / "grouped.py", "nv_qsa_grouped")
    selector = load_source(source / "selector.py", "nv_qsa_selector")
    torch.manual_seed(20261016)
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    device, dtype, scale = "cuda", torch.bfloat16, 256**-0.5
    q = torch.randn((total, 12, 256), device=device, dtype=dtype, requires_grad=True)
    k = torch.randn((total, 1, 256), device=device, dtype=dtype, requires_grad=True)
    v = torch.randn_like(k, requires_grad=True)
    upstream = torch.randn_like(q)
    index_q = torch.randn((total, 4, 128), device=device, dtype=dtype)
    pooled_keys = [torch.randn((length // 4, 128), device=device, dtype=dtype) for length in lengths]
    bounds = [0]
    for length in lengths:
        bounds.append(bounds[-1] + length)

    @torch.no_grad()
    def select(first: int, last: int) -> Any:
        selected = torch.full((last - first, 512), -1, device=device, dtype=torch.int32)
        for doc, (start, end) in enumerate(zip(bounds, bounds[1:])):
            begin, finish = max(first, start), min(last, end)
            for a in range(begin, finish, 512):
                b = min(a + 512, finish)
                positions = torch.arange(a - start, b - start, device=device)
                keys = pooled_keys[doc]
                if args.selector == "compile":
                    ids = selector.compiled_score_topk(index_q[a:b], keys, positions, 512)
                else:
                    scores = selector.index_scores(index_q[a:b], keys)
                    logits = scores.relu().sum(-1) / 128**0.5
                    visible = torch.arange(keys.shape[0], device=device)[None, :] < (positions[:, None] + 1) // 4
                    values, ids = logits.masked_fill(~visible, -torch.inf).topk(512, dim=-1)
                    ids = ids.masked_fill(values == -torch.inf, -1).to(torch.int32)
                selected[a - first : b - first] = ids
        return selected

    @torch.no_grad()
    def expand(selected: Any) -> Any:
        # Same document-local 4-token blocks + causal partial tail as the adapter.
        starts = torch.cat(
            [torch.full((length,), start, device=device, dtype=torch.int32) for length, start in zip(lengths, bounds)]
        )
        positions = torch.arange(total, device=device, dtype=torch.int32) - starts
        offsets = torch.arange(4, device=device, dtype=torch.int32)
        complete = starts[:, None, None] + selected[:, :, None] * 4 + offsets
        complete = complete.masked_fill(selected[:, :, None] < 0, -1).flatten(1)
        tail = starts[:, None] + ((positions[:, None] + 1) // 4) * 4 + offsets
        tail = tail.masked_fill(tail > (starts + positions)[:, None], -1)
        return torch.cat((complete, tail), dim=1).contiguous()

    selected = select(0, total)
    indices = expand(selected)

    def forward() -> Any:
        return grouped.attention_bf16(q, k, v, indices, scale, block_k=64, gradient_group=None)

    def gradients(output: Any) -> Any:
        return torch.autograd.grad(output, (q, k, v), upstream)

    def measure(call: Any, setup: Any = None) -> dict[str, float]:
        for _ in range(args.warmup):
            result = call(setup()) if setup else call()
            del result
        torch.cuda.synchronize()
        times = []
        for _ in range(args.repeats):
            inputs = setup() if setup else None
            begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            begin.record()
            result = call(inputs) if setup else call()
            end.record()
            end.synchronize()
            times.append(begin.elapsed_time(end))
            del result, inputs
        return {
            "mean_ms": sum(times) / len(times),
            "median_ms": statistics.median(times),
            "min_ms": min(times),
            "max_ms": max(times),
        }

    @torch.no_grad()
    def inference_forward() -> Any:
        return forward()

    def forward_backward() -> Any:
        return gradients(forward())

    shard = total // args.selector_tp_size
    first = args.selector_tp_rank * shard
    results = {
        "selector_score_topk_one_tp_shard": measure(lambda: select(first, first + shard)),
        "route_expansion_full_pack": measure(lambda: expand(selected)),
        "attention_forward": measure(inference_forward),
        "attention_backward_including_grad_allocation": measure(gradients, setup=forward),
        "attention_forward_backward": measure(forward_backward),
    }
    properties = torch.cuda.get_device_properties(0)
    report = {
        "backend": "grouped_bf16_forced",
        "gpu": properties.name,
        "torch": torch.__version__,
        "triton": triton.__version__,
        "dtype": str(dtype),
        "lengths": lengths,
        "packed_tokens": total,
        "selection_block_tokens": 4,
        "topk_blocks": 512,
        "scale": scale,
        "selector_mode": args.selector,
        "selector_tp_size": args.selector_tp_size,
        "selector_tp_rank": args.selector_tp_rank,
        "tensors": {
            name: {"shape": list(t.shape), "stride_elements": list(t.stride())}
            for name, t in (("q", q), ("k", k), ("v", v), ("indices", indices))
        },
        "launch": {
            key: os.environ[key]
            for key in (
                "RELAX_QWEN38_QSA_GROUPED_BACKWARD_WARPS",
                "RELAX_QWEN38_QSA_GROUPED_BACKWARD_STAGES",
                "RELAX_QWEN38_QSA_GROUPED_BACKWARD_BLOCK_K",
            )
        },
        "timings": results,
        "limitations": "Synthetic post-RoPE indexer inputs; no full indexer projection/pooling/RoPE, "
        "TP collective, mixed dispatch, Flash/Flex path or optimizer. Not a correctness test.",
    }
    sys.stdout.write(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
