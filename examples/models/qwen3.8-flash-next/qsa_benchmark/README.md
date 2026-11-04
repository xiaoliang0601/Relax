# H20 packed QSA benchmark

This benchmark compares Triton v2, Grouped, the original mixed dispatcher, and NVIDIA PR2080 using **fully synthetic BF16 inputs**. It targets 6Q/1KV/D256 packed attention on H20 and also covers the historical 12Q/128K geometry. It runs without model weights, a tokenizer, Megatron, or a Relax installation.

The checked-in data consists of random document layouts, seeds, reference measurements, and synthetic profiler traces. There are no customer samples or training artifacts. Large Q/K/V tensors are generated locally rather than checked into Git.

## Results and the optimization question

All numbers below are H20 forward-plus-backward milliseconds, including fresh routing/metadata, using 27 samples per cell:

| 6Q workload             | Triton v2 | Grouped core | NV whole indexed | Flash plus NV per document |
| ----------------------- | --------: | -----------: | ---------------: | -------------------------: |
| 64K short-document pack |     89.15 |       182.28 |           203.24 |                      99.84 |
| 64K mixed-length pack   |    195.42 |       182.83 |           206.35 |                     146.55 |
| Single 64K document     |    942.30 |       205.78 |           264.81 |                     266.91 |

The short and mixed packs have the same 65,536-token physical length. Adding a long document changes the best backend. Tuned Grouped is also a strong baseline; a comparison against v2 alone misses that result.

We would appreciate investigation of native variable-length packed SM90 execution, backward scheduling for 6Q/1KV, and frozen-selection metadata reuse. Diagnostic traces identify dense SM90 backward as the main cost for NV in the short-document pack. They do not establish an atomic-contention bottleneck.

See [RESULTS.md](RESULTS.md) for all 6Q and 12Q cases, including 128K, and [the reference data](data/reference/h20_20261104).

## Run from a checkout

The runner loads the existing repository operator files directly. It verifies their file hashes against Relax baseline `d16f2e3ec45be059db6377295090b1d49257892c`; adding a benchmark commit does not invalidate the check. No operator sources are copied into this directory. The older sibling `benchmark_qsa.py` remains unchanged.

The reference environment was Python 3.12, PyTorch `2.11.0+cu129`, Triton `3.6.0`, NumPy `1.26.4`, FlashAttention `2.7.4.post1`, cuDNN Frontend `1.33.0`, and CUTLASS DSL `4.8.1` with CUDA 12 support. NVIDIA QSA is pinned to PR2080 `f4fa4747f7e48a436cf138023b321cef9b615e67` and its 31 Python source blobs are checked.

From the repository root:

```bash
python examples/models/qwen3.8-flash-next/qsa_benchmark/benchmark.py \
  --describe --cases packed-short64,packed-mixed64

CUDA_VISIBLE_DEVICES=0 python examples/models/qwen3.8-flash-next/qsa_benchmark/benchmark.py \
  --heads 6 --cases packed-short64,packed-mixed64,single-32768,single-65536 \
  --selector-mode compile --selector-timing --output /tmp/qsa-h20-6q

CUDA_VISIBLE_DEVICES=0 python examples/models/qwen3.8-flash-next/qsa_benchmark/benchmark.py \
  --heads 12 --cases packed-short64,packed-mixed64,packed-short128,packed-mixed128 \
  --selector-mode compile --selector-timing --output /tmp/qsa-h20-12q
```

The default repository location is resolved from the script, so it also works from another working directory. `--repo` can select a different checkout. Use `--backends v2,grouped,mixed0` when NVIDIA QSA is unavailable.

For an already installed cuDNN Frontend environment, `--qsa-source CHECKOUT/python/cudnn/QSA` overlays the pinned QSA Python sources from a NVIDIA checkout without installing binary dependencies. When testing an optimized NVIDIA tree, use a separate benchmark copy and `record_nv_sources.py --qsa-source CHECKOUT/python/cudnn/QSA --commit COMMIT --output nv-source-manifest.json` to record that experimental baseline explicitly.

The selected GPU must be idle. The runner reads its process state and never changes another process or resource job.

## Synthetic data

[data/synthetic_cases.json](data/synthetic_cases.json) is the input dataset: ten explicit document-length layouts, padding, geometry, seeds, and the RNG recipe. It includes short-only, ragged-boundary, mixed-length 64K/128K packs and single-document controls. The 64K cases use 16,384-token physical alignment; the 128K cases use 8,192-token alignment.

Generate a new random layout dataset, using only Python's standard library:

```bash
python examples/models/qwen3.8-flash-next/qsa_benchmark/generate_cases.py \
  --seed 12345 --output /tmp/qsa-cases.json

python examples/models/qwen3.8-flash-next/qsa_benchmark/benchmark.py \
  --cases-file /tmp/qsa-cases.json --describe
```

The layout file determines document lengths. `--seed` on the benchmark changes tensor/indexer randomness while retaining those lengths; without it, the layout file's seed is used. Exact tensor/top-k fingerprints can differ across PyTorch/CUDA versions, so versions and fingerprints are retained in the reference data.

To materialize the random tensor data locally, use an existing CUDA environment:

```bash
CUDA_VISIBLE_DEVICES=0 python examples/models/qwen3.8-flash-next/qsa_benchmark/benchmark.py \
  --heads 6 --cases packed-short64 --export-only \
  --export-inputs /tmp/qsa-generated --output /tmp/qsa-generated/metadata
```

The exported `.pt` contains Q/K/V, dO, selected IDs/bitset, positions, document IDs, boundaries, post-RoPE indexer queries, and pooled keys. It can be read with `torch.load(path, map_location="cpu", weights_only=True)`. Export occurs outside timing. Generated tensor files are ignored by Git.

Selection uses document-local four-token blocks: at most 512 complete blocks plus the current zero-to-three-token causal tail, shared across attention heads. Visible complete groups are all selected below the budget. Beyond the budget, score/top-k uses random post-RoPE indexer features. Expanded routes have 2,052 INT32 slots with `-1` for invalid entries. Documents and tail padding remain isolated; padding has zero output gradient but still incurs attention work.

## Checked-in measurements and provenance

- `data/reference/h20_20261104/*-medians.csv`: directly viewable pooled median tables.
- `data/reference/h20_20261104/samples/*.csv`: all **7,830 individual samples**, split into small files by run and case.
- `data/reference/h20_20261104/h20-6q.json.gz` and `h20-12q.json.gz`: complete original measurement reports, including numerical checks, settings, versions, shapes, fingerprints, and all timing samples.
- `data/reference/h20_20261104/traces/`: four synthetic Chrome traces and kernel summaries. Filename labels are portable; device UUIDs are omitted. Event hashes show the metadata adjustment did not alter recorded events.
- `provenance/measured_runner.py.txt` and `provenance/reference.json`: the exact original measurement script and hashes. The original formal reports retain that script's SHA256.

The formal 3-by-9 tables were measured with the archived standalone predecessor. This repository integration changes source loading, explicit layout-file input, and export support. CPU checks and source validation are recorded in `provenance/integration.json`; the original H20 resource had no available containers for a new GPU smoke run. The formal measurements are not relabeled as new measurements of the integrated runner. The mathematical and timing functions have identical ASTs to the measured predecessor, apart from docstrings.

## Timing and limitations

Each supported cell checks sampled FP32 output and dQ/dK/dV, plus dense-gradient checkpoint/recompute with the original selection reused. The formal reference has 58 passing cells, six unsupported mixed long-document 6Q cases, and two 128K whole-bitmap cases skipped by a 2 GiB bitmap-body guard. These are operator checks, not a training-convergence validation.

Timing alternates backend order for three rounds of nine samples after compilation and warmup. Total forward/backward includes fresh metadata, routes, allocation, and public autograd wrappers. Prepared forward is separate. Each timed backward gets a fresh forward graph outside its interval, with no retained backward graph reused. Score/top-k and activation recomputation are excluded from the total metric. Optional selector timing covers one TP query-row shard after projection/norm/pooling/RoPE; those operations, TP communication, and selector backward are excluded.

`grouped` measures the direct core, including with 6Q. The unchanged production mixed selection adapter still requires 12Q. `mixed0` measures the original dispatch core with Flex disabled; its long-document branch retains the 12Q assertion. `nv-flash-per-doc` is an experimental adapter with routing, metadata, splitting, and merging included. It is not an unchanged production integration.

Grouped uses forward BK64 and backward BK32/four warps/three stages, as specified in the supplied workload details. The three backward settings are configurable and recorded. `--max-bitmap-gib` limits bitmap body bytes, not total workspace; increasing it explicitly permits larger bitmap experiments.

Add `--profile --rounds 1 --repeats 1` to export warm synthetic traces. Use unprofiled runs for latency comparisons.
