# Synthetic H20 benchmark results

All inputs below are generated randomly. No customer data or customer document-length lists are included.

Numbers are CUDA-event medians in milliseconds from three alternating-order rounds of nine samples. Forward plus backward includes fresh metadata/routes and allocation, excludes score/top-k and activation recomputation.

The source baseline is Relax `d16f2e3ec45be059db6377295090b1d49257892c`; NVIDIA QSA is PR2080 `f4fa4747f7e48a436cf138023b321cef9b615e67`.

## H20 with 6Q and 1KV

| Case           | Physical tokens | Documents incl padding |     v2 | Grouped | Mixed0 core | NV whole indexed | NV Flash plus per doc |
| -------------- | --------------: | ---------------------: | -----: | ------: | ----------: | ---------------: | --------------------: |
| boundaries     |           38167 |                     11 | 123.66 |  107.94 |         N/A |           131.09 |                113.33 |
| dense-ragged   |           11712 |                     13 |   5.04 |   29.78 |        3.13 |            14.89 |                  2.72 |
| packed-short64 |           65536 |                     14 |  89.15 |  182.28 |         N/A |           203.24 |                 99.84 |
| packed-mixed64 |           65536 |                     14 | 195.42 |  182.83 |         N/A |           206.35 |                146.55 |
| single-16384   |           16384 |                      1 |  61.53 |   47.17 |         N/A |            60.79 |                 48.72 |
| single-32768   |           32768 |                      1 | 236.83 |   95.55 |         N/A |           127.02 |                127.20 |
| single-65536   |           65536 |                      1 | 942.30 |  205.78 |         N/A |           264.81 |                266.91 |

Raw measurements: [JSON](data/reference/h20_20261104/h20-6q.json.gz), [CSV](data/reference/h20_20261104/h20-6q-medians.csv). Whole-bitmap timings, all five metrics, numeric errors, memory sizes and individual samples are retained there.

## H20 with 12Q and 1KV

| Case            | Physical tokens | Documents incl padding |     v2 | Grouped | Mixed0 core | NV whole indexed | NV Flash plus per doc |
| --------------- | --------------: | ---------------------: | -----: | ------: | ----------: | ---------------: | --------------------: |
| packed-short64  |           65536 |                     14 | 173.95 |  181.21 |      181.61 |           210.18 |                179.14 |
| packed-mixed64  |           65536 |                     14 | 381.99 |  181.54 |      184.56 |           213.20 |                179.87 |
| packed-short128 |          131072 |                     27 | 342.16 |  362.42 |      367.02 |           422.15 |                354.32 |
| packed-mixed128 |          131072 |                     14 | 921.85 |  369.23 |      384.35 |           478.80 |                468.30 |

Raw measurements: [JSON](data/reference/h20_20261104/h20-12q.json.gz), [CSV](data/reference/h20_20261104/h20-12q-medians.csv). Whole-bitmap timings, all five metrics, numeric errors, memory sizes and individual samples are retained there.

## Interpretation

For the 6Q short-document 64K pack, document-wise NV plus Flash takes 12.0% more time than v2. Adding a long document changes the comparison: document-wise NV reduces latency by 25.0% versus v2 in packed-mixed64.

The tuned Grouped baseline also matters. On the 6Q single-64K control, NV indexed takes 28.7% more time than Grouped, despite beating v2. On the 12Q mixed-128K pack, NV plus Flash takes 21.8% more time than the original mixed0 core. A v2-only comparison would miss these stronger baselines.

Compare individual document lengths and head geometry, as well as total packed length. The packed-short64 and packed-mixed64 cases both have a 65,536-token physical buffer but different document-length compositions.

N/A marks the unchanged 12Q-only mixed long-document assertion. Whole-buffer 128K bitmap cases are separately skipped by the benchmark's 2 GiB bitmap-body guard; this is not a claim that the NVIDIA API cannot support those shapes.

Each timed supported cell passed the sampled independent FP32 attention checks and the dense-gradient checkpoint/recompute check. These checks do not establish full training equivalence or convergence.

Profiles, when supplied, were collected in a separate one-round diagnostic run. Their durations are not used in these tables.
