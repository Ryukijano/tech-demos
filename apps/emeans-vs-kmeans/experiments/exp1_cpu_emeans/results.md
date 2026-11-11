# exp1 CPU e-means vs Lloyd — PARTIAL

Primary metric: train MSE ratio at 2.5 epochs versus FAISS Lloyd at 25 iterations.
Bar: ratio ≤ 1.01 and wall-clock ratio ≤ 0.1.
Mean ± sample std (ddof=1) over seeds 0, 1, 2.

| dataset | n, d, k | Lloyd train MSE | e-means train MSE | ratio | wall ratio | MSE bar | time bar |
|---|---|---:|---:|---:|---:|---|---|
| mixture | 20000, 32, 512 | 11.4711 ± 0.0766 | 11.3739 ± 0.0896 | 0.9916 ± 0.0127 | 0.475 ± 0.023 | met | miss |
| stretched | 20000, 64, 512 | 33.4539 ± 0.0196 | 33.4184 ± 0.0273 | 0.9989 ± 0.0006 | 0.165 ± 0.005 | met | miss |
| mnist | 10000, 784, 256 | 23.9264 ± 0.0500 | 23.9253 ± 0.0376 | 1.0000 ± 0.0021 | 0.208 ± 0.009 | met | miss |

Test MSE is report-only.

| dataset | Lloyd test MSE | e-means test MSE |
|---|---:|---:|
| mixture | 12.2062 ± 0.1125 | 12.0508 ± 0.0899 |
| stretched | 36.4132 ± 0.0236 | 36.3798 ± 0.0371 |
| mnist | 25.9057 ± 0.1930 | 25.8333 ± 0.0812 |
