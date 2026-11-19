# exp1_cpu_state_replay

**Status: PASS**

Storing only stable-envs simulator states at the default appearance and re-rendering on the fly gives at least 90% disk saving versus lossless PNG and versus uncompressed uint8 npz of the same episodes, the re-rendered frames are bit-identical to the frames drawn at collection, and state-replay throughput (torch.load of the states plus env.render) is at least the faster of the two frame-loading baselines.

Config SHA-256 `8332d484bed43d70d9c9ed9fed569d56f2670d74345691f3e66d6c95b4eccefb`.

| Env | PNG saving % | npz saving % | Lossless min | Replay fps | Load fps | H1 | H2 | H4 |
|---|---:|---:|---:|---:|---:|---|---|---|
| TwoRoom | 99.10 ± 0.01 | 99.98 ± 0.00 | 1.000 | 43297.9 ± 515.9 | 23177.8 ± 889.1 | yes | yes | yes |
| CartPole | 91.68 ± 0.41 | 99.97 ± 0.00 | 1.000 | 8686.5 ± 469.9 | 3503.0 ± 23.4 | yes | yes | yes |

H3 (every cell lossless): yes.

## Cells

| Env | Seed | Frames | State B | PNG B | npz B | PNG % | npz % | Lossless | Replay fps | npz fps | PNG fps |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| TwoRoom | 0 | 128 | 4061 | 454365 | 19267850 | 99.11 | 99.98 | 1.000 | 42837.0 | 24145.1 | 2847.0 |
| TwoRoom | 1 | 128 | 4061 | 449556 | 19267850 | 99.10 | 99.98 | 1.000 | 43201.5 | 22991.9 | 2844.1 |
| TwoRoom | 2 | 128 | 4061 | 449752 | 19267850 | 99.10 | 99.98 | 1.000 | 43855.1 | 22396.3 | 2593.3 |
| CartPole | 0 | 128 | 27801 | 350438 | 92160266 | 92.07 | 99.97 | 1.000 | 9137.6 | 3529.2 | 849.6 |
| CartPole | 1 | 128 | 27801 | 318014 | 92160266 | 91.26 | 99.97 | 1.000 | 8722.0 | 3484.1 | 885.5 |
| CartPole | 2 | 128 | 27801 | 335914 | 92160266 | 91.72 | 99.97 | 1.000 | 8199.8 | 3495.7 | 864.4 |
