# stable-envs-state-replay

CPU experiment on [stable-envs](https://github.com/galilai-group/stable-envs) 0.1.0
(GalilAI, [arXiv:2611.23151](https://arxiv.org/abs/2611.23151)): store simulator
states instead of frames, then render those states again.

The pass bars are frozen in
[`experiments/exp1_cpu_state_replay/config.yaml`](experiments/exp1_cpu_state_replay/config.yaml)
**before** the measured run. This README's result table is filled in from that
run; do not treat this file as the pre-registration.

## Run

```bash
cd apps/stable-envs-state-replay
bash run_all.sh
```

`stable-envs` is installed from PyPI. It is not vendored. Upstream licence
texts are in `LICENSE`, `NOTICE`, `THIRD_PARTY_NOTICES.md`, and `LICENSES/`.
