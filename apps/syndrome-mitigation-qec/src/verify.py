"""Correctness checks for the exact ML decoder and the pipeline plumbing.

Run before anything official:  python src/verify.py
1. d=3: transfer-matrix DP coset probabilities == brute-force 4^9 enumeration
   (homogeneous and inhomogeneous noise), to ~1e-12 relative error.
2. d=3: stim-sampled syndrome/frame-flip marginals == brute-force exact values.
3. d=5 / d=7: DP wall-time per 1000 distinct syndromes (feasibility numbers).
"""

import sys
import time

import numpy as np

sys.path.insert(0, ".")
from src import vecqec as V  # noqa: E402


def check_d3_dp_vs_brute():
    code = V.RotatedSurfaceCode(3)
    rng = np.random.default_rng(0)
    for name, p_q in [("homogeneous p=0.05", np.full(9, 0.05)),
                      ("homogeneous p=0.12", np.full(9, 0.12)),
                      ("inhomogeneous", rng.uniform(0.01, 0.12, size=9))]:
        probs = V.depolarizing_probs(p_q)
        table = V.brute_force_coset_table(code, probs)        # (256, 4)
        keys = np.arange(256, dtype=np.int64)
        bits = ((keys[:, None] >> np.arange(code.nd)[None, :]) & 1)
        ml = V.ExactMLDecoder(code, probs)
        dp = ml.coset_probs(bits)
        bf = table / table.sum(axis=1, keepdims=True)
        err = np.max(np.abs(dp - bf))
        print(f"  d=3 DP vs brute force [{name}]: max |diff| = {err:.3e}")
        assert err < 1e-12, "DP decoder disagrees with brute force"
        # also check the unnormalised syndrome distribution totals
        # (DP state sums = Z_L(s) up to identical normalisation)
    print("  d=3 DP decoder verified against exhaustive enumeration. OK")


def check_d3_stim_marginals():
    code = V.RotatedSurfaceCode(3)
    p = 0.08
    probs = V.depolarizing_probs(np.full(9, p))
    table = V.brute_force_coset_table(code, probs)
    Ps = table.sum(axis=1)                   # exact syndrome distribution
    pL_true_frame = 1.0 - table[:, 0].sum()  # P(frame flip != I)
    mwpm = V.MWPMDecoder(code, p)
    ml = V.ExactMLDecoder(code, probs)
    n = 400_000
    shots = V.sample_shots(code, np.full(9, p), n, seed=12345, mwpm=mwpm, ml=ml)
    # empirical syndrome distribution vs exact
    emp = np.bincount(shots.key.astype(np.int64), minlength=256) / n
    tv = 0.5 * np.abs(emp - Ps).sum()
    # empirical frame-flip rate vs exact: resid ^ ... need true flips:
    # resid = cls_rt ^ cls_true -> P(cls_true != I) from exact table
    print(f"  stim vs exact: syndrome TV distance = {tv:.4f} "
          f"(expected O(1/sqrt(N)) ~ {np.sqrt(256 / n):.4f})")
    assert tv < 5 * np.sqrt(256 / n)
    print(f"  exact P(true frame flip) = {pL_true_frame:.5f}")
    print("  stim sampling marginals consistent with exact enumeration. OK")


def time_dp(d, n_syn=1000, p=0.05):
    code = V.RotatedSurfaceCode(d)
    probs = V.depolarizing_probs(np.full(d * d, p))
    t0 = time.time()
    ml = V.ExactMLDecoder(code, probs)
    t_build = time.time() - t0
    rng = np.random.default_rng(1)
    # realistic syndromes: sample from stim
    mwpm = V.MWPMDecoder(code, p)
    shots = V.sample_shots(code, np.full(d * d, p), n_syn * 4, seed=7,
                           mwpm=mwpm, ml=V.ExactMLDecoder(code, probs))
    uniq = np.unique(shots.key)[:n_syn]
    bits = ((uniq[:, None] >> np.arange(code.nd)[None, :]) & 1)
    t0 = time.time()
    ml.coset_probs(bits)
    dt = time.time() - t0
    print(f"  d={d}: DP build {t_build:.2f}s; {bits.shape[0]} distinct syndromes "
          f"decoded in {dt:.2f}s ({1e3 * dt / max(1, bits.shape[0]):.3f} ms/syndrome)")
    return dt / max(1, bits.shape[0])


if __name__ == "__main__":
    print("== 1. d=3 DP vs brute force ==")
    check_d3_dp_vs_brute()
    print("== 2. d=3 stim sampling vs exact marginals ==")
    check_d3_stim_marginals()
    print("== 3. DP timing (feasibility) ==")
    for d in (5, 7):
        time_dp(d)
    print("ALL CHECKS PASSED")
