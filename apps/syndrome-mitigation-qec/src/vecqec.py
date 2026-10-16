"""Core library for the syndrome-mitigation-qec CPU demo.

Independent re-implementation (toy form) of the dual-decoder coarse-grained
virtual error cancellation (VEC) protocol of:

  Shtanko, Roy, Kumar, Minev (Google Quantum AI),
  "Measure Now, Mitigate Later: Virtual Error Cancellation for Logical
  Quantum Circuits", arXiv:2610.12400 (CC BY 4.0).

Scope (code-capacity toy, main text + Supplementary Sec. II of the paper):
  * rotated surface code, distance d, one logical qubit, d^2 data qubits;
  * independent single-qubit depolarizing noise E_p on each data qubit;
  * one round of perfect syndrome extraction, sampled with stim; the true
    logical frame flip Q' is recorded via a noiseless reference ancilla
    (benchmarking only, as in the paper's Supplementary Sec. II.A);
  * real-time decoder D_rt: PyMatching MWPM on the decoupled X/Z graphs
    with log-likelihood weights;
  * offline decoder D_off: exact maximum-likelihood coset probabilities
    (Eq. S.1), via brute force (d=3) or an exact transfer-matrix /
    MPS-style contraction with bond dimension <= 2^d x 4 (any odd d),
    verified against brute force at d=3;
  * dual-decoder sectors S_Q (Eq. 10), coarse-grained VEC weights (Eq. 11),
    sampling overhead C (Eq. 9/12), and the estimator <O>_VEC = N^-1 sum w m.

Logical-class encoding used throughout: 2-bit integer with
  bit0 = a = X_L-type flip (anticommutation of the error with Z_L),
  bit1 = b = Z_L-type flip (anticommutation of the error with X_L),
so class 0=I, 1=X, 2=Z, 3=Y, and classes compose by XOR.
Report order [I, X, Y, Z] is obtained with REPORT_ORDER.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import pymatching
import stim
from scipy.sparse import csr_matrix

CLASS_NAMES = ["I", "X", "Z", "Y"]        # index = (a | b<<1)
REPORT_ORDER = [0, 1, 3, 2]               # -> I, X, Y, Z
REPORT_NAMES = ["I", "X", "Y", "Z"]
DIGIT_TO_CLASS = np.array([0, 1, 3, 2], dtype=np.int64)  # pauli digit I,X,Y,Z -> class


# --------------------------------------------------------------------------
# Rotated surface code construction
# --------------------------------------------------------------------------

def _f2_rank(mat: np.ndarray) -> int:
    m = mat.copy().astype(np.uint8) % 2
    rank = 0
    rows, cols = m.shape
    for c in range(cols):
        piv = None
        for r in range(rank, rows):
            if m[r, c]:
                piv = r
                break
        if piv is None:
            continue
        m[[rank, piv]] = m[[piv, rank]]
        for r in range(rows):
            if r != rank and m[r, c]:
                m[r] ^= m[rank]
        rank += 1
    return rank


@dataclass
class Check:
    ctype: str              # 'Z' or 'X'
    qubits: list            # data-qubit indices
    x: int                  # plaquette lattice coords (2x, 2y)
    y: int
    syn_idx: int = -1       # global syndrome bit index


@dataclass
class RotatedSurfaceCode:
    """Rotated surface code following the layout of arXiv:2610.12400 Supp. II.A.

    Data qubit (col c, row r) at lattice (2c+1, 2r+1), index q = r*d + c.
    Z_L = Z along row r=0 (qubits 0..d-1); X_L = X along column c=0.
    Weight-2 boundary checks: Z-type on left/right edges, X-type on top/bottom.
    """

    d: int
    checks: list = field(default_factory=list)
    _dp_struct: dict = field(default_factory=dict, repr=False)

    def __post_init__(self):
        d = self.d
        assert d % 2 == 1 and d >= 3
        for x in range(d + 1):
            for y in range(d + 1):
                qs = []
                for cc in (x - 1, x):
                    for rr in (y - 1, y):
                        if 0 <= cc < d and 0 <= rr < d:
                            qs.append(rr * d + cc)
                if len(qs) < 2:
                    continue
                ctype = "Z" if (x + y) % 2 == 0 else "X"
                if len(qs) == 2:
                    if x in (0, d) and ctype != "Z":
                        continue
                    if y in (0, d) and ctype != "X":
                        continue
                self.checks.append(Check(ctype, sorted(qs), x, y))
        # global syndrome order: all Z checks first, then all X checks
        self.z_checks = [c for c in self.checks if c.ctype == "Z"]
        self.x_checks = [c for c in self.checks if c.ctype == "X"]
        self.checks = self.z_checks + self.x_checks
        for i, c in enumerate(self.checks):
            c.syn_idx = i
        self.n = d * d
        self.n_z = len(self.z_checks)
        self.n_x = len(self.x_checks)
        self.nd = self.n_z + self.n_x
        self.zl_support = np.array([c for c in range(d)], dtype=np.int64)          # row 0
        self.xl_support = np.array([r * d for r in range(d)], dtype=np.int64)      # col 0
        self.H_Z = np.zeros((self.n_z, self.n), dtype=np.uint8)
        for i, c in enumerate(self.z_checks):
            self.H_Z[i, c.qubits] = 1
        self.H_X = np.zeros((self.n_x, self.n), dtype=np.uint8)
        for i, c in enumerate(self.x_checks):
            self.H_X[i, c.qubits] = 1
        self._self_check()

    def _self_check(self):
        d = self.d
        assert self.n_z == self.n_x == (d * d - 1) // 2, (self.n_z, self.n_x)
        # CSS commutation: every Z check overlaps every X check evenly
        overlap = (self.H_Z.astype(int) @ self.H_X.T.astype(int)) % 2
        assert not overlap.any(), "Z and X checks do not commute"
        # stabilizers are independent
        assert _f2_rank(self.H_Z) == self.n_z
        assert _f2_rank(self.H_X) == self.n_x
        # logicals commute with checks, anticommute with each other
        zl = np.zeros(self.n, dtype=int); zl[self.zl_support] = 1
        xl = np.zeros(self.n, dtype=int); xl[self.xl_support] = 1
        assert not ((self.H_X.astype(int) @ zl) % 2).any()   # Z_L commutes with X checks
        assert not ((self.H_Z.astype(int) @ xl) % 2).any()   # X_L commutes with Z checks
        assert (zl @ xl) % 2 == 1                            # Z_L anticommutes with X_L
        # Z_L not a stabilizer product
        aug = np.vstack([self.H_Z, zl])
        assert _f2_rank(aug) == self.n_z + 1

    # -- stim circuit: 1 perfect extraction round, true frame flip via noiseless
    #    reference ancilla (Bell pair with the logical qubit) ------------------
    def stim_circuit(self, p_per_qubit: np.ndarray) -> stim.Circuit:
        """Code-capacity depolarizing circuit.

        Qubits 0..n-1 are data, qubit n is the noiseless reference ancilla.
        Observable 0 = flip of MPP(Z_L * Z_anc)  -> a (X_L-type frame flip),
        observable 1 = flip of MPP(X_L * X_anc)  -> b (Z_L-type frame flip).
        Both logical-pair MPPs commute with each other and with all checks,
        so stim's detector sampler accepts them; DEPOLARIZE1 only touches
        data qubits, keeping the reference ancilla noiseless.
        """
        n = self.n
        anc = n
        lines = [f"R {' '.join(str(q) for q in range(n + 1))}"]

        def mpp_round():
            prods = []
            for c in self.checks:
                prods.append("*".join(f"{c.ctype}{q}" for q in c.qubits))
            prods.append("*".join([f"Z{q}" for q in self.zl_support] + [f"Z{anc}"]))
            prods.append("*".join([f"X{q}" for q in self.xl_support] + [f"X{anc}"]))
            return "MPP " + " ".join(prods)

        lines.append(mpp_round())
        p_per_qubit = np.asarray(p_per_qubit, dtype=float)
        assert p_per_qubit.shape == (n,)
        for q in range(n):
            lines.append(f"DEPOLARIZE1({p_per_qubit[q]:.12g}) {q}")
        lines.append(mpp_round())
        m = self.nd + 2  # measurements per round
        for k in range(self.nd):
            lines.append(f"DETECTOR rec[{k - m}] rec[{k - 2 * m}]")
        lines.append(f"OBSERVABLE_INCLUDE(0) rec[{self.nd - m}] rec[{self.nd - 2 * m}]")
        lines.append(f"OBSERVABLE_INCLUDE(1) rec[{self.nd + 1 - m}] rec[{self.nd + 1 - 2 * m}]")
        return stim.Circuit("\n".join(lines))

    # -- DP (transfer-matrix) structure, cached per code ----------------------
    def dp_struct(self):
        if self._dp_struct:
            return self._dp_struct
        d = self.d
        cols = []
        for c in range(d):
            own = [ch for ch in self.checks
                   if (c == 0 and ch.x == 0) or (c == d - 1 and ch.x == d)]
            prev = [ch for ch in self.checks if 1 <= ch.x <= d - 1 and ch.x == c]
            nxt = [ch for ch in self.checks if 1 <= ch.x <= d - 1 and ch.x == c + 1]
            cols.append(dict(own=own, prev=prev, nxt=nxt,
                             qubits=[r * d + c for r in range(d)]))
        # consistency: interface carried out of column c == interface into c+1
        for c in range(d - 1):
            assert [ch.syn_idx for ch in cols[c]["nxt"]] == \
                   [ch.syn_idx for ch in cols[c + 1]["prev"]]
        # every check is emitted exactly once
        covered = sorted(ch.syn_idx for col in cols for ch in col["own"] + col["prev"])
        assert covered == list(range(self.nd)), "DP check coverage broken"
        self._dp_struct = dict(cols=cols)
        return self._dp_struct


# --------------------------------------------------------------------------
# Noise model helpers
# --------------------------------------------------------------------------

def depolarizing_probs(p_per_qubit: np.ndarray) -> np.ndarray:
    """Per-qubit Pauli digit probabilities [I, X, Y, Z] for E_p."""
    p = np.asarray(p_per_qubit, dtype=float)
    out = np.empty((p.size, 4))
    out[:, 0] = 1.0 - p
    out[:, 1] = p / 3.0
    out[:, 2] = p / 3.0
    out[:, 3] = p / 3.0
    return out


# --------------------------------------------------------------------------
# Exact ML coset probabilities, brute force (d=3 scale)
# --------------------------------------------------------------------------

def brute_force_coset_table(code: RotatedSurfaceCode, probs_q: np.ndarray,
                            batch: int = 1 << 18) -> np.ndarray:
    """Enumerate all 4^n Pauli errors once -> table (2^ND, 4) of coset
    probabilities Z_L(s) (Eq. S.1, normalised by the total so each row of the
    table sums to P(s)).  Feasible for d=3 (4^9 = 262144 errors)."""
    n, nd = code.n, code.nd
    assert n <= 13, "brute force only intended for d=3"
    total = 4 ** n
    table = np.zeros((1 << nd, 4))
    syn_weights = (np.int64(1) << np.arange(nd))
    zl_mask = np.zeros(n, dtype=np.int64); zl_mask[code.zl_support] = 1
    xl_mask = np.zeros(n, dtype=np.int64); xl_mask[code.xl_support] = 1
    HZ = code.H_Z.astype(np.int64)
    HX = code.H_X.astype(np.int64)
    for start in range(0, total, batch):
        idx = np.arange(start, min(start + batch, total), dtype=np.int64)
        digits = (idx[:, None] >> (2 * np.arange(n)[None, :])) & 3   # 0I 1X 2Y 3Z
        xpart = ((digits == 1) | (digits == 2)).astype(np.int64)
        zpart = ((digits == 2) | (digits == 3)).astype(np.int64)
        s_z = (xpart @ HZ.T) % 2
        s_x = (zpart @ HX.T) % 2
        s_all = np.concatenate([s_z, s_x], axis=1)
        key = s_all @ syn_weights
        a = (xpart @ zl_mask) % 2
        b = (zpart @ xl_mask) % 2
        cls = a | (b << 1)
        w = np.prod(probs_q[np.arange(n)[None, :], digits], axis=1)
        np.add.at(table, (key, cls), w)
    return table


# --------------------------------------------------------------------------
# Exact ML coset probabilities, transfer-matrix DP (any odd d, bond <= 2^d x 4)
# --------------------------------------------------------------------------

class ExactMLDecoder:
    """Exact coset probabilities Z_L(s) by a column-by-column transfer-matrix
    contraction (BSV-style MPS with exact bond dimension <= 2^d x 4).

    State after column c = (partial parities of the d straddling checks of the
    interface c -> c+1) x (accumulated logical class).  All checks are emitted
    exactly once and matched against the target syndrome, so the contraction
    is exact (no truncation).  Verified against brute force at d=3.
    """

    def __init__(self, code: RotatedSurfaceCode, probs_q: np.ndarray):
        self.code = code
        self.probs_q = np.asarray(probs_q, dtype=float)
        self._build()
        self.cache: dict[int, np.ndarray] = {}   # packed syndrome -> prob 4-vector

    def _build(self):
        code, d = self.code, self.code.d
        st = code.dp_struct()
        self.columns = []
        n_cfg = 4 ** d
        digits_all = (np.arange(n_cfg, dtype=np.int64)[:, None]
                      >> (2 * np.arange(d)[None, :])) & 3
        xpart = ((digits_all == 1) | (digits_all == 2)).astype(np.int64)
        zpart = ((digits_all == 2) | (digits_all == 3)).astype(np.int64)
        for c, col in enumerate(st["cols"]):
            qcol = col["qubits"]           # qubit index of row r in this column
            def contrib(checks):
                bits = np.zeros((n_cfg, len(checks)), dtype=np.int64)
                for j, ch in enumerate(checks):
                    rows = [r for r in range(d) if qcol[r] in ch.qubits]
                    part = xpart if ch.ctype == "Z" else zpart
                    for r in rows:
                        bits[:, j] ^= part[:, r]
                return bits

            u_bits = contrib(col["prev"])
            o_bits = contrib(col["own"])
            v_bits = contrib(col["nxt"])
            npv, nov, nnx = len(col["prev"]), len(col["own"]), len(col["nxt"])
            pack = lambda bits: bits @ (np.int64(1) << np.arange(bits.shape[1])) \
                if bits.shape[1] else np.zeros(n_cfg, dtype=np.int64)
            U, O, V = pack(u_bits), pack(o_bits), pack(v_bits)
            # logical class contribution of this column
            a = xpart[:, 0]                                  # Z_L support = row 0
            b = zpart.sum(axis=1) % 2 if c == 0 else np.zeros(n_cfg, dtype=np.int64)
            L = a | (b << 1)
            w = np.prod(self.probs_q[np.array(qcol)[None, :], digits_all], axis=1)
            # A[e_O, u, v, dl] = sum of w over matching configs
            A = np.zeros((1 << nov, 1 << npv, 1 << nnx, 4))
            np.add.at(A, (O, U, V, L), w)
            # B[e_O][(u,Lin),(v,Lout)] = A[e_O, u, v, Lin ^ Lout]
            B = np.zeros((1 << nov, (1 << npv) * 4, (1 << nnx) * 4))
            for lin in range(4):
                for lout in range(4):
                    B[:, lin::4, lout::4] = A[:, :, :, lin ^ lout]
            # XOR permutations of the input-state index for each prev pattern
            perms = np.empty((1 << npv, (1 << npv) * 4), dtype=np.int64)
            for ep in range(1 << npv):
                perms[ep] = (((np.arange(1 << npv)[:, None] ^ ep) * 4
                              + np.arange(4)[None, :]).ravel())
            self.columns.append(dict(
                B=B, perms=perms, n_prev=npv, n_own=nov,
                prev_idx=np.array([ch.syn_idx for ch in col["prev"]], dtype=np.int64),
                own_idx=np.array([ch.syn_idx for ch in col["own"]], dtype=np.int64),
            ))

    def coset_probs(self, syn_bits: np.ndarray) -> np.ndarray:
        """syn_bits: (n_syn, ND) 0/1 array -> (n_syn, 4) normalised coset probs."""
        syn_bits = np.asarray(syn_bits, dtype=np.int64)
        n_syn = syn_bits.shape[0]
        S = np.zeros((n_syn, 4))
        S[:, 0] = 1.0
        for colinfo in self.columns:
            npv, nov = colinfo["n_prev"], colinfo["n_own"]
            e_p = (syn_bits[:, colinfo["prev_idx"]]
                   @ (np.int64(1) << np.arange(npv))) if npv else \
                np.zeros(n_syn, dtype=np.int64)
            e_o = (syn_bits[:, colinfo["own_idx"]]
                   @ (np.int64(1) << np.arange(nov))) if nov else \
                np.zeros(n_syn, dtype=np.int64)
            gkey = e_p | (e_o << npv)
            out = np.empty((n_syn, colinfo["B"].shape[2]))
            for g in np.unique(gkey):
                idx = np.nonzero(gkey == g)[0]
                ep, eo = int(g) & ((1 << npv) - 1), int(g) >> npv
                out[idx] = S[idx][:, colinfo["perms"][ep]] @ colinfo["B"][eo]
            S = out
        Z = S  # final state has no carried checks: shape (n_syn, 4)
        tot = Z.sum(axis=1, keepdims=True)
        return Z / tot

    def probs_for_keys(self, keys: np.ndarray) -> np.ndarray:
        """Cached coset probabilities for packed syndrome keys (global bit
        order: Z checks then X checks, bit k = check syn_idx k)."""
        keys = np.asarray(keys, dtype=np.int64)
        uniq = np.unique(keys)
        missing = np.array([k for k in uniq if int(k) not in self.cache],
                           dtype=np.int64)
        if missing.size:
            bits = ((missing[:, None] >> np.arange(self.code.nd)[None, :]) & 1)
            probs = self.coset_probs(bits)
            for k, pr in zip(missing.tolist(), probs):
                self.cache[int(k)] = pr
        return np.stack([self.cache[int(k)] for k in keys])


# --------------------------------------------------------------------------
# Real-time MWPM decoder (PyMatching, decoupled X/Z graphs)
# --------------------------------------------------------------------------

class MWPMDecoder:
    """D_rt: MWPM on the decoupled graphs with weights ln[(1-q)/q], q = 2p/3.

    For d<=5 a full lookup table over all sub-syndromes is precomputed
    (exactly the paper's precomputed tables T_X(s_Z), T_Z(s_X)); for larger d
    unique observed syndromes are decoded on demand and cached.
    """

    def __init__(self, code: RotatedSurfaceCode, p_nominal: float):
        self.code = code
        q = 2.0 * p_nominal / 3.0   # marginal X-flip (and Z-flip) rate per qubit
        wts = np.full(code.n, np.log((1 - q) / q))
        zl = np.zeros((1, code.n), dtype=np.uint8); zl[0, code.zl_support] = 1
        xl = np.zeros((1, code.n), dtype=np.uint8); xl[0, code.xl_support] = 1
        self.m_z = pymatching.Matching.from_check_matrix(
            csr_matrix(code.H_Z), weights=wts, faults_matrix=csr_matrix(zl))
        self.m_x = pymatching.Matching.from_check_matrix(
            csr_matrix(code.H_X), weights=wts, faults_matrix=csr_matrix(xl))
        self.table_a = self.table_b = None
        if code.n_z <= 12:
            all_s = ((np.arange(1 << code.n_z)[:, None]
                      >> np.arange(code.n_z)[None, :]) & 1).astype(np.uint8)
            self.table_a = self.m_z.decode_batch(all_s).ravel().astype(np.int64)
            self.table_b = self.m_x.decode_batch(all_s).ravel().astype(np.int64)
        self._cache_a: dict[int, int] = {}
        self._cache_b: dict[int, int] = {}

    def _decode_side(self, packed, nbits, matcher, table, cache):
        if table is not None:
            return table[packed]
        out = np.empty(packed.size, dtype=np.int64)
        uniq = np.unique(packed)
        missing = np.array([k for k in uniq if int(k) not in cache], dtype=np.int64)
        if missing.size:
            bits = ((missing[:, None] >> np.arange(nbits)[None, :]) & 1).astype(np.uint8)
            preds = matcher.decode_batch(bits).ravel().astype(np.int64)
            for k, v in zip(missing.tolist(), preds.tolist()):
                cache[int(k)] = v
        lut = {int(k): cache[int(k)] for k in uniq}
        for i, k in enumerate(packed.tolist()):
            out[i] = lut[k]
        return out

    def decode(self, sz_packed: np.ndarray, sx_packed: np.ndarray) -> np.ndarray:
        """Return real-time logical class (a_rt | b_rt<<1) per shot."""
        a = self._decode_side(sz_packed, self.code.n_z, self.m_z,
                              self.table_a, self._cache_a)
        b = self._decode_side(sx_packed, self.code.n_x, self.m_x,
                              self.table_b, self._cache_b)
        return a | (b << 1)


# --------------------------------------------------------------------------
# Sampling + per-shot labels
# --------------------------------------------------------------------------

@dataclass
class ShotData:
    """Per-shot labels. sector and resid are logical classes (0=I,1=X,2=Z,3=Y);
    resid is the residual logical error in the committed real-time frame."""
    key: np.ndarray       # packed full syndrome (int64)
    sector: np.ndarray    # D_rt(s) * D_off(s)
    resid: np.ndarray     # D_rt(s) * Q'_true


def sample_shots(code: RotatedSurfaceCode, p_per_qubit: np.ndarray,
                 n_shots: int, seed: int, mwpm: MWPMDecoder,
                 ml: ExactMLDecoder, chunk: int = 1 << 19) -> ShotData:
    circuit = code.stim_circuit(p_per_qubit)
    sampler = circuit.compile_detector_sampler(seed=seed)
    wz = (np.int64(1) << np.arange(code.n_z))
    wx = (np.int64(1) << np.arange(code.n_x))
    keys, sectors, resids = [], [], []
    done = 0
    while done < n_shots:
        m = min(chunk, n_shots - done)
        dets, obs = sampler.sample(m, separate_observables=True)
        dets = dets.astype(np.int64)
        sz = dets[:, :code.n_z] @ wz
        sx = dets[:, code.n_z:] @ wx
        cls_rt = mwpm.decode(sz, sx)
        key = sz | (sx << code.n_z)
        probs = ml.probs_for_keys(key)
        cls_off = np.argmax(probs, axis=1)
        cls_true = obs[:, 0].astype(np.int64) | (obs[:, 1].astype(np.int64) << 1)
        keys.append(key)
        sectors.append(cls_rt ^ cls_off)
        resids.append(cls_rt ^ cls_true)
        done += m
    return ShotData(np.concatenate(keys), np.concatenate(sectors),
                    np.concatenate(resids))


def joint_counts(shots: ShotData, n_shards: int) -> np.ndarray:
    """(n_shards, 4 sectors, 4 resid classes) counts; shards are contiguous."""
    n = shots.sector.size
    counts = np.zeros((n_shards, 4, 4), dtype=np.int64)
    edges = np.linspace(0, n, n_shards + 1).astype(np.int64)
    for b in range(n_shards):
        sl = slice(edges[b], edges[b + 1])
        np.add.at(counts[b], (shots.sector[sl], shots.resid[sl]), 1)
    return counts


# --------------------------------------------------------------------------
# VEC weights (Eq. 11), overhead (Eq. 9/12), residual bias, observables
# --------------------------------------------------------------------------

def sector_stats(counts: np.ndarray):
    """counts: (4,4) joint (sector, resid) -> shares q (4,), centroids p (4,3).

    Centroid component order follows REPORT_ORDER[1:] = [X, Y, Z] i.e.
    classes [1, 3, 2]."""
    tot = counts.sum()
    q = counts.sum(axis=1) / tot
    comp = [1, 3, 2]  # X, Y, Z in class encoding
    p = np.zeros((4, 3))
    for s in range(4):
        ns = counts[s].sum()
        if ns > 0:
            p[s] = counts[s, comp] / ns
    return q, p


def vec_weights(q: np.ndarray, p: np.ndarray):
    """Coarse-grained VEC weights, Eq. (11) of arXiv:2610.12400.

    q, p ordered by sector class [I=0, X=1, Z=2, Y=3]; centroid components
    [X, Y, Z].  V columns ordered [X, Y, Z] sectors (classes 1, 3, 2).
    Returns (w (4,), C, degenerate_flag)."""
    sec = [1, 3, 2]  # X, Y, Z sectors in class encoding
    if np.any(q[sec] <= 0):
        return None, None, True
    V = np.stack([p[s] - p[0] for s in sec], axis=1)  # 3x3, columns X,Y,Z
    try:
        vinv_pI = np.linalg.solve(V, p[0])
    except np.linalg.LinAlgError:
        return None, None, True
    w = np.zeros(4)
    w[0] = (1.0 + vinv_pI.sum()) / q[0]
    for j, s in enumerate(sec):
        w[s] = -vinv_pI[j] / q[s]
    C = float(np.sum(q * w ** 2))
    return w, C, False


def residual_bias(counts: np.ndarray, w: np.ndarray) -> float:
    """p_res = || sum_k q_k w_k p_k ||_1 evaluated on held-out counts (4,4)."""
    tot = counts.sum()
    comp = [1, 3, 2]
    vec = np.zeros(3)
    for s in range(4):
        vec += w[s] * counts[s, comp] / tot
    return float(np.abs(vec).sum())


def plain_pl(counts: np.ndarray) -> float:
    """Unmitigated (real-time MWPM) logical error rate = ||p_bar||_1."""
    tot = counts.sum()
    return float((tot - counts[:, 0].sum()) / tot)


def ml_pl(counts: np.ndarray) -> float:
    """Offline-ML logical error rate from the same joint counts:
    resid_off = resid_rt XOR sector, so success <=> resid == sector."""
    tot = counts.sum()
    ok = sum(counts[s, s] for s in range(4))
    return float((tot - ok) / tot)


def postselect_bias(counts: np.ndarray) -> float:
    """Agreement postselection: keep sector I, renormalise -> bias ||p_I||_1."""
    nI = counts[0].sum()
    if nI == 0:
        return float("nan")
    return float((nI - counts[0, 0]) / nI)


def observable_estimates(counts: np.ndarray, w: np.ndarray):
    """(<Z_L> on |0_L>, <X_L> on |+_L>) for weights w, from joint counts.

    Z_L readout flips iff resid has an X component (classes 1=X, 3=Y);
    X_L readout flips iff resid has a Z component (classes 2=Z, 3=Y)."""
    tot = counts.sum()
    z_est = x_est = 0.0
    for s in range(4):
        for r in range(4):
            mz = -1.0 if (r & 1) else 1.0
            mx = -1.0 if (r & 2) else 1.0
            z_est += w[s] * mz * counts[s, r]
            x_est += w[s] * mx * counts[s, r]
    return z_est / tot, x_est / tot


def postselect_weights(q: np.ndarray) -> np.ndarray:
    w = np.zeros(4)
    w[0] = 1.0 / q[0]
    return w


# --------------------------------------------------------------------------
# Paired jackknife over shards
# --------------------------------------------------------------------------

def jackknife_pres(cal_counts: np.ndarray, eval_counts: np.ndarray):
    """Paired delete-1 jackknife over B shards (cal and eval dropped jointly).

    Returns dict with point estimates and jackknife standard errors for
    p_res_vec, C, and point estimates recomputed on the full data."""
    B = min(cal_counts.shape[0], eval_counts.shape[0])
    calB = _regroup(cal_counts, B)
    evalB = _regroup(eval_counts, B)
    cal_tot, eval_tot = calB.sum(axis=0), evalB.sum(axis=0)

    def stat(cal_c, eval_c):
        q, p = sector_stats(cal_c)
        w, C, degen = vec_weights(q, p)
        if degen:
            return np.nan, np.nan
        return residual_bias(eval_c, w), C

    full = stat(cal_tot, eval_tot)
    reps = np.array([stat(cal_tot - calB[b], eval_tot - evalB[b])
                     for b in range(B)])
    mean_rep = np.nanmean(reps, axis=0)
    se = np.sqrt((B - 1) / B * np.nansum((reps - mean_rep) ** 2, axis=0))
    return dict(p_res_vec=full[0], p_res_vec_se=float(se[0]),
                C=full[1], C_se=float(se[1]), B=B)


def _regroup(counts: np.ndarray, B: int) -> np.ndarray:
    """Regroup (n_shards,4,4) counts into B contiguous super-shards."""
    n = counts.shape[0]
    edges = np.linspace(0, n, B + 1).astype(int)
    return np.stack([counts[edges[b]:edges[b + 1]].sum(axis=0) for b in range(B)])


def jackknife_scalar(shard_counts: np.ndarray, func, B: int = 20):
    """Delete-1 jackknife of func(counts_4x4) over regrouped shards."""
    grp = _regroup(shard_counts, B)
    tot = grp.sum(axis=0)
    full = func(tot)
    reps = np.array([func(tot - grp[b]) for b in range(B)])
    se = np.sqrt((B - 1) / B * np.nansum((reps - np.nanmean(reps)) ** 2))
    return float(full), float(se)


# --------------------------------------------------------------------------
# Syndrome-only noise learning (Eqs. 16-17 / S.2-S.9), d=3 stretch goal
# --------------------------------------------------------------------------

def character_ensemble(code: RotatedSurfaceCode):
    """Degree-1 characters for all checks plus weight-2 XOR pairs of adjacent
    same-sector checks (sharing >=1 data qubit).  Returns the character
    matrix X (K, ND) acting on syndrome bits."""
    nd = code.nd
    chars = [np.eye(nd, dtype=np.int64)[i] for i in range(nd)]
    groups = [range(0, code.n_z), range(code.n_z, nd)]
    H = np.vstack([code.H_Z, code.H_X]).astype(np.int64)
    for grp in groups:
        grp = list(grp)
        for i in range(len(grp)):
            for j in range(i + 1, len(grp)):
                u, v = grp[i], grp[j]
                if (H[u] & H[v]).sum() > 0:
                    chi = np.zeros(nd, dtype=np.int64)
                    chi[u] = chi[v] = 1
                    chars.append(chi)
    return np.stack(chars)


def learn_qubit_rates(code: RotatedSurfaceCode, syn_bits: np.ndarray):
    """Learn per-qubit depolarizing rates p_q from unlabelled syndromes via
    Walsh-Hadamard fidelities (Eq. 16/S.2), Jensen-corrected attenuations
    (S.3) and weighted NNLS on the tied design matrix (S.5-S.9)."""
    from scipy.optimize import nnls
    X = character_ensemble(code)
    H = np.vstack([code.H_Z, code.H_X]).astype(np.int64)   # H_deg1 (ND x n)
    A_eff = (X @ H) % 2                                     # (K x n)
    T = syn_bits.shape[0]
    par = (syn_bits.astype(np.int64) @ X.T) % 2
    F_hat = 1.0 - 2.0 * par.mean(axis=0)
    keep = F_hat >= 3.0 / np.sqrt(T)
    F_k = F_hat[keep]
    var_b = (1.0 - F_k ** 2) / (T * F_k ** 2)
    b_hat = -np.log(F_k) - 0.5 * var_b
    Wh = 1.0 / np.sqrt(np.maximum(var_b, 1e-300))
    theta, _ = nnls(A_eff[keep] * Wh[:, None], b_hat * Wh)
    p_hat = 0.75 * (1.0 - np.exp(-theta))
    return p_hat, dict(n_chars=int(X.shape[0]), n_used=int(keep.sum()),
                       rank=int(np.linalg.matrix_rank(A_eff.astype(float))))


def model_calibration_d3(code: RotatedSurfaceCode, probs_q: np.ndarray,
                         mwpm: MWPMDecoder):
    """Analytic sector shares/centroids under a noise MODEL (no shots):
    enumerate the exact syndrome distribution and posteriors (d=3 brute
    force), decode every syndrome with both decoders, and average.
    Returns (q (4,), p (4,3), model_off_decoder_table)."""
    table = brute_force_coset_table(code, probs_q)           # (256, 4) ~ P(s, L)
    Ps = table.sum(axis=1)
    post = np.where(Ps[:, None] > 0, table / np.maximum(Ps[:, None], 1e-300), 0.0)
    cls_off = np.argmax(table, axis=1)
    nkeys = table.shape[0]
    keys = np.arange(nkeys)
    sz = keys & ((1 << code.n_z) - 1)
    sx = keys >> code.n_z
    cls_rt = mwpm.decode(sz, sx)
    sector = cls_rt ^ cls_off
    comp = [1, 3, 2]
    q = np.zeros(4)
    p = np.zeros((4, 3))
    for s in range(4):
        m = sector == s
        q[s] = Ps[m].sum()
        if q[s] > 0:
            # residual-class posterior in the rt frame: p_resid[r] = post[r ^ cls_rt]
            pr = np.zeros((int(m.sum()), 4))
            post_m, rt_m = post[m], cls_rt[m]
            for r in range(4):
                pr[:, r] = post_m[np.arange(post_m.shape[0]), r ^ rt_m]
            p[s] = (Ps[m][:, None] * pr[:, comp]).sum(axis=0) / q[s]
    return q, p, cls_off
