"""Sampling and minimum-weight matching for the stability experiment."""

from __future__ import annotations

import numpy as np
import pymatching
import stim

from src.circuit import Noise, StabilityExperiment, build_stability_circuit
from src.leakage import Leakage, apply_mask, burst_mask


def per_cycle_error(p_shot: float, cycles: int) -> float:
    """eps = 1 - (1 - P_shot)^(1/cycles)."""

    if cycles < 1:
        raise ValueError("cycles must be positive")
    p = float(min(max(p_shot, 0.0), 1.0))
    if p >= 1.0:
        return 1.0
    return float(1.0 - (1.0 - p) ** (1.0 / cycles))


def rule_of_three_upper(shots: int) -> float:
    """One-sided ~95% upper bound on a rate with zero observed failures."""

    if shots < 1:
        raise ValueError("shots must be positive")
    return 3.0 / shots


def augment_observable(dem: stim.DetectorErrorModel) -> stim.DetectorErrorModel:
    """Turn observable L0 into one extra detector, for the complementary gap."""

    n = dem.num_detectors
    text = str(dem).replace("L0", f"D{n}") + f"\ndetector D{n}\n"
    return stim.DetectorErrorModel(text)


class Decoder:
    def __init__(self, experiment: StabilityExperiment) -> None:
        self.experiment = experiment
        self.dem = experiment.circuit.detector_error_model(decompose_errors=True)
        self.matching = pymatching.Matching.from_detector_error_model(self.dem)
        self._augmented: pymatching.Matching | None = None

    def _gap_matching(self) -> pymatching.Matching:
        if self._augmented is None:
            self._augmented = pymatching.Matching.from_detector_error_model(
                augment_observable(self.dem)
            )
        return self._augmented

    def failures(self, detections: np.ndarray, observable: np.ndarray) -> int:
        prediction = self.matching.decode_batch(detections)
        obs = observable[:, 0] if observable.ndim == 2 else observable
        return int(np.sum(prediction[:, 0] != obs))

    def complementary_gap(
        self, detections: np.ndarray, observable: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return (prediction, gap) using two matchings of the logical classes.

        The gap is |W_wrong-class - W_right-class| in the decoder's
        log-likelihood weights (nats when edges are log((1-p)/p)).
        """

        n = len(detections)
        zeros = np.zeros((n, 1), dtype=np.uint8)
        ones = np.ones((n, 1), dtype=np.uint8)
        syn0 = np.concatenate([detections, zeros], axis=1)
        syn1 = np.concatenate([detections, ones], axis=1)
        matching = self._gap_matching()
        _p0, w0 = matching.decode_batch(syn0, return_weights=True)
        _p1, w1 = matching.decode_batch(syn1, return_weights=True)
        prediction = np.where(w1 < w0, np.uint8(1), np.uint8(0))
        return prediction, np.abs(w1 - w0)


def sample_pauli(decoder: Decoder, shots: int, seed: int, batch: int = 50_000) -> tuple[int, int]:
    sampler = decoder.experiment.circuit.compile_detector_sampler(seed=seed)
    fails = 0
    done = 0
    while done < shots:
        n = min(batch, shots - done)
        detections, observable = sampler.sample(n, separate_observables=True)
        fails += decoder.failures(detections, observable)
        done += n
    return fails, shots


def sample_correlated(
    decoder: Decoder,
    shots: int,
    seed: int,
    leakage: Leakage,
    batch: int = 20_000,
) -> tuple[int, int]:
    circuit = decoder.experiment.circuit
    sampler = circuit.compile_sampler(seed=seed)
    converter = circuit.compile_m2d_converter()
    n_stab = len(decoder.experiment.z_tile_index)
    fails = 0
    done = 0
    while done < shots:
        n = min(batch, shots - done)
        measurements = sampler.sample(n)
        rng = np.random.default_rng(seed + 1_000_003 + done)
        mask = burst_mask(n, n_stab, decoder.experiment.rounds, leakage, rng)
        measurements = apply_mask(measurements, mask, decoder.experiment)
        detections, observable = converter.convert(
            measurements=measurements, separate_observables=True
        )
        fails += decoder.failures(detections, observable)
        done += n
    return fails, shots


def build_decoder(
    distance: int,
    rounds: int,
    p: float,
    close_data: bool = True,
) -> Decoder:
    experiment = build_stability_circuit(
        distance, rounds, Noise(p), boundary="Z", close_data=close_data
    )
    return Decoder(experiment)
