"""Fast checks for the quantizer, the Hadamard fold, Eq. (8), and S4."""

from __future__ import annotations

import unittest

import torch

from model import TinyGeoTransformer, sample_batch
from quant import (
    fit_affine,
    fold_weight,
    qmax_for_bits,
    quantize,
    random_signs,
    rotate_activations,
    rotation_identity_error,
    snr_proxy_ratio,
    walsh_hadamard,
)


class QuantTests(unittest.TestCase):
    def test_qmax(self) -> None:
        self.assertEqual(qmax_for_bits(2), 1)
        self.assertEqual(qmax_for_bits(4), 7)
        self.assertEqual(qmax_for_bits(8), 127)

    def test_hadamard_and_fold_are_lossless(self) -> None:
        generator = torch.Generator().manual_seed(0)
        hadamard = walsh_hadamard(32, device=torch.device("cpu"), dtype=torch.float32)
        eye = hadamard @ hadamard.T
        self.assertLess(float((eye - torch.eye(32)).abs().max()), 1e-5)
        signs = random_signs(32, generator, device=torch.device("cpu"), dtype=torch.float32)
        self.assertTrue(torch.equal(signs.square(), torch.ones_like(signs)))
        x = torch.randn(4, 7, 32)
        weight = torch.randn(16, 32)
        err = rotation_identity_error(x, weight, signs, hadamard)
        self.assertLess(float(err), 1e-4)
        rotated = rotate_activations(x, signs, hadamard)
        base = x.norm(dim=-1)
        self.assertLess(float((rotated.norm(dim=-1) - base).abs().max()), 1e-4)

    def test_prop1_ratio_is_invariant(self) -> None:
        generator = torch.Generator().manual_seed(1)
        tokens = torch.randn(16, 9, 32) * 0.2
        tokens[:, :-1, 0] = 12.0
        tokens[:, -1] *= 0.15
        signs = random_signs(32, generator, device=torch.device("cpu"), dtype=torch.float32)
        hadamard = walsh_hadamard(32, device=torch.device("cpu"), dtype=torch.float32)
        rotated = rotate_activations(tokens, signs, hadamard)
        delta = float(tokens.abs().max()) / 7
        delta_r = float(rotated.abs().max()) / 7
        pre = snr_proxy_ratio(tokens, delta)
        post = snr_proxy_ratio(rotated, delta_r)
        self.assertLess(abs(post / pre - 1.0), 1e-4)
        self.assertGreater(delta / delta_r, 1.0)

    def test_eq8_removes_the_special_row_mean(self) -> None:
        special = torch.tensor([0.4, -0.2, 0.1])
        shared = torch.tensor([0.05, 0.0, -0.05])
        error = shared.view(1, 1, 3).repeat(6, 4, 1).clone()
        error[:, -1] = special
        fit = fit_affine(error, has_special=True)
        self.assertIsNotNone(fit.b_s)
        assert fit.b_s is not None
        corrected_special = error[:, -1] + fit.beta + fit.b_s
        self.assertLess(float(corrected_special.abs().max()), 1e-6)
        # Token-uniform beta leaves the special-row residual (Prop. 3).
        leftover = error[:, -1] + fit.beta
        self.assertGreater(float(leftover.abs().max()), 0.1)
        self.assertGreater(float(leftover.abs().max()), float(corrected_special.abs().max()) + 0.1)

    def test_quantize_round_trip_of_a_grid_point(self) -> None:
        x = torch.tensor([0.2, -0.36, 0.14])
        hat = quantize(x, 0.1, 7)
        self.assertTrue(torch.allclose(hat, torch.tensor([0.2, -0.4, 0.1])))

    def test_frame_block_does_not_write_the_scale_token(self) -> None:
        cfg = {
            "model": {
                "d": 32,
                "ff": 64,
                "n_heads": 4,
                "n_patches": 8,
                "block_kinds": ["frame", "global"],
                "kappa_inv_target": 6.02,
                "range_loss_weight": 0.0,
                "n_outlier_channels": 2,
                "outlier_scale": 1.0,
                "outlier_gain": 1.0,
                "patch_std": 0.5,
                "m_lo": 0.5,
                "m_hi": 2.0,
            }
        }
        torch.manual_seed(0)
        model = TinyGeoTransformer(cfg)
        batch = sample_batch(cfg, 4, torch.Generator().manual_seed(2), torch.device("cpu"))
        # Run only the frame block by calling it directly.
        frame = model.blocks[0]
        hidden = torch.cat([batch[0], batch[1]], dim=1)
        before = hidden[:, -1].clone()
        after, delta = frame(hidden, batch[2], "b0", None, model.outlier_gain, model.kappa_inv_target)
        self.assertEqual(delta, 0.0)
        self.assertTrue(torch.equal(after[:, -1], before))
        folded = fold_weight(torch.eye(4), torch.ones(4), walsh_hadamard(4, device=torch.device("cpu"), dtype=torch.float32))
        self.assertEqual(tuple(folded.shape), (4, 4))


if __name__ == "__main__":
    unittest.main()
