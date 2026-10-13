"""Unit and end-to-end tests. Run from this folder with:

    python -m unittest test_pipeline -v

The tests build a small fake LOBSTER-format order book in a temporary directory,
so they do not need the real (uncommitted) dataset.
"""
import ast
import json
import os
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import torch

import metrics
from dataset import (LEVELS, N_FEATURES, PREFIX, PRICE_SCALE, book_to_levels,
                     build_features, chronological_split, denormalise,
                     features_to_book, get_dataloaders, load_lob, resample_to_grid)
from modules import RecurrentGAN, TimeGAN, build_model, count_parameters

HERE = os.path.dirname(os.path.abspath(__file__))
TICK_UNITS = 100  # one $0.01 tick in LOBSTER price units (1/10,000 dollar)


def make_fake_lob(directory, n_events=9000, seconds=3000, seed=0):
    """Write a valid random-walk order book in LOBSTER layout; return the arrays."""
    rng = np.random.default_rng(seed)
    times = np.sort(rng.uniform(34200, 34200 + seconds, n_events))
    bid1 = 2239500 + np.cumsum(rng.integers(-1, 2, n_events)) * TICK_UNITS
    spread = rng.integers(1, 6, n_events) * TICK_UNITS
    ask = bid1[:, None] + spread[:, None] + np.concatenate(
        [np.zeros((n_events, 1)), np.cumsum(rng.integers(1, 4, (n_events, LEVELS - 1)), 1)],
        axis=1) * TICK_UNITS
    bid = bid1[:, None] - np.concatenate(
        [np.zeros((n_events, 1)), np.cumsum(rng.integers(1, 4, (n_events, LEVELS - 1)), 1)],
        axis=1) * TICK_UNITS
    book = np.zeros((n_events, 4 * LEVELS))
    book[:, 0::4], book[:, 2::4] = ask, bid
    book[:, 1::4] = rng.integers(1, 500, (n_events, LEVELS))
    book[:, 3::4] = rng.integers(1, 500, (n_events, LEVELS))
    messages = np.zeros((n_events, 6))
    messages[:, 0] = times
    np.savetxt(f"{directory}/{PREFIX}orderbook_10.csv", book, fmt="%d", delimiter=",")
    np.savetxt(f"{directory}/{PREFIX}message_10.csv", messages, fmt="%.9f", delimiter=",")
    return times, book


class TestDataset(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.times, cls.book = make_fake_lob(cls.tmp.name)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_resample_uses_last_event_at_or_before_each_tick(self):
        times, book = load_lob(self.tmp.name)
        grid, snaps = resample_to_grid(times, book, step=1.0)
        self.assertTrue(np.allclose(np.diff(grid), 1.0))
        for k in (0, 10, len(grid) // 2, len(grid) - 1):
            last = np.where(times <= grid[k])[0].max()
            np.testing.assert_array_equal(snaps[k], book[last])

    def test_feature_roundtrip_recovers_book(self):
        times, book = load_lob(self.tmp.name)
        _, snaps = resample_to_grid(times, book)
        feats, mid = build_features(snaps)
        self.assertEqual(feats.shape[1], N_FEATURES)
        s = 100
        ask_p, ask_v, bid_p, bid_v = features_to_book(feats[s:s + 24], mid[s - 1])
        r_ask_p, r_ask_v, r_bid_p, r_bid_v = book_to_levels(snaps)
        np.testing.assert_allclose(ask_p, r_ask_p[s:s + 24], atol=1e-8)
        np.testing.assert_allclose(bid_p, r_bid_p[s:s + 24], atol=1e-8)
        np.testing.assert_allclose(ask_v, r_ask_v[s:s + 24], atol=1e-6)
        np.testing.assert_allclose(bid_v, r_bid_v[s:s + 24], atol=1e-6)

    def test_split_is_chronological_and_complete(self):
        s = chronological_split(1000)
        self.assertEqual(s["train"][0], 0)
        self.assertEqual(s["train"][1], s["val"][0])
        self.assertEqual(s["val"][1], s["test"][0])
        self.assertEqual(s["test"][1], 1000)

    def test_loaders_shapes_and_train_only_normalisation(self):
        loaders, info = get_dataloaders(self.tmp.name, seq_len=24, batch_size=16)
        batch = next(iter(loaders["train"]))
        self.assertEqual(tuple(batch.shape), (16, 24, N_FEATURES))
        train = loaders["train"].dataset.windows.numpy()
        # normalisation statistics come from the training split, so it is ~zero-mean
        self.assertLess(np.abs(train.mean(axis=(0, 1))).max(), 0.1)
        # windows never cross a split boundary: counts follow split lengths
        for name, (a, b) in info["splits"].items():
            self.assertEqual(len(loaders[name].dataset), (b - a) - 24 + 1)

    def test_denormalise_inverts_normalisation(self):
        loaders, info = get_dataloaders(self.tmp.name, seq_len=24)
        raw = denormalise(loaders["train"].dataset.windows.numpy(), info)
        a, _ = info["splits"]["train"]
        np.testing.assert_allclose(raw[0], info["feats_raw"][a:a + 24], atol=1e-4)


class TestModules(unittest.TestCase):
    def test_shapes_and_sampling(self):
        x = torch.randn(4, 24, N_FEATURES)
        t, b = TimeGAN(N_FEATURES, 16, 2), RecurrentGAN(N_FEATURES, 16, 2)
        h = t.embedder(x)
        self.assertEqual(tuple(h.shape), (4, 24, 16))
        self.assertTrue(0 <= float(h.min()) and float(h.max()) <= 1)  # sigmoid latent
        self.assertEqual(tuple(t.recovery(h).shape), (4, 24, N_FEATURES))
        self.assertEqual(tuple(t.supervisor(h).shape), (4, 24, 16))
        self.assertEqual(tuple(t.discriminator(h).shape), (4, 24, 1))
        self.assertEqual(tuple(b.discriminator(x).shape), (4, 24, 1))
        self.assertEqual(tuple(t.sample(5, 24).shape), (5, 24, N_FEATURES))
        self.assertEqual(tuple(b.sample(5, 24).shape), (5, 24, N_FEATURES))

    def test_gradients_flow_through_autoencoder(self):
        t = TimeGAN(N_FEATURES, 16, 2)
        x = torch.randn(4, 24, N_FEATURES)
        t.recovery(t.embedder(x)).pow(2).mean().backward()
        self.assertTrue(all(p.grad is not None for p in t.embedder.parameters()))
        self.assertGreater(count_parameters(t), 0)

    def test_build_model(self):
        self.assertIsInstance(build_model("timegan", N_FEATURES), TimeGAN)
        self.assertIsInstance(build_model("baseline", N_FEATURES), RecurrentGAN)
        with self.assertRaises(ValueError):
            build_model("nope", N_FEATURES)

    def test_modules_does_not_import_numpy(self):
        """Assessment rule: the model code must depend only on PyTorch."""
        tree = ast.parse(open(os.path.join(HERE, "modules.py")).read())
        imported = {n.names[0].name.split(".")[0] for n in ast.walk(tree)
                    if isinstance(n, ast.Import)}
        imported |= {n.module.split(".")[0] for n in ast.walk(tree)
                     if isinstance(n, ast.ImportFrom)}
        self.assertEqual(imported & {"numpy", "scipy", "pandas"}, set())


class TestMetrics(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        make_fake_lob(cls.tmp.name)
        loaders, info = get_dataloaders(cls.tmp.name, seq_len=24)
        cls.raw = denormalise(loaders["test"].dataset.windows.numpy(), info)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_valid_book_has_no_violations(self):
        a = metrics.microstructure_audit(self.raw)
        self.assertEqual(a["any"], 0.0)

    def test_detects_crossed_book(self):
        bad = self.raw.copy()
        # Both best distances negative, so ask_dist + bid_dist < 0: spread <= 0 (crossed)
        bad[..., metrics.ASK_DIST][..., 0] = -5.0
        bad[..., metrics.BID_DIST][..., 0] = -5.0
        self.assertEqual(metrics.microstructure_audit(bad)["crossed"], 1.0)

    def test_detects_invalid_ladder_and_negative_depth(self):
        bad = self.raw.copy()
        bad[..., metrics.ASK_DIST] = bad[..., metrics.ASK_DIST][..., ::-1]
        self.assertGreater(metrics.microstructure_audit(bad)["ladder"], 0.9)
        bad2 = self.raw.copy()
        bad2[..., metrics.ASK_VOL][:, 0, 0] = -1.0
        self.assertGreater(metrics.microstructure_audit(bad2)["neg_depth"], 0.0)

    def test_kl_zero_for_identical_and_large_for_different(self):
        x = np.random.default_rng(0).normal(size=20000)
        self.assertLess(metrics.kl_divergence(x, x), 1e-6)
        self.assertGreater(metrics.kl_divergence(x, x + 3.0), 1.0)

    def test_acf_detects_volatility_clustering(self):
        rng = np.random.default_rng(0)
        white = np.zeros((500, 24, N_FEATURES))
        white[..., 0] = rng.normal(size=(500, 24))
        scale = np.exp(np.cumsum(rng.normal(0, 0.3, (500, 24)), axis=1) * 0.5)
        clustered = np.zeros_like(white)
        clustered[..., 0] = rng.normal(size=(500, 24)) * scale
        self.assertLess(abs(metrics.abs_return_acf(white)[1]), 0.05)
        self.assertGreater(metrics.abs_return_acf(clustered)[1], 0.2)

    def test_ssim_identical_heatmaps_is_one(self):
        h = metrics.depth_heatmaps(self.raw[:50])
        self.assertAlmostEqual(metrics.ssim_to_nearest(h, h, n=20), 1.0, places=6)


class TestEndToEnd(unittest.TestCase):
    """Tiny train -> predict run on fake data for both models."""

    def test_train_and_predict(self):
        with tempfile.TemporaryDirectory() as data, tempfile.TemporaryDirectory() as out:
            make_fake_lob(data)
            for model in ("timegan", "baseline"):
                subprocess.run(
                    [sys.executable, "train.py", "--model", model, "--data-dir", data,
                     "--out-dir", out, "--ae-epochs", "1", "--sup-epochs", "1",
                     "--epochs", "2", "--max-batches", "3", "--eval-every", "1",
                     "--n-eval", "100", "--hidden", "16", "--layers", "2"],
                    cwd=HERE, check=True, capture_output=True)
                summary = json.load(open(f"{out}/{model}_summary.json"))
                self.assertIn("kl_spread", summary["test"])
                self.assertTrue(os.path.exists(f"{out}/{model}.pt"))
                self.assertTrue(os.path.exists(f"{out}/{model}_training.png"))
            subprocess.run([sys.executable, "predict.py", "--data-dir", data,
                            "--model-dir", out, "--n-samples", "100"],
                           cwd=HERE, check=True, capture_output=True)
            for fig in ("depth_heatmaps", "distributions", "volatility", "violations",
                        "autopsy"):
                self.assertTrue(os.path.exists(f"{out}/figures/{fig}.png"), fig)


if __name__ == "__main__":
    unittest.main()
