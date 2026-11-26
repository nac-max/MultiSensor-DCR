"""Focused data, ablation, corruption and metric tests (no downloads)."""

from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image
import torch

from common import Confusion, apply_corruption, seed_everything
from data import PairedDataset, RGB_MEAN, RGB_STD
from models import build_model


class ProjectTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_metrics(self):
        meter = Confusion(["a", "b"])
        meter.update(torch.tensor([0, 1, 0, 1]), torch.tensor([0, 1, 1, 255]))
        result = meter.result()
        self.assertAlmostEqual(result["miou"], 0.5)
        self.assertAlmostEqual(result["pixel_accuracy"], 2 / 3)
        self.assertIsNone(Confusion(["a"]).result()["miou"])

    def test_corruptions(self):
        rgb = (torch.ones(2, 3, 8, 10) * 0.8 - RGB_MEAN) / RGB_STD
        thermal = torch.linspace(-1, 1, 10).view(1, 1, 1, 10).expand(2, 1, 8, 10)
        r0, t0 = rgb.clone(), thermal.clone()
        r, t = apply_corruption(rgb, thermal, "rgb_missing")
        torch.testing.assert_close(r * RGB_STD + RGB_MEAN, torch.zeros_like(r))
        torch.testing.assert_close(t, thermal)
        r, t = apply_corruption(rgb, thermal, "thermal_missing")
        torch.testing.assert_close(t, -torch.ones_like(t))
        r, t = apply_corruption(rgb, thermal, "rgb_dark:0.5")
        torch.testing.assert_close(r * RGB_STD + RGB_MEAN, torch.full_like(r, 0.4))
        for spec in ("none", "rgb_noise:0.1", "thermal_noise:0.1", "thermal_shift:2", "occlusion:0.25"):
            seed_everything(3)
            a = apply_corruption(rgb, thermal, spec)
            seed_everything(3)
            b = apply_corruption(rgb, thermal, spec)
            for x, y in zip(a, b):
                torch.testing.assert_close(x, y)
                self.assertTrue(torch.isfinite(x).all())
        shifted = apply_corruption(rgb, thermal, "thermal_shift:2")[1]
        torch.testing.assert_close(shifted[..., 2:], thermal[..., :-2])
        torch.testing.assert_close(rgb, r0)
        torch.testing.assert_close(thermal, t0)
        for invalid in ("bad", "rgb_noise", "thermal_shift:-1", "occlusion:2", "rgb_dark:nan"):
            with self.assertRaises(ValueError):
                apply_corruption(rgb, thermal, invalid)

    def test_pst900_and_invalid_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for folder in ("rgb", "thermal", "labels"):
                (root / "train" / folder).mkdir(parents=True)
            for i in range(5):
                Image.fromarray(np.full((8, 10, 3), 128, dtype=np.uint8)).save(root / "train/rgb" / f"{i}.png")
                Image.fromarray(np.full((8, 10), 32768, dtype=np.uint16)).save(root / "train/thermal" / f"{i}.png")
                Image.fromarray(np.full((8, 10), i, dtype=np.uint8)).save(root / "train/labels" / f"{i}.png")
            train = PairedDataset(root, "train", "pst900", size=(8, 10), augment=True)
            val = PairedDataset(root, "val", "pst900", size=(8, 10))
            self.assertFalse(set(train.names) & set(val.names))
            self.assertEqual((len(train), len(val)), (4, 1))
            sample = val[0]
            self.assertEqual(sample["rgb"].shape, (3, 8, 10))
            self.assertEqual(sample["label"].dtype, torch.long)
            self.assertLess(float(sample["thermal"].abs().max()), 0.001)
            seed_everything(2)
            a = train[0]
            seed_everything(2)
            b = train[0]
            torch.testing.assert_close(a["label"], b["label"])
            Image.fromarray(np.full((8, 10), 99, dtype=np.uint8)).save(root / "train/labels/0.png")
            with self.assertRaisesRegex(ValueError, "invalid class"):
                val[0]
            with self.assertRaisesRegex(ValueError, "empty"):
                PairedDataset(root, "test", "pst900")

    def test_rcaf_ablations(self):
        rgb, thermal = torch.randn(2, 3, 64, 64), torch.randn(2, 1, 64, 64)
        for flag in ("no_align", "no_reliability", "no_interaction"):
            model = build_model({"model": "rcaf", "num_classes": 9, "pretrained": 0, flag: True})
            out = model(rgb, thermal)
            self.assertEqual(out["logits"].shape, (2, 9, 64, 64))
            out["logits"].mean().backward()
            if flag == "no_align":
                self.assertTrue(all(not x.any() for x in out["flows"]))
                self.assertTrue(all(block.offset is None for block in model.fusions))
            elif flag == "no_reliability":
                self.assertEqual(model.fusions[0].local_gate[0].in_channels, 128)
            else:
                self.assertTrue(all(block.gamma is None for block in model.fusions))


if __name__ == "__main__":
    unittest.main(verbosity=2)
