"""RCAF v2 initialization, displacement, intensity and compatibility checks."""

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch.nn import functional as F

from common import apply_corruption, load_model
from data import RGB_MEAN, RGB_STD
from models import AblatedFusion, build_model
from rcaf import RCAFNet, RCAFusion, corrupt_modality, rcaf_loss


class V2Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        torch.manual_seed(23)

    def test_additive_initialization_and_bounds(self):
        model = RCAFNet(9, max_disp=0.75).eval()
        with torch.no_grad():
            rgb = model._encode(model.rgb_encoder, torch.randn(2, 3, 96, 128))
            thermal = model._encode(model.thermal_encoder, torch.randn(2, 1, 96, 128))
            errors = []
            for block, r, t in zip(model.fusions, rgb, thermal):
                fused, _, gate, flow = block(r, t)
                errors.append(float((fused - (r + t)).abs().max()))
                torch.testing.assert_close(fused, r + t, atol=1e-5, rtol=1e-5)
                self.assertTrue((gate == 0.5).all())
                self.assertFalse(flow.any())
            print("Additive init max abs diff (s1..s4): " + ",".join(f"{x:.8f}" for x in errors))
            previous, bound = None, 0
            for i in range(3, -1, -1):
                block = model.fusions[i]
                block.offset[-1].bias.copy_(torch.tensor([100., -100.]))
                flow = block.predict_flow(torch.cat((rgb[i], thermal[i]), 1), previous)
                inherited = torch.zeros_like(flow) if previous is None else 2 * F.interpolate(
                    previous, size=flow.shape[-2:], mode="bilinear", align_corners=False)
                self.assertLessEqual(float((flow - inherited).abs().max()), 0.75 + 1e-6)
                bound = 2 * bound + 0.75
                self.assertLessEqual(float(flow.abs().max()), bound + 1e-6)
                previous = flow
            print(f"Flow bounds: max |residual|<=0.75; saturated s1 cumulative max={float(previous.abs().max()):.2f} (15*0.75)")
        r, t = torch.randn(2, 8, 7, 9), torch.randn(2, 8, 7, 9)
        for flag in ("no_align", "no_reliability", "no_interaction"):
            block = AblatedFusion(8, 3, **{flag: True})
            torch.testing.assert_close(block(r, t)[0], r + t, atol=1e-5, rtol=1e-5)
        for invalid in (-1, float("nan")):
            with self.assertRaises(ValueError):
                RCAFusion(8, 3, max_disp=invalid)

    def test_magnitude(self):
        flow = torch.ones(2, 2, 3, 4, requires_grad=True)
        logits = torch.randn(2, 3, 6, 8, requires_grad=True)
        out = {"logits": logits, "aux": [], "gates": [],
               "flows": [flow, torch.ones(2, 2, 6, 8) * 2]}
        labels = torch.full((2,), -1)
        target = torch.zeros(2, 6, 8, dtype=torch.long)
        loss, parts = rcaf_loss(out, target, labels, lambdas={"mag": 1})
        self.assertAlmostEqual(float(parts["mag"].detach()), 0.25)
        self.assertAlmostEqual(float((loss - parts["main"]).detach()), 0.25, places=6)
        self.assertEqual(float(parts["tv"].detach()), 0)
        loss.backward()
        self.assertTrue((flow.grad > 0).all())
        out["flows"] = []
        self.assertEqual(float(rcaf_loss(out, target, labels)[1]["mag"].detach()), 0)

    def test_training_intensity_corruption(self):
        rgb = (torch.full((1, 3, 16, 20), 0.8) - RGB_MEAN) / RGB_STD
        thermal = torch.full((1, 1, 16, 20), 0.6)
        for modality in (0, 1):
            # Force missing and compare exactly to the evaluation protocol.
            with patch("rcaf.torch.randint", side_effect=[torch.tensor(modality), torch.tensor(0)]):
                r, t, labels = corrupt_modality(rgb, thermal, 1)
            expected = apply_corruption(rgb, thermal, "rgb_missing" if modality == 0 else "thermal_missing")
            self.assertEqual(labels.item(), modality)
            for actual, reference in zip((r, t), expected):
                torch.testing.assert_close(actual, reference)
        for operation in (1, 2, 3, 4):
            choices = [torch.tensor(0), torch.tensor(operation)]
            if operation == 3:
                choices.extend([torch.tensor(0), torch.tensor(0)])
            with patch("rcaf.torch.randint", side_effect=choices):
                r, t, _ = corrupt_modality(rgb, thermal, 1)
            raw = r * RGB_STD + RGB_MEAN
            self.assertTrue(((raw >= -1e-6) & (raw <= 1 + 1e-6)).all())
            torch.testing.assert_close(t, thermal)
            if operation == 3:
                torch.testing.assert_close(raw[..., :5, :6], torch.zeros_like(raw[..., :5, :6]))
            elif operation == 4:
                self.assertTrue(((raw >= 0.08 - 1e-6) & (raw <= 0.4 + 1e-6)).all())
        r, t, labels = corrupt_modality(rgb, thermal, 0)
        torch.testing.assert_close(r, rgb, rtol=0, atol=0)
        torch.testing.assert_close(t, thermal, rtol=0, atol=0)
        self.assertEqual(labels.item(), -1)
        print("Intensity corruption: train/eval RGB and thermal missing agree; noise/blur/erase/dark checks passed")

    def test_legacy_config(self):
        config = {"model": "rcaf", "num_classes": 9, "pretrained": 0}
        for ablation in (False, True):
            config["no_reliability"] = ablation
            old = build_model({**config, "model_version": 1}).eval()
            with tempfile.TemporaryDirectory() as folder:
                path = Path(folder) / "best.pt"
                torch.save({"model": old.state_dict(), "config": config, "epoch": 1, "best_miou": 0.1}, path)
                restored, _ = load_model(path, torch.device("cpu"))
            self.assertEqual(restored.fusions[0].model_version, 1)
            rgb, thermal = torch.randn(1, 3, 64, 64), torch.randn(1, 1, 64, 64)
            with torch.inference_mode():
                torch.testing.assert_close(old(rgb, thermal)["logits"], restored(rgb, thermal)["logits"])
        print("Legacy configs: v1 full/ablation checkpoint forward semantics preserved")


if __name__ == "__main__":
    unittest.main(verbosity=2)
