"""CPU v3 unit, legacy-reference, and command-line integration tests."""

import csv
import gc
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F

from common import load_model
from models import MODEL_NAMES, build_model
from v3 import (DCRFusion, StrongDecoder, boundary_target, class_statistics,
                discrepancy_map, edge_loss, segmentation_ce, training_loss)


PROJECT = Path(__file__).resolve().parent
LEGACY_WEIGHTS = {"aux": 0.4, "gate": 0.1, "tv": 0.01, "mag": 0.1}


class V3Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        torch.manual_seed(12)
        (PROJECT / "results").mkdir(exist_ok=True)

    def test_init_equivalence_and_gradients(self):
        config = {"model": "add_fusion", "recipe": "v3", "pretrained": 0, "num_classes": 9}
        additive = build_model(config).eval()
        rgb, thermal = torch.randn(2, 3, 64, 80), torch.randn(2, 1, 64, 80)
        with torch.inference_mode():
            reference = additive(rgb, thermal)
        errors = []
        for option in ({}, {"no_dwl": True}, {"no_spatial": True}, {"no_channel": True},
                       {"no_cross": True}, {"dcr_stages": "1234"}):
            model = build_model({**config, "model": "dcr", **option}).eval()
            incompatible = model.load_state_dict(additive.state_dict(), strict=False)
            self.assertFalse(incompatible.unexpected_keys)
            self.assertTrue(all(k.startswith("dcr_fusions.") for k in incompatible.missing_keys))
            with torch.inference_mode():
                out = model(rgb, thermal)
            for key in ("logits", "deep_logits", "edge_logits"):
                self.assertTrue(torch.equal(reference[key], out[key]), key)
            errors.append(float((reference["logits"] - out["logits"]).abs().max()))
            self.assertFalse(out["dwl_map"].requires_grad)
            self.assertTrue(((out["dwl_map"] >= 0) & (out["dwl_map"] <= 1)).all())
            if option.get("no_spatial"):
                self.assertTrue(all((x["spatial"] == 1).all() for x in out["dcr"]))
            del model, out
        block = DCRFusion(16).train()
        r, t = torch.randn(2, 16, 7, 9), torch.randn(2, 16, 7, 9)
        fused, _, _ = block(r, t)
        self.assertTrue(torch.equal(fused, r + t))
        fused.square().mean().backward()
        self.assertGreater(float(block.alpha.grad.abs().sum()), 0)
        self.assertGreater(float(block.correction[-1].weight.detach().abs().sum()), 0)
        with torch.no_grad():
            block.alpha.fill_(0.1)
        block.zero_grad()
        fused, _, info = block(r, t)
        fused.square().mean().backward()
        for module in (block.correction, block.channel, block.spatial):
            self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in module.parameters()))
        self.assertTrue((info["ratio"] > 0).all())
        print("DCR/add init max abs diff (full,no_dwl,no_spatial,no_channel,no_cross,1234): "
              + ",".join(f"{x:.8f}" for x in errors))
        print("DCR gradients: alpha nonzero at init; correction/channel/spatial learn after alpha opens")

    def test_boundary_dwl_and_empty(self):
        target = torch.tensor([[[0, 0, 1], [0, 255, 1]]])
        edge, valid = boundary_target(target)
        self.assertTrue(torch.equal(edge[:, 0].bool(), torch.tensor([[[False, True, True], [False, False, True]]])))
        self.assertFalse(valid[0, 0, 1, 1])
        logits = torch.randn(1, 2, 2, 3, requires_grad=True)
        weights = torch.tensor([0.5, 1.5])
        torch.testing.assert_close(segmentation_ce(logits, target, weights),
                                   F.cross_entropy(logits, target, weight=weights, ignore_index=255))
        m = torch.tensor([[[0., 0.2, 0.4], [0.6, 0.8, 1.]]])
        pixels = 1 + m
        mask = target != 255
        normalized = pixels / pixels[mask].mean()
        torch.testing.assert_close(normalized[mask].sum(), mask.sum().float())
        expected = (F.cross_entropy(logits, target, reduction="none", ignore_index=255) * normalized).sum() / mask.sum()
        torch.testing.assert_close(segmentation_ce(logits, target, pixel_weights=pixels), expected)
        out = {"logits": logits, "aux": [], "gates": [], "flows": [], "dwl_map": m[:, None]}
        config = {"recipe": "v3", "model": "dcr", "no_dwl": True}
        _, parts = training_loss(out, target, torch.tensor([-1]), config, LEGACY_WEIGHTS)
        torch.testing.assert_close(parts["main"], F.cross_entropy(logits, target, ignore_index=255))
        self.assertEqual(float(parts["dwl"].detach()), 0)
        config["no_dwl"] = False
        _, parts = training_loss(out, target, torch.tensor([-1]), config, LEGACY_WEIGHTS)
        torch.testing.assert_close(parts["main"], expected)
        self.assertFalse(discrepancy_map(torch.ones(2, 1, 3, 4), (7, 9)).any())
        sparse = torch.zeros(1, 32, 32, dtype=torch.long)
        sparse[:, 16, 16] = 1
        boundary, valid = boundary_target(sparse)
        pos, neg = boundary.sum(), valid.sum() - boundary.sum()
        self.assertGreater(float(neg / pos), 20)
        expected_edge = ((neg + 20 * pos) * np.log(2)) / valid.sum()
        torch.testing.assert_close(edge_loss(torch.zeros_like(boundary), sparse), expected_edge.float())
        blank_logits = torch.zeros(1, 1, 4, 5, requires_grad=True)
        self.assertGreater(float(edge_loss(blank_logits, torch.zeros(1, 4, 5, dtype=torch.long)).detach()), 0)
        ignored = torch.full_like(target, 255)
        out.update(deep_logits=logits, edge_logits=torch.randn(1, 1, 2, 3, requires_grad=True))
        loss, parts = training_loss(out, ignored, torch.tensor([-1]), config, LEGACY_WEIGHTS)
        self.assertEqual(float(loss.detach()), 0)
        self.assertTrue(all(torch.isfinite(x) and x.detach() == 0 for x in parts.values()))
        loss.backward()
        print("V3 losses: weighted CE, valid-pixel-normalized DWL, clipped edge BCE, constant M/all-ignore passed")

    def test_class_weights_and_errors(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            (folder / "labels").mkdir()
            dataset = SimpleNamespace(folder=folder, names=["a"], class_names=["a", "b", "c"], ignore_index=255)
            label = np.array([[0, 0, 1], [0, 255, 1]], dtype=np.uint8)
            path = folder / "labels/a.png"
            Image.fromarray(label).save(path)
            stats = class_statistics(dataset)
            self.assertEqual(stats["counts"], [3, 2, 0])
            expected = 1 / np.log(1.10 + np.array([0.6, 0.4, 0.]))
            np.testing.assert_allclose(stats["weights"], expected / expected.mean())
            self.assertAlmostEqual(np.mean(stats["weights"]), 1)
            Image.fromarray(np.full((2, 3), 255, dtype=np.uint8)).save(path)
            with self.assertRaisesRegex(ValueError, "no valid pixels"):
                class_statistics(dataset)
            Image.fromarray(np.full((2, 3), 99, dtype=np.uint8)).save(path)
            with self.assertRaisesRegex(ValueError, "invalid class"):
                class_statistics(dataset)
        for option in ({"no_dwl": True}, {"no_channel": True}, {"no_spatial": True},
                       {"no_cross": True}, {"dcr_stages": "1234"}):
            with self.assertRaisesRegex(ValueError, "require --model dcr"):
                build_model({"model": "add_fusion", "pretrained": 0, **option})
        for stages in ("", "22", "05", "abc"):
            with self.assertRaisesRegex(ValueError, "unique stage"):
                build_model({"model": "dcr", "dcr_stages": stages, "pretrained": 0})
        for key in ("lambda_dwl", "lambda_edge", "lambda_deep"):
            with self.assertRaisesRegex(ValueError, "nonnegative"):
                build_model({"model": "dcr", key: -1, "pretrained": 0})

    def test_shared_recipe_all_models(self):
        rgb, thermal = torch.randn(2, 3, 64, 80), torch.randn(2, 1, 64, 80)
        y, x = torch.meshgrid(torch.arange(64), torch.arange(80), indexing="ij")
        target = ((x // 16 + 3 * (y // 16)) % 9)[None].expand(2, -1, -1).clone()
        target[:, :2, :3] = 255
        for name in MODEL_NAMES:
            config = {"model": name, "recipe": "v3", "pretrained": 0, "class_weights": [1.] * 9}
            model = build_model(config).train()
            self.assertIsInstance(model.decoder, StrongDecoder)
            out = model(rgb, thermal)
            self.assertEqual(out["deep_logits"].shape, (2, 9, 64, 80))
            self.assertEqual(out["edge_logits"].shape, (2, 1, 64, 80))
            loss, parts = training_loss(out, target, torch.tensor([-1, -1]), config, LEGACY_WEIGHTS)
            self.assertTrue(torch.isfinite(loss))
            self.assertGreater(float(parts["deep"].detach()), 0)
            self.assertGreater(float(parts["edge"].detach()), 0)
            loss.backward()
            self.assertIsNotNone(model.decoder.deep_head.weight.grad)
            self.assertIsNotNone(model.decoder.boundary_head.weight.grad)
            del model, out, loss, parts
            gc.collect()
        decoder = StrongDecoder(9).train()
        features = [torch.randn(1, c, h, w) for c, h, w in ((64, 13, 17), (128, 7, 9), (256, 3, 5), (512, 1, 1))]
        self.assertTrue(torch.isfinite(decoder(features, (51, 67))["logits"]).all())
        print("Shared v3 recipe: all 8 models forward/loss/backward; singleton PPM and odd sizes passed")

    def test_original_checkpoint_parity(self):
        original = PROJECT.with_name("rcaf")
        if not (original / "models.py").exists():
            self.skipTest("original sibling rcaf directory unavailable for external reference")
        with tempfile.TemporaryDirectory(dir=PROJECT / "results", prefix="legacy_reference_") as tmp:
            script = '''
import sys, torch
from pathlib import Path
from models import build_model
torch.set_num_threads(2)
torch.manual_seed(37)
folder=Path(sys.argv[1])
rgb,thermal=torch.randn(1,3,64,80),torch.randn(1,1,64,80)
configs=[dict(model=n) for n in ['rgb_only','thermal_only','early_fusion','late_fusion','add_fusion','concat_fusion']]
configs += [dict(model='rcaf',model_version=v,**o) for v in [1,2] for o in [{},{'no_align':True},{'no_reliability':True},{'no_interaction':True}]]
for i,config in enumerate(configs):
 config.update(pretrained=0,num_classes=9)
 model=build_model(config).eval()
 with torch.inference_mode(): logits=model(rgb,thermal)['logits']
 saved=dict(config)
 if saved.get('model_version')==1: saved.pop('model_version')
 torch.save(dict(model=model.state_dict(),config=saved,rgb=rgb,thermal=thermal,expected=logits),folder/f'{i}.pt')
 del model
'''
            result = subprocess.run([sys.executable, "-c", script, tmp], cwd=original,
                                    capture_output=True, text=True, timeout=180)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            paths = list(Path(tmp).glob("*.pt"))
            for path in paths:
                model, checkpoint = load_model(path, torch.device("cpu"))
                with torch.inference_mode():
                    actual = model(checkpoint["rgb"], checkpoint["thermal"])["logits"]
                self.assertTrue(torch.equal(actual, checkpoint["expected"]), str(checkpoint["config"]))
                del model, checkpoint
            self.assertEqual(len(paths), 14)
        print("Original-folder compatibility: 14 baseline/v1/v2/ablation checkpoints load with bitwise-identical logits")

    def test_cli_v3_workflows_and_resume(self):
        with tempfile.TemporaryDirectory(dir=PROJECT / "results", prefix="v3_smoke_") as tmp:
            folder = Path(tmp)
            def run(script, *args):
                result = subprocess.run([sys.executable, script, *map(str, args)], cwd=PROJECT,
                                        capture_output=True, text=True, timeout=180)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                return result.stdout
            data = folder / "data"
            print(run("scripts/make_tiny_dataset.py", "--root", data).strip())
            runtime = ["--root", data, "--device", "cpu", "--num_workers", "0", "--threads", "2",
                       "--height", "96", "--width", "128", "--batch_size", "2"]
            cases = [("add", "add_fusion", []), ("dcr", "dcr", []),
                     ("no_dwl", "dcr", ["--no_dwl"]), ("no_spatial", "dcr", ["--no_spatial"]),
                     ("no_channel", "dcr", ["--no_channel", "--class_weight", "none"]),
                     ("no_cross", "dcr", ["--no_cross"]), ("1234", "dcr", ["--dcr_stages", "1234"])]
            for label, name, options in cases:
                with tempfile.TemporaryDirectory(dir=folder, prefix=label + "_") as exp:
                    exp = Path(exp)
                    stdout = run("train.py", *runtime, "--recipe", "v3", "--model", name, "--pretrained", "0",
                                 "--exp", exp, "--epochs", "1", *options)
                    print(f"{label}: " + next(x for x in stdout.splitlines() if x.startswith("epoch=1/")))
                    checkpoint = torch.load(exp / "last.pt", weights_only=True)
                    self.assertEqual(checkpoint["config"]["decoder"], "strong")
                    self.assertEqual(checkpoint["step"], 2)
                    if label == "no_channel":
                        self.assertFalse((exp / "class_weights.json").exists())
                    else:
                        stats = json.loads((exp / "class_weights.json").read_text())
                        self.assertEqual(stats["weights"], checkpoint["config"]["class_weights"])
                        self.assertAlmostEqual(sum(stats["weights"]) / 9, 1)
                        if label == "add":
                            print("Synthetic class counts: " + ",".join(map(str, stats["counts"])))
                            print("Synthetic class weights: " + ",".join(f"{x:.6f}" for x in stats["weights"]))
                    if label == "dcr":
                        saved_weights = checkpoint["config"]["class_weights"]
                        (exp / "class_weights.json").write_text(json.dumps({"weights": [999.] * 9}))
                        resumed = run("train.py", "--resume", exp / "last.pt", "--epochs", "2",
                                      "--device", "cpu", "--threads", "2", "--num_workers", "0",
                                      "--recipe", "v2", "--model", "rgb_only", "--no_cross", "--lambda_deep", "0")
                        restored = torch.load(exp / "last.pt", weights_only=True)
                        self.assertEqual((restored["epoch"], restored["step"]), (2, 4))
                        self.assertEqual(restored["config"]["recipe"], "v3")
                        self.assertFalse(restored["config"]["no_cross"])
                        self.assertEqual(restored["config"]["lambda_deep"], 0.4)
                        self.assertEqual(restored["config"]["class_weights"], saved_weights)
                        self.assertEqual(json.loads((exp / "class_weights.json").read_text())["weights"], saved_weights)
                        print(next(x for x in resumed.splitlines() if x.startswith("epoch=2/")))
                        print("V3 resume: checkpoint recipe/ablations/loss weights restored; cached class weights reused")
                        del restored
                    with (exp / "history.csv").open() as stream:
                        history = list(csv.DictReader(stream))
                    self.assertEqual([int(row["epoch"]) for row in history], [1, 2] if label == "dcr" else [1])
                    for row in history:
                        self.assertTrue(all(np.isfinite(float(row[k])) for k in ("deep", "edge", "dwl", "loss")))
                    if label == "no_dwl":
                        self.assertEqual(float(history[0]["dwl"]), 0)
                    evaluation = run("eval.py", *runtime, "--checkpoint", exp / "last.pt", "--output", folder / "eval.json")
                    self.assertIn("overall: mIoU=", evaluation)
                    run("robustness_sweep.py", *runtime, "--checkpoint", exp / "last.pt", "--output", folder / "robustness.md")
                    robust = json.loads((folder / "robustness.json").read_text())["results"]
                    self.assertEqual(len(robust), 8)
                    diagnostics = run("diag.py", *runtime, "--checkpoint", exp / "last.pt", "--stride", "1")
                    self.assertTrue(all(f"{mode}:" in diagnostics for mode in ("clean", "rgb_missing", "thermal_missing")))
                    if name == "dcr":
                        self.assertIn("residual/sum=", diagnostics)
                    else:
                        self.assertIn("DCR=N/A", diagnostics)
                    if label in ("add", "dcr"):
                        print(diagnostics.strip())
                        efficiency = run("efficiency.py", *runtime, "--model", name, "--recipe", "v3",
                                         "--warmup", "0", "--iterations", "1", "--output", folder / f"{label}_eff.json")
                        print(label + " v3 efficiency: " + next(x for x in efficiency.splitlines() if x.startswith("params=")))
                    if label == "dcr":
                        vis = PROJECT / "results/v3_visual_check"
                        run("visualize.py", *runtime, "--checkpoint", exp / "last.pt", "--count", "1", "--output", vis)
                        self.assertEqual(len(list(vis.glob("*_dcr_M.png"))), 1)
                        self.assertEqual(len(list(vis.glob("*_dcr_s2_spatial.png"))), 1)
                    print(f"{label}: eval + 8-corruption robustness + clean/missing diagnostics PASS")
                    del checkpoint
                    gc.collect()
        print("V3 integration: 7 training configurations, DCR resume, eval/sweep/diag and DCR visualization passed; temporary checkpoints removed")


if __name__ == "__main__":
    unittest.main(verbosity=2)
