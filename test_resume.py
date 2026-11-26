"""CPU integration checks for full-state and legacy resume; cleans its runs."""

import csv
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch

from train import atomic_save, reconcile_history


PROJECT = Path(__file__).resolve().parent


class ResumeTests(unittest.TestCase):
    def test_atomic_and_history(self):
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            path = folder / "last.pt"
            atomic_save({"epoch": 1}, path)
            with patch("train.torch.save", side_effect=OSError("simulated disk failure")):
                with self.assertRaisesRegex(OSError, "disk failure"):
                    atomic_save({"epoch": 2}, path)
            self.assertEqual(torch.load(path, weights_only=True)["epoch"], 1)
            self.assertFalse(list(folder.glob("*.tmp")))
            history = folder / "history.csv"
            reconcile_history(history, 1)
            self.assertFalse(history.exists())
            history.touch()
            reconcile_history(history, 1)
            self.assertEqual(history.stat().st_size, 0)
            with history.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.writer(stream)
                writer.writerows([["epoch", "loss"], [1, 0.5], [2, 0.4], [3, 0.3]])
            reconcile_history(history, 1)
            with history.open(newline="", encoding="utf-8") as stream:
                self.assertEqual([row["epoch"] for row in csv.DictReader(stream)], ["1"])

    def test_cpu_training_resume(self):
        (PROJECT / "runs").mkdir(exist_ok=True)
        with (tempfile.TemporaryDirectory(prefix="resume_test_A_", dir=PROJECT / "runs") as a,
              tempfile.TemporaryDirectory(prefix="resume_test_legacy_", dir=PROJECT / "runs") as b,
              tempfile.TemporaryDirectory(prefix="resume_test_data_") as data):
            a, b = Path(a), Path(b)
            subprocess.run([sys.executable, "scripts/make_tiny_dataset.py", "--root", data],
                           cwd=PROJECT, check=True, capture_output=True, text=True)

            def run(*args):
                result = subprocess.run([sys.executable, "train.py", *args], cwd=PROJECT,
                                        capture_output=True, text=True, timeout=180)
                if result.returncode:
                    self.fail(result.stdout + result.stderr)
                print(result.stdout, end="")
                return result.stdout

            run("--root", data, "--exp", str(a), "--model", "rcaf", "--pretrained", "0",
                "--epochs", "2", "--device", "cpu", "--num_workers", "0", "--threads", "2",
                "--batch_size", "2", "--height", "96", "--width", "128")
            checkpoint = torch.load(a / "last.pt", weights_only=True)
            self.assertTrue({"model", "optimizer", "scaler", "epoch", "step", "best", "config"} <= checkpoint.keys())
            self.assertEqual((checkpoint["epoch"], checkpoint["step"]), (2, 4))
            legacy = {key: checkpoint[key] for key in ("model", "config", "epoch")}
            legacy["best_miou"] = 0.99
            atomic_save(legacy, b / "best.pt")

            # A CSV row may have been flushed after the saved checkpoint.
            with (a / "history.csv").open(newline="", encoding="utf-8") as stream:
                reader = csv.DictReader(stream)
                fields, rows = reader.fieldnames, list(reader)
            with (a / "history.csv").open("a", newline="", encoding="utf-8") as stream:
                csv.DictWriter(stream, fieldnames=fields).writerow({**rows[-1], "epoch": 3})
            (b / "history.csv").write_bytes((a / "history.csv").read_bytes())
            stdout = run("--resume", str(a / "last.pt"), "--epochs", "3", "--device", "cpu",
                         "--num_workers", "0", "--threads", "2", "--model", "rgb_only", "--no_align")
            self.assertIn("model=rcaf", stdout)
            self.assertIn("epoch=3, step=4", stdout)
            self.assertNotIn("WARNING", stdout)
            restored = torch.load(a / "last.pt", weights_only=True)
            self.assertEqual((restored["epoch"], restored["step"]), (3, 6))
            self.assertEqual(restored["config"]["model"], "rcaf")
            self.assertFalse(restored["config"]["no_align"])
            self.assertTrue(all(int(state["step"]) == 6 for state in restored["optimizer"]["state"].values()))
            print("Full-state check: optimizer steps=6; checkpoint architecture overrides conflicting CLI flags")
            stdout = run("--resume", str(b / "best.pt"), "--epochs", "3", "--device", "cpu",
                         "--num_workers", "0", "--threads", "2")
            self.assertIn("WARNING: checkpoint has no optimizer state", stdout)
            self.assertIn("WARNING: checkpoint has no scaler state", stdout)
            restored = torch.load(b / "last.pt", weights_only=True)
            self.assertEqual((restored["epoch"], restored["step"], restored["best"]), (3, 6, 0.99))
            self.assertTrue(all(int(state["step"]) == 2 for state in restored["optimizer"]["state"].values()))
            print("Legacy check: inferred step=4; optimizer restarted; best_miou=0.99 preserved")
            for folder in (a, b):
                with (folder / "history.csv").open(newline="", encoding="utf-8") as stream:
                    history = list(csv.DictReader(stream))
                self.assertEqual([int(row["epoch"]) for row in history], [1, 2, 3])
                expected_lr = 1e-4 * (1 - 5 / 6) ** 0.9
                self.assertAlmostEqual(float(history[-1]["lr"]), expected_lr)
                self.assertFalse(list(folder.glob("*.tmp")))
            print("CSV/LR check: both histories=[1, 2, 3] exactly once; poly LR continued from step 4")
            failed = subprocess.run([sys.executable, "train.py", "--resume", str(a / "last.pt"),
                                     "--epochs", "3"], cwd=PROJECT, capture_output=True, text=True)
            self.assertNotEqual(failed.returncode, 0)
            self.assertIn("must exceed the checkpoint epoch", failed.stderr)
        print("Temporary resume runs and data removed")


if __name__ == "__main__":
    unittest.main(verbosity=2)
