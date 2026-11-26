"""Shared CLI, reproducibility, metrics, checkpoint and robustness utilities."""

import argparse
import json
from pathlib import Path
import random

import numpy as np
import torch
from torch.utils.data import DataLoader

from data import PairedDataset, RGB_MEAN, RGB_STD
from models import MODEL_NAMES, build_model
from rcaf import _warp


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id):
    seed = torch.initial_seed() % (2 ** 32)
    random.seed(seed)
    np.random.seed(seed)


def base_parser(description, checkpoint=False):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--root", default="data/MFNet")
    parser.add_argument("--dataset", choices=["mfnet", "pst900"], default="mfnet")
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ignore_index", type=int, default=255)
    if checkpoint:
        parser.add_argument("--checkpoint", required=True)
        parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    else:
        parser.add_argument("--model", choices=MODEL_NAMES, default="rcaf")
        parser.add_argument("--pretrained", type=int, choices=[0, 1], default=1)
        parser.add_argument("--max_disp", type=float, default=1.0)
        parser.add_argument("--recipe", choices=["v2", "v3"], default="v2")
        parser.add_argument("--class_weight", choices=["none", "log"], default=None)
        parser.add_argument("--lambda_deep", type=float, default=0.4)
        parser.add_argument("--lambda_edge", type=float, default=0.5)
        parser.add_argument("--lambda_dwl", type=float, default=1.0)
        parser.add_argument("--dcr_stages", default="234")
        for flag in ("no_dwl", "no_spatial", "no_channel", "no_cross"):
            parser.add_argument(f"--{flag}", action="store_true")
        for flag in ("no_align", "no_reliability", "no_interaction", "no_gate_sup", "no_corrupt"):
            parser.add_argument(f"--{flag}", action="store_true")
    return parser


def setup(args):
    torch.set_num_threads(args.threads)
    seed_everything(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; use --device cpu")
    return device


def make_loader(args, split, train=False):
    dataset = PairedDataset(args.root, split, args.dataset, (args.height, args.width),
                            augment=train, ignore_index=args.ignore_index)
    generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=train,
                        num_workers=args.num_workers, pin_memory=str(args.device).startswith("cuda"),
                        worker_init_fn=seed_worker, generator=generator)
    return dataset, loader


def load_model(path, device):
    # Only load checkpoints from trusted local sources.
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    config = dict(checkpoint["config"])
    config.setdefault("model_version", 1)
    config["pretrained"] = 0  # Never download weights when restoring.
    model = build_model(config)
    model.load_state_dict(checkpoint["model"])
    return model.to(device).eval(), checkpoint


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


class Confusion:
    def __init__(self, classes, ignore_index=255):
        self.classes, self.ignore_index = classes, ignore_index
        self.matrix = torch.zeros(len(classes), len(classes), dtype=torch.int64)

    def update(self, prediction, target):
        n = len(self.classes)
        prediction, target = prediction.detach().cpu().long(), target.detach().cpu().long()
        valid = (target != self.ignore_index) & (target >= 0) & (target < n)
        self.matrix += torch.bincount(n * target[valid] + prediction[valid], minlength=n * n).reshape(n, n)

    def result(self):
        matrix = self.matrix.double()
        intersection = matrix.diag()
        union = matrix.sum(0) + matrix.sum(1) - intersection
        valid = union > 0
        iou = intersection / union.clamp_min(1)
        return {"per_class_iou": {name: float(iou[i]) if valid[i] else None for i, name in enumerate(self.classes)},
                "miou": float(iou[valid].mean()) if valid.any() else None,
                "pixel_accuracy": float(intersection.sum() / matrix.sum()) if matrix.sum() else None,
                "valid_pixels": int(matrix.sum())}


def apply_corruption(rgb, thermal, specification):
    """Eval corruption in [0,1] intensity space; sigma is also in [0,1] units."""
    if specification == "none":
        return rgb, thermal
    parts = specification.split(":")
    name = parts[0]
    needs_value = name in ("rgb_noise", "thermal_noise", "thermal_shift", "occlusion", "rgb_dark")
    if len(parts) != (2 if needs_value else 1):
        raise ValueError(f"invalid corruption specification: {specification}")
    value = float(parts[1]) if needs_value else 0.0
    if not np.isfinite(value) or value < 0:
        raise ValueError("corruption strength must be finite and nonnegative")
    mean, std = RGB_MEAN.to(rgb), RGB_STD.to(rgb)
    r, t = (rgb * std + mean).clone(), ((thermal + 1) / 2).clone()
    if name == "rgb_missing":
        r.zero_()
    elif name == "thermal_missing":
        t.zero_()
    elif name == "rgb_noise":
        r = (r + torch.randn_like(r) * value).clamp(0, 1)
    elif name == "thermal_noise":
        t = (t + torch.randn_like(t) * value).clamp(0, 1)
    elif name == "thermal_shift":
        flow = t.new_zeros((t.shape[0], 2, *t.shape[-2:]))
        flow[:, 0] = -value  # Shift visible thermal contents right, without wraparound.
        t = _warp(t, flow)
    elif name == "occlusion":
        if value > 1:
            raise ValueError("occlusion area ratio must be in [0,1]")
        h, w = r.shape[-2:]
        oh, ow = round(h * value ** 0.5), round(w * value ** 0.5)
        top, left = (h - oh) // 2, (w - ow) // 2
        r[..., top:top + oh, left:left + ow] = 0
    elif name == "rgb_dark":
        if value > 1:
            raise ValueError("rgb_dark factor must be in [0,1]")
        r = r * value
    else:
        raise ValueError(f"unknown corruption: {name}")
    return (r - mean) / std, t * 2 - 1


@torch.inference_mode()
def evaluate(model, loader, device, classes, corruption="none", ignore_index=255, day_night=False):
    model.eval()
    meters = {key: Confusion(classes, ignore_index) for key in ("overall", "day", "night")}
    for batch in loader:
        rgb, thermal = batch["rgb"].to(device), batch["thermal"].to(device)
        rgb, thermal = apply_corruption(rgb, thermal, corruption)
        prediction = model(rgb, thermal)["logits"].argmax(1).cpu()
        meters["overall"].update(prediction, batch["label"])
        if day_night:
            for i, period in enumerate(batch["period"]):
                if period in ("day", "night"):
                    meters[period].update(prediction[i], batch["label"][i])
    return {key: meter.result() for key, meter in meters.items() if key == "overall" or day_night}
