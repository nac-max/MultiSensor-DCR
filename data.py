"""Paired MFNet / PST900 loading and synchronized spatial augmentation."""

from pathlib import Path
import random

import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset


MFNET_CLASSES = ["unlabeled", "car", "person", "bike", "curve", "car_stop",
                 "guardrail", "color_cone", "bump"]
PST900_CLASSES = ["background", "fire_extinguisher", "backpack", "hand_drill", "survivor"]
RGB_MEAN = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
RGB_STD = torch.tensor([0.229, 0.224, 0.225])[:, None, None]


def _thermal_tensor(array):
    if array.ndim == 3:
        if array.shape[2] != 1:
            raise ValueError("thermal PNG must be single-channel, not colorized")
        array = array[..., 0]
    # PIL may expose 16-bit PNG as int32; use bit depth rather than observed maximum.
    maximum = 65535.0 if array.dtype.itemsize > 1 else 255.0
    return torch.from_numpy(array.astype(np.float32) / maximum)[None]


class PairedDataset(Dataset):
    def __init__(self, root, split="train", dataset="mfnet", size=(480, 640),
                 augment=False, ignore_index=255):
        self.root, self.dataset = Path(root), dataset
        self.size, self.augment, self.ignore_index = tuple(size), augment, ignore_index
        self.class_names = MFNET_CLASSES if dataset == "mfnet" else PST900_CLASSES
        if dataset == "mfnet":
            self.names = [line.strip() for line in (self.root / f"{split}.txt").read_text().splitlines()
                          if line.strip()]
            self.folder = self.root
            # 官方 train.txt 含 *_flip 条目（需 make_flip.py 生成）；训练时已有随机翻转，缺失文件直接跳过
            missing = [n for n in self.names if not (self.root / "images" / f"{n}.png").exists()]
            if missing:
                print(f"[data] {split}: skip {len(missing)} names without image files (e.g. {missing[0]})")
                self.names = [n for n in self.names if n not in set(missing)]
        elif dataset == "pst900":
            # Optional real val directory; otherwise use a deterministic train holdout.
            if split in ("train", "val") and not (self.root / "val").exists():
                self.folder = self.root / "train"
                names = sorted(p.stem for p in (self.folder / "rgb").glob("*.png"))
                if len(names) < 2:
                    raise ValueError("PST900 needs >=2 training images for a train/val holdout")
                n_val = max(1, len(names) // 5)
                self.names = names[:n_val] if split == "val" else names[n_val:]
            else:
                self.folder = self.root / split
                self.names = sorted(p.stem for p in (self.folder / "rgb").glob("*.png"))
        else:
            raise ValueError(f"unknown dataset: {dataset}")
        if not self.names:
            raise ValueError(f"empty {dataset} split: {split} at {root}")

    def __len__(self):
        return len(self.names)

    def __getitem__(self, index):
        name = self.names[index]
        if self.dataset == "mfnet":
            with Image.open(self.folder / "images" / f"{name}.png") as im:
                array = np.array(im)
            if array.ndim != 3 or array.shape[2] != 4:
                raise ValueError(f"{name}: MFNet images must be RGBA (thermal in alpha)")
            rgb_array, thermal_array = array[..., :3], array[..., 3]
        else:
            with Image.open(self.folder / "rgb" / f"{name}.png") as im:
                rgb_array = np.array(im.convert("RGB"))
            with Image.open(self.folder / "thermal" / f"{name}.png") as im:
                thermal_array = np.array(im)
        with Image.open(self.folder / "labels" / f"{name}.png") as im:
            label_array = np.array(im)
        if label_array.ndim != 2:
            raise ValueError(f"{name}: labels must be class-id PNGs, not RGB masks")
        valid = (label_array < len(self.class_names)) | (label_array == self.ignore_index)
        if not valid.all():
            raise ValueError(f"{name}: invalid class ids {np.unique(label_array[~valid]).tolist()}")
        rgb = torch.from_numpy(rgb_array.copy()).permute(2, 0, 1).float() / 255
        thermal = _thermal_tensor(thermal_array)
        label = torch.from_numpy(label_array.astype(np.int64))
        if rgb.shape[-2:] != thermal.shape[-2:] or label.shape != rgb.shape[-2:]:
            raise ValueError(f"{name}: RGB, thermal and label sizes differ")
        pair = torch.cat((rgb, thermal))
        h, w = self.size
        if self.augment:
            scale = random.uniform(0.5, 2.0)
            resized = (max(1, round(h * scale)), max(1, round(w * scale)))
        else:
            resized = (h, w)
        pair = F.interpolate(pair[None], size=resized, mode="bilinear", align_corners=False)[0]
        label = F.interpolate(label[None, None].float(), size=resized, mode="nearest")[0, 0].long()
        if self.augment:
            pad_h, pad_w = max(0, h - resized[0]), max(0, w - resized[1])
            pair = F.pad(pair, (0, pad_w, 0, pad_h))
            label = F.pad(label, (0, pad_w, 0, pad_h), value=self.ignore_index)
            top = random.randint(0, pair.shape[-2] - h)
            left = random.randint(0, pair.shape[-1] - w)
            pair, label = pair[:, top:top + h, left:left + w], label[top:top + h, left:left + w]
            if random.random() < 0.5:
                pair, label = pair.flip(-1), label.flip(-1)
        rgb = (pair[:3] - RGB_MEAN) / RGB_STD
        thermal = (pair[3:] - 0.5) / 0.5
        period = ("day" if name.endswith("D") else "night" if name.endswith("N") else "unknown")
        return {"rgb": rgb, "thermal": thermal, "label": label,
                "name": name, "period": period}
