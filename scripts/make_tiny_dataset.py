"""Generate paired synthetic data for plumbing tests, not accuracy claims."""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="data/tiny_mfnet")
    parser.add_argument("--count", type=int, default=8)
    parser.add_argument("--height", type=int, default=96)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()
    if args.count < 8:
        parser.error("count must be >=8 (train/val/test each need day and night)")
    root = Path(args.root)
    (root / "images").mkdir(parents=True, exist_ok=True)
    (root / "labels").mkdir(exist_ok=True)
    rng = np.random.default_rng(args.seed)
    names = []
    y, x = np.indices((args.height, args.width))
    for i in range(args.count):
        name = f"sample_{i:03d}{'D' if i % 2 == 0 else 'N'}"
        label = ((x // max(1, args.width // 3) + 3 * (y // max(1, args.height // 3)) + i) % 9).astype(np.uint8)
        ids = label.astype(np.int32)
        rgb = np.stack(((ids * 29 + 30) % 256, (ids * 47 + 20) % 256,
                        (ids * 13 + 80) % 256), -1).astype(np.float32)
        if name.endswith("N"):
            rgb *= 0.35
        rgb = np.clip(rgb + rng.normal(0, 5, rgb.shape), 0, 255).astype(np.uint8)
        thermal = np.clip(label.astype(float) * 25 + rng.normal(0, 3, label.shape), 0, 255).astype(np.uint8)
        label[:2, :4] = 255
        Image.fromarray(np.concatenate((rgb, thermal[..., None]), -1)).save(root / "images" / f"{name}.png")
        Image.fromarray(label).save(root / "labels" / f"{name}.png")
        names.append(name)
    splits = {"train": names[:-4], "val": names[-4:-2], "test": names[-2:]}
    for split, values in splits.items():
        (root / f"{split}.txt").write_text("\n".join(values) + "\n", encoding="utf-8")
    print(f"created {root}: " + ", ".join(f"{k}={len(v)}" for k, v in splits.items()) + f" size={args.height}x{args.width}")


if __name__ == "__main__":
    main()
