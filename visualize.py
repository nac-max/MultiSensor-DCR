"""Save RGB, labels, predictions and per-stage RCAF diagnostics."""

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import torch
from torch.nn import functional as F

from common import base_parser, load_model, make_loader, setup, write_json
from data import RGB_MEAN, RGB_STD
from rcaf import RCAFusion


PALETTE = np.array([[0, 0, 0], [64, 128, 64], [255, 64, 64], [255, 192, 0],
                    [128, 128, 128], [64, 64, 255], [192, 64, 192], [0, 192, 192],
                    [192, 128, 64]], dtype=np.uint8)


def colorize(label):
    label = np.asarray(label)
    result = np.full((*label.shape, 3), 255, dtype=np.uint8)
    valid = (label >= 0) & (label < len(PALETTE))
    result[valid] = PALETTE[label[valid]]
    return Image.fromarray(result)


def heatmap(tensor, size):
    tensor = F.interpolate(tensor.detach().float().cpu(), size=size, mode="bilinear", align_corners=False)[0, 0]
    array = tensor.clamp(0, 1).numpy()
    return Image.fromarray(np.stack((array, np.zeros_like(array), 1 - array), -1).__mul__(255).astype(np.uint8))


@torch.inference_mode()
def main():
    parser = base_parser("Visualize predictions and RCAF maps", checkpoint=True)
    parser.add_argument("--count", type=int, default=4)
    parser.add_argument("--output", default="results/visualizations")
    args = parser.parse_args()
    device = setup(args)
    args.batch_size = 1
    dataset, loader = make_loader(args, args.split)
    model, checkpoint = load_model(args.checkpoint, device)
    if checkpoint["config"]["num_classes"] != len(dataset.class_names):
        raise ValueError("checkpoint classes do not match dataset")
    directory = Path(args.output)
    directory.mkdir(parents=True, exist_ok=True)
    scales, count = {}, 0
    for batch in loader:
        if count >= args.count:
            break
        name = batch["name"][0]
        out = model(batch["rgb"].to(device), batch["thermal"].to(device))
        rgb = ((batch["rgb"][0] * RGB_STD + RGB_MEAN).clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        images = {"rgb": Image.fromarray(rgb), "target": colorize(batch["label"][0]),
                  "prediction": colorize(out["logits"][0].argmax(0).cpu())}
        size = batch["rgb"].shape[-2:]
        if "dwl_map" in out:
            images["dcr_M"] = heatmap(out["dwl_map"], size)
            for info in out["dcr"]:
                if info["stage"] == 2:
                    images["dcr_s2_spatial"] = heatmap(info["spatial"], size)
        for i, (gates, flow, aux) in enumerate(zip(out["gates"], out["flows"], out["aux"]), 1):
            images[f"s{i}_thermal_gate"] = heatmap(gates[:, 1:], size)
            magnitude = flow.square().sum(1, keepdim=True).sqrt()
            maximum = float(magnitude.max())
            scales[f"{name}_s{i}_flow_max_feature_pixels"] = maximum
            images[f"s{i}_flow_magnitude"] = heatmap(magnitude / max(maximum, 1e-8), size)
            for modality, logits in zip(("rgb", "thermal"), aux):
                images[f"s{i}_{modality}_reliability"] = heatmap(RCAFusion.reliability(logits), size)
        for key, im in images.items():
            im.save(directory / f"{name}_{key}.png")
        # Overview has RGB/target/prediction and stage-1 maps.
        keys = ["rgb", "target", "prediction"] + [k for k in images if k.startswith(("s1_", "dcr_"))]
        w, h = images["rgb"].size
        overview = Image.new("RGB", (w * 3, (h + 20) * 3), "white")
        draw = ImageDraw.Draw(overview)
        for j, key in enumerate(keys):
            x, y = (j % 3) * w, (j // 3) * (h + 20)
            draw.text((x + 2, y + 2), key, fill="black")
            overview.paste(images[key], (x, y + 20))
        overview.save(directory / f"{name}_overview.png")
        count += 1
    write_json(directory / "legend.json", {"classes": dataset.class_names, "palette": PALETTE.tolist(),
               "heatmap": "blue=0, red=1; flow normalized per image/stage, see scales", "scales": scales})
    print(f"saved {count} images with predictions and {len(out['gates']) if count else 0}-stage diagnostics to {directory}")


if __name__ == "__main__":
    main()
