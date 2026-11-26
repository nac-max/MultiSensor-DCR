"""Sample every eighth test image and report gate/flow health by stage."""

import torch
from torch.utils.data import DataLoader, Subset

from common import (Confusion, apply_corruption, base_parser, load_model,
                    make_loader, seed_everything, setup)


@torch.inference_mode()
def main():
    parser = base_parser("RCAF gate/flow diagnostics", checkpoint=True)
    parser.add_argument("--stride", type=int, default=8)
    args = parser.parse_args()
    if args.stride < 1:
        parser.error("stride must be positive")
    device = setup(args)
    dataset, _ = make_loader(args, args.split)
    model, checkpoint = load_model(args.checkpoint, device)
    if checkpoint["config"]["num_classes"] != len(dataset.class_names):
        raise ValueError("checkpoint classes do not match dataset")
    indices = list(range(0, len(dataset), args.stride))
    loader = DataLoader(Subset(dataset, indices), batch_size=1, num_workers=0)
    print(f"checkpoint={args.checkpoint} model_version={checkpoint['config'].get('model_version', 1)} "
          f"images={len(indices)}/{len(dataset)} stride={args.stride}")
    for mode in ("clean", "rgb_missing", "thermal_missing"):
        seed_everything(args.seed)
        meter = Confusion(dataset.class_names, args.ignore_index)
        gates, flows, dcr = [], [], {}
        for batch in loader:
            rgb, thermal = batch["rgb"].to(device), batch["thermal"].to(device)
            rgb, thermal = apply_corruption(rgb, thermal, "none" if mode == "clean" else mode)
            out = model(rgb, thermal)
            meter.update(out["logits"].argmax(1), batch["label"])
            if out.get("gates"):
                gates.append([float(x[:, 1].mean()) for x in out["gates"]])
            if out.get("flows"):
                flows.append([float(x.abs().mean()) for x in out["flows"]])
            for info in out.get("dcr", []):
                dcr.setdefault(info["stage"], []).append([
                    float(info["alpha_abs"]), float(info["ratio"].mean()), float(info["spatial"].mean())])
        miou = meter.result()["miou"]
        summary = f"mIoU={miou * 100:.2f}%" if miou is not None else "mIoU=N/A"
        for label, values in (("w_t(s1..s4)", gates), ("|flow|(feature px)", flows)):
            means = torch.tensor(values).mean(0).tolist() if values else []
            summary += f" {label}=" + (",".join(f"{x:.6f}" for x in means) if means else "N/A (baseline)")
        for stage in range(1, 5):
            if stage in dcr:
                alpha, ratio, spatial = torch.tensor(dcr[stage]).mean(0).tolist()
                summary += f" s{stage}[|alpha|={alpha:.6f},residual/sum={ratio:.6f},A_s={spatial:.6f}]"
            else:
                summary += f" s{stage}[DCR=N/A]"
        print(f"{mode}: {summary}", flush=True)


if __name__ == "__main__":
    main()
