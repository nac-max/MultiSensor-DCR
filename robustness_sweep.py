"""Evaluate a fixed robustness suite with one checkpoint load."""

from pathlib import Path

from common import base_parser, evaluate, load_model, make_loader, seed_everything, setup, write_json


CORRUPTIONS = ["none", "rgb_missing", "thermal_missing", "rgb_noise:0.1", "thermal_noise:0.1",
               "thermal_shift:10", "occlusion:0.25", "rgb_dark:0.2"]


def main():
    parser = base_parser("Robustness sweep", checkpoint=True)
    parser.add_argument("--corruptions", nargs="+", default=CORRUPTIONS)
    parser.add_argument("--output", default="results/robustness.md")
    args = parser.parse_args()
    device = setup(args)
    dataset, loader = make_loader(args, args.split)
    model, checkpoint = load_model(args.checkpoint, device)
    if checkpoint["config"]["num_classes"] != len(dataset.class_names):
        raise ValueError("checkpoint classes do not match dataset")
    results = {}
    lines = ["| Corruption | mIoU (%) | Pixel acc (%) | Day mIoU (%) | Night mIoU (%) |",
             "|---|---:|---:|---:|---:|"]
    def percent(value):
        return "N/A" if value is None else f"{100 * value:.2f}"
    for corruption in args.corruptions:
        seed_everything(args.seed)
        result = evaluate(model, loader, device, dataset.class_names, corruption,
                          args.ignore_index, args.dataset == "mfnet")
        results[corruption] = result
        overall = result["overall"]
        day, night = result.get("day", {}).get("miou"), result.get("night", {}).get("miou")
        lines.append(f"| {corruption} | {percent(overall['miou'])} | {percent(overall['pixel_accuracy'])} | {percent(day)} | {percent(night)} |")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_json(output.with_suffix(".json"), {"config": vars(args), "results": results})
    print("\n".join(lines))


if __name__ == "__main__":
    main()
