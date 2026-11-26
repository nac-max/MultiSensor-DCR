"""Evaluate clean or corrupted data; metrics are fractions, not percentages."""

from common import base_parser, evaluate, load_model, make_loader, setup, write_json


def main():
    parser = base_parser("Evaluate RGB-T segmentation", checkpoint=True)
    parser.add_argument("--corrupt", default="none")
    parser.add_argument("--output", default="results/eval.json")
    args = parser.parse_args()
    device = setup(args)
    dataset, loader = make_loader(args, args.split)
    model, checkpoint = load_model(args.checkpoint, device)
    if checkpoint["config"]["num_classes"] != len(dataset.class_names):
        raise ValueError("checkpoint classes do not match dataset")
    results = evaluate(model, loader, device, dataset.class_names, args.corrupt,
                       args.ignore_index, day_night=args.dataset == "mfnet")
    write_json(args.output, {"checkpoint": args.checkpoint, "dataset": args.dataset,
                             "split": args.split, "corruption": args.corrupt, "seed": args.seed,
                             "resolution": [args.height, args.width], **results})
    for split, result in results.items():
        print(f"{split}: mIoU={result['miou']} pixel_acc={result['pixel_accuracy']}")
        print(f"  IoU: {result['per_class_iou']}")
    print(f"saved {args.output}")


if __name__ == "__main__":
    main()
