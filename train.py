"""Train any baseline or RCAF; safe for Windows DataLoader spawn."""

import csv
import os
from pathlib import Path
import tempfile

import torch

from common import base_parser, evaluate, make_loader, setup, write_json
from models import build_model
from rcaf import corrupt_modality
from v3 import class_statistics, training_loss, validate_v3


def atomic_save(state, path):
    path = Path(path)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=path.name + ".", suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            torch.save(state, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def reconcile_history(path, epoch):
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        return
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        fields = reader.fieldnames
        rows = [row for row in reader if int(row["epoch"]) <= epoch]
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, prefix=path.name + ".",
                                         suffix=".tmp", newline="", encoding="utf-8", delete=False) as stream:
            temporary = Path(stream.name)
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main():
    parser = base_parser("Train RGB-T segmentation")
    parser.add_argument("--exp", default="rcaf")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--optimizer", choices=["adamw", "sgd"], default="adamw")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--amp", choices=["auto", "bf16", "fp16", "off"], default="auto")
    parser.add_argument("--corrupt_p", type=float, default=0.5)
    parser.add_argument("--lambda_aux", type=float, default=0.4)
    parser.add_argument("--lambda_gate", type=float, default=0.1)
    parser.add_argument("--lambda_tv", type=float, default=0.01)
    parser.add_argument("--lambda_mag", type=float, default=0.1)
    parser.add_argument("--tensorboard", action="store_true")
    parser.add_argument("--resume", help="Resume last.pt or legacy best.pt in its existing run directory")
    args = parser.parse_args()
    checkpoint = None
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=True)
        saved = checkpoint["config"]
        runtime = {"resume", "epochs", "device", "threads", "num_workers", "tensorboard", "exp"}
        parser.set_defaults(**{key: value for key, value in saved.items()
                               if key in vars(args) and key not in runtime})
        args = parser.parse_args()
        # Model structure and optimizer type must match their saved states.
        for key in ("model", "pretrained", "no_align", "no_reliability", "no_interaction",
                    "no_gate_sup", "no_corrupt", "optimizer", "lr", "weight_decay", "max_disp",
                    "recipe", "class_weight", "lambda_deep", "lambda_edge", "lambda_dwl",
                    "dcr_stages", "no_dwl", "no_spatial", "no_channel", "no_cross"):
            setattr(args, key, saved.get(key, parser.get_default(key)))
        directory = Path(args.resume).resolve().parent
        args.exp = directory.name
        if args.epochs <= checkpoint["epoch"]:
            parser.error("--epochs is the total target and must exceed the checkpoint epoch")
        print(f"Resume uses checkpoint model/ablation and optimizer config: model={args.model}, "
              f"optimizer={args.optimizer}; directory={directory}")
    else:
        directory = Path("runs") / args.exp
    if args.class_weight is None:
        args.class_weight = "log" if args.recipe == "v3" else "none"
    try:
        validate_v3(vars(args))
    except ValueError as error:
        parser.error(str(error))
    if args.epochs < 1 or args.batch_size < 1:
        parser.error("epochs and batch_size must be positive")
    if args.model != "rcaf" and any(getattr(args, k) for k in
                                   ("no_align", "no_reliability", "no_interaction", "no_gate_sup")):
        parser.error("RCAF architecture/gate ablations require --model rcaf")
    device = setup(args)
    dataset, loader = make_loader(args, "train", train=True)
    val_dataset, val_loader = make_loader(args, "val")
    config = vars(args).copy()
    config["num_classes"] = len(dataset.class_names)
    config["model_version"] = checkpoint["config"].get("model_version", 1) if checkpoint is not None else 2
    config["decoder"] = (checkpoint["config"].get("decoder", "legacy") if checkpoint is not None
                         else "strong" if args.recipe == "v3" else "legacy")
    directory.mkdir(parents=True, exist_ok=True)
    if checkpoint is None and (directory / "history.csv").exists():
        raise FileExistsError(f"{directory} already contains a run; choose a new --exp")
    if args.class_weight == "log":
        if checkpoint is not None and "class_weights" in checkpoint["config"]:
            config["class_weights"] = checkpoint["config"]["class_weights"]
            statistics = checkpoint["config"].get("class_statistics", {"weights": config["class_weights"]})
        else:
            statistics = class_statistics(dataset)
            config["class_weights"] = statistics["weights"]
        if len(config["class_weights"]) != config["num_classes"]:
            raise ValueError("class weights do not match dataset classes")
        config["class_statistics"] = statistics
        write_json(directory / "class_weights.json", statistics)
        print("Class weights (log, mean=1): " + ",".join(f"{x:.6f}" for x in config["class_weights"]))
    if checkpoint is not None and checkpoint["config"]["num_classes"] != config["num_classes"]:
        raise ValueError("checkpoint classes do not match dataset")
    model_config = dict(config)
    if checkpoint is not None:
        model_config["pretrained"] = 0
    model = build_model(model_config).to(device)
    if checkpoint is not None:
        model.load_state_dict(checkpoint["model"])
    optimizer = (torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
                 if args.optimizer == "adamw" else
                 torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay))
    use_amp = device.type == "cuda" and args.amp != "off"
    dtype = torch.bfloat16 if args.amp in ("auto", "bf16") and torch.cuda.is_bf16_supported() else torch.float16
    if use_amp and args.amp == "bf16" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("bf16 not supported; choose --amp fp16")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and dtype == torch.float16)
    start_epoch, best, step = 1, -1.0, 0
    if checkpoint is not None:
        start_epoch = checkpoint["epoch"] + 1
        step = checkpoint.get("step", checkpoint["epoch"] * len(loader))
        best = checkpoint.get("best", checkpoint.get("best_miou", -1.0))
        if "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
        else:
            print("WARNING: checkpoint has no optimizer state; using a fresh optimizer (legacy best.pt).")
        if "scaler" in checkpoint:
            if scaler.is_enabled() and not checkpoint["scaler"]:
                print("WARNING: checkpoint has no active scaler state; using a fresh scaler.")
            else:
                scaler.load_state_dict(checkpoint["scaler"])
        else:
            print("WARNING: checkpoint has no scaler state; using a fresh scaler.")
        reconcile_history(directory / "history.csv", checkpoint["epoch"])
        print(f"Resuming at epoch={start_epoch}, step={step}, best={best:.6f}, target_epochs={args.epochs}")
    write_json(directory / "config.json", config)
    writer = None
    if args.tensorboard:
        try:
            from torch.utils.tensorboard import SummaryWriter
            writer = SummaryWriter(str(directory / "tensorboard"))
        except ImportError:
            print("TensorBoard not installed; CSV logging remains enabled.")
    weights = {"aux": args.lambda_aux, "gate": 0 if args.no_gate_sup else args.lambda_gate,
               "tv": args.lambda_tv, "mag": args.lambda_mag}
    max_steps = args.epochs * len(loader)
    print(f"model={args.model} device={device} amp={str(dtype) if use_amp else 'off'} "
          f"train={len(dataset)} val={len(val_dataset)}")
    history = directory / "history.csv"
    needs_header = not history.exists() or history.stat().st_size == 0
    extended = args.recipe == "v3" or args.model == "dcr" or args.class_weight == "log"
    with history.open("a" if checkpoint is not None else "w", newline="", encoding="utf-8") as stream:
        fields = ["epoch", "lr", "loss", "main", "aux", "gate", "tv", "mag", "val_miou", "val_pixel_acc"]
        if extended:
            fields += ["deep", "edge", "dwl"]
        if checkpoint is not None and not needs_header:
            with history.open(newline="", encoding="utf-8") as existing:
                fields = next(csv.reader(existing))
        log = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        if checkpoint is None or needs_header:
            log.writeheader()
        for epoch in range(start_epoch, args.epochs + 1):
            model.train()
            totals = {key: 0.0 for key in ("loss", "main", "aux", "gate", "tv", "mag")}
            if extended:
                totals.update({key: 0.0 for key in ("deep", "edge", "dwl")})
            seen = 0
            try:
                from tqdm import tqdm
                iterator = tqdm(loader, desc=f"epoch {epoch}/{args.epochs}", leave=False, dynamic_ncols=True)
            except ImportError:
                iterator = loader
            for it, batch in enumerate(iterator, 1):
                rgb, thermal, target = (batch[key].to(device, non_blocking=True) for key in ("rgb", "thermal", "label"))
                # A final singleton batch at 1x1 stage-4 cannot train BatchNorm.
                if rgb.shape[0] == 1:
                    for module in model.modules():
                        if isinstance(module, torch.nn.BatchNorm2d):
                            module.eval()
                labels = torch.full((rgb.shape[0],), -1, device=device, dtype=torch.long)
                if not args.no_corrupt:
                    rgb, thermal, labels = corrupt_modality(rgb, thermal, args.corrupt_p)
                lr = args.lr * max(0, 1 - step / max_steps) ** 0.9
                for group in optimizer.param_groups:
                    group["lr"] = lr
                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast(device_type=device.type, dtype=dtype, enabled=use_amp):
                    out = model(rgb, thermal)
                # BCE and entropy-related losses run in fp32, outside autocast.
                out = {**out, "logits": out["logits"].float(),
                       "aux": [(r.float(), t.float()) for r, t in out["aux"]],
                       "gates": [x.float() for x in out["gates"]],
                       "flows": [x.float() for x in out["flows"]]}
                for key in ("deep_logits", "edge_logits", "dwl_map"):
                    if key in out:
                        out[key] = out[key].float()
                loss, parts = training_loss(out, target, labels, config, weights)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"nonfinite loss at epoch={epoch}, step={step}")
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                n = rgb.shape[0]
                seen += n
                for key, value in {"loss": loss, **parts}.items():
                    totals[key] += float(value.detach()) * n
                step += 1
                if iterator is loader and (it % 20 == 0 or it == len(loader)):
                    print(f"  epoch {epoch} iter {it}/{len(loader)} loss={totals['loss'] / seen:.4f}", flush=True)
                elif iterator is not loader:
                    iterator.set_postfix(loss=f"{totals['loss'] / seen:.4f}")
            metrics = evaluate(model, val_loader, device, dataset.class_names,
                               ignore_index=args.ignore_index)["overall"]
            if metrics["miou"] is None:
                raise ValueError("validation has no valid pixels")
            row = {"epoch": epoch, "lr": lr, **{k: v / seen for k, v in totals.items()},
                   "val_miou": metrics["miou"], "val_pixel_acc": metrics["pixel_accuracy"]}
            log.writerow(row)
            stream.flush()
            if writer:
                for key, value in row.items():
                    if key != "epoch":
                        writer.add_scalar(key, value, epoch)
            if metrics["miou"] > best:
                best = metrics["miou"]
                atomic_save({"model": model.state_dict(), "config": config, "epoch": epoch,
                             "best_miou": best}, directory / "best.pt")
            atomic_save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                         "scaler": scaler.state_dict(), "epoch": epoch, "step": step,
                         "best": best, "config": config}, directory / "last.pt")
            print(f"epoch={epoch}/{args.epochs} loss={row['loss']:.4f} val_mIoU={metrics['miou']:.4f} best={best:.4f}")
    if writer:
        writer.close()
    print(f"saved {directory / 'last.pt'} (best checkpoint: {directory / 'best.pt'})")


if __name__ == "__main__":
    main()
