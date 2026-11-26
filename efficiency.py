"""Inference-only throughput; excludes data loading and preprocessing."""

import time

import torch

from common import base_parser, load_model, setup, write_json
from models import build_model


def main():
    parser = base_parser("Model efficiency (default resolution 480x640)")
    parser.set_defaults(batch_size=1, pretrained=0)
    parser.add_argument("--checkpoint")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--amp", choices=["off", "bf16", "fp16"], default="off")
    parser.add_argument("--output", default="results/efficiency.json")
    args = parser.parse_args()
    if args.iterations < 1 or args.warmup < 0:
        parser.error("iterations must be positive and warmup nonnegative")
    device = setup(args)
    if args.checkpoint:
        model, _ = load_model(args.checkpoint, device)
    else:
        config = vars(args).copy()
        config["num_classes"] = 9 if args.dataset == "mfnet" else 5
        model = build_model(config).to(device).eval()
    rgb = torch.randn(args.batch_size, 3, args.height, args.width, device=device)
    thermal = torch.randn(args.batch_size, 1, args.height, args.width, device=device)
    gflops, flop_note = None, "No FLOPs profiler installed; install thop or fvcore."
    with torch.inference_mode():
        try:
            from thop import profile
        except ImportError:
            try:
                from fvcore.nn import FlopCountAnalysis
            except ImportError:
                pass
            else:
                try:
                    analysis = FlopCountAnalysis(model, (rgb, thermal))
                    gflops = analysis.total() / args.batch_size / 1e9
                    flop_note = "fvcore: one multiply-add counts as one; unsupported=" + str(dict(analysis.unsupported_ops()))
                except Exception as error:
                    flop_note = f"fvcore profiling skipped: {error}"
        else:
            try:
                macs, _ = profile(model, inputs=(rgb, thermal), verbose=False)
                gflops = 2 * macs / args.batch_size / 1e9
                flop_note = "thop MACs x2; approximate, excludes unsupported functional ops (e.g. grid_sample)."
            except Exception as error:
                flop_note = f"thop profiling skipped: {error}"
        enabled = device.type == "cuda" and args.amp != "off"
        dtype = torch.bfloat16 if args.amp == "bf16" else torch.float16
        if enabled and dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
            raise RuntimeError("bf16 not supported on this device")
        def sync():
            if device.type == "cuda":
                torch.cuda.synchronize(device)
        with torch.amp.autocast(device_type=device.type, dtype=dtype, enabled=enabled):
            for _ in range(args.warmup):
                model(rgb, thermal)
            sync()
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            start = time.perf_counter()
            for _ in range(args.iterations):
                model(rgb, thermal)
            sync()
            elapsed = time.perf_counter() - start
    result = {"model": args.model if not args.checkpoint else args.checkpoint,
              "device": str(device), "torch": str(torch.__version__), "resolution": [args.height, args.width],
              "batch_size": args.batch_size, "amp": args.amp if enabled else "off",
              "params_M": sum(p.numel() for p in model.parameters()) / 1e6,
              "GFLOPs_per_image": gflops, "flop_note": flop_note,
              "fps": args.iterations * args.batch_size / elapsed,
              "latency_ms_per_batch": elapsed / args.iterations * 1000,
              "peak_cuda_memory_MB": torch.cuda.max_memory_allocated(device) / 1e6 if device.type == "cuda" else None}
    write_json(args.output, result)
    print(f"params={result['params_M']:.3f} M FPS={result['fps']:.2f} latency={result['latency_ms_per_batch']:.2f} ms/batch")
    print(f"GFLOPs/image={gflops}; {flop_note}")
    print(f"saved {args.output}")


if __name__ == "__main__":
    main()
