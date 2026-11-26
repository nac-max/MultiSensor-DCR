#!/usr/bin/env bash
set -euo pipefail
# Run from project root. First argument is MFNet root; optional second is run prefix.
ROOT="${1:?Usage: bash scripts/reproduce.sh DATA_ROOT [PREFIX]}"
PREFIX="${2:-mfnet}"
for MODEL in rgb_only thermal_only early_fusion late_fusion add_fusion concat_fusion rcaf; do
  python train.py --root "$ROOT" --model "$MODEL" --exp "${PREFIX}_${MODEL}"
done
for FLAG in no_align no_reliability no_interaction no_gate_sup no_corrupt; do
  python train.py --root "$ROOT" --model rcaf --exp "${PREFIX}_${FLAG}" "--${FLAG}"
done
for MODEL in rgb_only thermal_only early_fusion late_fusion add_fusion concat_fusion rcaf; do
  python eval.py --root "$ROOT" --checkpoint "runs/${PREFIX}_${MODEL}/best.pt" --output "results/${PREFIX}_${MODEL}.json"
  python robustness_sweep.py --root "$ROOT" --checkpoint "runs/${PREFIX}_${MODEL}/best.pt" --output "results/${PREFIX}_${MODEL}_robustness.md"
done
