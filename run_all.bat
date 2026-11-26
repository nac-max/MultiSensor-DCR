@echo off
for %%E in ("add_fusion 1 v3_add_s1" "dcr 1 v3_dcr_s1" "add_fusion 2 v3_add_s2" "dcr 2 v3_dcr_s2") do (
  for /f "tokens=1-3" %%a in (%%E) do (
    python train.py --model %%a --recipe v3 --seed %%b --exp %%c
    python eval.py --checkpoint runs/%%c/best.pt --output results/%%c.json
    python robustness_sweep.py --checkpoint runs/%%c/best.pt --output results/%%c_robust.md
  )
)
for %%A in (no_cross no_spatial no_dwl) do (
  python train.py --model dcr --recipe v3 --seed 42 --%%A --exp v3_dcr_%%A
  python eval.py --checkpoint runs/v3_dcr_%%A/best.pt --output results/v3_dcr_%%A.json
  python robustness_sweep.py --checkpoint runs/v3_dcr_%%A/best.pt --output results/v3_dcr_%%A_robust.md
)