@echo off
setlocal
rem Run from project root: scripts\reproduce.bat D:\datasets\MFNet prefix
if "%~1"=="" exit /b 1
set "ROOT=%~1"
set "PREFIX=%~2"
if "%PREFIX%"=="" set "PREFIX=mfnet"
for %%M in (rgb_only thermal_only early_fusion late_fusion add_fusion concat_fusion rcaf) do (
  python train.py --root "%ROOT%" --model %%M --exp %PREFIX%_%%M
  if errorlevel 1 exit /b 1
)
for %%F in (no_align no_reliability no_interaction no_gate_sup no_corrupt) do (
  python train.py --root "%ROOT%" --model rcaf --exp %PREFIX%_%%F --%%F
  if errorlevel 1 exit /b 1
)
for %%M in (rgb_only thermal_only early_fusion late_fusion add_fusion concat_fusion rcaf) do (
  python eval.py --root "%ROOT%" --checkpoint runs/%PREFIX%_%%M/best.pt --output results/%PREFIX%_%%M.json
  if errorlevel 1 exit /b 1
  python robustness_sweep.py --root "%ROOT%" --checkpoint runs/%PREFIX%_%%M/best.pt --output results/%PREFIX%_%%M_robustness.md
  if errorlevel 1 exit /b 1
)
