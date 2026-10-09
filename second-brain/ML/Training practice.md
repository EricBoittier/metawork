---
tags: [ml, training]
---
# Training practice

## Before a long run
- [ ] Overfit one batch (loss → ~0). If it can't, it's a bug, not a hyperparameter.
- [ ] Smoke run on `--qos=debug` with the exact config (see [[Kuma and SLURM]]).
- [ ] Checkpoint + resume tested (kill the job, restart, curves continue).
- [ ] Validation set fixed and never touched for tuning decisions you'll report.
- [ ] Log: config, commit, venv versions, seed (the `hpc_run.py` manifest).

## Loss going wrong
| Symptom | Check |
|---|---|
| NaN/Inf | LR too high, fp16 overflow (use bf16), divide-by-zero in distances (atoms overlapping, `r=0` self-pairs), `torch.autograd.set_detect_anomaly(True)` to locate |
| Loss flat from step 0 | LR too low / grads not flowing (`p.grad is None`), frozen params, wrong target units |
| Train ↓, val ↑ | overfitting → more data, weight decay, smaller model, early stopping |
| Spiky | LR too high, no warmup, bad samples (outlier energies) → gradient clipping `clip_grad_norm_(params, 1.0)` |
| Energy fine, forces bad | force loss weight too low; check force sign and units |

## Atomistic specifics
- Normalise per-atom energies (subtract composition baseline / per-element reference) — huge effect on convergence.
- Units: eV, Å, eV/Å consistently; dataset conversions (Hartree, Bohr, kcal/mol) are a classic bug.
- Energy–force loss weighting; forces usually dominate the useful signal.
- Test energy conservation in short NVE MD — good RMSE ≠ stable MD.
- Report MAE/RMSE per element / per subset, not just global.

## Optimiser defaults
- AdamW, LR 1e-3–1e-4 for MLIPs; warmup (few hundred–thousand steps) then cosine or reduce-on-plateau.
- EMA of weights often gives better validation for MLIPs.
- Gradient clipping as a safety net.

## Hyperparameter search
- Change one thing at a time; keep a table (run id → change → result).
- Log-uniform for LR/weight decay; random search beats grid.
- Small model / data subset first; check trends transfer before burning GPU-days.

## Checkpoints
Save `{model, optimizer, scheduler, scaler, epoch, step, rng states, config}`; keep last + best. For FSDP see [[FSDP]].

Related: [[PyTorch]], [[DDP]], [[Benchmarking and profiling]], [[Atomistic ML stack]]
