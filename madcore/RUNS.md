# RUNS — madcore / LOREM by periodicity, with and without long-range

Code: `~/metawork/metatrain/lorem-sr-switch`, local-only branch `local/lorem-long-range-switch` @ `060f69c3` (PR #1265 head `42e25756` + `long_range` switch; never pushed).
Data: `/work/cosmo/boittier/kuma/madcore-lorem/data/{mixed,periodic,nonperiodic}` from `mts-2pow18-conservative` via `filter_pbc_zip.py` (slabs dropped: 45, all in train).
Model: paper defaults (cutoff 5 Å, 128 features, max_degree 6, max_degree_lr 2, no message passing). Training: `options-lorem-mts.yaml` (10 epochs, lr 1e-3, ≤1024 atoms/batch).

| date | job | name | cluster partition / QOS | wall | CHF bound | SHA | status | outcome | actual CHF |
|---|---|---|---|---|---|---|---|---|---|
| 10-01 | 4487925 | lorem-pbc-prep | kuma h100 / build (CPU) | 2 h | 0 | bb8860ce | FAILED 2 s | `metatrain._version` missing in new worktree (generated file); copied from lorem-review | 0 |
| 10-01 | 4487931 | lorem-pbc-prep | kuma h100 / build (CPU) | 2 h | 0 | bb8860ce | FAILED 67 s | pytest run from repo root (refs are cwd-relative) + checkpoint-structure test trips on the new hyper by design; fixed: run from tests/, deselect that one test, PR ckpts load via .get | 0 |
| 10-01 | 4487942 | lorem-pbc-prep | kuma h100 / build (CPU) | 2 h | 0 | 9169ff14 | FAILED 56 s | --deselect did not match the checkpoint-structure test; now -k 'not test_checkpoint_did_not_change' | 0 |
| 10-01 | 4487950 | lorem-pbc-prep | kuma h100 / build (CPU) | 2 h | 0 | 9169ff14 | FAILED 62 s | test_paper_defaults asserted long_range absent; now expects True (d77e99b4); pytest -x dropped | 0 |
| 10-01 | 4487951 | canary-mixed8k-lr | kuma h100 / debug | 1 h | 0 | 9169ff14 | CANCELLED (prep failed) | 1 epoch, throughput | |
| 10-01 | 4487952 | canary-mixed8k-sr | kuma h100 / debug | 1 h | 0 | 9169ff14 | CANCELLED (prep failed) | 1 epoch, throughput | |
| 10-01 | 4487953 | lorem-pbc-dispatch | kuma mig12gb / debug | 15 min | 0 | — | CANCELLED (prep failed) | submits the 6 runs if bound ≤ CHF 100 | |
| 10-02 | 4490073 | lorem-pbc-prep | kuma h100 / build (CPU) | 2 h | 0 | d77e99b4 | FAILED 3 min | 4 tests: 2 test bugs (scripted-model helper, lr=False yaml) fixed in 060f69c3; test_regression_init + _energies_forces_train fail identically on PR head 42e25756 (job 4490113), now skipped | 0 |
| 10-02 | 4490074 (CANCELLED, prep failed) | canary-mixed8k-lr | kuma h100 / debug | 1 h | 0 | d77e99b4 | CANCELLED | 1 epoch, throughput | |
| 10-02 | 4490075 (CANCELLED, prep failed) | canary-mixed8k-sr | kuma h100 / debug | 1 h | 0 | d77e99b4 | CANCELLED | 1 epoch, throughput | |
| 10-02 | 4490076 (CANCELLED, prep failed) | lorem-pbc-dispatch | kuma mig12gb / debug | 15 min | 0 | — | CANCELLED | submits the 6 runs if bound ≤ CHF 100 | |
| 10-02 | 4490113 | lorem-regcheck | kuma h100 / build (CPU) | 30 min | 0 | 42e25756 vs d77e99b4 | COMPLETED | both regression tests fail identically on the PR head: pre-existing, not the switch | 0 |
| 10-02 | 4490116 | lorem-pbc-prep | kuma h100 / build (CPU) | 2 h | 0 | 060f69c3 | submitted | tests + subsets | |
| 10-02 | 4490117 | canary-mixed8k-lr | kuma h100 / debug | 1 h | ≤0.52 | 060f69c3 | pending afterok prep | 1 epoch, throughput | |
| 10-02 | 4490118 | canary-mixed8k-sr | kuma h100 / debug | 1 h | ≤0.52 | 060f69c3 | pending afterok prep | 1 epoch, throughput | |
| 10-02 | 4490119 | lorem-pbc-dispatch | kuma mig12gb / debug | 15 min | ≤0.02 | — | pending afterok both canaries | submits the 6 runs if bound ≤ CHF 100 | |
| — | — | lorem-{mixed,periodic,nonperiodic}-{lr,sr} | kuma h100 / normal | from canary | ≤ 12.42 each at 24 h | 060f69c3 | planned after canary | | |

## dispatch 2026-10-02 08:28

canary s/structure: lr 0.0146, sr 0.0057 (n=13472)  
projected (x1.3 + 0.5 h), 12 h segments, bound CHF 49.92, ceiling CHF 100.00

- lorem-mixed-lr: 14.2 h -> 2 segment(s)
- lorem-mixed-sr: 5.9 h -> 1 segment(s)
- lorem-periodic-lr: 13.4 h -> 2 segment(s)
- lorem-periodic-sr: 5.6 h -> 1 segment(s)
- lorem-nonperiodic-lr: 1.3 h -> 1 segment(s)
- lorem-nonperiodic-sr: 0.8 h -> 1 segment(s)
  - submitted lorem-mixed-lr segment 1/2: job 4490148
  - submitted lorem-mixed-lr segment 2/2: job 4490149
  - submitted lorem-mixed-sr segment 1/1: job 4490150
  - submitted lorem-periodic-lr segment 1/2: job 4490151
  - submitted lorem-periodic-lr segment 2/2: job 4490152
  - submitted lorem-periodic-sr segment 1/1: job 4490153
  - submitted lorem-nonperiodic-lr segment 1/1: job 4490154
  - submitted lorem-nonperiodic-sr segment 1/1: job 4490155

## canary timing (10-02, kuma H100, mixed-8192: 8192 train + 2584 val, 1 epoch)

| | long-range | short-range only | LR / SR |
|---|---:|---:|---:|
| job total (ELAPSED, incl. ~14 s setup + test eval) | 197 s | 77 s | 2.6× |
| training epoch (train + val, log timestamps) | 94 s → 8.7 ms/structure | 22 s → 2.0 ms/structure | 4.3× |
| test eval (2696 structures) | 8.0 s | 1.8 s | 4.4× |

Epoch-based projection for 10 epochs, without the dispatcher's ×1.3 margin: mixed lr ≈ 6.3 h, sr ≈ 1.5 h; periodic lr ≈ 5.9 h, sr ≈ 1.4 h; nonperiodic minutes. The dispatcher's segments (bound CHF 49.92) are about 2× generous; billing is by elapsed time, so expect roughly CHF 8–10 total. Second mixed/periodic-lr segments should exit at once.

## outcomes (10-02, from logs; details in ~/scitas/journal/2026-10-02.md)

| job | name | status | outcome |
|---|---|---|---|
| 4490116 | lorem-pbc-prep | COMPLETED 3 min | tests + subsets ok |
| 4490117 / 4490118 | canary-mixed8k-lr / -sr | COMPLETED 208 s / 81 s | timing above |
| 4490119 | lorem-pbc-dispatch | COMPLETED | 8 jobs submitted |
| 4490154 | lorem-nonperiodic-lr | COMPLETED 8 min | test E RMSE 1363.9 meV/atom, F RMSE 4268 meV/Å (suspiciously large) |
| 4490155 | lorem-nonperiodic-sr | COMPLETED 4 min | test E RMSE 1497.3 meV/atom, F RMSE 4422 meV/Å |
| 4490148 | lorem-mixed-lr | FAILED 39.8 min | NaN loss in epoch 0 → `_get_digits` ValueError at epoch end |
| 4490150 | lorem-mixed-sr | FAILED 10.5 min | same |
| 4490151 | lorem-periodic-lr | FAILED 39.2 min | same |
| 4490153 | lorem-periodic-sr | FAILED 10.1 min | same |
| 4490149 / 4490152 | segment 2 (mixed-lr, periodic-lr) | FAILED 2–3 s | guard: previous segment failed |
| 4490253 | lorem-nanscan | CANCELLED 42 s | first 256 structures finite |
| 4490258 / 4490274 | lorem-nandebug | FAILED ~7.5 min | grad norm NaN at epoch 0 step 301, E/F finite |
| 4490294 | lorem-bisect | FAILED 6 s | nan_batch.pt float32 neighbours vs float64 system |
| 4490295 | lorem-nandebug + bisect | COMPLETED 7.5 min | **system 215511** (9 atoms, pbc, min_dist 1.71 Å): NaN grads in sr.* with saved and fresh weights → structure-specific |

normal-QOS cost of the 6 runs ≈ 1.86 GPU-h ≈ CHF 0.97 (1 GPU each assumed). Mixed/periodic runs need resubmission after 215511 (and any similar structures) is handled.

## NaN root cause + fix (10-02, details in ~/scitas/journal/2026-10-02.md)

| job | name | QOS | status | outcome |
|---|---|---|---|---|
| 4490705 | lorem-nanfix | h100 / debug | completed epoch | with fix: saved NaN batch clean; 1-epoch SR repro clean past step 301 (to 1500) |
| 4490725 | lorem-fixsuite | h100 / build | COMPLETED | PR tip 551b296d ± fix: same 3 pre-existing failures, +1 pass (new test) |

Cause: `bernstein_basis` NaN second derivative at r == cutoff exactly (GPU rounding, high-symmetry cells). Fix `0a2d45d4` pushed to origin/experimental/lorem-clean (PR #1265); local branch has it as `47c25cc3`. **Mixed/periodic × lr/sr runs need resubmitting from 47c25cc3** (not done yet).

## resubmission 10-02 ~11:40 (code 47c25cc3, same 12 h segmenting as dispatch, bound CHF 37.4)

  - lorem-mixed-lr: jobs 4490775 → 4490776 (afterany)
  - lorem-mixed-sr: job 4490777
  - lorem-periodic-lr: jobs 4490778 → 4490779 (afterany)
  - lorem-periodic-sr: job 4490780
  - nonperiodic not resubmitted (completed 10-02 on 060f69c3; the fix does not change values, only gradients at r == cutoff)

## convergence test 10-02 (alongside the resubmission)

  - lorem-nonperiodic-lr-e100: job 4490805, 100 epochs (cosine over 100), code 47c25cc3, wall 3 h (~70 min expected, ~CHF 0.6). Question: where does the error flatten out, and do train/val diverge (= data-limited)? Compare against the 10-epoch run 4490154 (test E 1363.9 meV/atom, F 4268 meV/Å).

## timings 10-02 (kuma H100, code 47c25cc3/060f69c3, batch_size 64, ≤1024 atoms/batch; train+val per epoch)

| run | s/epoch | ms/structure | µs/atom | structures/s | atoms/s |
|---|---:|---:|---:|---:|---:|
| mixed-lr (epoch 0 of 4490148) | ~1985 | 7.65 | 1190 | 131 | 840 |
| periodic-lr (epoch 0 of 4490151) | ~1969 | 8.0 | 1210 | 125 | 820 |
| mixed-sr 4490777 | 221 | 0.85 | 133 | 1174 | 7549 |
| periodic-sr 4490780 | 212 | 0.86 | 131 | 1158 | 7661 |
| nonperiodic-lr 4490154 / e100 4490805 | 37–38 | 2.6 | 780 | ~390 | ~1280 |
| nonperiodic-sr 4490155 | 12 | 0.82 | 250 | 1215 | 3999 |

Setup before training: ~3.8 min on mixed/periodic (composition + scaler weights), 14 s on nonperiodic. LR/SR: ~9× on mixed/periodic, ~3× on nonperiodic (canary mixed-8k: 4.3×).
Live sample (srun --overlap, 8 s): GPU util 25–35 % for both mixed-lr and mixed-sr; main `mtt` process pinned at ~100 % of one core; mixed-lr 53.6 GB GPU memory, mixed-sr 20.8 GB. → host-bound.

## batched long-range + cache, 10-02 ~12:40

- Code `local/lorem-perf` in `~/metawork/metatrain/lorem-perf` (push-blocked): 47c25cc3 + batched Ewald 91fdc0b2 + one-by-one route for huge k-grids 2c5099bb + TorchScript fix a5f412dc. Suite: same 3 pre-existing failures, 108 passed.
- Cache: `/work/.../madcore-lorem/cache/<subset>/{composition_model,scaler}.ckpt` (weights identical across runs; only Z=0 slot of type_to_index is uninitialised). sbatch passes them as atomic_baseline / fixed_scaling_weights. Setup 6.5 → 4 min (rest is dataset scan).
- bench_lr.py (debug, 40 mixed batches): LR step 422 → 54 ms, 44k → 4.4k kernel launches; SR 24 ms either way.
- 4490944 perfcheck FAILED: OOM (9.4 GiB alloc) on huge-cell batch → fixed in 2c5099bb.
- 4491020 perfcheck (2c5099bb, mixed-lr): epoch 0 in 370 s vs ~1985 s old (5.4×). train E 1968 vs 1962, val E 1165 vs 1238 meV/atom (fp reordering). Cancelled after epoch 0 as planned.
- Eric approved restarting LR runs on the new code: cancelled 4490775/6, 4490778/9 (old dirs moved to runs/lorem-*-lr-oldcode-47c25cc3). Resubmitted mixed-lr 4491073, periodic-lr 4491074 (CODE=lorem-perf, SHA=a5f412dc, 4 h, one segment).

## training-speed work 10-02 afternoon (details: ~/scitas/journal/2026-10-02.md)

nonperiodic-lr-e100 4490805 COMPLETED 64 min: test E 522.7 meV/atom, F 2122 meV/A (10 epochs: 1363.9 / 4268). train/val E 530/554 (not data-limited), F 1398/1938 (forces data-limited).
mixed-sr 4490777 COMPLETED 54 min, periodic-sr 4490780 COMPLETED 52 min (old code 47c25cc3).

Variants (frozen worktrees ~/metawork/metatrain/bench-*): v0 a5f412dc batched Ewald; v1 097b80e8 + #1275 + timing; v2 f2c65f52 + no per-step syncs + on-device metric sums; v3 172a9287 + index_select gathers. Further: 061675ec parallel get_atomic_types/get_stats (startup).
- 4491092 matrix (mixed-8192, 4 epochs, s/epoch): LR w8 v0 17 → v1 14 → v2 14; SR w8 11 → 9 → 9; LR w0 29 → 25 → 25; SR w0 26 → 22 → 23. Timing (v2, w8): loader 0.1 ms; unpack 17 + h2d 26 ms/step = 35 % (SR) / 23 % (LR); forward 48/92, backward 23/42.
- 4491125 bench_lr model-only: LR 53.3 → 42.0 ms/step with index_select (index_put was 45 % of GPU time); SR 23.6 → 23.2.
- 4491147 batch 256/4096 vs 64/1024 (v3, w8): s/epoch SR 10 → 9, LR 13 → 10; but after 4 epochs val E SR 3619 → 4856, LR 3732 → 4517 meV/atom (4x fewer steps, same lr) → NOT confirmed; item 3 (MIG) stays on hold.
- 4491216 bench_unpack (real pinned batches, 407 atoms): unpack+batch_to 20.5 ms; assume_unique 19.1; move flat + build on device 14.3 ms (identical content).
- 4491242 big batch + lr scaling (v3, 4 epochs mixed-8192, val E / F; s/epoch): SR default 3695/7453 (10 s); big lr 2e-3 4437/8501 (8 s); big lr 4e-3 4521/8316 (8 s). LR default 3696/7319 (13 s); big 2e-3 4313/8510 (11 s); big 4e-3 6148/9322 (10 s, unstable). → big batches never match at equal epochs; 10–23 % speed does not pay for it. Item 1 not confirmed; item 3 (MIG) stays on hold.
- v4 a73363f3 (unpack straight to device), 4491257: SR 10 → 8 s/epoch, LR 13 → 12; in-loop unpack+h2d 44 → 30 ms/step. Suites: LOREM 3 pre-existing failures/108 passed; tests/utils (5 files need requests/spex, ignored) 4 env failures vs 7 on the tree without #1275 (PET scatter ones fixed by the stack).
- PR-bound branch perf/lorem-batched-ewald (worktree lorem-clean-perf, push-blocked): 4 commits on origin/experimental/lorem-clean 0a2d45d4 (batched Ewald, huge-cell route, index_select, no per-step sync); no long_range switch. Suite vs PR head: same 3 failures, 107 vs 106 passed.
- Pushed by Eric (`!` in session, auto mode blocked Claude's push): origin/experimental/lorem-clean 0a2d45d4 → 324e5c56 (PR #1265 head confirmed).

## results: all six 10-epoch runs (test set)

| subset | long-range E / F | short-range E / F | (meV/atom, meV/A) |
|---|---|---|---|
| mixed | 515.9 / 1541 (4491073, a5f412dc, 76 min) | 509.4 / 1506 (4490777, 47c25cc3) | |
| periodic | 509.5 / 1510 (4491074, a5f412dc, 71 min) | 493.4 / 1497 (4490780, 47c25cc3) | |
| nonperiodic | 1363.9 / 4268 (4490154) | 1497.3 / 4422 (4490155) | |

All undertrained at 10 epochs (nonperiodic-lr 100 epochs: 522.7 / 2122). No LR benefit visible on mixed/periodic at this budget.

## 100-epoch runs, submitted 10-02 ~13:50 (Eric: "set up the 100-epoch runs for all six")

Code: frozen detached worktree ~/metawork/metatrain/runs-e100 @ a73363f3 (local/lorem-pipeline v4: batched Ewald + #1275 + no per-step sync + index_select + unpack to device; includes the long_range switch). Cached composition/scaler. 100 epochs, cosine lr, otherwise options-lorem-mts.yaml.
Bound: mixed/periodic 2 × 12 h segments, nonperiodic 1 × 4 h → 104 GPU-h ≈ CHF 54 worst case; expected ~26 GPU-h ≈ CHF 14.

| run | segment jobs |
|---|---|
| lorem-mixed-lr-100ep | 4491394 → 4491395 |
| lorem-mixed-sr-100ep | 4491396 → 4491397 |
| lorem-periodic-lr-100ep | 4491398 → 4491399 |
| lorem-periodic-sr-100ep | 4491400 → 4491401 |
| lorem-nonperiodic-lr-100ep | 4491402 |
| lorem-nonperiodic-sr-100ep | 4491403 |
- Startup on full mixed with a73363f3 + cache: 44 s (was 3 min 50 s; types 98 → 15 s, stats 123 → 16 s).
- Epoch time on full data, a73363f3 (min/epoch): mixed LR 5.0, periodic LR 4.5, mixed SR 3.1, periodic SR 2.9, nonperiodic LR 0.18, SR 0.15. (Original code: mixed LR ~33; batched Ewald only: 6.2.) Expected: mixed LR ~8.3 h, periodic LR ~7.5 h, SR ~5 h, nonperiodic < 30 min — all inside the first 12 h segment.
- nonperiodic 100 epochs done (a73363f3): LR 4491402 test E 518.4 meV/atom, F 2114 meV/A; SR 4491403 test E 615.7, F 2355 → LR −16 % E, −10 % F (val, 126 structures: −29 % / +2 %). Comparison script: madcore/compare_runs.py.
