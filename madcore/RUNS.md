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

# RUNS — PET training: metatrain vs mlip.cpp (MAD-CORE 2^18, 2 GPUs, batch 64/GPU)

Plan: `~/.claude/plans/iridescent-jumping-llama.md`. metatrain = fork `origin/main` `c3bc0cf2` (worktree `metatrain/main-baseline`, no dipole-like force head), `.venv-blackwell`; options `options-pet-bs64.yaml` (PET defaults, 10 epochs, seed 42). Data staged to node `$TMPDIR`; caches in `/work/cosmo/boittier/cache`.

| date | job | name | cluster partition / QOS | wall | CHF bound | SHA | status | outcome | actual CHF |
|---|---|---|---|---|---|---|---|---|---|
| 10-14 | 145086 | pet-bs64-2gpu (metatrain) | lyra rtx6000 / normal, 2 GPU | 3 h | 2.32 | c3bc0cf2 | FAILED 3 s | `metatrain._version` missing in new worktree; copied from the submodule checkout | 0 |
| 10-14 | 145087 | pet-bs64-2gpu (metatrain) | lyra rtx6000 / normal, 2 GPU (lr011) | 3 h | 2.32 | c3bc0cf2 | COMPLETED 26:33 | ~105 s/epoch; test RMSE E 288.08 meV/atom, F 1431.0 meV/Å (val 322.32 / 1626.1); peak GPU 94.6 GiB | ~0.34 |
| 10-14 | 145093 | mlipcpp-dev (build, init, grad parity, smoke) | lyra rtx6000 / debug, 1 GPU | 1 h | 0.39 | mlip.cpp 35fb4b0+dirty (train/pet) | FAILED 2:42 | build: `std::reverse` without `<algorithm>` in graph_interpreter.cpp (pre-existing; gcc 14); fixed, all sources gcc-14 syntax-checked | ~0.02 |
| 10-14 | 145096 | mlipcpp-dev (build, init, grad parity, smoke) | lyra rtx6000 / debug, 1 GPU | 1 h | 0.39 | mlip.cpp 35fb4b0+dirty (train/pet) | FAILED 2:29 | init+export OK (3.04M params, 349-node graph); check aborted: interpreter SUM_ROWS ignored `dims` (nc-force neighbour sum); fixed | ~0.02 |
| 10-14 | 145097 | mlipcpp-dev (grad parity, smoke; init from 145096) | lyra rtx6000 / debug, 1 GPU | 1 h | 0.39 | mlip.cpp 35fb4b0+dirty (train/pet) | COMPLETED (gate FAIL) | CPU: loss = PyTorch to 2e-8, but grads of params upstream of the transformer tokens wrong (rel ~1); GPU: CUDA GET_ROWS_BACK grid.y > 65535 (and O(rows^2)) -> scatter-add patch; smoke 25 steps, 441 ms/step | ~0.03 |
| 10-14 | 145098 | mlipcpp-dev #4 (ggml GET_ROWS_BACK patch, node-grad diagnostic) | lyra rtx6000 / debug, 1 GPU | 1 h | 0.39 | mlip.cpp 35fb4b0+dirty (train/pet) | CANCELLED in smoke | patch works (smoke 100+ steps, 365-413 ms/step, 1 GPU); GPU check segfault (null node tensor in diagnostic hook), CPU check OOM (node dump of 64 structures); both fixed; NL cache now content-keyed | ~0.1 |
| 10-14 | 145099 | mlipcpp-dev #5 (gates n=64, node-grad diagnostic n=8 CPU) | lyra rtx6000 / debug, 1 GPU | 1 h | 0.39 | mlip.cpp 35fb4b0+dirty (train/pet) | COMPLETED | **GPU gate PASS** (127 tensors, worst rel 8.8e-4; loss rel 3.9e-4); CPU gate FAIL (87/127 tensors, CPU-backend only); node diagnostic segfault (unallocated grads, guarded); smoke segfault right after --max-steps 300 (end-of-epoch path) | ~0.08 |
| 10-14 | 145101 | mlipcpp-bs64-2gpu (mlip.cpp trainer, init = metatrain's initial PET) | lyra rtx6000 / normal, 2 GPU | 4 h | 3.12 | mlip.cpp 35fb4b0+dirty (train/pet) | CANCELLED ~12 min (epoch 0, step ~1300) | would crash at epoch-end checkpoint: write_gguf used gguf_set_tensor_data on a loaded ctx (writer read through stale buffer; backtrace from dev 145102); fixed. 241 struct/s on 2 GPUs | ~0.16 |
| 10-14 | 145102 | mlipcpp-dev #6 (node-grad diagnostic, crash backtrace, smoke 30 steps; build-dev RelWithDebInfo) | lyra rtx6000 / debug, 1 GPU | 1 h | 0.39 | mlip.cpp 35fb4b0+dirty (train/pet) | COMPLETED | GPU gate PASS again; backtraces: smoke crash in write_gguf (fixed), node diagnostic crash in its own dump of unallocated view grads (fixed) | ~0.06 |
| 10-14 | 145104 | mlipcpp-bs64-2gpu (rebuild + 2-GPU canary through checkpoint/eval-only reload, then 10 epochs) | lyra rtx6000 / normal, 2 GPU | 4 h | 3.12 | mlip.cpp 35fb4b0+dirty (train/pet) | FAILED 14:52 (epoch 0 step ~1700) | canary OK (checkpoint write + exact eval-only reload); then OOM: `cannot allocate the graph (468 atoms x 1523 neighbours)` -- a batch padded to one 8-atom compressed crystal with 1523 neighbours (manual attention materialises n*8*(m+1)^2). Fix: neighbour-sorted micro-batches with full-batch loss normalisation (exact gradient) | ~0.19 |
| 10-14 | 145107 | mlipcpp-diag (per-node CPU gradient diagnostic, n=8) | lyra rtx6000 / debug, 1 GPU | 40 min | 0.26 | mlip.cpp 35fb4b0+dirty (train/pet) | FAILED | use-after-free in the dump (tensor shapes read after ggml_free(ctx)); fixed | ~0.03 |
| 10-14 | 145108 | mlipcpp-diag (retry; standing OK until it reports) | lyra rtx6000 / debug, 1 GPU | 40 min | 0.26 | mlip.cpp 35fb4b0+dirty (train/pet) | COMPLETED | ran, but node names misaligned (re-export without dynamic shapes); compare now exports via the shared example_export_args | ~0.03 |
| 10-14 | 145110 | mlipcpp-bs64-2gpu (micro-batching; job gated on grad parity + micro-batch equivalence; canary; 10 epochs) | lyra rtx6000 / normal, 2 GPU | 4 h | 3.12 | mlip.cpp 35fb4b0+dirty (train/pet) | FAILED ~6 min | gate PASS (8.8e-4), micro-batch == full batch (6 mbs, grads 4.3e-4, loss 1.7e-6), canary OK; then GGML_ASSERT in ggml_concat: a 1-atom micro-batch, interpreter CONCAT/TRANSPOSE took the PyTorch rank from ggml_n_dims; fixed (IR rank) + micro-batches >= 2 atoms | ~0.08 |
| 10-14 | 145111 | mlipcpp-diag (aligned node names) | lyra rtx6000 / debug, 1 GPU | 40 min | 0.26 | mlip.cpp 35fb4b0+dirty (train/pet) | COMPLETED | names aligned; first BADs are view gradients whose memory was reused (dump artefact); dump needs view roots kept | ~0.01 |
| 10-14 | 145112 | mlipcpp-bs64-2gpu (rank fix; gate + 30-step canary; 10 epochs) | lyra rtx6000 / normal, 2 GPU | 4 h | 3.12 | mlip.cpp 35fb4b0+dirty (train/pet) | COMPLETED 1:28:54 | gate PASS, canary OK; unoptimized build (host Adam, pre-REPEAT_BACK patch): 10 epochs, ~528 s/epoch (486 struct/s, 2 GPU) — 5x slower than metatrain, superseded by 145128; **test RMSE E 291.74 meV/atom, F 1416.69 meV/Å** (val 316.21 / 1597.59), matches 145128 to <0.1 %; peak GPU 7.0 GiB | ~1.16 |
| 10-14 | 145114 | mlipcpp-diag (node grads, view roots kept; pre CPU in-place fix) | lyra rtx6000 / debug, 1 GPU | 40 min | 0.26 | mlip.cpp 35fb4b0+dirty (train/pet) | COMPLETED | 215 nodes compared; CPU node grads BAD from add_25 back through attention/RMS_NORM to embeddings — root cause found by op test in 145118 (CPU RMS_NORM_BACK in place) and fixed (145124 CPU gate PASS) | ~0.01 |
| 10-14 | 145115 | mlipcpp-diag (op test rms_norm*w backward CPU vs CUDA; node grads with view roots) | lyra rtx6000 / debug, 1 GPU | 40 min | 0.26 | snapshot 20261014-162142 | FAILED 3 s | cmake: build dir configured for another source tree; now one build dir per snapshot | 0 |
| 10-14 | 145118 | mlipcpp-diag (op test + node grads; includes GPU-optimizer/accumulator and WHERE changes) | lyra rtx6000 / debug, 1 GPU | 40 min | 0.26 | snapshot 20261014-162940 | COMPLETED | **op test: CPU RMS_NORM_BACK input-gradient wrong (rel 0.72), CUDA exact** -- ggml-alloc runs it in place, CPU kernel copied x over dz before reading dz; patched (also SOFT_MAX_BACK) | ~0.02 |
| 10-14 | 145119 | mlipcpp-bench #1 (GPU optimizer + device grad accumulation, WHERE w/o zeros, profile) | lyra rtx6000 / debug, 2 GPU | 45 min | 0.59 | snapshot 20261014-163115 | COMPLETED | gate PASS; GPU optimizer trajectory == host-optimizer canary (5.323/2.582/1.942/2.337); sync 15 -> 3.3 ms; 531 struct/s (2 GPU, 300 steps); **profile: REPEAT_BACK 75.7% of a step** (attention ~1%) | ~0.3 |
| 10-14 | 145124 | mlipcpp-diag (op test + CPU gate after ggml-cpu in-place patch) | lyra rtx6000 / debug, 1 GPU | 40 min | 0.26 | snapshot 20261014-163305 | COMPLETED | op test CPU+CUDA ok (dL/dx 4.7e-8); **CPU gate PASS** (127 tensors, worst rel 1.0e-4; micro-batch == full batch 9e-5) | ~0.03 |
| 10-14 | 145125 | mlipcpp-bench #2 (CUDA REPEAT_BACK split-reduction patch) | lyra rtx6000 / debug, 2 GPU | 45 min | 0.59 | snapshot 20261014-163806 | COMPLETED | gate PASS; REPEAT_BACK 321 -> 21 ms; step compute 362 -> 62 ms (6 mbs, 1 GPU); 2163 struct/s (2 GPU, 300 steps) | ~0.2 |
| 10-14 | 145126 | mlipcpp-bench #3 (REPEAT_BACK patch + 2-D LINEAR; growth sweep 2/3/4) | lyra rtx6000 / debug, 2 GPU | 45 min | 0.59 | snapshot 20261014-163858 | COMPLETED | gate PASS (8.9e-4); step compute 50 ms; 300-step 2-GPU throughput: growth 2: 3010, growth 3: 3364, growth 4: 3450 struct/s (metatrain 2-GPU: ~2440) | ~0.25 |
| 10-14 | 145128 | mlipcpp-bs64-2gpu-fast (optimized: REPEAT_BACK patch, 2-D LINEAR, GPU Adam, growth 4; gate + canary + 10 epochs) | lyra rtx6000 / debug, 2 GPU | 45 min | 0.59 | snapshot 20261014-164702 | COMPLETED 14:16 | gate PASS; 10 epochs in 707.6 s (70.8 s/epoch, 3629 struct/s) vs metatrain 1054 s (105.4 s/epoch): 1.49x; **test RMSE E 291.83 meV/atom, F 1417.37 meV/Å** (metatrain 288.08 / 1431.0); peak GPU 13.0 GB vs 94.6 GB | ~0.19 |
| 10-14 | 145153 | mlipcpp-bench (tiles + shared-buffer CUDA graphs, Muon, ggml MUL_MAT-backward-as-OUT_PROD patch) | lyra rtx6000 / debug, 2 GPU | 45 min | 0.58 | snapshot 20261014-184629 (dc31736+dirty) | COMPLETED | gate PASS (8.9e-4); tiles == full batch (loss 2.6e-6 rel, grads 2e-4, metrics exact, stable replays); tiled 30-step trajectory == dynamic (2.338); **OUT_PROD patch regression**: g4 2244 struct/s vs 3450 (145126) -> reverted; tiles slower than dynamic in the same build (1700-1850 struct/s: padding costs more than launch overhead saves; buffer growth flushes); Muon (48 tensors) val after 600 steps E 2194 / F 10734 vs Adam 2074 / 10810, +6 ms optimizer step | ~0.1 |
| 10-14 | 145154 | mlipcpp-bench (OUT_PROD reverted, REPEAT_BACK carry kernel) | lyra rtx6000 / debug, 2 GPU | 45 min | 0.58 | snapshot 20261014-185207 | CANCELLED 2:17 | REPEAT_BACK launch: gridDim.y > 65535 when E is tiny (2^18/E splits); clamped | ~0.01 |
| 10-14 | 145155 | mlipcpp-bench (clamp fix, batched Muon NS; g4 / muon / muon x3 / tiles 1.1) | lyra rtx6000 / debug, 2 GPU | 45 min | 0.58 | snapshot 20261014-185551 (= 4a6d06f) | COMPLETED | gate PASS (8.9e-4); step 50.7 ms (6 mbs) = 145126, regression gone; 600 steps: g4 Adam **3648 struct/s** (val E 2060 / F 10792), Muon x1 3417 (2194 / 10734), **Muon x3 3486 struct/s, val E 1330 / F 5907**, tiles 1.1 3300 (157 shapes, 9 flushes); batched NS: optimizer +2.8 ms (was +6.2) | ~0.07 |
| 10-14 | 145156 | mlipcpp-bs64-2gpu-muon3 (Muon, --muon-lr-scale 3; gate + canary + 10 epochs) | lyra rtx6000 / debug, 2 GPU | 45 min | 0.58 | snapshot 20261014-190034 (= 4a6d06f) | COMPLETED 14:34 | gate PASS; 10 epochs in 726 s (3560 struct/s; Adam 145128: 710 s, metatrain 1054 s); **test RMSE E 204.87 meV/atom, F 902.80 meV/Å** (Adam 291.83 / 1417.37, metatrain 288.08 / 1431.0); val F beats Adam's final val after epoch 3, val E after epoch 4; peak GPU 13.0 GB | ~0.19 |
| 10-14 | 145157 | mlipcpp-bench (controls: Adam lr 3e-4, Muon x10; 600 steps) | lyra rtx6000 / debug, 2 GPU | 45 min | 0.58 | snapshot 20261014-190034 | COMPLETED | 600-step val: Adam lr 3e-4 E 1152 / F 5094 (3800 struct/s) -- better than Muon x3 (1330 / 5907), i.e. early gain is mostly step size; Muon x10 E 926 / F 3759 (3500 struct/s); full-run controls 145161/145162 | ~0.07 |
| 10-14 | 145161 | mlipcpp-bs64-2gpu-adam3e-4 (Adam, lr 3e-4: step-size control for Muon) | lyra rtx6000 / debug, 2 GPU | 45 min | 0.58 | snapshot 20261014-190034 | COMPLETED 11:34 | gate PASS; 10 epochs in 665 s (3890 struct/s); **test RMSE E 238.79 meV/atom, F 1057.72 meV/Å** (val 259.72 / 1291.37) | ~0.15 |
| 10-14 | 145162 | mlipcpp-bs64-2gpu-muon10 (Muon, --muon-lr-scale 10) | lyra rtx6000 / debug, 2 GPU | 45 min | 0.58 | snapshot 20261014-190034 | COMPLETED 12:33 | gate PASS; 10 epochs (3570 struct/s); **test RMSE E 182.49 meV/atom, F 827.54 meV/Å** (val 186.41 / 994.25): best so far; vs Adam lr 3e-4 -24 % E, -22 % F; vs metatrain default -37 % / -42 % | ~0.16 |
| 10-14 | 145168 | mlipcpp-2pow20-muon10 (first try) | lyra rtx6000 / normal, 2 GPU | 3 h | 2.32 | snapshot 20261014-190034 | FAILED 0:07 | CMake race: configured the same snapshot build dir as 145167 at the same time; concurrent jobs now get their own BUILD; resubmitted as 145169 | 0 |
| 10-14 | 145175 | mlipcpp-bench (step packing by fitted cost model + overlapped gradient sync / one-copy weight broadcast) | lyra rtx6000 / debug, 2 GPU | 45 min | 0.58 | snapshot 20261014-201632 (= ac15115) | COMPLETED 4:30 | gates PASS (8.9e-4; step packing for 2 workers == full batch, 8.3e-4); sync 3.3 -> 1.0 ms; 600 steps: per-worker packing 3828 struct/s, **step packing 5801 struct/s** (+52 %; 1.36 vs 3.02 micro-batches/worker, compute 16.7 vs 26.2 ms; val E 2050 / F 10798 vs 2142 / 10728), Muon x10 step 5008 (sync 3.9 ms), Muon x10 worker 3680; fitted cost ~4.2 ms + 1.7e-6 n(m+1)^2 + 5.3e-4 n(m+1) | ~0.05 |
| 10-14 | 145166 | pet-bs64-2gpu-2pow20 (metatrain reference, memmap-2pow20, 10 epochs) | lyra rtx6000 / normal, 2 GPU | 3 h | 2.32 | metatrain c3bc0cf2 | RUNNING (as of 20:31) | ~397 s/epoch (epoch 0 end 20:16:41 -> epoch 1 20:23:18) vs mlip.cpp Adam 145167 ~260 s: 1.53x; shares lr011 with 3 other jobs | - |
| 10-14 | 145167 | mlipcpp-2pow20-adam (Adam 1e-4, memmap-2pow20, 10 epochs) | lyra rtx6000 / normal, 2 GPU | 3 h | 2.32 | snapshot 20261014-190034 (= 4a6d06f; pre step-packing) | CANCELLED 20:32 (superseded by step-packing rerun) | epoch 5: val E 191.75 / F 866.42; 3970 struct/s (~260 s/epoch) | - |
| 10-14 | 145169 | mlipcpp-2pow20-muon10 (Muon x10, memmap-2pow20, 10 epochs; resubmit of 145168) | lyra rtx6000 / normal, 2 GPU | 3 h | 2.32 | snapshot 20261014-190034 (= 4a6d06f; pre step-packing) | CANCELLED 20:32 (superseded by step-packing rerun) | epoch 4: val E 161.22 / F 865.95 (Adam 145167 at epoch 4: 191.77 / 897.83); 3636 struct/s | - |
| 10-14 | 145170 | mlipcpp-2pow18-e50-adam (Adam 1e-4, 50 epochs) | lyra rtx6000 / normal, 2 GPU | 2 h | ~1.5 | snapshot 20261014-190034 (= 4a6d06f; pre step-packing) | CANCELLED 20:32 (superseded by step-packing rerun) | epoch 21: val E 253.28 / F 1263.38; 3865 struct/s | - |
| 10-14 | 145171 | mlipcpp-2pow18-e50-adam3e-4 (Adam 3e-4, 50 epochs) | lyra rtx6000 / normal, 2 GPU | 2 h | ~1.5 | snapshot 20261014-190034 (= 4a6d06f; pre step-packing) | CANCELLED 20:32 (superseded by step-packing rerun) | epoch 21: val E 290.12 / F 1222.59; 3872 struct/s | - |
| 10-14 | 145172 | mlipcpp-2pow18-e50-muon10 (Muon x10, 50 epochs) | lyra rtx6000 / normal, 2 GPU | 2 h | ~1.5 | snapshot 20261014-190034 (= 4a6d06f; pre step-packing) | CANCELLED 20:32 (superseded by step-packing rerun) | epoch 19: val E 224.95 / F 1027.04; 3544 struct/s | - |
| 10-14 | 145173 | mlipcpp-2pow20-e30-adam3e-4 (Adam 3e-4, memmap-2pow20, 30 epochs) | lyra rtx6000 / normal, 2 GPU | 4.5 h | ~3.5 | snapshot 20261014-190034 (= 4a6d06f; pre step-packing) | CANCELLED 20:32 (superseded by step-packing rerun) | epoch 4: val E 198.13 / F 974.79; 3970 struct/s | - |
| 10-14 | 145174 | mlipcpp-2pow20-e30-muon10 (Muon x10, memmap-2pow20, 30 epochs) | lyra rtx6000 / normal, 2 GPU | 4.5 h | ~3.5 | snapshot 20261014-190034 (= 4a6d06f; pre step-packing) | CANCELLED 20:32 (superseded by step-packing rerun) | epoch 4: val E 209.70 / F 948.64; 3634 struct/s | - |
| 10-14 | 145177 | mlipcpp-bench (ZeRO-1 sharded optimizer: per-device Adam / Muon NS shards, async all-gather) | lyra rtx6000 / debug, 2 GPU | 45 min | 0.58 | snapshot 20261014-202534 | FAILED 2:34 | gate PASS; aborted in the node-by-node profile (CUDA illegal address): one-node graph views with CUDA graphs on can replay a stale capture keyed by a reused node address; check now disables CUDA graphs when profiling, bench profile step non-fatal | ~0.02 |
| 10-14 | 145178 | mlipcpp-bench (retry of 145177: sharded optimizer e2506dd) | lyra rtx6000 / debug, 2 GPU | 45 min | 0.58 | snapshot 20261014-202841 | COMPLETED 3:17 | gate PASS; trajectories identical to 145175 (1.93 / 1.979); **slower than ac15115**: step Adam 5091 struct/s (sync 2.8 ms) vs 5801 (1.0), step Muon x10 4894 (4.4 ms) vs 5008 (3.9); shards 64/63 (Adam), 78/49 (Muon) -> reverted (9160970) | ~0.04 |
| 10-14 | jed 40387863 | madcore-memmaps (memmaps 2^15, 17, 19, 21, 22, 23, 24 -> /work/cosmo/boittier/lyra/madcore-data, symlinked in ~/data/madcore) | jed standard / serial, 16 cores | 3 h | <=0.26 | madcore-to-diskdataset, split seed 0 | PENDING | - | - |
| 10-14 | 145179 | mlipcpp-step-2pow20-adam (Adam 1e-4, 2pow20, 10 ep) | lyra rtx6000 / normal, 2 GPU | 3 h | - | snapshot 20261014-201632 (= ac15115, as benchmarked in 145175) | RUNNING (started 20:32) | step packing rerun of 145167 | - |
| 10-14 | 145180 | mlipcpp-step-2pow20-muon10 (Muon x10, 2pow20, 10 ep) | lyra rtx6000 / normal, 2 GPU | 3 h | - | snapshot 20261014-201632 (= ac15115, as benchmarked in 145175) | RUNNING (started 20:32) | step packing rerun of 145169 | - |
| 10-14 | 145181 | mlipcpp-step-2pow18-e50-adam (Adam 1e-4, 2pow18, 50 ep) | lyra rtx6000 / normal, 2 GPU | 2 h | - | snapshot 20261014-201632 (= ac15115, as benchmarked in 145175) | RUNNING (started 20:32) | step packing rerun of 145170 | - |
| 10-14 | 145182 | mlipcpp-step-2pow18-e50-adam3e-4 (Adam 3e-4, 2pow18, 50 ep) | lyra rtx6000 / normal, 2 GPU | 2 h | - | snapshot 20261014-201632 (= ac15115, as benchmarked in 145175) | RUNNING (started 20:32) | step packing rerun of 145171 | - |
| 10-14 | 145183 | mlipcpp-step-2pow18-e50-muon10 (Muon x10, 2pow18, 50 ep) | lyra rtx6000 / normal, 2 GPU | 2 h | - | snapshot 20261014-201632 (= ac15115, as benchmarked in 145175) | RUNNING (started 20:32) | step packing rerun of 145172 | - |
| 10-14 | 145184 | mlipcpp-step-2pow20-e30-adam3e-4 (Adam 3e-4, 2pow20, 30 ep) | lyra rtx6000 / normal, 2 GPU | 4.5 h | - | snapshot 20261014-201632 (= ac15115, as benchmarked in 145175) | RUNNING (started 20:32) | step packing rerun of 145173 | - |
| 10-14 | 145185 | mlipcpp-step-2pow20-e30-muon10 (Muon x10, 2pow20, 30 ep) | lyra rtx6000 / normal, 2 GPU | 4.5 h | - | snapshot 20261014-201632 (= ac15115, as benchmarked in 145175) | RUNNING (started 20:32) | step packing rerun of 145174 | - |

**Summary (10 epochs, test RMSE E meV/atom / F meV/Å, wall):** metatrain Adam 1e-4: 288.1 / 1431, 1054 s, 94.6 GB · mlip.cpp Adam 1e-4 (145128): 291.8 / 1417, 710 s, 13.0 GB · Adam 3e-4 (145161): 238.8 / 1058, 665 s · Muon x3 (145156): 204.9 / 903, 726 s · **Muon x10 (145162): 182.5 / 828, 724 s**. Single seed each. Tiles + CUDA graphs: correct but slower (opt-in); OUT_PROD weight-grad patch: regression, dropped.
