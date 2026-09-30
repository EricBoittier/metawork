# Running the GROMACS ONIOM jobs on the clusters

Two job sets use this: the cyclic-peptide campaign (`gromacs-oniom-cyclic/campaign`, 27 runs)
and hemoglobin with four ML heme sites (`gromacs-oniom-hemoglobin`, eight replicas). Both need
the `oniom-stress-fixes` branch of `EricBoittier/gromacs_metatensor` built with metatomic.

| File | What it does |
|---|---|
| `env.sh` | paths for the jobs (`GMX`, `GMX_D`, `ONIOM_PY`, `ONIOM_MODELS`, `NT`), guessed per cluster, each overridable |
| `build-gromacs.sh [--double]` | clones and builds the branch for this cluster: single precision with non-bondeds and PME on the GPU, or double precision with MM on the CPU |
| `export-models.sh` | exports PET-MAD xs v1.5.0 and PET-OMol s v1.0.0 into `$ONIOM_MODELS` |
| `worker.sh TASKS [MAX]` | runs tasks from a task list; any number of workers can share one list |
| `submit.sh TASKS [sbatch options]` | submits a task list the way this cluster allocates GPUs |
| `single-gpu.sbatch`, `alps-8gpu.sbatch` | the two batch layouts `submit.sh` uses |

## The clusters

| Cluster | GPUs | Python | Allocation | Layout |
|---|---|---|---|---|
| kuma | L40S (sm_89), H100 (sm_90) | `.venv` | 1 GPU | job array, one run per task |
| lyra | RTX PRO 6000, B200 (sm_120, sm_100) | `.venv-blackwell` | 1 GPU | job array, one run per task |
| daint, clariden | GH200 (sm_90, aarch64), 4 per node | PyTorch uenv | 8 GPUs (2 nodes) | 8 workers per allocation, one per GPU |

On kuma and lyra `submit.sh` submits one array task per pending run, each with one GPU and
16 cores. On daint and clariden a job must reserve 8 GPUs, so it starts eight workers, one
per GPU with 64 cores (32 OpenMP threads per run), and they take runs from the shared list
until none are left. The eight hemoglobin replicas fill one allocation. The campaign's 27
runs take about four rounds: resubmit until `submit.sh` says there is nothing to do.

Each run takes a claim in `TASKS.d/NAME/`. A claim whose SLURM job has ended is taken over,
and the run scripts resume from their stage markers and checkpoints, so a run cut off by
the time limit carries on in the next allocation. After three failures a run is skipped;
its log is `TASKS.d/NAME/log`.

### Which kuma partition

`submit.sh` uses `h100` on kuma unless you pass `--partition`. `l40s` is often full. The MIG
partitions `mig12gb` and `mig24gb` are slices of an H100 with a fraction of its compute and
12 or 24 GB of memory; they suit the peptide campaign (well under 1 GB of GPU memory per run,
and a model call that is latency-bound) and are usually idle. Check how the slices are
requested there (`sinfo -o '%P %G'`) before submitting to them; if the gres is not plain
`gpu`, pass it, e.g. `--partition=mig24gb --gres=gpu:<name>:1`. The whole-protein ML test
needs a full GPU (7 GB with PET-MAD xs, far more with larger models).

## Once per cluster

```bash
cd $METAWORK                          # your metawork checkout
etc/setup-metawork.sh                 # the venvs, if not there yet (Alps: use the uenv)
# on a compute node (the CUDA architectures and the CPU SIMD are chosen for it), ~30 min:
etc/oniom/build-gromacs.sh            # -> builds/gromacs-oniom-$CLUSTER-$ARCH/bin/gmx
etc/oniom/build-gromacs.sh --double   # -> ...-double/bin/gmx_d (only the shellres runs use it)
etc/oniom/export-models.sh            # needs .venv-upet and network; or copy models/oniom/ over
```

On Alps, run the build inside the uenv that the jobs use, so that torch, CUDA and the
compilers match: `uenv start pytorch/v2.6.0:v1@clariden --view=default`. The build
downloads metatomic-torch and FFTW, so the node needs network access (or set
`GMX_BUILD_OWN_FFTW=OFF` where a system FFTW exists).

## Running

```bash
# kuma: one run per H100
gromacs-oniom-cyclic/campaign/tasks.sh > gromacs-oniom-cyclic/campaign/tasks.txt
etc/oniom/submit.sh gromacs-oniom-cyclic/campaign/tasks.txt --partition=h100
etc/oniom/submit.sh gromacs-oniom-hemoglobin/tasks.txt --partition=h100

# lyra
etc/oniom/submit.sh gromacs-oniom-hemoglobin/tasks.txt --partition=b200

# daint or clariden: 8 GPUs per job
export ONIOM_ACCOUNT=<project>
etc/oniom/submit.sh gromacs-oniom-hemoglobin/tasks.txt
etc/oniom/submit.sh gromacs-oniom-cyclic/campaign/tasks.txt    # again until nothing is left
```

Anywhere else (a workstation), `submit.sh` runs the tasks one after another. The campaign's
old cron runner (`runner.sh`, `install_cron.sh`) still works on a single machine.

## What to expect on each GPU

These are estimates from runs on an RTX 4070 Ti SUPER, not yet measured on the clusters. The
single-precision build moves the MM non-bondeds and PME to the GPU, which the workstation
runs never did (about 23 ms of each hemoglobin step was MM on the CPU).

| System | Workstation (MM on CPU) | H100 / GH200 (estimate) |
|---|---|---|
| Peptide, 87 ML atoms | 1.5-3 ns/day | 4-6 ns/day; the model call is latency-bound (about 10 ms) |
| Hemoglobin, 4 PET-OMol sites | 0.67 ns/day | 1.5-2.5 ns/day |
| Whole protein ML, PET-MAD xs | 0.042 ns/day | 3-5x faster; PET-MAD s or PET-OMol s also fit in 96 GB |

Checked on the workstation with a CUDA build from `build-gromacs.sh`: with non-bondeds and PME
on the GPU, the ONIOM energies match the CPU build (PET-MAD within 0.004 kJ/mol; the four
PET-OMol hemoglobin sites identical), and a 20 ps peptide NVE drifts -0.23 kJ/mol/ps (CPU
build: -0.45) at 15.9 instead of 18.7 ms/step, on a GPU shared with other jobs.

The GH200's 96 GB lets larger models run with the whole protein as ML (PET-MAD s needs about
26 GB for 9,026 atoms, PET-OMol s about 18 GB); the 16 GB workstation GPU fits only PET-MAD xs.
The double-precision build keeps MM on the CPU, as GROMACS has no GPU support in double, so
the shellres runs gain less. Several ML sites (the PET-OMol hemoglobin runs) need a single
rank, which is why each run gets one GPU.
