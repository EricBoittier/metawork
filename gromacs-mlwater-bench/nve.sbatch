#!/usr/bin/env bash
# NVE validation of GMX_METATOMIC_DISTRIBUTED: pure-ML water, PET-MAD xs, a BOX nm cube (default
# 5.85, about 20k atoms), NSTEPS steps at 0.5 fs from the same coordinates and velocities, on 1 GPU
# and on 2 GPUs distributed. Conserved-energy drift and fluctuation, temperature, and the
# step-by-step difference between the two runs. Free debug QOS, 2 GPUs.
#SBATCH --job-name=mlwater-nve
#SBATCH --partition=h100
#SBATCH --qos=debug
#SBATCH --gpus=2
#SBATCH --cpus-per-task=32
#SBATCH --mem=180G
#SBATCH --time=01:00:00
#SBATCH --output=/home/boittier/metawork/builds/logs/mlwater-nve-%j.out
set -uo pipefail
M=/home/boittier/metawork
W=$M/gromacs-mlwater-bench
. $M/builds/kuma-env.sh
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export GMX=$M/builds/gromacs-capi-halo-kuma-x86_64/bin/gmx ONIOM_EXTENSIONS=$M/builds/capi/build-plugin/libmetatomic_torch_plugin.so
. $M/etc/oniom/env.sh
L=${BOX:-5.85} N=${NSTEPS:-2000}
R=$W/runs/nve-$SLURM_JOB_ID
mkdir -p $R && cd $R
echo "== $SLURM_JOB_ID $(hostname) $(date -Is) $($GMX --version 2>/dev/null | grep 'GROMACS version')"; nvidia-smi -L
$GMX solvate -cs spc216.gro -box $L $L $L -o box.gro -quiet > /dev/null 2>&1
nw=$(( $(sed -n 2p box.gro) / 3 ))
printf '#include "%s/mlwater.itp"\n\n[ system ]\nPET-MAD xs water NVE\n\n[ molecules ]\nSOL %d\n' $W $nw > box.top
oniom_mdp $W/bench.mdp nve.mdp
sed -i -e "s/^nsteps .*/nsteps                  = $N/" -e 's/^nstlog .*/nstlog                  = 100/' nve.mdp
$GMX grompp -f nve.mdp -c box.gro -p box.top -o nve.tpr -maxwarn 3 -quiet > grompp.log 2>&1 || { tail -12 grompp.log; exit 1; }
echo "   $(sed -n 2p box.gro) atoms, $N steps"
for n in 1 2; do
    dist=0 gpus=0; [[ $n == 2 ]] && dist=1 gpus=01
    GMX_METATOMIC_DISTRIBUTED=$dist GMX_METATOMIC_DEVICE='cuda:{rank}' $GMX mdrun -s nve.tpr -deffnm gpu$n \
        -ntmpi $n -ntomp 8 -npme 0 -nb gpu -pme cpu -bonded cpu -update cpu -gpu_id $gpus -dlb no -pin on \
        > gpu$n.out 2>&1 && st=ok || st=FAILED
    echo "   gpu$n $st $(grep -h '^Performance:' gpu$n.log | awk '{print $2" ns/day"}') $(grep -h 'distributed, first step' gpu$n.log | sed 's/.*rows: //')"
done
$ONIOM_PY -I - "$GMX" <<'EOF'
import subprocess, sys
import numpy as np
gmx = sys.argv[1]
def terms(tag):
    out = f"_{tag}.xvg"
    subprocess.run([gmx, "energy", "-f", f"{tag}.edr", "-o", out, "-quiet"],
                   input="Metatomic-Potential\nKinetic-En.\nTotal-Energy\nConserved-En.\nTemperature\nVir-XX\nVir-YY\nVir-ZZ\n0\n",
                   text=True, capture_output=True, check=True)
    lines = open(out).read().splitlines()
    names = [l.split('"')[1] for l in lines if l.startswith("@ s") and 'legend "' in l]
    a = np.loadtxt([l for l in lines if l and l[0] not in "#@"], ndmin=2)
    return a[:, 0], {n: a[:, k + 1] for k, n in enumerate(names)}
t1, e1 = terms("gpu1")
t2, e2 = terms("gpu2")
n = min(len(t1), len(t2))
for tag, t, e in (("1 GPU", t1, e1), ("2 GPUs distributed", t2, e2)):
    c = e["Conserved En."]
    print(f"   {tag}: {t[-1]:.2f} ps, conserved-energy drift {np.polyfit(t, c, 1)[0]:+.2f} kJ/mol/ps, "
          f"std {c.std():.2f} kJ/mol, T {e['Temperature'].mean():.1f} K")
for k in (0, 1, 10, 50, 100, 500, n - 1):
    if k < n:
        print(f"   step {k * 1:5d} ({t1[k]:.4f} ps): dE_ML {e2['Metatomic Potential'][k] - e1['Metatomic Potential'][k]:+9.3f}  "
              f"dVir-XX {e2['Vir-XX'][k] - e1['Vir-XX'][k]:+9.3f}  dT {e2['Temperature'][k] - e1['Temperature'][k]:+7.3f} K")
print(f"   step 0 virial, 1 GPU: {e1['Vir-XX'][0]:.2f} {e1['Vir-YY'][0]:.2f} {e1['Vir-ZZ'][0]:.2f}; "
      f"2 GPUs: {e2['Vir-XX'][0]:.2f} {e2['Vir-YY'][0]:.2f} {e2['Vir-ZZ'][0]:.2f} kJ/mol")
EOF
echo "== done $(date -Is)"
