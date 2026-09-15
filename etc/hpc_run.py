#!/usr/bin/env python3
"""Prepare (and optionally submit) an HPC job from a small YAML spec.

Wraps an arbitrary shell command (a metatrain training run, an i-pi/LAMMPS MD
run, anything) with three things: a SLURM batch script sized to the target
cluster's QOS limits, a per-run output directory, and a manifest recording
everything needed to reproduce the run later -- the resolved job config, the
exact command, and the git branch/commit/dirty-state of every repo in this
workspace at the moment the job was prepared.

Deliberately simple: one script, one YAML spec per job, no job DAG / database.
Never submits by default -- it prepares runs/<timestamp>_<name>/job.sbatch and
tells you the `sbatch` command to run; pass --submit to have it call sbatch
itself (only useful on a node that actually has it, e.g. a real HPC login
node, not this workstation).

Usage:
    etc/hpc_run.py JOB_SPEC.yaml [--submit] [--runs-dir runs]

Job spec fields (see etc/hpc-jobs/examples/*.yaml):
    name            run name, used in the run directory name and job-name
    type            free-form label for the manifest, e.g. "train" or "md"
    cluster         key into CLUSTERS below (default: "kuma")
    qos             QOS name for that cluster, e.g. "normal" / "debug"
    nodes, gpus, cpus_per_task, time, mem   SLURM resource requests
    venv            "shared" (.venv) or "upet" (.venv-upet)
    workdir         cwd for `command`, relative to the metawork repo root
    inputs          list of files (relative to workdir) copied into the run
                    directory for reproducibility, e.g. the training options
    command         the shell command to run (multi-line YAML block scalar)
"""
import argparse
import datetime
import getpass
import shutil
import socket
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent

# Repos this workspace tracks that are worth recording the git state of in
# every run's manifest -- keep in sync with INSTALL_REPOS/CLONE_ONLY_REPOS in
# setup-metawork.sh. Missing ones (e.g. chemiscope on a machine that skipped
# it) are silently omitted rather than erroring.
TRACKED_REPOS = [
    ".",  # the metawork superproject itself
    "metatensor",
    "metatomic",
    "featomic",
    "metatrain",
    "i-pi",
    "chemiscope",
    "upet",
    "lammps",
    "gromacs",
    "eOn",
    "plumed2",
]

# Per-cluster QOS limits, used only to warn/error on an obviously-over-limit
# request before wasting a submission. Add clusters here as you get access to
# them; unknown clusters just skip validation.
CLUSTERS = {
    "kuma": {
        "qos": {
            "normal": {"priority": "normal", "max_time": "3-00:00:00", "max_nodes": 8},
            "long": {"priority": "low", "max_time": "7-00:00:00", "max_nodes": 8},
            "build": {
                "priority": "high",
                "max_time": "04:00:00",
                "max_nodes": 1,
                "max_gpus": 0,
                "max_cpus": 16,
            },
            "debug": {"priority": "high", "max_time": "01:00:00", "max_gpus": 2},
        },
    },
}


def parse_slurm_time(spec):
    """Parse a SLURM time string ("D-HH:MM:SS", "HH:MM:SS", ...) into minutes."""
    days = 0
    if "-" in spec:
        day_part, spec = spec.split("-", 1)
        days = int(day_part)
    parts = [int(p) for p in spec.split(":")]
    while len(parts) < 3:
        parts.insert(0, 0)
    hours, minutes, seconds = parts
    return days * 24 * 60 + hours * 60 + minutes + (1 if seconds else 0)


def validate_against_qos(job):
    cluster = CLUSTERS.get(job.get("cluster", "kuma"))
    if cluster is None:
        return  # unknown cluster, nothing to validate against
    qos = cluster["qos"].get(job.get("qos"))
    if qos is None:
        return
    problems = []
    if "time" in job and "max_time" in qos:
        if parse_slurm_time(job["time"]) > parse_slurm_time(qos["max_time"]):
            problems.append(f"time {job['time']} exceeds qos max {qos['max_time']}")
    if "nodes" in job and "max_nodes" in qos and job["nodes"] > qos["max_nodes"]:
        problems.append(f"nodes {job['nodes']} exceeds qos max {qos['max_nodes']}")
    if "gpus" in job and "max_gpus" in qos and job["gpus"] > qos["max_gpus"]:
        problems.append(f"gpus {job['gpus']} exceeds qos max {qos['max_gpus']}")
    if "cpus_per_task" in job and "max_cpus" in qos and job["cpus_per_task"] > qos["max_cpus"]:
        problems.append(
            f"cpus_per_task {job['cpus_per_task']} exceeds qos max {qos['max_cpus']}"
        )
    if problems:
        raise SystemExit(
            f"job spec violates {job.get('cluster', 'kuma')}/{job['qos']} QOS limits:\n  "
            + "\n  ".join(problems)
        )


def git_state(path):
    if not (path / ".git").exists():
        return None
    def git(*args):
        return subprocess.run(
            ["git", "-C", str(path), *args],
            capture_output=True, encoding="utf8", check=False,
        ).stdout.strip()
    branch = git("rev-parse", "--abbrev-ref", "HEAD")
    commit = git("rev-parse", "--short", "HEAD")
    dirty = bool(git("status", "--porcelain"))
    return {"branch": branch, "commit": commit, "dirty": dirty}


def collect_repo_states():
    states = {}
    for name in TRACKED_REPOS:
        path = ROOT / name
        state = git_state(path)
        if state is not None:
            states[Path(name).name if name != "." else "metawork"] = state
    return states


def python_env_info(venv):
    py = ROOT / (".venv" if venv == "shared" else ".venv-upet") / "bin" / "python"
    if not py.exists():
        return {"venv": venv, "python": None}
    result = subprocess.run(
        [str(py), "-c",
         "import sys, importlib\n"
         "print(sys.version.split()[0])\n"
         "try:\n"
         "    import torch\n"
         "    print(torch.__version__)\n"
         "    print(torch.cuda.is_available())\n"
         "except Exception:\n"
         "    print('?'); print('?')\n"],
        capture_output=True, encoding="utf8", check=False,
    )
    lines = result.stdout.strip().splitlines() or ["?", "?", "?"]
    lines += ["?"] * (3 - len(lines))
    return {
        "venv": venv,
        "python": lines[0],
        "torch": lines[1],
        "cuda_available": lines[2],
    }


def render_sbatch(job, run_dir, manifest_path):
    venv = ROOT / (".venv" if job.get("venv", "shared") == "shared" else ".venv-upet")
    workdir = ROOT / job.get("workdir", ".")
    lines = [
        "#!/usr/bin/env bash",
        f"#SBATCH --job-name={job['name']}",
        f"#SBATCH --output={run_dir}/slurm-%j.out",
        f"#SBATCH --error={run_dir}/slurm-%j.err",
    ]
    if "qos" in job:
        lines.append(f"#SBATCH --qos={job['qos']}")
    if "time" in job:
        lines.append(f"#SBATCH --time={job['time']}")
    if "nodes" in job:
        lines.append(f"#SBATCH --nodes={job['nodes']}")
    if job.get("gpus"):
        lines.append(f"#SBATCH --gres=gpu:{job['gpus']}")
    if "cpus_per_task" in job:
        lines.append(f"#SBATCH --cpus-per-task={job['cpus_per_task']}")
    if "mem" in job:
        lines.append(f"#SBATCH --mem={job['mem']}")
    lines += [
        "",
        "set -euo pipefail",
        "",
        f"echo \"job started on $(hostname) at $(date)\" | tee -a {run_dir}/run.log",
        f"source {venv}/bin/activate",
        f"cd {workdir}",
        "",
        job["command"].rstrip(),
        "",
        f"echo \"job finished at $(date)\" | tee -a {run_dir}/run.log",
    ]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("spec", type=Path, help="job spec YAML file")
    parser.add_argument("--runs-dir", type=Path, default=ROOT / "runs")
    parser.add_argument("--submit", action="store_true", help="call sbatch after preparing the run")
    args = parser.parse_args()

    job = yaml.safe_load(args.spec.read_text())
    if "name" not in job or "command" not in job:
        raise SystemExit("job spec needs at least 'name' and 'command'")
    job.setdefault("cluster", "kuma")

    validate_against_qos(job)

    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = args.runs_dir / f"{timestamp}_{job['name']}"
    run_dir.mkdir(parents=True, exist_ok=False)

    workdir = ROOT / job.get("workdir", ".")
    for rel in job.get("inputs", []):
        src = workdir / rel
        if src.exists():
            shutil.copy2(src, run_dir / Path(rel).name)
        else:
            print(f"warning: input '{rel}' not found at {src}, not snapshotted", file=sys.stderr)

    manifest = {
        "run": {
            "name": job["name"],
            "type": job.get("type", "unspecified"),
            "created": datetime.datetime.now().isoformat(timespec="seconds"),
            "user": getpass.getuser(),
            "host": socket.gethostname(),
            "spec_file": str(args.spec.resolve()),
        },
        "slurm": {
            "cluster": job["cluster"],
            "qos": job.get("qos"),
            "nodes": job.get("nodes"),
            "gpus": job.get("gpus"),
            "cpus_per_task": job.get("cpus_per_task"),
            "time": job.get("time"),
            "mem": job.get("mem"),
            "job_id": None,
        },
        "workdir": job.get("workdir", "."),
        "command": job["command"],
        "python_env": python_env_info(job.get("venv", "shared")),
        "repos": collect_repo_states(),
    }
    manifest_path = run_dir / "manifest.yaml"
    manifest_path.write_text(yaml.safe_dump(manifest, sort_keys=False))

    sbatch_path = run_dir / "job.sbatch"
    sbatch_path.write_text(render_sbatch(job, run_dir, manifest_path))

    print(f"prepared run: {run_dir}")
    print(f"  manifest: {manifest_path}")
    print(f"  sbatch:   {sbatch_path}")

    if args.submit:
        if shutil.which("sbatch") is None:
            raise SystemExit("--submit given but 'sbatch' is not on PATH here")
        result = subprocess.run(["sbatch", "--parsable", str(sbatch_path)],
                                 capture_output=True, encoding="utf8", check=False)
        if result.returncode != 0:
            raise SystemExit(f"sbatch failed:\n{result.stderr}")
        job_id = result.stdout.strip()
        manifest["slurm"]["job_id"] = job_id
        manifest_path.write_text(yaml.safe_dump(manifest, sort_keys=False))
        print(f"  submitted: job {job_id}")
    else:
        print(f"  not submitted (pass --submit, or run: sbatch {sbatch_path})")


if __name__ == "__main__":
    main()
