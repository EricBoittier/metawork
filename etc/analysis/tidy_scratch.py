"""One-time housekeeping: move the engine scratch/restart noise that
`hourglass_engines.py`'s `run_ipi`/`run_lammps`/`run_gromacs` (and i-PI's own
automatic backup mechanism) write into `analysis/`'s top level into
`analysis/scratch/<project>/<simulation>/<date>/`, out of the way of the actual
notebooks/scripts/results. Purely a filesystem tidy -- none of this is read back
by anything (every run rewrites its own inputs before reading its own outputs),
so nothing here is a dependency of any notebook.

`<project>` is the model the file's name identifies (`sn2-matched-lorem`,
`sn2-matched-pet`), or `_unattributed` for i-PI's own generic backup files
(`RESTART`, `#RESTART#N#`) and GROMACS's generic `mdout.mdp`, which don't carry a
model/box tag in their name. `<simulation>` is `{engine}-{ensemble}-box{N}`
(e.g. `ipi-nve-box50`), taken from the filename. `<date>` is the file's own
modification date (`YYYY-MM-DD`), so a future run's noise sorts under its own day
without needing this script to change.

Run once from this directory:
    python tidy_scratch.py
"""
import datetime
import re
import shutil
from pathlib import Path

ANALYSIS_DIR = Path(__file__).parent
SCRATCH_ROOT = ANALYSIS_DIR / "scratch"

MODEL_KEYS = ["sn2-matched-lorem-dipole", "sn2-matched-pet-dipole", "sn2-matched-lorem", "sn2-matched-pet"]

# engine-tagged scratch: sn2-{engine}-{model}-{ensemble}-box{N}[.ext | -stuff]
TAGGED_RE = re.compile(r"^#?sn2-(ipi|lammps|gmx)-.*-(nve|nvt)-box[\d.]+")

UNATTRIBUTED_NAMES = {"mdout.mdp"}
UNATTRIBUTED_RE = re.compile(r"^#?RESTART(#\d+#)?$")


def classify(path: Path):
    name = path.name
    if UNATTRIBUTED_RE.match(name) or name in UNATTRIBUTED_NAMES:
        engine = "gromacs" if name in UNATTRIBUTED_NAMES else "i-pi"
        return "_unattributed", f"{engine}-backups"
    if not TAGGED_RE.match(name):
        return None
    engine_tag = {"ipi": "i-pi", "lammps": "lammps", "gmx": "gromacs"}[TAGGED_RE.match(name).group(1)]
    model = next((m for m in MODEL_KEYS if m in name), "_unattributed")
    ensemble_box = re.search(r"(nve|nvt)-box\d+(?:\.\d+)?", name)
    simulation = f"{engine_tag}-{ensemble_box.group(0)}" if ensemble_box else engine_tag
    return model, simulation


def main():
    moved = 0
    for path in sorted(ANALYSIS_DIR.iterdir()):
        if path.is_dir():
            continue
        result = classify(path)
        if result is None:
            continue
        project, simulation = result
        date = datetime.date.fromtimestamp(path.stat().st_mtime).isoformat()
        dest_dir = SCRATCH_ROOT / project / simulation / date
        dest_dir.mkdir(parents=True, exist_ok=True)
        shutil.move(str(path), str(dest_dir / path.name))
        moved += 1
    print(f"Moved {moved} scratch files under {SCRATCH_ROOT}/")


if __name__ == "__main__":
    main()
