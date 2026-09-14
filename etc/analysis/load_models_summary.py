"""
Loads every model in `model_registry.MODELS` via `MetatomicCalculator` and
records each one's declared capabilities (outputs, length unit, interaction
range) -- the "does it even load, and what does it claim to do" check that
`01_load_models.ipynb` opens with, before the benchmark section.

Run with the standard torch env:
    /home/boittier/metawork/.venv/bin/python load_models_summary.py
"""
import json
from pathlib import Path

import sphericart.torch  # pre-import before loading any model
from metatomic_ase import MetatomicCalculator

from model_registry import MODELS, RESULTS

RESULTS.mkdir(exist_ok=True)

rows = []
lines = []
for m in MODELS:
    model_path = m["dir"] / "model.pt"
    if not model_path.exists():
        continue
    calc = MetatomicCalculator(str(model_path))
    caps = calc.model().capabilities()
    row = {
        "key": m["key"],
        "family": m["family"],
        "dipole": m["dipole"],
        "outputs": list(caps.outputs.keys()),
        "length_unit": caps.length_unit,
        "interaction_range": caps.interaction_range,
    }
    rows.append(row)
    line = (f"{m['key']:28s} outputs={row['outputs']}  "
            f"length_unit={row['length_unit']}  interaction_range={row['interaction_range']:.2f}")
    print(line)
    lines.append(line)

Path("results/load_models_summary.json").write_text(json.dumps(rows, indent=2))
Path("results/load_models_summary.txt").write_text("\n".join(lines) + "\n")
print(f"\nWrote results/load_models_summary.{{json,txt}} ({len(rows)} models)")
