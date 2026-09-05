"""POV-Ray snapshots on a shared-ξ matplotlib SN2 reaction coordinate.

Uses ASE ``write_pov`` + ``bondatoms`` (C–H and C–halogen sticks). POV-Ray
must be on PATH. Run from the metatrain venv that has ASE:

    metatrain/.tox/lorem-tests/bin/python \\
        etc/sn2_zenodo/plot_reaction_coordinate.py
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from ase import Atoms
from ase.data import chemical_symbols, covalent_radii
from ase.io import read
from ase.io.pov import write_pov
from matplotlib.offsetbox import AnnotationBbox, OffsetImage

HALOGENS = {9, 17, 35, 53}
C_H_CUT = 1.25
C_X_CUT = 2.80


def six_atom_indices(atoms: Atoms) -> tuple[int, list[int], list[int]] | None:
    numbers = atoms.get_atomic_numbers()
    c_idx = np.flatnonzero(numbers == 6)
    h_idx = np.flatnonzero(numbers == 1)
    x_idx = [i for i, z in enumerate(numbers) if int(z) in HALOGENS]
    if len(c_idx) != 1 or len(h_idx) != 3 or len(x_idx) != 2:
        return None
    return int(c_idx[0]), [int(i) for i in h_idx], x_idx


def reaction_coordinate(atoms: Atoms) -> dict | None:
    ids = six_atom_indices(atoms)
    if ids is None:
        return None
    carbon, _hydrogens, x_idx = ids
    pos = atoms.get_positions()
    z1, z2 = (int(atoms.numbers[x_idx[0]]), int(atoms.numbers[x_idx[1]]))
    r1 = float(np.linalg.norm(pos[x_idx[0]] - pos[carbon]))
    r2 = float(np.linalg.norm(pos[x_idx[1]] - pos[carbon]))
    if z1 <= z2:
        light, heavy, r_light, r_heavy = x_idx[0], x_idx[1], r1, r2
        pair = (z1, z2)
    else:
        light, heavy, r_light, r_heavy = x_idx[1], x_idx[0], r2, r1
        pair = (z2, z1)
    return {
        "carbon": carbon,
        "light": light,
        "heavy": heavy,
        "r_light": r_light,
        "r_heavy": r_heavy,
        "xi": r_heavy - r_light,
        "pair": pair,
        "label": f"{chemical_symbols[pair[0]]}–{chemical_symbols[pair[1]]}",
    }


def sn2_bondpairs(atoms: Atoms) -> list[tuple[int, int, tuple[int, int, int]]]:
    """C–H and C–halogen bonds only (no spurious H–X from a large cutoff)."""
    ids = six_atom_indices(atoms)
    if ids is None:
        return []
    carbon, hydrogens, x_idx = ids
    pos = atoms.get_positions()
    pairs: list[tuple[int, int, tuple[int, int, int]]] = []
    seen: set[tuple[int, int]] = set()

    def add(i: int, j: int) -> None:
        a, b = (i, j) if i < j else (j, i)
        if (a, b) in seen:
            return
        seen.add((a, b))
        pairs.append((a, b, (0, 0, 0)))

    for h in hydrogens:
        if float(np.linalg.norm(pos[h] - pos[carbon])) < C_H_CUT:
            add(carbon, h)
    for x in x_idx:
        if float(np.linalg.norm(pos[x] - pos[carbon])) < C_X_CUT:
            add(carbon, x)
    return pairs


def align_sn2(atoms: Atoms) -> Atoms:
    """C at the origin, lighter halogen on −x, methyl hydrogens toward +z."""
    rc = reaction_coordinate(atoms)
    if rc is None:
        raise ValueError("not a six-atom SN2 complex")
    aligned = atoms.copy()
    aligned.positions -= aligned.positions[rc["carbon"]]
    aligned.rotate(aligned.positions[rc["light"]], (-1.0, 0.0, 0.0), rotate_cell=False)
    h_mean = aligned.positions[aligned.numbers == 1].mean(axis=0)
    aligned.rotate(-np.degrees(np.arctan2(h_mean[1], h_mean[2])), "x")
    aligned.positions -= aligned.positions[rc["carbon"]]
    aligned.pbc = False
    return aligned


def select_along_xi(rows: list[dict], targets: list[float]) -> list[dict]:
    xs = np.asarray([row["xi"] for row in rows], dtype=float)
    used: set[int] = set()
    chosen: list[dict] = []
    for target in targets:
        for index in np.argsort(np.abs(xs - target)):
            j = int(index)
            if j not in used:
                used.add(j)
                chosen.append(rows[j])
                break
    chosen.sort(key=lambda row: row["xi"])
    return chosen


# Tight window around C after align_sn2 so every thumbnail is the same scale.
POV_BBOX = (-4.4, -2.3, 4.4, 2.3)


def render_pov(atoms: Atoms, dest: Path, canvas_width: int = 380) -> Path:
    """Write and render a POV-Ray PNG. Run povray in ``dest.parent`` (ASE ini)."""
    dest = dest.resolve()
    dest.parent.mkdir(parents=True, exist_ok=True)
    radii = 0.42 * covalent_radii[atoms.numbers]
    renderer = write_pov(
        str(dest.with_suffix(".pov")),
        atoms,
        povray_settings={
            "canvas_width": int(canvas_width),
            "transparent": True,
            "display": False,
            "bondatoms": sn2_bondpairs(atoms),
            "bondlinewidth": 0.09,
            "camera_type": "orthographic",
            "background": "White",
        },
        radii=radii,
        show_unit_cell=0,
        bbox=list(POV_BBOX),
    )
    ini = renderer.path
    subprocess.check_call(
        ["povray", ini.name],
        cwd=ini.parent,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    png = ini.with_suffix(".png")
    if not png.is_file():
        raise RuntimeError(f"POV-Ray left no PNG at {png}")
    return png


def _load_pred_map(path: Path) -> dict[tuple[float, float], float]:
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text())
    mapping: dict[tuple[float, float], float] = {}
    for row in payload.get("rc", []):
        mapping[(round(float(row["xi"]), 3), round(float(row["e_ref"]), 3))] = float(
            row["e_pred"]
        )
    return mapping


def _predict_selected(frames: list[Atoms], model_pt: Path) -> np.ndarray:
    from metatomic.torch import ModelOutput
    from metatomic.torch.ase_calculator import MetatomicCalculator

    calc = MetatomicCalculator(str(model_pt))
    energies = []
    for atoms in frames:
        atoms.calc = calc
        energies.append(float(atoms.get_potential_energy()))
    return np.asarray(energies, dtype=float)


def collect_pair_rows(frames: list[Atoms], pair_label: str | None) -> tuple[str, list[dict]]:
    rows: list[dict] = []
    counts: dict[str, int] = {}
    for index, atoms in enumerate(frames):
        rc = reaction_coordinate(atoms)
        if rc is None:
            continue
        counts[rc["label"]] = counts.get(rc["label"], 0) + 1
        rows.append(
            {
                **rc,
                "i": index,
                "e_ref": float(atoms.get_potential_energy()),
            }
        )
    if not rows:
        raise SystemExit("no six-atom SN2 complexes in the XYZ")
    label = pair_label or max(counts, key=counts.get)
    picked = [row for row in rows if row["label"] == label]
    if not picked:
        available = ", ".join(sorted(counts))
        raise SystemExit(f"no frames for {pair_label!r}; have {available}")
    return label, picked


def plot_coordinate(
    rows: list[dict],
    snapshots: list[dict],
    pngs: list[Path],
    out: Path,
    pair_label: str,
) -> Path:
    xi = np.asarray([row["xi"] for row in rows], dtype=float)
    e_ref = np.asarray([row["e_ref"] for row in rows], dtype=float)
    e_pred = np.asarray([row.get("e_pred", np.nan) for row in rows], dtype=float)
    r_light = np.asarray([row["r_light"] for row in rows], dtype=float)
    r_heavy = np.asarray([row["r_heavy"] for row in rows], dtype=float)
    has_pred = np.isfinite(e_pred).any()

    light, heavy = pair_label.split("–")
    fig, (ax_mol, ax_e, ax_r) = plt.subplots(
        3,
        1,
        sharex=True,
        figsize=(11.2, 8.4),
        gridspec_kw={"height_ratios": [1.15, 2.15, 1.45], "hspace": 0.06},
    )
    snap_x = np.asarray([snap["xi"] for snap in snapshots], dtype=float)
    x_pad = 1.8
    xlim = (float(snap_x.min()) - x_pad, float(snap_x.max()) + x_pad)

    ax_mol.set_xlim(*xlim)
    ax_mol.set_ylim(0.0, 1.0)
    ax_mol.set_yticks([])
    ax_mol.tick_params(axis="x", labelbottom=False)
    for spine in ax_mol.spines.values():
        spine.set_visible(False)
    ax_mol.set_ylabel("POV-Ray")

    zoom = 0.34 if len(snapshots) <= 5 else 0.28
    for snap, png in zip(snapshots, pngs, strict=True):
        image = plt.imread(png)
        box = OffsetImage(image, zoom=zoom)
        ax_mol.add_artist(
            AnnotationBbox(
                box,
                (snap["xi"], 0.52),
                frameon=False,
                annotation_clip=False,
                pad=0.0,
            )
        )
        ax_mol.text(
            snap["xi"],
            0.04,
            f"ξ={snap['xi']:+.2f}",
            ha="center",
            va="bottom",
            fontsize=8,
            color="0.35",
        )

    inside = (xi >= xlim[0]) & (xi <= xlim[1])
    ax_e.scatter(
        xi[inside], e_ref[inside], s=18, c="0.45", alpha=0.75, label="DFT", zorder=2
    )
    if has_pred:
        ax_e.scatter(
            xi[inside], e_pred[inside], s=18, c="C0", alpha=0.7, label="LOREM", zorder=3
        )
    ax_e.set_ylabel("atomization energy (eV)")
    ax_e.legend(frameon=False, loc="best")
    ax_e.axvline(0.0, color="0.75", ls="--", lw=0.8)

    ax_r.scatter(
        xi[inside], r_light[inside], s=18, c="C2", alpha=0.75, label=f"r(C–{light})"
    )
    ax_r.scatter(
        xi[inside], r_heavy[inside], s=18, c="C3", alpha=0.75, label=f"r(C–{heavy})"
    )
    ax_r.set_ylabel("distance (Å)")
    ax_r.set_xlabel(f"ξ = r(C–{heavy}) − r(C–{light}) (Å)")
    ax_r.legend(frameon=False, loc="best")
    ax_r.axvline(0.0, color="0.75", ls="--", lw=0.8)

    for snap in snapshots:
        for axis in (ax_mol, ax_e, ax_r):
            axis.axvline(snap["xi"], color="0.85", lw=0.7, zorder=0)

    fig.suptitle(
        f"{pair_label} SN2  ·  {int(inside.sum())}/{len(rows)} frames in this ξ window  ·  "
        "ASE POV-Ray (C–H and C–X bonds added)",
        fontsize=12,
    )
    fig.text(
        0.5,
        0.01,
        "Source: PhysNet SN2 (Zenodo 2605341) subset in ~/data/sn2/sn2.xyz. "
        "Negative ξ is CH₃Br + Cl⁻-like; ξ≈0 is pentacoordinate.",
        ha="center",
        fontsize=8,
        color="0.35",
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path.home() / "data" / "sn2",
    )
    parser.add_argument("--xyz", type=Path, default=None)
    parser.add_argument("--pair", default=None, help="e.g. Cl–Br (default: most common)")
    parser.add_argument(
        "--xi-targets",
        default="-2.5,-1.2,0,1.2,2.5",
        help="comma-separated ξ values (Å) for POV-Ray annotations",
    )
    parser.add_argument("--pred-json", type=Path, default=Path("/tmp/sn2_plot_payload.json"))
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()

    xyz = args.xyz or (args.data_dir / "sn2.xyz")
    out_dir = args.out_dir or (args.data_dir / "figures")
    frames = read(xyz, index=":")
    pair_label, rows = collect_pair_rows(frames, args.pair)

    pred_map = _load_pred_map(args.pred_json)
    for row in rows:
        row["e_pred"] = pred_map.get((round(row["xi"], 3), round(row["e_ref"], 3)), np.nan)

    if not np.isfinite([row["e_pred"] for row in rows]).any():
        model_pt = args.data_dir / "model.pt"
        if model_pt.is_file():
            print(f"predicting {len(rows)} {pair_label} frames with {model_pt}", flush=True)
            subset = [frames[row["i"]] for row in rows]
            energies = _predict_selected(subset, model_pt)
            for row, energy in zip(rows, energies, strict=True):
                row["e_pred"] = float(energy)

    targets = [float(part) for part in args.xi_targets.split(",") if part.strip()]
    snapshots = select_along_xi(rows, targets)
    pov_dir = out_dir / "pov"
    stem = pair_label.replace("–", "")
    if pov_dir.is_dir():
        for leftover in pov_dir.glob(f"{stem}_*"):
            leftover.unlink()
    pngs: list[Path] = []
    for k, snap in enumerate(snapshots):
        aligned = align_sn2(frames[snap["i"]])
        png = render_pov(aligned, pov_dir / f"{pair_label.replace('–', '')}_{k:02d}.pov")
        pngs.append(png)
        print(
            f"rendered {png.name}  ξ={snap['xi']:+.3f}  "
            f"r_light={snap['r_light']:.3f}  r_heavy={snap['r_heavy']:.3f}  "
            f"bonds={len(sn2_bondpairs(aligned))}",
            flush=True,
        )

    figure = plot_coordinate(
        rows,
        snapshots,
        pngs,
        out_dir / f"{pair_label.replace('–', '')}_reaction_coordinate.png",
        pair_label,
    )
    print(f"wrote {figure}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
