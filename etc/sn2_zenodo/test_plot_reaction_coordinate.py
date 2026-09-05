"""Unit tests for SN2 POV-Ray bond lists and alignment (no POV-Ray, no download)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from ase import Atoms

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from plot_reaction_coordinate import (
    align_sn2,
    reaction_coordinate,
    select_along_xi,
    sn2_bondpairs,
)


def _ch3cl_br(*, r_cl: float = 1.85, r_br: float = 2.15) -> Atoms:
    return Atoms(
        "CHHHClBr",
        positions=[
            [0.0, 0.0, 0.0],
            [0.36, 0.62, 0.72],
            [0.36, 0.62, -0.72],
            [0.36, -0.88, 0.0],
            [-r_cl, 0.0, 0.0],
            [r_br, 0.0, 0.0],
        ],
    )


def test_sn2_bondpairs_skips_distant_halogen_and_h_x() -> None:
    atoms = _ch3cl_br(r_cl=1.85, r_br=8.0)
    pairs = {(a, b) for a, b, _offset in sn2_bondpairs(atoms)}
    assert (0, 1) in pairs and (0, 2) in pairs and (0, 3) in pairs
    assert (0, 4) in pairs
    assert (0, 5) not in pairs
    assert not any(1 <= a <= 3 and b in (4, 5) for a, b in pairs)


def test_sn2_bondpairs_includes_both_halogens_near_ts() -> None:
    atoms = _ch3cl_br(r_cl=2.2, r_br=2.3)
    pairs = {(a, b) for a, b, _offset in sn2_bondpairs(atoms)}
    assert (0, 4) in pairs and (0, 5) in pairs


def test_align_sn2_puts_light_left_of_heavy() -> None:
    atoms = _ch3cl_br()
    atoms.rotate(37.0, "z")
    atoms.positions += [3.0, -1.5, 2.0]
    aligned = align_sn2(atoms)
    rc = reaction_coordinate(aligned)
    assert rc is not None
    assert np.linalg.norm(aligned.positions[rc["carbon"]]) < 1e-8
    light = aligned.positions[rc["light"]]
    assert light[0] < 0
    assert abs(light[1]) < 1e-6 and abs(light[2]) < 1e-6
    assert aligned.positions[rc["light"], 0] < aligned.positions[rc["heavy"], 0]


def test_select_along_xi_picks_nearest_unused() -> None:
    rows = [{"xi": x, "i": i} for i, x in enumerate((-8.0, -3.0, -0.1, 2.0, 7.0))]
    chosen = select_along_xi(rows, [-3.0, 0.0, 2.0])
    assert [row["xi"] for row in chosen] == [-3.0, -0.1, 2.0]
