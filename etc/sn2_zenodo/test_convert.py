"""Smoke-test the SN2 npz → XYZ mapping without downloading Zenodo."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from ase.io import read, write

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from convert import npz_to_atoms


class _FakeNpz:
    def __init__(self) -> None:
        self._data = {
            "N": np.array([3, 2], dtype=int),
            "Z": np.array([[6, 1, 1, 0], [9, 17, 0, 0]], dtype=int),
            "R": np.array(
                [
                    [
                        [0.0, 0.0, 0.0],
                        [1.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0],
                        [0.0, 0.0, 0.0],
                    ],
                    [
                        [0.0, 0.0, 0.0],
                        [1.8, 0.0, 0.0],
                        [0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0],
                    ],
                ],
                dtype=float,
            ),
            "E": np.array([-10.0, -4.0], dtype=float),
            "Q": np.array([-1.0, 0.0], dtype=float),
            "D": np.array([[0.1, 0.2, 0.3], [0.0, 0.0, 1.0]], dtype=float),
            "F": np.array(
                [
                    [
                        [0.0, 0.0, 1.0],
                        [0.1, 0.0, 0.0],
                        [0.0, 0.1, 0.0],
                        [0.0, 0.0, 0.0],
                    ],
                    [
                        [0.2, 0.0, 0.0],
                        [-0.2, 0.0, 0.0],
                        [0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0],
                    ],
                ],
                dtype=float,
            ),
        }

    def __getitem__(self, key: str) -> np.ndarray:
        return self._data[key]


def test_npz_to_atoms_strips_padding(tmp_path) -> None:
    data = _FakeNpz()
    first = npz_to_atoms(data, 0)
    assert first.get_atomic_numbers().tolist() == [6, 1, 1]
    assert first.info["energy"] == -10.0
    assert first.info["charge"] == -1.0
    assert np.allclose(first.info["dipole_moment"], [0.1, 0.2, 0.3])
    assert first.arrays["forces"].shape == (3, 3)

    xyz = tmp_path / "sn2.xyz"
    write(xyz, [npz_to_atoms(data, 0), npz_to_atoms(data, 1)], format="extxyz")
    frames = read(xyz, index=":")
    assert len(frames) == 2
    assert frames[1].get_atomic_numbers().tolist() == [9, 17]
    assert "dipole" not in frames[0].info
    assert np.allclose(frames[0].info["dipole_moment"], [0.1, 0.2, 0.3])
