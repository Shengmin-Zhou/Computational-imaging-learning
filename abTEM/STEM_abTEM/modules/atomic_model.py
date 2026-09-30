"""Periodic atomic model preparation and structural diagnostics."""
from __future__ import annotations

import numpy as np
import abtem
from abtem.atoms import is_cell_valid, orthogonalize_cell, pretty_print_transform
from ase.io import read


def build_atomic_model(config):
    hkl = tuple(config.atomic.hkl)
    if hkl not in ((1, 0, 0), (0, 1, 0), (0, 0, 1)):
        raise ValueError("Supported orientations are (100), (010), and (001).")
    atoms = read(str(config.input_cif))
    if not all(atoms.pbc):
        raise ValueError("Input CIF must be periodic along x, y, and z.")
    atoms.wrap()
    normal = np.asarray(hkl, dtype=float) @ np.asarray(atoms.cell.reciprocal())
    if np.linalg.norm(normal) == 0:
        raise ValueError("The selected reciprocal plane normal is zero.")
    atoms.rotate(normal, (0, 0, 1), rotate_cell=True)
    atoms.wrap()
    atoms, transform = orthogonalize_cell(
        atoms,
        max_repetitions=config.atomic.orthogonalize_max_repetitions,
        return_transform=True,
    )
    repetitions = tuple(config.atomic.repetitions)
    if len(repetitions) != 3 or any(int(x) != x or x < 1 for x in repetitions):
        raise ValueError("repetitions must contain three positive integers")
    atoms = atoms.repeat(repetitions)
    atoms.wrap()
    if not all(atoms.pbc) or not is_cell_valid(atoms):
        raise ValueError("Final orthogonalized cell is not valid for abTEM.")
    metadata = {
        "source_cif": str(config.input_cif), "hkl": hkl,
        "repetitions": repetitions,
        "plane_normal_before_rotation": normal.tolist(),
        "orthogonalization_transform": [np.asarray(item).tolist() for item in transform],
        "final_cell_A": np.asarray(atoms.cell).tolist(),
        "atom_count": len(atoms), "pbc": atoms.pbc.tolist(),
    }
    return atoms, metadata


def show_projections(atoms, config):
    import matplotlib.pyplot as plt

    figures = {}
    for plane in ("xy", "xz", "yz"):
        fig, ax = plt.subplots(figsize=(5, 5))
        abtem.show_atoms(atoms, plane=plane, ax=ax, title=f"PbS {plane}")
        fig.savefig(config.run_dir / "figures" / f"atoms_{plane}.png", dpi=160, bbox_inches="tight")
        figures[plane] = fig
    return figures


def print_transform(metadata):
    transform = tuple(np.asarray(item) for item in metadata["orthogonalization_transform"])
    pretty_print_transform(transform)
