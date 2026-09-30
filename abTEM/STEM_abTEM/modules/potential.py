"""Projected electrostatic potential construction and plots."""
from __future__ import annotations

import abtem


def build_potential(atoms_ortho, config):
    if not all(atoms_ortho.pbc):
        raise ValueError("Potential requires a fully periodic Atoms object.")
    settings = config.potential
    potential = abtem.Potential(
        atoms_ortho,
        sampling=settings.sampling_A,
        parametrization=settings.parametrization,
        slice_thickness=settings.slice_thickness_A,
        projection=settings.projection,
    )
    metadata = {
        "sampling_requested_A": settings.sampling_A,
        "sampling_effective_A": tuple(potential.sampling),
        "gpts": tuple(potential.gpts),
        "parametrization": settings.parametrization,
        "slice_thickness_requested_A": settings.slice_thickness_A,
        "projection": settings.projection,
        "num_slices": potential.num_slices,
        "total_thickness_A": potential.thickness,
        "slice_limits_A": [[float(a), float(b)] for a, b in potential.slice_limits],
    }
    return potential, metadata


def show_potential(potential, config):
    import matplotlib.pyplot as plt

    # abTEM 1.0.6 can reject a lazy partial ``build(first_slice, last_slice)``
    # for finite projections because the shortened slice-thickness sequence no
    # longer sums to the original cell thickness. Build once, then index the
    # materialized PotentialArray for the selected-slice diagnostic plot.
    built = potential.build(lazy=True).compute()
    projected = built.project().compute()
    projected.show()
    plt.gcf().savefig(config.run_dir / "figures" / "potential_projected.png", dpi=160, bbox_inches="tight")
    subset = built[: min(3, built.num_slices)]
    subset.show(project=False)
    plt.gcf().savefig(config.run_dir / "figures" / "potential_first_slices.png", dpi=160, bbox_inches="tight")
    return projected, subset
