"""Incident probe, wave function, and aperture diagnostics."""
from __future__ import annotations

import numpy as np
import abtem
from config import ABERRATION_UNITS


def build_probe(potential, config):
    p = config.probe
    if p.max_angle_mrad < p.semiangle_cutoff_mrad:
        raise ValueError("Probe diffraction display max_angle must cover the aperture semiangle.")
    if set(p.aberrations) != set(ABERRATION_UNITS):
        raise ValueError("Probe aberration keys must match the canonical Cnm/phi_nm set.")
    kwargs = dict(
        energy=p.energy_eV,
        semiangle_cutoff=p.semiangle_cutoff_mrad,
        soft=p.soft_aperture,
        tilt=p.tilt_mrad,
        device=p.device,
        **p.aberrations,
    )
    for key, value in (("extent", p.extent_A), ("gpts", p.gpts), ("sampling", p.sampling_A)):
        if value is not None:
            kwargs[key] = value
    probe = abtem.Probe(**kwargs)
    # Potential is authoritative: any optional constructor grid values are
    # replaced by the potential's transverse grid before building waves.
    probe.match_grid(potential)
    waves = probe.build(lazy=True)
    metadata = {
        "energy_eV": p.energy_eV,
        "semiangle_cutoff_mrad": p.semiangle_cutoff_mrad,
        "soft_aperture": p.soft_aperture,
        "aperture_rolloff": "abTEM soft edge set by angular grid sampling; no independent Probe rolloff parameter",
        "tilt_mrad": p.tilt_mrad,
        "device_requested": p.device,
        "device_effective": probe.device,
        "constructor_extent_A": p.extent_A,
        "constructor_gpts": p.gpts,
        "constructor_sampling_A": p.sampling_A,
        "grid_precedence": "potential grid after probe.match_grid(potential)",
        "effective_extent_A": tuple(probe.extent),
        "effective_gpts": tuple(probe.gpts),
        "effective_sampling_A": tuple(probe.sampling),
        "aberrations": {key: {"value": value, "unit": ABERRATION_UNITS[key]}
                        for key, value in p.aberrations.items()},
        "defocus_convention": "C10 = -physical defocus in Å; defocus alias maps to C10, Cs to C30, astigmatism to C12 and astigmatism_angle to phi12",
        "diffraction_display_max_angle_mrad": p.max_angle_mrad,
    }
    return probe, waves, metadata


def show_probe(waves, config):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle

    wave = waves.compute().to_cpu()
    array = np.asarray(wave.array).squeeze()
    if array.ndim != 2:
        raise ValueError("Expected one 2D incident probe wave function.")
    extent = tuple(wave.extent)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].imshow(np.abs(array) ** 2, origin="lower", aspect="equal",
                   extent=(0, extent[0], 0, extent[1]))
    axes[0].set_title("Incident probe intensity\n(rectangular simulation grid)")
    axes[1].imshow(np.angle(array), origin="lower", aspect="equal",
                   extent=(0, extent[0], 0, extent[1]), cmap="twilight")
    axes[1].set_title("Incident probe phase (rad)\n(rectangular simulation grid)")
    for ax in axes[:2]:
        ax.set(xlabel="x (Å)", ylabel="y (Å)")
    max_angle = config.probe.max_angle_mrad
    diffraction = wave.diffraction_patterns(max_angle=max_angle).compute().to_cpu()
    pattern = np.asarray(diffraction.array).squeeze()
    sx, sy = diffraction.angular_sampling
    nx, ny = pattern.shape[1], pattern.shape[0]
    axes[2].imshow(pattern, origin="lower", aspect="equal",
                   extent=(-nx*sx/2, nx*sx/2, -ny*sy/2, ny*sy/2))
    axes[2].add_patch(Circle((0, 0), config.probe.semiangle_cutoff_mrad,
                             fill=False, edgecolor="red", linewidth=1.5))
    axes[2].set(xlabel="angle x (mrad)", ylabel="angle y (mrad)",
                title="Incident aperture (red circle)")
    fig.tight_layout()
    fig.savefig(config.run_dir / "figures" / "probe.png", dpi=160, bbox_inches="tight")
    return fig
