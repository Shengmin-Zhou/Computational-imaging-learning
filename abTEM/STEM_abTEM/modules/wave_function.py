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
    """Visualize the incident probe in real and reciprocal space.

    Notes
    -----
    abTEM wave arrays use the transverse axis order (x, y), whereas
    matplotlib.imshow() interprets a 2D array as (row=y, column=x).
    Therefore, the arrays are transposed before plotting.

    This is particularly important for rectangular simulation grids,
    otherwise a physically circular probe/aperture may appear elliptical.
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle

    # ------------------------------------------------------------------
    # 1. Build / compute the incident probe wave
    # ------------------------------------------------------------------
    wave = waves.compute().to_cpu()

    array = np.asarray(wave.array).squeeze()

    if array.ndim != 2:
        raise ValueError(
            f"Expected one 2D incident probe wave function, "
            f"but got array shape {array.shape}."
        )

    # abTEM convention:
    #   array.shape = (nx, ny)
    #
    # Physical real-space extent:
    #   extent = (Lx, Ly)
    nx, ny = array.shape
    extent_x, extent_y = tuple(wave.extent)

    # ------------------------------------------------------------------
    # 2. Diagnostics
    # ------------------------------------------------------------------
    print(
        "Probe real-space array:",
        f"shape=(nx={nx}, ny={ny}), "
        f"extent=({extent_x:.4f}, {extent_y:.4f}) Å"
    )

    # ------------------------------------------------------------------
    # 3. Prepare figure
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(15, 4),
        constrained_layout=True,
    )

    # ------------------------------------------------------------------
    # 4. Real-space probe intensity
    # ------------------------------------------------------------------
    intensity = np.abs(array) ** 2

    # IMPORTANT:
    # abTEM -> array[x, y]
    # imshow -> array[y, x]
    #
    # Therefore transpose before plotting.
    im0 = axes[0].imshow(
        intensity.T,
        origin="lower",
        aspect="equal",
        extent=(0.0, extent_x, 0.0, extent_y),
    )

    axes[0].set_title("Incident probe intensity")
    axes[0].set_xlabel("x (Å)")
    axes[0].set_ylabel("y (Å)")

    fig.colorbar(
        im0,
        ax=axes[0],
        fraction=0.046,
        pad=0.04,
        label="Intensity (a.u.)",
    )

    # ------------------------------------------------------------------
    # 5. Real-space probe phase
    # ------------------------------------------------------------------
    phase = np.angle(array)

    im1 = axes[1].imshow(
        phase.T,
        origin="lower",
        aspect="equal",
        extent=(0.0, extent_x, 0.0, extent_y),
        cmap="twilight",
        vmin=-np.pi,
        vmax=np.pi,
    )

    axes[1].set_title("Incident probe phase")
    axes[1].set_xlabel("x (Å)")
    axes[1].set_ylabel("y (Å)")

    fig.colorbar(
        im1,
        ax=axes[1],
        fraction=0.046,
        pad=0.04,
        label="Phase (rad)",
    )

    # ------------------------------------------------------------------
    # 6. Reciprocal-space probe / aperture
    # ------------------------------------------------------------------
    max_angle = config.probe.max_angle_mrad

    diffraction = (
        wave
        .diffraction_patterns(max_angle=max_angle)
        .compute()
        .to_cpu()
    )

    pattern = np.asarray(diffraction.array).squeeze()

    if pattern.ndim != 2:
        raise ValueError(
            f"Expected one 2D diffraction pattern, "
            f"but got array shape {pattern.shape}."
        )

    # abTEM transverse convention is again (x, y)
    nkx, nky = pattern.shape

    angular_sampling_x, angular_sampling_y = (
        diffraction.angular_sampling
    )

    print(
        "Probe reciprocal-space array:",
        f"shape=(nx={nkx}, ny={nky}), "
        f"angular sampling="
        f"({angular_sampling_x:.4f}, "
        f"{angular_sampling_y:.4f}) mrad"
    )

    # Physical angular limits corresponding to pixel centres / image extent.
    angle_x_half = nkx * angular_sampling_x / 2.0
    angle_y_half = nky * angular_sampling_y / 2.0

    reciprocal_extent = (
        -angle_x_half,
        +angle_x_half,
        -angle_y_half,
        +angle_y_half,
    )

    # Again transpose because imshow expects (y, x).
    im2 = axes[2].imshow(
        pattern.T,
        origin="lower",
        aspect="equal",
        extent=reciprocal_extent,
    )

    # Draw the nominal circular convergence aperture.
    aperture_circle = Circle(
        (0.0, 0.0),
        config.probe.semiangle_cutoff_mrad,
        fill=False,
        edgecolor="red",
        linewidth=1.5,
    )

    axes[2].add_patch(aperture_circle)

    axes[2].set_title("Incident probe aperture")
    axes[2].set_xlabel("angle x (mrad)")
    axes[2].set_ylabel("angle y (mrad)")

    fig.colorbar(
        im2,
        ax=axes[2],
        fraction=0.046,
        pad=0.04,
        label="Intensity (a.u.)",
    )

    # ------------------------------------------------------------------
    # 7. Save
    # ------------------------------------------------------------------
    output_path = config.run_dir / "figures" / "probe.png"

    fig.savefig(
        output_path,
        dpi=160,
        bbox_inches="tight",
    )

    return fig
