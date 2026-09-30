"""Small STEM scans and detector measurements."""
from __future__ import annotations

import warnings
import abtem
import numpy as np


def configure_scan(potential, probe, atoms_ortho, config):
    if not all(atoms_ortho.pbc):
        raise ValueError("Scan input must be fully periodic.")
    ranges = {
        "BF": config.detector.bf_mrad,
        "MAADF": config.detector.maadf_mrad,
        "HAADF": config.detector.haadf_mrad,
    }
    supported = float(min(probe.cutoff_angles))
    requested_outer = max(outer for _, outer in (
        config.detector.bf_mrad,
        config.detector.maadf_mrad,
        config.detector.haadf_mrad,
    ))
    suggested_sampling = min(probe.sampling) * supported / requested_outer
    print(
        f"Antialias cutoff: {supported:.1f} mrad (current probe sampling "
        f"{min(probe.sampling):.4f} Å); detector angles above this are not "
        "quantitatively supported."
    )
    if requested_outer > supported:
        print(
            f"To support {requested_outer:.1f} mrad approximately, use "
            f"potential sampling_A <= {suggested_sampling:.4f} Å "
            "(this increases the simulation grid size)."
        )
    detectors = {}
    for name, (inner, outer) in ranges.items():
        if not (0 <= inner < outer):
            raise ValueError(f"Invalid {name} detector angular range: {(inner, outer)}")
        if outer > supported:
            warnings.warn(f"{name} outer angle {outer} mrad exceeds antialias cutoff "
                          f"{supported:.1f} mrad; high-angle signal may be invalid.", stacklevel=2)
        detectors[name] = abtem.AnnularDetector(inner=inner, outer=outer)
    pixel_detector = None
    if config.detector.enable_4d_stem:
        pixel_max = config.detector.pixelated_max_angle_mrad
        if pixel_max > supported:
            warnings.warn(f"Pixelated max angle {pixel_max} mrad exceeds "
                          f"antialias cutoff {supported:.1f} mrad; high-angle data may be invalid. "
                          "Reduce potential sampling_A for quantitative high-angle 4D-STEM.", stacklevel=2)
        pixel_detector = abtem.PixelatedDetector(max_angle=pixel_max)
    sampling = config.scan.sampling_A
    if sampling is None:
        sampling = float(probe.aperture.nyquist_sampling)
    if sampling <= 0:
        raise ValueError("Scan sampling must be positive Å per position.")
    scan = abtem.GridScan(
        start=config.scan.start, end=config.scan.end, sampling=sampling,
        fractional=config.scan.fractional, potential=potential, endpoint=False,
    )
    shape = tuple(int(x) for x in scan.gpts)
    count = int(np.prod(shape))
    if count > config.scan.max_positions:
        raise ValueError(f"Scan has {count} positions; limit is {config.scan.max_positions}. "
                         "Shrink scan bounds or increase the explicit limit.")
    print(f"Manual STEM scan: {shape[0]} × {shape[1]} = {count} positions; step {sampling:.3f} Å")
    metadata = {
        "start": config.scan.start, "end": config.scan.end,
        "fractional": config.scan.fractional,
        "sampling_A": sampling, "shape": shape, "position_count": count,
        "detector_ranges_mrad": ranges,
        "pixelated_enabled": config.detector.enable_4d_stem,
        "pixelated_max_angle_mrad": config.detector.pixelated_max_angle_mrad if pixel_detector else None,
        "supported_antialias_cutoff_mrad": supported,
        "probe_sampling_A": tuple(float(x) for x in probe.sampling),
        "antialias_explanation": (
            "The antialias cutoff is the largest scattering angle retained by "
            "the current real-space grid after abTEM's antialiasing margin. "
            "Detector outer angles above it are outside the quantitatively "
            "supported reciprocal-space region."
        ),
        "suggested_sampling_for_requested_outer_A": suggested_sampling,
        "angular_warning": {name: outer > supported for name, (_, outer) in ranges.items()},
        "pixelated_angular_warning": bool(pixel_detector and
                                            config.detector.pixelated_max_angle_mrad > supported),
    }
    return scan, detectors, pixel_detector, metadata


def show_scan_and_detectors(scan, detectors, atoms_ortho, probe, config):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, Patch, Wedge

    fig, ax = plt.subplots(figsize=(5, 5))
    abtem.show_atoms(atoms_ortho, plane="xy", ax=ax)
    scan.add_to_plot(ax)
    ax.set_title("Scan grid over atomic projection")
    fig.savefig(config.run_dir / "figures" / "scan_grid.png", dpi=160, bbox_inches="tight")

    ranges = {
        "BF": config.detector.bf_mrad,
        "MAADF": config.detector.maadf_mrad,
        "HAADF": config.detector.haadf_mrad,
    }
    colors = {"BF": "#377eb8", "MAADF": "#ff9f00", "HAADF": "#e41a1c"}
    supported = float(min(probe.cutoff_angles))
    limit = max(max(outer for _, outer in ranges.values()), supported) * 1.08

    def draw_regions(ax, selected=None):
        for name, (inner, outer) in ranges.items():
            if selected is not None and name != selected:
                continue
            ax.add_patch(Wedge(
                (0, 0), outer, 0, 360, width=outer - inner,
                facecolor=colors[name], edgecolor=colors[name], alpha=0.55,
                label=f"{name}: {inner:g}–{outer:g} mrad",
            ))
        ax.add_patch(Circle((0, 0), supported, fill=False, linestyle="--",
                            linewidth=1.5, color="black",
                            label=f"antialias cutoff: {supported:.1f} mrad"))
        ax.set(xlim=(-limit, limit), ylim=(-limit, limit), aspect="equal",
               xlabel="angle x (mrad)", ylabel="angle y (mrad)")
        ax.grid(alpha=0.2)

    # The three individual panels make overlapping MAADF/HAADF ranges clear;
    # the combined panel retains the official-style colored detector overview.
    fig2, axes = plt.subplots(2, 2, figsize=(11, 10), constrained_layout=True)
    for ax, selected in zip(axes.ravel()[:3], ranges):
        draw_regions(ax, selected=selected)
        ax.set_title(f"{selected} detector")
    draw_regions(axes.flat[3])
    axes.flat[3].set_title("Combined detector regions")
    axes.flat[3].legend(handles=[
        Patch(facecolor=colors[name], edgecolor=colors[name], alpha=0.55,
              label=f"{name}: {inner:g}–{outer:g} mrad")
        for name, (inner, outer) in ranges.items()
    ] + [Patch(facecolor="none", edgecolor="black", linestyle="--",
               label=f"supported cutoff: {supported:.1f} mrad")],
        loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=9)
    fig2.suptitle("STEM detector angular regions (filled colors)")
    fig2.savefig(config.run_dir / "figures" / "detectors.png", dpi=160, bbox_inches="tight")
    fig2.savefig(config.run_dir / "figures" / "detectors_colored.png", dpi=160, bbox_inches="tight")
    # Newer abTEM releases expose AnnularDetector.show(). abTEM 1.0.6 does
    # not, so keep the portable matplotlib angular-region diagram above and
    # use the API only when it exists.
    for name, detector in detectors.items():
        show_detector = getattr(detector, "show", None)
        if callable(show_detector):
            show_detector(gpts=probe.gpts, sampling=probe.sampling, energy=probe.energy)
            plt.gcf().savefig(config.run_dir / "figures" / f"detector_{name}.png",
                              dpi=160, bbox_inches="tight")
    return fig, fig2


def run_annular_scan(probe, potential, scan, detectors, config):
    names = ("BF", "MAADF", "HAADF")
    results = probe.scan(potential=potential, scan=scan,
                         detectors=[detectors[name] for name in names], lazy=True)
    if len(results) != len(names):
        raise RuntimeError("abTEM returned an unexpected number of detector measurements")
    # Compute the detector list together so one multislice run feeds all three.
    computed = results.compute(progress_bar=True)
    measurements = {}
    for name, image in zip(names, computed):
        image.to_zarr(str(config.run_dir / "measurements" / f"{name}_raw.zarr"))
        image.show()
        import matplotlib.pyplot as plt
        plt.gcf().savefig(config.run_dir / "figures" / f"{name}_raw.png", dpi=160, bbox_inches="tight")
        measurements[name] = image
    return measurements


def run_4d_scan(probe, potential, scan, pixel_detector, config):
    if not config.detector.enable_4d_stem:
        return None
    if pixel_detector is None:
        raise RuntimeError("4D-STEM was enabled after detector setup; rerun configure_scan.")
    from dask.diagnostics import ProgressBar
    from abtem.array import from_zarr

    result = probe.scan(potential=potential, scan=scan, detectors=pixel_detector, lazy=True)
    # zarr stores chunked raw diffraction data and abTEM axis metadata.
    store = config.run_dir / "4d_stem" / "pixelated_raw.zarr"
    with ProgressBar():
        result.to_zarr(str(store), compute=True)
    # Read only selected patterns from disk; do not repeat the full scan.
    saved = from_zarr(str(store))
    for iy, ix in config.detector.cbed_scan_indices:
        if not (0 <= iy < saved.shape[0] and 0 <= ix < saved.shape[1]):
            raise IndexError(f"CBED scan index {(iy, ix)} is outside {saved.shape[:2]}")
        pattern = saved[iy, ix].compute(progress_bar=True)
        pattern.show()
        import matplotlib.pyplot as plt
        plt.gcf().savefig(config.run_dir / "figures" / f"cbed_{iy}_{ix}.png",
                          dpi=160, bbox_inches="tight")
    return saved
