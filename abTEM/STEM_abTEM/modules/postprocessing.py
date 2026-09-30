"""Quantitative annular-image processing without intensity rescaling."""
from __future__ import annotations


def _materialize_image(image):
    """Return a CPU numpy image for plotting without changing measurement data."""
    computed = image.compute() if getattr(image, "is_lazy", False) else image
    if hasattr(computed, "to_cpu"):
        computed = computed.to_cpu()
    array = computed.array if hasattr(computed, "array") else computed
    return array.compute() if hasattr(array, "compute") else array


def show_postprocessing_summary(stages, config):
    """Save one figure containing all detectors and all processing stages."""
    import matplotlib.pyplot as plt
    import numpy as np

    detector_names = ("BF", "MAADF", "HAADF")
    stage_names = ("raw", "interpolated", "blurred", "noisy")
    fig, axes = plt.subplots(3, 4, figsize=(15, 10), constrained_layout=True)
    for row, detector_name in enumerate(detector_names):
        arrays = [np.asarray(_materialize_image(stages[detector_name][stage])).squeeze()
                  for stage in stage_names]
        if any(array.ndim != 2 for array in arrays):
            raise ValueError(f"Expected 2D {detector_name} measurements for summary plot")
        finite = np.concatenate([array[np.isfinite(array)].ravel() for array in arrays])
        vmin, vmax = np.percentile(finite, (1, 99))
        if vmax <= vmin:
            vmin, vmax = float(finite.min()), float(finite.max() + 1.0)
        for col, (stage_name, array) in enumerate(zip(stage_names, arrays)):
            ax = axes[row, col]
            image = ax.imshow(array, origin="lower", cmap="inferno", vmin=vmin, vmax=vmax)
            ax.set_title(f"{detector_name} — {stage_name}")
            ax.set_xticks([])
            ax.set_yticks([])
            fig.colorbar(image, ax=ax, fraction=0.046, pad=0.03)
    fig.suptitle("BF / MAADF / HAADF postprocessing summary\n"
                 "raw → interpolated → blurred → Poisson noisy")
    path = config.run_dir / "figures" / "postprocessed_summary.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    return path


def process_measurements(measurements, config):
    p = config.postprocessing
    if p.interpolation_sampling_A <= 0 or p.gaussian_sigma_A < 0 or p.dose_e_per_A2 <= 0:
        raise ValueError("Interpolation sampling and dose must be positive; blur cannot be negative.")
    stages = {}
    for name in ("BF", "MAADF", "HAADF"):
        raw = measurements[name]
        interpolated = raw.interpolate(sampling=p.interpolation_sampling_A,
                                        method=p.interpolation_method)
        blurred = interpolated.gaussian_filter(sigma=p.gaussian_sigma_A,
                                               boundary=p.gaussian_boundary)
        # abTEM uses quantitative fraction of incident intensity; no min-max
        # normalization is applied before the Poisson electron-count draw.
        noisy = blurred.poisson_noise(dose_per_area=p.dose_e_per_A2,
                                      seed=p.random_seed)
        stages[name] = {"raw": raw, "interpolated": interpolated,
                        "blurred": blurred, "noisy": noisy}
        for stage, image in stages[name].items():
            if stage != "raw":
                image.to_zarr(str(config.run_dir / "measurements" / f"{name}_{stage}.zarr"))
                image.show()
                import matplotlib.pyplot as plt
                plt.gcf().savefig(config.run_dir / "figures" / f"{name}_{stage}.png",
                                  dpi=160, bbox_inches="tight")
    summary_path = show_postprocessing_summary(stages, config)
    metadata = {
        "interpolation_sampling_A": p.interpolation_sampling_A,
        "interpolation_method": p.interpolation_method,
        "gaussian_sigma_A": p.gaussian_sigma_A,
        "gaussian_boundary": p.gaussian_boundary,
        "dose_e_per_A2": p.dose_e_per_A2,
        "random_seed": p.random_seed,
        "normalization_before_noise": "none",
        "summary_figure": str(summary_path),
    }
    return stages, metadata
