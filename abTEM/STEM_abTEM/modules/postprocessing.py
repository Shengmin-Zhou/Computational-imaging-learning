"""Postprocessing and visualization of annular STEM measurements."""
from __future__ import annotations


def _materialize_image(image):
    """Return a CPU NumPy array without modifying the measurement."""
    computed = image.compute() if getattr(image, "is_lazy", False) else image

    if hasattr(computed, "to_cpu"):
        computed = computed.to_cpu()

    array = computed.array if hasattr(computed, "array") else computed

    if hasattr(array, "compute"):
        array = array.compute()

    return array


def _get_xy_axes(image, shape):
    """Return physical x/y coordinates from abTEM scan metadata when available."""
    import numpy as np

    axes_metadata = getattr(image, "ensemble_axes_metadata", [])

    x_axis = next(
        (axis for axis in axes_metadata if getattr(axis, "label", None) == "x"),
        None,
    )
    y_axis = next(
        (axis for axis in axes_metadata if getattr(axis, "label", None) == "y"),
        None,
    )

    if x_axis is None or y_axis is None:
        return None

    nx, ny = shape

    try:
        x = np.asarray(x_axis.coordinates(nx), dtype=float)
        y = np.asarray(y_axis.coordinates(ny), dtype=float)
    except Exception:
        dx = float(x_axis.sampling)
        dy = float(y_axis.sampling)
        x0 = float(getattr(x_axis, "offset", 0.0))
        y0 = float(getattr(y_axis, "offset", 0.0))

        x = x0 + np.arange(nx) * dx
        y = y0 + np.arange(ny) * dy

    return x, y


def _coordinate_extent(coordinates):
    """Convert pixel-center coordinates to imshow pixel-edge extent."""
    import numpy as np

    coordinates = np.asarray(coordinates, dtype=float)

    if len(coordinates) == 1:
        step = 1.0
    else:
        step = float(np.mean(np.diff(coordinates)))

    return (
        coordinates[0] - step / 2,
        coordinates[-1] + step / 2,
    )


def _prepare_for_imshow(image):
    """
    Convert an abTEM x-y scan array to Matplotlib's row(y)-column(x) convention.
    """
    import numpy as np

    array = np.asarray(_materialize_image(image)).squeeze()

    if array.ndim != 2:
        raise ValueError(f"Expected a 2D STEM image, got shape {array.shape}")

    # abTEM GridScan axes are (x, y); imshow expects array axes as (y, x).
    display_array = array.T

    xy = _get_xy_axes(image, array.shape)

    if xy is None:
        extent = None
    else:
        x, y = xy
        xmin, xmax = _coordinate_extent(x)
        ymin, ymax = _coordinate_extent(y)
        extent = (xmin, xmax, ymin, ymax)

    return display_array, extent


def _adaptive_limits(array, lower=1.0, upper=99.0):
    """Return robust display limits without modifying image intensity."""
    import numpy as np

    finite = np.asarray(array)
    finite = finite[np.isfinite(finite)]

    if finite.size == 0:
        raise ValueError("Image contains no finite values.")

    vmin, vmax = np.percentile(finite, (lower, upper))

    if vmax <= vmin:
        vmin = float(finite.min())
        vmax = float(finite.max())

        if vmax <= vmin:
            vmax = vmin + 1.0

    return float(vmin), float(vmax)


def _plot_image(ax, image, title, cmap):
    """Plot one abTEM scan image with correct x/y orientation."""
    array, extent = _prepare_for_imshow(image)
    vmin, vmax = _adaptive_limits(array)

    kwargs = dict(
        origin="lower",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        aspect="equal",
    )

    if extent is not None:
        kwargs["extent"] = extent

    plotted = ax.imshow(array, **kwargs)

    ax.set_title(title)

    if extent is not None:
        ax.set_xlabel("x (Å)")
        ax.set_ylabel("y (Å)")
    else:
        ax.set_xlabel("x pixel")
        ax.set_ylabel("y pixel")

    return plotted


def save_raw_images(measurements, config):
    """Save raw BF/MAADF/HAADF images as heatmaps and grayscale images."""
    import matplotlib.pyplot as plt

    for name in ("BF", "MAADF", "HAADF"):
        image = measurements[name]

        # Heatmap
        fig, ax = plt.subplots(figsize=(6, 5))
        plotted = _plot_image(
            ax,
            image,
            title=f"{name} — raw",
            cmap="inferno",
        )
        fig.colorbar(plotted, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        fig.savefig(
            config.run_dir / "figures" / f"{name}_raw.png",
            dpi=180,
            bbox_inches="tight",
        )
        plt.close(fig)

        # Grayscale image for direct visual comparison with experimental STEM.
        fig, ax = plt.subplots(figsize=(6, 5))
        plotted = _plot_image(
            ax,
            image,
            title=f"{name} — raw grayscale",
            cmap="gray",
        )
        fig.colorbar(plotted, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        fig.savefig(
            config.run_dir / "figures" / f"{name}_raw_gray.png",
            dpi=180,
            bbox_inches="tight",
        )
        plt.close(fig)


def show_postprocessing_summary(stages, config):
    """Save raw → interpolated → blurred → noisy comparison."""
    import matplotlib.pyplot as plt

    detector_names = ("BF", "MAADF", "HAADF")
    stage_names = ("raw", "interpolated", "blurred", "noisy")

    fig, axes = plt.subplots(
        3,
        4,
        figsize=(16, 11),
        constrained_layout=True,
    )

    for row, detector_name in enumerate(detector_names):
        for col, stage_name in enumerate(stage_names):
            ax = axes[row, col]
            image = stages[detector_name][stage_name]

            plotted = _plot_image(
                ax,
                image,
                title=f"{detector_name} — {stage_name}",
                cmap="inferno",
            )

            fig.colorbar(
                plotted,
                ax=ax,
                fraction=0.046,
                pad=0.03,
            )

    fig.suptitle(
        "BF / MAADF / HAADF postprocessing summary\n"
        "raw → interpolated → blurred → Poisson noisy",
        fontsize=14,
    )

    path = config.run_dir / "figures" / "postprocessed_summary.png"

    fig.savefig(
        path,
        dpi=180,
        bbox_inches="tight",
    )
    plt.show()
    plt.close(fig)

    return path


def process_measurements(measurements, config):
    p = config.postprocessing

    if (
        p.interpolation_sampling_A <= 0
        or p.gaussian_sigma_A < 0
        or p.dose_e_per_A2 <= 0
    ):
        raise ValueError(
            "Interpolation sampling and dose must be positive; "
            "blur cannot be negative."
        )

    save_raw_images(measurements, config)

    stages = {}

    for name in ("BF", "MAADF", "HAADF"):
        raw = measurements[name]

        interpolated = raw.interpolate(
            sampling=p.interpolation_sampling_A,
            method=p.interpolation_method,
        )

        blurred = interpolated.gaussian_filter(
            sigma=p.gaussian_sigma_A,
            boundary=p.gaussian_boundary,
        )

        # Keep quantitative detector intensity; no min-max normalization.
        noisy = blurred.poisson_noise(
            dose_per_area=p.dose_e_per_A2,
            seed=p.random_seed,
        )

        stages[name] = {
            "raw": raw,
            "interpolated": interpolated,
            "blurred": blurred,
            "noisy": noisy,
        }

        # Preserve numerical results; visualization is handled by the summary.
        for stage_name, image in stages[name].items():
            if stage_name == "raw":
                continue

            image.to_zarr(
                str(
                    config.run_dir
                    / "measurements"
                    / f"{name}_{stage_name}.zarr"
                )
            )

    summary_path = show_postprocessing_summary(stages, config)

    metadata = {
        "interpolation_sampling_A": p.interpolation_sampling_A,
        "interpolation_method": p.interpolation_method,
        "gaussian_sigma_A": p.gaussian_sigma_A,
        "gaussian_boundary": p.gaussian_boundary,
        "dose_e_per_A2": p.dose_e_per_A2,
        "random_seed": p.random_seed,
        "normalization_before_noise": "none",
        "summary_display_scaling": "independent 1st–99th percentile per panel",
        "matplotlib_axis_conversion": (
            "abTEM GridScan array axes (x, y) are transposed to "
            "Matplotlib image axes (row=y, column=x)"
        ),
        "summary_figure": str(summary_path),
    }

    return stages, metadata