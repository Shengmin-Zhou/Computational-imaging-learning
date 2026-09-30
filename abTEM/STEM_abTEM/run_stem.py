"""Command-line Python entry point for the modular abTEM STEM workflow.

Examples
--------
Preparation and diagnostic plots only (no scan computation)::

    python run_stem.py

Run the configured small BF/MAADF/HAADF scan::

    python run_stem.py --run-scan

Run the scan and optional postprocessing::

    python run_stem.py --run-scan --postprocess

Enable raw 4D-STEM output and selected CBED plots::

    python run_stem.py --enable-4d-stem --run-4d

The input CIF is never modified. Every invocation creates a new run directory.
"""
from __future__ import annotations

import argparse

from config import Config
from modules.atomic_model import build_atomic_model, print_transform, show_projections
from modules.potential import build_potential, show_potential
from modules.wave_function import build_probe, show_probe
from modules.scan_detector import (
    configure_scan,
    run_4d_scan,
    run_annular_scan,
    show_scan_and_detectors,
)
from modules.postprocessing import process_measurements


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-scan",
        action="store_true",
        help="compute and save the small BF/MAADF/HAADF scan",
    )
    parser.add_argument(
        "--postprocess",
        action="store_true",
        help="interpolate, blur, and add quantitative Poisson noise after --run-scan",
    )
    parser.add_argument(
        "--enable-4d-stem",
        action="store_true",
        help="create a PixelatedDetector and the 4D-STEM output directory",
    )
    parser.add_argument(
        "--run-4d",
        action="store_true",
        help="compute/save raw pixelated data; implies --enable-4d-stem",
    )
    return parser


def run(config: Config, *, run_scan: bool = False, postprocess: bool = False,
        run_4d: bool = False):
    """Execute the selected workflow stages and return stage objects.

    This function is intentionally explicit so it can also be imported and
    called from another Python driver on a cluster.
    """
    if postprocess:
        config.postprocessing.enabled = True
    run_dir = config.prepare_output()
    print(f"Input CIF: {config.input_cif}")
    print(f"Run output: {run_dir}")

    atoms_ortho, atomic_metadata = build_atomic_model(config)
    config.save_metadata("atomic_model", atomic_metadata)
    print(f"Atomic model: {atomic_metadata['atom_count']} atoms; PBC={atomic_metadata['pbc']}")
    print_transform(atomic_metadata)
    show_projections(atoms_ortho, config)

    potential, potential_metadata = build_potential(atoms_ortho, config)
    config.save_metadata("potential", potential_metadata)
    print(f"Potential: {potential_metadata['num_slices']} slices, "
          f"thickness={potential_metadata['total_thickness_A']:.3f} Å")
    show_potential(potential, config)

    probe, waves, probe_metadata = build_probe(potential, config)
    config.save_metadata("probe", probe_metadata)
    print(f"Probe grid: {probe_metadata['effective_gpts']} points, "
          f"sampling={probe_metadata['effective_sampling_A']} Å")
    show_probe(waves, config)

    scan, detectors, pixel_detector, scan_metadata = configure_scan(
        potential, probe, atoms_ortho, config
    )
    config.save_metadata("scan_detector", scan_metadata)
    show_scan_and_detectors(scan, detectors, atoms_ortho, probe, config)

    measurements = None
    processing = None
    pixelated = None
    if run_scan:
        measurements = run_annular_scan(probe, potential, scan, detectors, config)
        if postprocess:
            processing = process_measurements(measurements, config)
            config.save_metadata("postprocessing", processing[1])
    elif postprocess:
        raise ValueError("--postprocess requires --run-scan")

    if run_4d:
        pixelated = run_4d_scan(probe, potential, scan, pixel_detector, config)

    return {
        "config": config,
        "run_dir": run_dir,
        "atoms": atoms_ortho,
        "potential": potential,
        "probe": probe,
        "waves": waves,
        "scan": scan,
        "detectors": detectors,
        "measurements": measurements,
        "postprocessing": processing,
        "pixelated": pixelated,
    }


def main() -> None:
    args = make_parser().parse_args()
    if args.run_4d:
        args.enable_4d_stem = True

    config = Config()
    config.detector.enable_4d_stem = args.enable_4d_stem
    result = run(
        config,
        run_scan=args.run_scan,
        postprocess=args.postprocess,
        run_4d=args.run_4d,
    )
    print(f"Completed. Outputs are in {result['run_dir']}")
    if not args.run_scan and not args.run_4d:
        print("No scan was computed. Use --run-scan for BF/MAADF/HAADF data.")


if __name__ == "__main__":
    main()


# ---------------------------------------------------------------------------
# Linux cluster test commands (manual execution; not run on import)
# ---------------------------------------------------------------------------
#
# Activate the existing environment, then run the preparation/diagnostic stage:
#
#   conda activate <abtem-1.0.6-environment>
#   cd /path/to/stem_abTEM
#   python run_stem.py
#
# Run the deliberately small BF/MAADF/HAADF STEM scan:
#
#   python run_stem.py --run-scan
#
# Run the scan followed by interpolation, Gaussian blur, and Poisson noise:
#
#   python run_stem.py --run-scan --postprocess
#
# Enable and run raw 4D-STEM pixelated diffraction data:
#
#   python run_stem.py --enable-4d-stem --run-4d
#
# The commands above create a new data/output/<sample>/<hkl>/<run_id>/ directory
# for each invocation. No pytest, notebook execution, or simulation is launched
# automatically by this source-file comment block.
