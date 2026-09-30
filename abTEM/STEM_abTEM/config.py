"""Central configuration for a small, manually launched abTEM STEM run."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
import json

PROJECT_ROOT = Path(__file__).resolve().parent

ABERRATION_UNITS = {name: ("rad" if name.startswith("phi") else "Å") for name in (
    "C10", "C12", "phi12", "C21", "phi21", "C23", "phi23",
    "C30", "C32", "phi32", "C34", "phi34", "C41", "phi41",
    "C43", "phi43", "C45", "phi45", "C50", "C52", "phi52",
    "C54", "phi54", "C56", "phi56",
)}


@dataclass
class AtomicConfig:
    hkl: tuple[int, int, int] = (0, 0, 1)
    repetitions: tuple[int, int, int] = (2, 2, 1)
    orthogonalize_max_repetitions: int = 5


@dataclass
class PotentialConfig:
    sampling_A: float = 0.05
    slice_thickness_A: float = 1.0
    parametrization: str = "lobato"
    projection: str = "finite"


@dataclass
class ProbeConfig:
    energy_eV: float = 200_000.0
    semiangle_cutoff_mrad: float = 20.0
    # The potential grid wins after match_grid(); these are optional constructor inputs.
    extent_A: tuple[float, float] | None = None
    gpts: tuple[int, int] | None = None
    sampling_A: float | None = None
    soft_aperture: bool = True  # abTEM Probe soft edge; rolloff follows angular grid sampling
    tilt_mrad: tuple[float, float] = (0.0, 0.0)
    device: str = "cpu"
    max_angle_mrad: float = 30.0  # visualization only, never passed to Probe
    # All unassigned aberrations use the documented zero default. C10 is the
    # abTEM defocus coefficient; positive physical defocus maps to negative C10.
    aberrations: dict[str, float] = field(default_factory=lambda: {
    "C10": -50.0,   # defocus，单位 Å
    "C12": 10.0,    # 二阶像散，单位 Å
    "phi12": 0.0,   # 方位角，单位 rad
    "C21": 0.0,
    "phi21": 0.0,
    "C23": 0.0,
    "phi23": 0.0,
    "C30": 0.0,  # spherical aberration，单位 Å
    "C32": 0.0,
    "phi32": 0.0,
    "C34": 0.0,
    "phi34": 0.0,
    "C41": 0.0,
    "phi41": 0.0,
    "C43": 0.0,
    "phi43": 0.0,
    "C45": 0.0,
    "phi45": 0.0,
    "C50": 0.0,
    "C52": 0.0,
    "phi52": 0.0,
    "C54": 0.0,
    "phi54": 0.0,
    "C56": 0.0,
    "phi56": 0.0,
})


@dataclass
class ScanConfig:
    # Fractional xy coordinates refer to the FINAL repeated potential cell.
    start: tuple[float, float] = (0.0, 0.0)
    end: tuple[float, float] = (1, 1)
    fractional: bool = True
    sampling_A: float | None = None  # None = probe.aperture.nyquist_sampling
    max_positions: int = 2106


@dataclass
class DetectorConfig:
    bf_mrad: tuple[float, float] = (0.0, 20.0)
    maadf_mrad: tuple[float, float] = (50.0, 120.0)
    haadf_mrad: tuple[float, float] = (90.0, 200.0)
    enable_4d_stem: bool = False
    pixelated_max_angle_mrad: float = 200.0
    cbed_scan_indices: tuple[tuple[int, int], ...] = ((0, 0),)


@dataclass
class PostprocessingConfig:
    enabled: bool = False
    interpolation_sampling_A: float = 0.1
    interpolation_method: str = "spline"  # small nonperiodic scan
    gaussian_sigma_A: float = 0.3
    gaussian_boundary: str = "reflect"
    dose_e_per_A2: float = 100_000.0
    random_seed: int = 100


@dataclass
class Config:
    project_root: Path = PROJECT_ROOT
    input_cif: Path = PROJECT_ROOT / "data" / "input" / "PbS.cif"
    output_root: Path = PROJECT_ROOT / "data" / "output"
    sample_name: str = "PbS"
    run_id: str = field(default_factory=lambda: datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    atomic: AtomicConfig = field(default_factory=AtomicConfig)
    potential: PotentialConfig = field(default_factory=PotentialConfig)
    probe: ProbeConfig = field(default_factory=ProbeConfig)
    scan: ScanConfig = field(default_factory=ScanConfig)
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    postprocessing: PostprocessingConfig = field(default_factory=PostprocessingConfig)

    @property
    def run_dir(self) -> Path:
        hkl = "".join(str(i) for i in self.atomic.hkl)
        return self.output_root / self.sample_name / hkl / self.run_id

    def prepare_output(self) -> Path:
        if not self.input_cif.is_file():
            raise FileNotFoundError(f"Input CIF missing: {self.input_cif}")
        for name in ("figures", "measurements", "metadata", "logs"):
            (self.run_dir / name).mkdir(parents=True, exist_ok=True)
        if self.detector.enable_4d_stem:
            (self.run_dir / "4d_stem").mkdir(exist_ok=True)
        self.save_metadata("effective_config", asdict(self))
        return self.run_dir

    def save_metadata(self, name: str, data: dict) -> Path:
        target = self.run_dir / "metadata" / f"{name}.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
        return target
