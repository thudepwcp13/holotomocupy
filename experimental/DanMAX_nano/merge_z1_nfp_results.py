#!/usr/bin/env python3
"""Merge DanMAX nano-NFP results from a z1 parameter scan.

Expected directory layout
-------------------------
ROOT/
  z1_0p12610/DanMAX_nano_nfp_results.h5
  z1_0p12620/DanMAX_nano_nfp_results.h5
  ...

The script streams one z1 result at a time into the output HDF5 file, so it
never stacks all 3712 x 3712 arrays in RAM.

Examples
--------
python merge_z1_nfp_results_probe_projection_v3.py \
    /zhome/64/c/214423/BioToBank/raw_data_extern/XHIST/output/z1_scan_step0 \
    --z1-range 0.12610:0.12710:0.00010 \
    --output merged_z1_results.h5 \
    --overwrite

By default, --z1-range follows Python/NumPy half-open semantics: stop is not
included. Add --include-stop when the final z1 value should be included.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
import tempfile
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Iterable, Sequence

import h5py
import numpy as np


SCRIPT_VERSION = "2026-07-20-probe-projection-split-v3"
INPUT_FILENAME_DEFAULT = "DanMAX_nano_nfp_results.h5"
SHARED_KEYS = (
    "corrected_preview",
    "flat_minus_dark",
    "tom_sam_x",
    "tom_y",
    "valid_detector_mask",
)
POSITION_KEYS = ("pos", "pos_err")
STACKED_KEYS = ("prb_amp", "prb_phase", "proj_beta", "proj_delta")

# The requested pixel range is 2001..3200 inclusive in one-based indexing.
# In Python indexing this is [2000:3200], giving exactly 1200 pixels.
DEFAULT_ROI = "2000:3200"


@dataclass(frozen=True)
class Z1Entry:
    value: Decimal
    directory: Path
    h5_path: Path


@dataclass(frozen=True)
class SliceSpec:
    start: int
    stop: int

    @property
    def size(self) -> int:
        return self.stop - self.start

    @property
    def as_slice(self) -> slice:
        return slice(self.start, self.stop)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge z1-scan DanMAX nano-NFP HDF5 result files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("root", type=Path, help="Root directory containing z1_* subdirectories.")
    parser.add_argument(
        "--z1-range",
        required=True,
        metavar="START:STOP:STEP",
        help="z1 values using Decimal arithmetic; stop is excluded unless --include-stop is set.",
    )
    parser.add_argument(
        "--include-stop",
        action="store_true",
        help="Include STOP when it lies exactly on the z1 grid.",
    )
    parser.add_argument(
        "--input-name",
        default=INPUT_FILENAME_DEFAULT,
        help="Input HDF5 filename inside each z1_* directory.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output HDF5 path. Relative paths are resolved under ROOT.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite an existing output file.")
    parser.add_argument(
        "--roi-y",
        default=DEFAULT_ROI,
        metavar="START:STOP",
        help="Zero-based half-open y slice for proj_*_roi.",
    )
    parser.add_argument(
        "--roi-x",
        default=DEFAULT_ROI,
        metavar="START:STOP",
        help="Zero-based half-open x slice for proj_*_roi.",
    )
    parser.add_argument(
        "--compression",
        choices=("lzf", "gzip", "none"),
        default="lzf",
        help="Compression for large output datasets. LZF is much faster than gzip.",
    )
    parser.add_argument(
        "--gzip-level",
        type=int,
        default=4,
        choices=range(0, 10),
        metavar="0..9",
        help="gzip compression level when --compression=gzip.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=512,
        help="Spatial chunk edge for stacked image datasets.",
    )
    parser.add_argument(
        "--verify-shared",
        action="store_true",
        help="Fully compare shared arrays in every source file against the first file.",
    )
    parser.add_argument(
        "--flat-pred-mode",
        choices=("auto", "copy", "propagate"),
        default="auto",
        help=(
            "How to obtain flat_pred_amplitude. auto first copies an existing prediction dataset, "
            "then propagates prb_amp*exp(i*prb_phase) if needed."
        ),
    )
    parser.add_argument(
        "--flat-pred-amplitude-key",
        default="flat_pred_amplitude",
        help="Preferred source key containing detector-plane predicted flat amplitude.",
    )
    parser.add_argument(
        "--flat-pred-intensity-key",
        default="flat_pred_intensity",
        help="Fallback source key containing detector-plane predicted flat intensity.",
    )
    parser.add_argument("--energy-kev", type=float, default=None, help="X-ray energy used for propagation.")
    parser.add_argument("--wavelength-m", type=float, default=None, help="X-ray wavelength used for propagation.")
    parser.add_argument("--distance-m", type=float, default=None, help="Effective propagation distance.")
    parser.add_argument("--pixel-size-m", type=float, default=None, help="Effective sample-plane pixel size.")
    parser.add_argument(
        "--propagation-device",
        choices=("auto", "cuda", "cpu"),
        default="auto",
        help="Torch device used only when flat prediction must be propagated.",
    )
    parser.add_argument(
        "--no-propagation-padding",
        action="store_true",
        help="Disable the 2x holotomocupy-style symmetric padding during propagation.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Resolve files and validate dataset shapes without creating output.",
    )
    return parser.parse_args()


def parse_decimal_range(spec: str, include_stop: bool) -> list[Decimal]:
    fields = spec.split(":")
    if len(fields) != 3:
        raise ValueError(f"Invalid --z1-range {spec!r}; expected START:STOP:STEP")
    try:
        start, stop, step = (Decimal(item.strip()) for item in fields)
    except InvalidOperation as exc:
        raise ValueError(f"Invalid decimal value in --z1-range {spec!r}") from exc

    if step == 0:
        raise ValueError("z1 step cannot be zero")
    if start < stop and step < 0:
        raise ValueError("z1 step must be positive when START < STOP")
    if start > stop and step > 0:
        raise ValueError("z1 step must be negative when START > STOP")

    values: list[Decimal] = []
    value = start
    comparator = (lambda x: x <= stop) if include_stop and step > 0 else None
    if include_stop and step < 0:
        comparator = lambda x: x >= stop
    if comparator is None:
        comparator = (lambda x: x < stop) if step > 0 else (lambda x: x > stop)

    # Protect against accidental gigantic ranges.
    max_count = 1_000_000
    while comparator(value):
        values.append(value)
        if len(values) > max_count:
            raise ValueError("z1 range contains more than 1,000,000 values; check the step")
        value += step

    if not values:
        raise ValueError("z1 range is empty")
    return values


def parse_slice_spec(text: str, name: str) -> SliceSpec:
    fields = text.split(":")
    if len(fields) != 2:
        raise ValueError(f"{name} must have format START:STOP")
    try:
        start, stop = map(int, fields)
    except ValueError as exc:
        raise ValueError(f"{name} must contain integer indices") from exc
    if start < 0 or stop <= start:
        raise ValueError(f"Invalid {name}={text!r}")
    return SliceSpec(start, stop)


def decode_z1_dir_name(name: str) -> Decimal | None:
    if not name.startswith("z1_"):
        return None
    token = name[3:]
    # Common scan naming: 0.12610 -> 0p12610; support m0p... for negatives.
    sign = ""
    if token.startswith("m"):
        sign = "-"
        token = token[1:]
    token = token.replace("p", ".")
    try:
        return Decimal(sign + token)
    except InvalidOperation:
        return None


def resolve_entries(root: Path, values: Sequence[Decimal], input_name: str) -> list[Z1Entry]:
    if not root.is_dir():
        raise FileNotFoundError(f"Root directory does not exist: {root}")

    available: dict[Decimal, list[Path]] = {}
    for child in root.iterdir():
        if not child.is_dir():
            continue
        value = decode_z1_dir_name(child.name)
        if value is not None:
            available.setdefault(value, []).append(child)

    entries: list[Z1Entry] = []
    missing: list[str] = []
    ambiguous: list[str] = []
    for value in values:
        candidates = available.get(value, [])
        if not candidates:
            missing.append(str(value))
            continue
        if len(candidates) > 1:
            ambiguous.append(f"{value}: {[p.name for p in candidates]}")
            continue
        directory = candidates[0]
        h5_path = directory / input_name
        if not h5_path.is_file():
            missing.append(f"{value} ({h5_path} missing)")
            continue
        entries.append(Z1Entry(value=value, directory=directory, h5_path=h5_path))

    if ambiguous:
        raise RuntimeError("Ambiguous z1 directories:\n  " + "\n  ".join(ambiguous))
    if missing:
        raise FileNotFoundError(
            "Missing requested z1 result(s):\n  " + "\n  ".join(missing)
            + "\nAvailable parsed z1 values: "
            + ", ".join(str(v) for v in sorted(available))
        )
    return entries


def require_dataset(h5: h5py.File, key: str) -> h5py.Dataset:
    if key not in h5:
        raise KeyError(f"Missing dataset {key!r} in {h5.filename}")
    obj = h5[key]
    if not isinstance(obj, h5py.Dataset):
        raise TypeError(f"{key!r} in {h5.filename} is not an HDF5 dataset")
    return obj


def shape_2d(dataset: h5py.Dataset, key: str) -> tuple[int, int]:
    shape = tuple(dataset.shape)
    if len(shape) == 2:
        return int(shape[0]), int(shape[1])
    if len(shape) == 3 and shape[0] == 1:
        return int(shape[1]), int(shape[2])
    raise ValueError(f"Dataset {key!r} in {dataset.file.filename} must be [H,W] or [1,H,W], got {shape}")


def read_2d(dataset: h5py.Dataset, key: str, dtype: np.dtype | type | None = None) -> np.ndarray:
    if dataset.ndim == 2:
        array = dataset[...]
    elif dataset.ndim == 3 and dataset.shape[0] == 1:
        array = dataset[0]
    else:
        raise ValueError(
            f"Dataset {key!r} in {dataset.file.filename} must be [H,W] or [1,H,W], got {dataset.shape}"
        )
    if dtype is not None:
        array = np.asarray(array, dtype=dtype)
    return np.asarray(array)


def copy_dataset_verbatim(src: h5py.Dataset, dst_file: h5py.File, key: str) -> h5py.Dataset:
    # h5py's copy preserves dtype, shape, attributes, chunks, and compression where possible.
    src.file.copy(src, dst_file, name=key)
    return dst_file[key]


def dataset_creation_kwargs(
    shape: tuple[int, ...],
    dtype: np.dtype,
    compression: str,
    gzip_level: int,
    chunk_size: int,
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {}
    if len(shape) == 3:
        _, height, width = shape
        kwargs["chunks"] = (1, min(chunk_size, height), min(chunk_size, width))
    elif len(shape) == 2:
        height, width = shape
        kwargs["chunks"] = (min(chunk_size, height), min(chunk_size, width))

    if compression == "lzf":
        kwargs["compression"] = "lzf"
        kwargs["shuffle"] = True
    elif compression == "gzip":
        kwargs["compression"] = "gzip"
        kwargs["compression_opts"] = gzip_level
        kwargs["shuffle"] = True
    return kwargs


def source_flat_prediction_kind(
    h5: h5py.File,
    amplitude_key: str,
    intensity_key: str,
) -> tuple[str, str] | None:
    amplitude_candidates = (
        amplitude_key,
        "pred_flat_amplitude",
        "flat_pred_amp",
        "predicted_flat_amplitude",
    )
    intensity_candidates = (
        intensity_key,
        "pred_flat_intensity",
        "predicted_flat_intensity",
    )
    for key in amplitude_candidates:
        if key in h5 and isinstance(h5[key], h5py.Dataset):
            return "amplitude", key
    for key in intensity_candidates:
        if key in h5 and isinstance(h5[key], h5py.Dataset):
            return "intensity", key
    return None


def scalar_from_h5(h5: h5py.File, candidates: Sequence[str]) -> float | None:
    """Find a scalar in root attrs or scalar datasets using common names."""
    for key in candidates:
        if key in h5.attrs:
            value = np.asarray(h5.attrs[key])
            if value.size == 1:
                return float(value.reshape(-1)[0])
    for key in candidates:
        if key in h5 and isinstance(h5[key], h5py.Dataset):
            value = np.asarray(h5[key][...])
            if value.size == 1:
                return float(value.reshape(-1)[0])
    return None


def resolve_propagation_parameters(h5: h5py.File, args: argparse.Namespace) -> tuple[float, float, float]:
    wavelength = args.wavelength_m
    if wavelength is None and args.energy_kev is not None:
        wavelength = 1.2398419840550367e-9 / args.energy_kev
    if wavelength is None:
        wavelength = scalar_from_h5(
            h5,
            ("wavelength_m", "wavelength", "lambda_m", "xray_wavelength_m"),
        )
    if wavelength is None:
        energy = scalar_from_h5(h5, ("energy_kev", "energy_keV", "xray_energy_kev"))
        if energy is not None:
            wavelength = 1.2398419840550367e-9 / energy

    distance = args.distance_m
    if distance is None:
        distance = scalar_from_h5(
            h5,
            ("effective_distance_m", "distance_m", "propagation_distance_m", "z_eff_m", "z_eff"),
        )

    pixel_size = args.pixel_size_m
    if pixel_size is None:
        pixel_size = scalar_from_h5(
            h5,
            ("effective_pixel_size_m", "pixel_size_m", "sample_pixel_size_m", "pixel_size"),
        )

    missing = []
    if wavelength is None:
        missing.append("wavelength (--wavelength-m or --energy-kev)")
    if distance is None:
        missing.append("effective distance (--distance-m)")
    if pixel_size is None:
        missing.append("effective pixel size (--pixel-size-m)")
    if missing:
        raise ValueError(
            "Cannot propagate probe because these parameters were not found: " + ", ".join(missing)
        )
    return float(wavelength), float(distance), float(pixel_size)


def torch_fresnel_amplitude(
    prb_amp: np.ndarray,
    prb_phase: np.ndarray,
    wavelength: float,
    distance: float,
    pixel_size: float,
    device_request: str,
    padding: bool,
) -> np.ndarray:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "PyTorch is required for --flat-pred-mode=propagate. "
            "Install torch or store flat_pred_amplitude in each source HDF5 file."
        ) from exc

    if device_request == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--propagation-device=cuda requested, but CUDA is unavailable")
        device = torch.device("cuda")
    elif device_request == "cpu":
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    amp = torch.from_numpy(np.asarray(prb_amp, dtype=np.float32)).to(device)
    phase = torch.from_numpy(np.asarray(prb_phase, dtype=np.float32)).to(device)
    wave = torch.polar(amp, phase).to(torch.complex64)[None, None]
    del amp, phase

    h, w = wave.shape[-2:]
    with torch.inference_mode():
        if padding:
            iy = torch.arange(2 * h, device=device)
            ix = torch.arange(2 * w, device=device)
            mapped_y = torch.empty_like(iy)
            mapped_x = torch.empty_like(ix)

            left = iy < h // 2
            center = (iy >= h // 2) & (iy < h + h // 2)
            right = iy >= h + h // 2
            mapped_y[left] = h // 2 - iy[left] - 1
            mapped_y[center] = iy[center] - h // 2
            mapped_y[right] = 2 * h - iy[right] + h // 2 - 1

            left = ix < w // 2
            center = (ix >= w // 2) & (ix < w + w // 2)
            right = ix >= w + w // 2
            mapped_x[left] = w // 2 - ix[left] - 1
            mapped_x[center] = ix[center] - w // 2
            mapped_x[right] = 2 * w - ix[right] + w // 2 - 1

            wave_work = wave.index_select(-2, mapped_y.long()).index_select(-1, mapped_x.long())
        else:
            wave_work = wave

        hp, wp = wave_work.shape[-2:]
        fy = torch.fft.fftfreq(hp, d=pixel_size, device=device)
        fx = torch.fft.fftfreq(wp, d=pixel_size, device=device)
        # Avoid storing both FX and FY as full 2D arrays.
        phase_kernel = -math.pi * wavelength * distance * (
            fy[:, None] ** 2 + fx[None, :] ** 2
        )
        kernel = torch.polar(torch.ones_like(phase_kernel), phase_kernel).to(torch.complex64)
        del phase_kernel, fy, fx

        propagated = torch.fft.ifft2(torch.fft.fft2(wave_work, dim=(-2, -1)) * kernel, dim=(-2, -1))
        del kernel, wave_work
        if padding:
            propagated = propagated[..., h // 2 : h // 2 + h, w // 2 : w // 2 + w]
        amplitude = torch.abs(propagated)[0, 0].float().cpu().numpy()

    del wave, propagated
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return np.asarray(amplitude, dtype=np.float32)


def obtain_flat_pred_amplitude(
    h5: h5py.File,
    prb_amp: np.ndarray,
    prb_phase: np.ndarray,
    args: argparse.Namespace,
) -> tuple[np.ndarray, str]:
    source = source_flat_prediction_kind(
        h5,
        args.flat_pred_amplitude_key,
        args.flat_pred_intensity_key,
    )

    if args.flat_pred_mode in ("auto", "copy") and source is not None:
        kind, key = source
        data = read_2d(require_dataset(h5, key), key, np.float32)
        if kind == "intensity":
            np.maximum(data, 0.0, out=data)
            np.sqrt(data, out=data)
        return data, f"source:{key} ({kind})"

    if args.flat_pred_mode == "copy":
        raise KeyError(
            f"No detector flat prediction dataset found in {h5.filename}; "
            "use --flat-pred-mode=propagate and supply propagation parameters"
        )

    wavelength, distance, pixel_size = resolve_propagation_parameters(h5, args)
    result = torch_fresnel_amplitude(
        prb_amp=prb_amp,
        prb_phase=prb_phase,
        wavelength=wavelength,
        distance=distance,
        pixel_size=pixel_size,
        device_request=args.propagation_device,
        padding=not args.no_propagation_padding,
    )
    method = (
        f"fresnel_propagation(wavelength_m={wavelength:.12g}, distance_m={distance:.12g}, "
        f"pixel_size_m={pixel_size:.12g}, padding={not args.no_propagation_padding})"
    )
    return result, method


def arrays_equal(a: h5py.Dataset, b: h5py.Dataset) -> bool:
    if a.shape != b.shape or a.dtype != b.dtype:
        return False
    arr_a = a[...]
    arr_b = b[...]
    if np.issubdtype(arr_a.dtype, np.floating):
        return bool(np.allclose(arr_a, arr_b, rtol=1e-6, atol=1e-7, equal_nan=True))
    return bool(np.array_equal(arr_a, arr_b))


def validate_all_sources(
    entries: Sequence[Z1Entry],
    roi_y: SliceSpec,
    roi_x: SliceSpec,
    args: argparse.Namespace,
) -> dict[str, Any]:
    reference: dict[str, Any] = {}
    first_shared: dict[str, h5py.Dataset] = {}

    with h5py.File(entries[0].h5_path, "r") as first:
        for key in SHARED_KEYS:
            first_shared[key] = require_dataset(first, key)
        for key in POSITION_KEYS:
            ds = require_dataset(first, key)
            if ds.ndim != 2 or ds.shape[1] != 2:
                raise ValueError(f"{key!r} must have shape [M,2], got {ds.shape} in {first.filename}")
            reference[f"{key}_shape"] = tuple(ds.shape)
            reference[f"{key}_dtype"] = ds.dtype
        for key in STACKED_KEYS:
            ds = require_dataset(first, key)
            reference[f"{key}_shape"] = shape_2d(ds, key)
            reference[f"{key}_dtype"] = ds.dtype

        # Probe/detector-plane arrays and object-plane projections do not have to
        # share the same spatial grid.  In the DanMAX result files, for example,
        # prb_* can be 3712x3712 while proj_* is 5152x5152.
        probe_shape = reference["prb_amp_shape"]
        projection_shape = reference["proj_beta_shape"]

        if reference["prb_phase_shape"] != probe_shape:
            raise ValueError(
                f"prb_phase shape {reference['prb_phase_shape']} != prb_amp shape {probe_shape}"
            )
        if reference["proj_delta_shape"] != projection_shape:
            raise ValueError(
                f"proj_delta shape {reference['proj_delta_shape']} "
                f"!= proj_beta shape {projection_shape}"
            )

        # The requested ROI is taken from proj_beta/proj_delta, so validate it
        # against the projection grid rather than the probe grid.
        if roi_y.stop > projection_shape[0] or roi_x.stop > projection_shape[1]:
            raise ValueError(
                f"ROI y={roi_y.start}:{roi_y.stop}, x={roi_x.start}:{roi_x.stop} "
                f"exceeds projection shape {projection_shape}"
            )

        # flat_minus_dark and flat predictions are detector-plane quantities and
        # must therefore match the probe grid.
        flat_shape = shape_2d(require_dataset(first, "flat_minus_dark"), "flat_minus_dark")
        if flat_shape != probe_shape:
            raise ValueError(f"flat_minus_dark shape {flat_shape} != probe shape {probe_shape}")
        reference["probe_shape"] = probe_shape
        reference["projection_shape"] = projection_shape
        reference["position_rows"] = reference["pos_shape"][0]

    # Reopen the first file during comparisons; h5py datasets cannot outlive their file.
    for index, entry in enumerate(entries):
        with h5py.File(entry.h5_path, "r") as h5:
            for key in SHARED_KEYS + POSITION_KEYS + STACKED_KEYS:
                require_dataset(h5, key)
            for key in POSITION_KEYS:
                ds = h5[key]
                if tuple(ds.shape) != reference[f"{key}_shape"]:
                    raise ValueError(
                        f"{key} shape mismatch at z1={entry.value}: {ds.shape} != {reference[f'{key}_shape']}"
                    )
            for key in STACKED_KEYS:
                current_shape = shape_2d(h5[key], key)
                if current_shape != reference[f"{key}_shape"]:
                    raise ValueError(
                        f"{key} shape mismatch at z1={entry.value}: {current_shape} != {reference[f'{key}_shape']}"
                    )

            if args.flat_pred_mode == "copy" and source_flat_prediction_kind(
                h5, args.flat_pred_amplitude_key, args.flat_pred_intensity_key
            ) is None:
                raise KeyError(f"No flat prediction dataset found at z1={entry.value} in {entry.h5_path}")

        if args.verify_shared and index > 0:
            with h5py.File(entries[0].h5_path, "r") as first, h5py.File(entry.h5_path, "r") as current:
                for key in SHARED_KEYS:
                    if not arrays_equal(first[key], current[key]):
                        raise ValueError(f"Shared dataset {key!r} differs at z1={entry.value}")

    return reference


def create_output_path(root: Path, requested: Path | None, range_spec: str) -> Path:
    if requested is None:
        safe_range = range_spec.replace(":", "_").replace(".", "p").replace("-", "m")
        return root / f"merged_z1_{safe_range}.h5"
    if requested.is_absolute():
        return requested
    return root / requested


def merge(entries: Sequence[Z1Entry], output: Path, reference: dict[str, Any], args: argparse.Namespace) -> None:
    n = len(entries)
    probe_height, probe_width = reference["probe_shape"]
    projection_height, projection_width = reference["projection_shape"]
    roi_y = parse_slice_spec(args.roi_y, "--roi-y")
    roi_x = parse_slice_spec(args.roi_x, "--roi-x")
    compression = args.compression

    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output exists: {output}; pass --overwrite to replace it")
        output.unlink()

    # Atomic write in the same directory, followed by os.replace.
    fd, temp_name = tempfile.mkstemp(prefix=output.name + ".tmp.", suffix=".h5", dir=output.parent)
    os.close(fd)
    temp_path = Path(temp_name)

    try:
        with h5py.File(temp_path, "w") as out, h5py.File(entries[0].h5_path, "r") as first:
            out.attrs["format"] = "DanMAX nano-NFP merged z1 scan"
            out.attrs["z1_range"] = args.z1_range
            out.attrs["z1_include_stop"] = bool(args.include_stop)
            out.attrs["z1_values"] = np.asarray([float(entry.value) for entry in entries], dtype=np.float64)
            out.attrs["z1_directory_names"] = np.asarray(
                [entry.directory.name for entry in entries], dtype=h5py.string_dtype("utf-8")
            )
            out.attrs["source_files"] = np.asarray(
                [str(entry.h5_path) for entry in entries], dtype=h5py.string_dtype("utf-8")
            )
            out.attrs["probe_grid_shape"] = np.asarray(
                [probe_height, probe_width], dtype=np.int64
            )
            out.attrs["projection_grid_shape"] = np.asarray(
                [projection_height, projection_width], dtype=np.int64
            )
            out.attrs["roi_y_python_slice"] = f"{roi_y.start}:{roi_y.stop}"
            out.attrs["roi_x_python_slice"] = f"{roi_x.start}:{roi_x.stop}"
            out.attrs["roi_description"] = (
                "Zero-based half-open Python slices. Defaults correspond to one-based pixels 2001..3200 inclusive."
            )

            for key in SHARED_KEYS:
                copy_dataset_verbatim(require_dataset(first, key), out, key)

            # Target detector amplitude is shared by all z1 values.
            flat_minus_dark = read_2d(require_dataset(first, "flat_minus_dark"), "flat_minus_dark", np.float32)
            np.maximum(flat_minus_dark, 0.0, out=flat_minus_dark)
            np.sqrt(flat_minus_dark, out=flat_minus_dark)
            target_kwargs = dataset_creation_kwargs(
                flat_minus_dark.shape, flat_minus_dark.dtype, compression, args.gzip_level, args.chunk_size
            )
            flat_target_ds = out.create_dataset(
                "flat_target_amplitude", data=flat_minus_dark, **target_kwargs
            )
            flat_target_ds.attrs["definition"] = "sqrt(max(flat_minus_dark, 0))"
            del flat_minus_dark

            # Concatenate [M,2] -> [M,2N].
            for key in POSITION_KEYS:
                src = require_dataset(first, key)
                pos_ds = out.create_dataset(
                    key,
                    shape=(src.shape[0], 2 * n),
                    dtype=src.dtype,
                    chunks=(src.shape[0], min(2 * n, 256)),
                )
                for attr_name, attr_value in src.attrs.items():
                    pos_ds.attrs[attr_name] = attr_value
                pos_ds.attrs["merge_axis"] = 1
                pos_ds.attrs["column_layout"] = "[x(z1_0), y(z1_0), x(z1_1), y(z1_1), ...]"

            # Create each stacked dataset using its own native spatial grid.
            # prb_* lives on the probe/detector grid, whereas proj_* may live on
            # a larger object-plane grid.
            stacked_out: dict[str, h5py.Dataset] = {}
            for key in STACKED_KEYS:
                src = require_dataset(first, key)
                dtype = src.dtype
                key_height, key_width = reference[f"{key}_shape"]
                shape = (n, key_height, key_width)
                kwargs = dataset_creation_kwargs(shape, dtype, compression, args.gzip_level, args.chunk_size)
                stacked_out[key] = out.create_dataset(key, shape=shape, dtype=dtype, **kwargs)
                for attr_name, attr_value in src.attrs.items():
                    stacked_out[key].attrs[attr_name] = attr_value
                stacked_out[key].attrs["axis_0"] = "z1 index"

            float_kwargs = dataset_creation_kwargs(
                (n, probe_height, probe_width),
                np.dtype(np.float32),
                compression,
                args.gzip_level,
                args.chunk_size,
            )
            flat_pred_ds = out.create_dataset(
                "flat_pred_amplitude",
                shape=(n, probe_height, probe_width),
                dtype=np.float32,
                **float_kwargs,
            )
            residual_ds = out.create_dataset(
                "flat_amplitude_residual",
                shape=(n, probe_height, probe_width),
                dtype=np.float32,
                **float_kwargs,
            )
            flat_pred_ds.attrs["axis_0"] = "z1 index"
            residual_ds.attrs["definition"] = "flat_pred_amplitude - flat_target_amplitude"
            residual_ds.attrs["axis_0"] = "z1 index"

            roi_shape = (n, roi_y.size, roi_x.size)
            beta_roi_kwargs = dataset_creation_kwargs(
                roi_shape, require_dataset(first, "proj_beta").dtype, compression, args.gzip_level, args.chunk_size
            )
            delta_roi_kwargs = dataset_creation_kwargs(
                roi_shape, require_dataset(first, "proj_delta").dtype, compression, args.gzip_level, args.chunk_size
            )
            beta_roi_ds = out.create_dataset(
                "proj_beta_roi", shape=roi_shape, dtype=require_dataset(first, "proj_beta").dtype, **beta_roi_kwargs
            )
            delta_roi_ds = out.create_dataset(
                "proj_delta_roi", shape=roi_shape, dtype=require_dataset(first, "proj_delta").dtype, **delta_roi_kwargs
            )
            for ds in (beta_roi_ds, delta_roi_ds):
                ds.attrs["axis_0"] = "z1 index"
                ds.attrs["source_y_slice"] = f"{roi_y.start}:{roi_y.stop}"
                ds.attrs["source_x_slice"] = f"{roi_x.start}:{roi_x.stop}"

            prediction_methods: list[str] = []
            target = flat_target_ds  # Read chunks lazily from output when computing residual.

            for index, entry in enumerate(entries):
                print(f"[{index + 1:>3}/{n}] z1={entry.value}  {entry.h5_path}", flush=True)
                with h5py.File(entry.h5_path, "r") as src:
                    for key in POSITION_KEYS:
                        out[key][:, 2 * index : 2 * index + 2] = require_dataset(src, key)[...]

                    # Read each large 2D array at most once for full stack output.
                    prb_amp = read_2d(require_dataset(src, "prb_amp"), "prb_amp")
                    prb_phase = read_2d(require_dataset(src, "prb_phase"), "prb_phase")
                    stacked_out["prb_amp"][index] = prb_amp
                    stacked_out["prb_phase"][index] = prb_phase

                    pred_amp, method = obtain_flat_pred_amplitude(src, prb_amp, prb_phase, args)
                    if pred_amp.shape != (probe_height, probe_width):
                        raise ValueError(
                            f"flat_pred_amplitude shape mismatch at z1={entry.value}: "
                            f"{pred_amp.shape} != {(probe_height, probe_width)}"
                        )
                    flat_pred_ds[index] = pred_amp
                    # h5py reads the target dataset as float32; only one residual image is held in RAM.
                    pred_amp -= target[...]
                    residual_ds[index] = pred_amp
                    prediction_methods.append(method)
                    del prb_amp, prb_phase, pred_amp

                    beta_src = require_dataset(src, "proj_beta")
                    delta_src = require_dataset(src, "proj_delta")
                    if beta_src.ndim == 2:
                        stacked_out["proj_beta"][index] = beta_src[...]
                        beta_roi_ds[index] = beta_src[roi_y.as_slice, roi_x.as_slice]
                    else:
                        stacked_out["proj_beta"][index] = beta_src[0]
                        beta_roi_ds[index] = beta_src[0, roi_y.as_slice, roi_x.as_slice]
                    if delta_src.ndim == 2:
                        stacked_out["proj_delta"][index] = delta_src[...]
                        delta_roi_ds[index] = delta_src[roi_y.as_slice, roi_x.as_slice]
                    else:
                        stacked_out["proj_delta"][index] = delta_src[0]
                        delta_roi_ds[index] = delta_src[0, roi_y.as_slice, roi_x.as_slice]

                out.flush()

            flat_pred_ds.attrs["generation_methods"] = np.asarray(
                prediction_methods, dtype=h5py.string_dtype("utf-8")
            )

        os.replace(temp_path, output)
    except Exception:
        temp_path.unlink(missing_ok=True)
        raise


def print_plan(entries: Sequence[Z1Entry], output: Path, reference: dict[str, Any], args: argparse.Namespace) -> None:
    n = len(entries)
    probe_h, probe_w = reference["probe_shape"]
    proj_h, proj_w = reference["projection_shape"]
    roi_y = parse_slice_spec(args.roi_y, "--roi-y")
    roi_x = parse_slice_spec(args.roi_x, "--roi-x")
    print("Resolved merge plan")
    print(f"  root:       {args.root.resolve()}")
    print(f"  z1 count:   {n}")
    print(f"  z1 first:   {entries[0].value}")
    print(f"  z1 last:    {entries[-1].value}")
    print(f"  probe grid: {probe_h} x {probe_w}")
    print(f"  proj grid:  {proj_h} x {proj_w}")
    print(f"  pos:        {reference['position_rows']} x {2 * n}")
    print(f"  ROI:        {roi_y.size} x {roi_x.size} (y={args.roi_y}, x={args.roi_x})")
    print(f"  output:     {output}")
    print(f"  compression:{args.compression}")
    print(f"  flat pred:  {args.flat_pred_mode}")


def main() -> int:
    args = parse_args()
    print(f"Script version: {SCRIPT_VERSION}", flush=True)
    try:
        values = parse_decimal_range(args.z1_range, args.include_stop)
        entries = resolve_entries(args.root.resolve(), values, args.input_name)
        roi_y = parse_slice_spec(args.roi_y, "--roi-y")
        roi_x = parse_slice_spec(args.roi_x, "--roi-x")
        reference = validate_all_sources(entries, roi_y, roi_x, args)
        output = create_output_path(args.root.resolve(), args.output, args.z1_range)
        print_plan(entries, output, reference, args)
        if args.dry_run:
            print("Dry run completed; no output file was created.")
            return 0
        merge(entries, output, reference, args)
        print(f"Done: {output}")
        return 0
    except (FileNotFoundError, FileExistsError, KeyError, TypeError, ValueError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
