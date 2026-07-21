#!/usr/bin/env python3
"""Compare z1-scan reconstructions using ring uniformity and resolution metrics.

This script is intended for the merged HDF5 file produced by
``merge_z1_nfp_results.py``.  It processes ``proj_beta_roi`` and
``proj_delta_roi`` one z1 slice at a time and writes CSV tables plus diagnostic
plots.

The implemented metrics are:

1. Annular spoke-uniformity metrics
   At configurable radii, the image is sampled on a circular annulus.  The
   periodic spoke signal is divided into individual spoke periods, and the
   local peak-to-trough amplitude is measured for every spoke.  The variance,
   coefficient of variation (CV), robust CV, and peak spread quantify how
   uniformly nominally identical radial bars are reconstructed.

2. Directional Siemens-star-like resolution
   The spoke harmonic is fitted as a function of radius in sectors on the
   left/right ("horizontal sides") and top/bottom ("vertical sides").  The
   innermost stable radius at which the harmonic remains detectable is
   converted into line-pair frequency, full period, and half-pitch.  Note that
   the left/right sectors mainly test tangential resolution along image y,
   whereas the top/bottom sectors mainly test tangential resolution along x.

3. Fourier power-spectrum resolution adapted from Modregger et al.
   Directional x/y and azimuthally averaged power spectra are calculated.  A
   high-frequency noise baseline is estimated, and the cutoff is the highest
   stable frequency where total spectral power reaches
   ``threshold_factor * noise_baseline`` (2 by default).  Frequencies are in
   cycles/pixel, so full-period resolution is 1/f and half-pitch is 1/(2f).

Example
-------
python compare_z1_roi_metrics.py merged_z1_results.h5 \
    --output-dir z1_roi_metrics \
    --keys proj_beta_roi proj_delta_roi \
    --center auto \
    --ring-radii auto \
    --ring-count 6 \
    --spoke-harmonic auto \
    --pixel-size 10 --pixel-unit nm

Using a center read from full-image coordinates (the merged file stores the
ROI origin):

python compare_z1_roi_metrics.py merged_z1_results.h5 \
    --center 2570,2540 --center-frame full \
    --ring-radii 70,110,160,220,300,400
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import h5py
import numpy as np

try:
    from scipy.ndimage import gaussian_filter1d, map_coordinates
    from scipy.signal import fftconvolve
    from scipy.stats import rankdata
except ImportError as exc:  # pragma: no cover - dependency error is explicit
    raise SystemExit(
        "This script requires SciPy. Install it with, for example, "
        "`python -m pip install scipy`."
    ) from exc


EPS = np.finfo(np.float64).eps


@dataclass(frozen=True)
class RangeSpec:
    start: float
    stop: float


@dataclass
class PSDResult:
    frequency: np.ndarray
    power: np.ndarray
    power_smooth: np.ndarray
    noise_baseline: float
    threshold: float
    cutoff_cyc_per_px: float
    cutoff_censored: bool


@dataclass
class StarDirectionResult:
    radii_px: np.ndarray
    amplitude: np.ndarray
    residual_rms: np.ndarray
    snr: np.ndarray
    amplitude_fraction: np.ndarray
    cutoff_radius_px: float
    cutoff_cyc_per_px: float
    cutoff_censored: bool


@dataclass(frozen=True)
class PolarSampler:
    radii_px: np.ndarray
    angles_rad: np.ndarray
    coordinates: np.ndarray

    @classmethod
    def create(
        cls,
        center_yx: tuple[float, float],
        radii_px: np.ndarray,
        n_angles: int,
    ) -> "PolarSampler":
        cy, cx = center_yx
        angles = np.linspace(0.0, 2.0 * np.pi, n_angles, endpoint=False, dtype=np.float64)
        radii = np.asarray(radii_px, dtype=np.float64)
        yy = cy + radii[:, None] * np.sin(angles)[None, :]
        xx = cx + radii[:, None] * np.cos(angles)[None, :]
        coordinates = np.stack((yy, xx), axis=0).astype(np.float32, copy=False)
        return cls(radii_px=radii, angles_rad=angles, coordinates=coordinates)

    def sample(self, image: np.ndarray) -> np.ndarray:
        return map_coordinates(
            image,
            self.coordinates,
            order=1,
            mode="nearest",
            prefilter=False,
        ).astype(np.float64, copy=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare z1-dependent proj_beta_roi/proj_delta_roi quality metrics.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("input", type=Path, help="Merged HDF5 file.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Default: <input_stem>_roi_metrics beside the input file.",
    )
    parser.add_argument(
        "--keys",
        nargs="+",
        default=("proj_beta_roi", "proj_delta_roi"),
        help="HDF5 datasets to analyze; each must have shape [N,H,W] or [H,W].",
    )
    parser.add_argument(
        "--z1-attr",
        default="z1_values",
        help="Root HDF5 attribute containing one z1 value per image.",
    )

    parser.add_argument(
        "--reference-key",
        default="proj_delta_roi",
        help="Dataset used for automatic center and spoke-harmonic estimation.",
    )
    parser.add_argument(
        "--reference-index",
        type=int,
        default=None,
        help="Use one z1 index as reference. By default a streaming mean over all z1 values is used.",
    )
    parser.add_argument(
        "--center",
        default="auto",
        metavar="auto|Y,X",
        help="Pattern center. Coordinates are interpreted according to --center-frame.",
    )
    parser.add_argument(
        "--center-frame",
        choices=("roi", "full"),
        default="roi",
        help="Whether an explicit --center is in ROI coordinates or original full-image coordinates.",
    )
    parser.add_argument(
        "--center-search-radius",
        type=int,
        default=120,
        help="Search radius around the ROI midpoint for automatic centrosymmetry fitting.",
    )
    parser.add_argument(
        "--center-downsample",
        type=int,
        default=4,
        help="Downsampling used for the coarse automatic center search.",
    )
    parser.add_argument(
        "--center-search-step",
        type=int,
        default=8,
        help="Coarse center-search step in full-resolution pixels.",
    )

    parser.add_argument(
        "--ring-radii",
        default="auto",
        metavar="auto|R1,R2,...",
        help="Annulus center radii in ROI pixels.",
    )
    parser.add_argument("--ring-count", type=int, default=6, help="Number of automatic annuli.")
    parser.add_argument(
        "--ring-radius-range",
        default="0.15:0.82",
        metavar="MIN_FRAC:MAX_FRAC",
        help="Automatic annulus range as fractions of the largest complete circle.",
    )
    parser.add_argument(
        "--ring-half-width",
        type=float,
        default=3.0,
        help="Half-width of each annulus in pixels.",
    )
    parser.add_argument(
        "--angular-samples",
        type=int,
        default=2048,
        help="Number of angular samples over 360 degrees.",
    )
    parser.add_argument(
        "--spoke-harmonic",
        default="auto",
        metavar="auto|INTEGER",
        help="Number of intensity periods/line pairs around a complete circle.",
    )
    parser.add_argument(
        "--harmonic-search",
        default="8:256",
        metavar="MIN:MAX",
        help="Integer harmonic search range for automatic spoke detection.",
    )
    parser.add_argument(
        "--harmonic-radius-range",
        default="0.28:0.78",
        metavar="MIN_FRAC:MAX_FRAC",
        help="Radial range used to estimate the common spoke harmonic.",
    )
    parser.add_argument(
        "--angular-background-periods",
        type=float,
        default=2.5,
        help="Gaussian background-removal width, in spoke periods, for ring uniformity.",
    )
    parser.add_argument(
        "--spoke-sample-width-fraction",
        type=float,
        default=0.24,
        help="Peak/trough averaging width as a fraction of one spoke period.",
    )

    parser.add_argument(
        "--star-radius-range",
        default="0.06:0.86",
        metavar="MIN_FRAC:MAX_FRAC",
        help="Radial range used for directional Siemens-star resolution.",
    )
    parser.add_argument(
        "--direction-half-width-deg",
        type=float,
        default=15.0,
        help="Half-width of each left/right or top/bottom angular sector.",
    )
    parser.add_argument(
        "--star-snr-threshold",
        type=float,
        default=2.0,
        help="Required fitted harmonic amplitude / residual RMS.",
    )
    parser.add_argument(
        "--star-amplitude-fraction",
        type=float,
        default=0.20,
        help="Required amplitude relative to the median outer-radius amplitude.",
    )
    parser.add_argument(
        "--star-smooth-sigma",
        type=float,
        default=2.0,
        help="Radial Gaussian smoothing sigma in pixels for star curves.",
    )
    parser.add_argument(
        "--star-min-run",
        type=int,
        default=7,
        help="Consecutive valid radii required for the innermost resolved radius.",
    )

    parser.add_argument(
        "--psd-roi",
        default="full",
        metavar="full|Y0:Y1,X0:X1",
        help="Rectangular ROI within each proj_*_roi used for power spectra.",
    )
    parser.add_argument(
        "--psd-window",
        choices=("hann", "none"),
        default="hann",
        help="Window applied before Fourier transformation.",
    )
    parser.add_argument(
        "--psd-noise-tail",
        default="0.80:0.98",
        metavar="MIN_FRAC:MAX_FRAC",
        help="Noise-baseline interval as fractions of the Nyquist frequency.",
    )
    parser.add_argument(
        "--psd-noise-stat",
        choices=("mean", "median"),
        default="mean",
        help="Statistic used for the high-frequency noise baseline.",
    )
    parser.add_argument(
        "--psd-threshold-factor",
        type=float,
        default=2.0,
        help="Cutoff threshold divided by the estimated noise baseline.",
    )
    parser.add_argument(
        "--psd-smooth-sigma",
        type=float,
        default=2.0,
        help="Gaussian smoothing sigma in frequency bins.",
    )
    parser.add_argument(
        "--psd-min-run",
        type=int,
        default=3,
        help="Minimum consecutive frequency bins above threshold.",
    )

    parser.add_argument(
        "--pixel-size",
        type=float,
        default=None,
        help="Optional physical size represented by one ROI pixel.",
    )
    parser.add_argument(
        "--pixel-unit",
        default="nm",
        help="Unit label used with --pixel-size.",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Write CSV/JSON only and skip PNG diagnostics.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow writing into an existing non-empty output directory.",
    )
    return parser.parse_args()


def parse_float_range(text: str, name: str) -> RangeSpec:
    parts = text.split(":")
    if len(parts) != 2:
        raise ValueError(f"{name} must have format START:STOP")
    start, stop = map(float, parts)
    if not np.isfinite(start) or not np.isfinite(stop) or stop <= start:
        raise ValueError(f"Invalid {name}={text!r}")
    return RangeSpec(start, stop)


def parse_int_range(text: str, name: str) -> tuple[int, int]:
    spec = parse_float_range(text, name)
    start, stop = int(round(spec.start)), int(round(spec.stop))
    if start < 1 or stop < start:
        raise ValueError(f"Invalid {name}={text!r}")
    return start, stop


def parse_center(text: str) -> tuple[float, float] | None:
    if text.lower() == "auto":
        return None
    parts = text.split(",")
    if len(parts) != 2:
        raise ValueError("--center must be 'auto' or 'Y,X'")
    y, x = map(float, parts)
    if not np.isfinite(y) or not np.isfinite(x):
        raise ValueError("--center must contain finite values")
    return y, x


def parse_slice(text: str, limit: int, name: str) -> slice:
    parts = text.split(":")
    if len(parts) != 2:
        raise ValueError(f"{name} must have format START:STOP")
    start, stop = map(int, parts)
    if start < 0 or stop <= start or stop > limit:
        raise ValueError(f"Invalid {name}={text!r} for size {limit}")
    return slice(start, stop)


def parse_psd_roi(text: str, shape: tuple[int, int]) -> tuple[slice, slice]:
    if text.lower() == "full":
        return slice(0, shape[0]), slice(0, shape[1])
    parts = text.split(",")
    if len(parts) != 2:
        raise ValueError("--psd-roi must be 'full' or 'Y0:Y1,X0:X1'")
    return parse_slice(parts[0], shape[0], "PSD y slice"), parse_slice(parts[1], shape[1], "PSD x slice")


def parse_roi_origin(h5: h5py.File) -> tuple[int, int] | None:
    def start_from_attr(name: str) -> int | None:
        if name not in h5.attrs:
            return None
        value = h5.attrs[name]
        if isinstance(value, bytes):
            value = value.decode("utf-8")
        text = str(value)
        try:
            return int(text.split(":", 1)[0])
        except (ValueError, IndexError):
            return None

    y0 = start_from_attr("roi_y_python_slice")
    x0 = start_from_attr("roi_x_python_slice")
    if y0 is None or x0 is None:
        return None
    return y0, x0


def dataset_shape_nhw(dataset: h5py.Dataset) -> tuple[int, int, int]:
    if dataset.ndim == 2:
        return 1, int(dataset.shape[0]), int(dataset.shape[1])
    if dataset.ndim == 3:
        return int(dataset.shape[0]), int(dataset.shape[1]), int(dataset.shape[2])
    raise ValueError(f"Dataset {dataset.name} must have shape [H,W] or [N,H,W], got {dataset.shape}")


def read_image(dataset: h5py.Dataset, index: int) -> np.ndarray:
    if dataset.ndim == 2:
        if index != 0:
            raise IndexError(index)
        array = dataset[...]
    else:
        array = dataset[index]
    image = np.asarray(array, dtype=np.float64)
    finite = np.isfinite(image)
    if not finite.all():
        if not finite.any():
            raise ValueError(f"Dataset {dataset.name}, index {index}, contains no finite pixels")
        fill = float(np.nanmedian(image[finite]))
        image = np.where(finite, image, fill)
    return image


def load_z1_values(h5: h5py.File, n: int, attr_name: str) -> np.ndarray:
    if attr_name in h5.attrs:
        values = np.asarray(h5.attrs[attr_name], dtype=np.float64).reshape(-1)
        if values.size == n:
            return values
        print(
            f"Warning: HDF5 attribute {attr_name!r} has {values.size} values but datasets have N={n}; "
            "using integer indices instead.",
            file=sys.stderr,
        )
    return np.arange(n, dtype=np.float64)


def build_reference(dataset: h5py.Dataset, reference_index: int | None) -> np.ndarray:
    n, _, _ = dataset_shape_nhw(dataset)
    if reference_index is not None:
        index = reference_index if reference_index >= 0 else n + reference_index
        if index < 0 or index >= n:
            raise IndexError(f"--reference-index {reference_index} is outside [0,{n - 1}]")
        return read_image(dataset, index)

    reference: np.ndarray | None = None
    for index in range(n):
        image = read_image(dataset, index)
        if reference is None:
            reference = np.zeros_like(image, dtype=np.float64)
        reference += image / n
    assert reference is not None
    return reference


def fft_centrosymmetry_center(
    image: np.ndarray,
    initial_yx: tuple[float, float],
    search_radius: float,
) -> tuple[tuple[float, float], float]:
    """Estimate the 180-degree symmetry center with FFT convolutions.

    For a candidate center c, the numerator is the correlation between I(x)
    and I(2c-x).  The corresponding local means and energies are also obtained
    by convolution, yielding a Pearson-like normalized score.  The full
    convolution grid naturally supports half-pixel centers.
    """
    work = np.asarray(image, dtype=np.float64)
    lo, hi = np.percentile(work, (1.0, 99.0))
    if hi > lo:
        work = np.clip(work, lo, hi)
    ones = np.ones(work.shape, dtype=np.float64)

    correlation = fftconvolve(work, work, mode="full")
    overlap_sum = fftconvolve(work, ones, mode="full")
    overlap_energy = fftconvolve(work * work, ones, mode="full")
    overlap_count = fftconvolve(ones, ones, mode="full")

    mean_term = overlap_sum * overlap_sum / np.maximum(overlap_count, 1.0)
    numerator = correlation - mean_term
    denominator = overlap_energy - mean_term
    score = numerator / np.maximum(denominator, EPS)

    iy, ix = initial_yx
    y0 = max(0, int(math.floor(2.0 * (iy - search_radius))))
    y1 = min(score.shape[0], int(math.ceil(2.0 * (iy + search_radius))) + 1)
    x0 = max(0, int(math.floor(2.0 * (ix - search_radius))))
    x1 = min(score.shape[1], int(math.ceil(2.0 * (ix + search_radius))) + 1)
    sub = score[y0:y1, x0:x1]
    if sub.size == 0 or not np.any(np.isfinite(sub)):
        raise ValueError("Automatic center search produced an empty/non-finite score region")
    local_index = np.unravel_index(np.nanargmax(sub), sub.shape)
    ky = y0 + local_index[0]
    kx = x0 + local_index[1]
    return (0.5 * ky, 0.5 * kx), float(sub[local_index])


def estimate_center(
    reference: np.ndarray,
    search_radius: int,
    downsample: int,
    search_step: int,
) -> tuple[tuple[float, float], dict[str, float]]:
    del search_step  # retained as a CLI compatibility parameter
    h, w = reference.shape
    downsample = max(1, int(downsample))
    small = reference[::downsample, ::downsample]
    initial_small = ((small.shape[0] - 1) / 2.0, (small.shape[1] - 1) / 2.0)
    coarse_center, coarse_score = fft_centrosymmetry_center(
        small,
        initial_small,
        max(1.0, search_radius / downsample),
    )
    coarse_full = (coarse_center[0] * downsample, coarse_center[1] * downsample)

    # Refine on a bounded full-resolution crop.  This keeps the FFT memory
    # independent of the full 1200x1200 ROI while retaining one-pixel accuracy.
    crop_half = int(min(320, coarse_full[0], coarse_full[1], h - 1 - coarse_full[0], w - 1 - coarse_full[1]))
    if crop_half < 32:
        return coarse_full, {
            "coarse_symmetry_score": float(coarse_score),
            "refined_symmetry_score": float("nan"),
        }
    center_int = (int(round(coarse_full[0])), int(round(coarse_full[1])))
    y0 = max(0, center_int[0] - crop_half)
    y1 = min(h, center_int[0] + crop_half + 1)
    x0 = max(0, center_int[1] - crop_half)
    x1 = min(w, center_int[1] + crop_half + 1)
    crop = reference[y0:y1, x0:x1]
    local_initial = (coarse_full[0] - y0, coarse_full[1] - x0)
    refine_radius = max(2.0 * downsample, 4.0)
    refined_local, refined_score = fft_centrosymmetry_center(crop, local_initial, refine_radius)
    refined = (refined_local[0] + y0, refined_local[1] + x0)
    cy = float(np.clip(refined[0], 0, h - 1))
    cx = float(np.clip(refined[1], 0, w - 1))
    return (cy, cx), {
        "coarse_symmetry_score": float(coarse_score),
        "refined_symmetry_score": float(refined_score),
    }

def max_complete_radius(shape: tuple[int, int], center_yx: tuple[float, float]) -> float:
    h, w = shape
    cy, cx = center_yx
    return float(min(cy, cx, h - 1 - cy, w - 1 - cx))


def choose_ring_radii(text: str, count: int, radius_range: RangeSpec, rmax: float) -> np.ndarray:
    if text.lower() == "auto":
        if count < 1:
            raise ValueError("--ring-count must be positive")
        if not (0.0 <= radius_range.start < radius_range.stop <= 1.0):
            raise ValueError("--ring-radius-range must lie inside 0:1")
        return np.linspace(radius_range.start * rmax, radius_range.stop * rmax, count)
    values = np.asarray([float(item) for item in text.split(",") if item.strip()], dtype=np.float64)
    if values.size == 0 or np.any(~np.isfinite(values)) or np.any(values <= 0) or np.any(values >= rmax):
        raise ValueError(f"Invalid --ring-radii={text!r}; all radii must be inside (0,{rmax:.3f})")
    return np.sort(values)


def estimate_spoke_harmonic(
    polar_reference: np.ndarray,
    radii_px: np.ndarray,
    rmax: float,
    harmonic_range: tuple[int, int],
    radial_range: RangeSpec,
) -> tuple[int, np.ndarray]:
    if not (0.0 <= radial_range.start < radial_range.stop <= 1.0):
        raise ValueError("--harmonic-radius-range must lie inside 0:1")
    radial_mask = (radii_px >= radial_range.start * rmax) & (radii_px <= radial_range.stop * rmax)
    data = polar_reference[radial_mask]
    if data.shape[0] < 3:
        raise ValueError("Too few radii for automatic spoke-harmonic estimation")
    data = data - data.mean(axis=1, keepdims=True)
    scale = data.std(axis=1, keepdims=True)
    data = data / np.maximum(scale, EPS)
    spectrum = np.mean(np.abs(np.fft.rfft(data, axis=1)) ** 2, axis=0)
    low, high = harmonic_range
    high = min(high, spectrum.size - 1)
    if low > high:
        raise ValueError("--harmonic-search does not fit inside --angular-samples")
    harmonic = int(low + np.argmax(spectrum[low : high + 1]))
    return harmonic, spectrum


def circular_weighted_mean(profile: np.ndarray, center_index: float, half_width_samples: float) -> float:
    n = profile.size
    indices = np.arange(n, dtype=np.float64)
    distance = (indices - center_index + 0.5 * n) % n - 0.5 * n
    sigma = max(half_width_samples / 2.0, 0.5)
    mask = np.abs(distance) <= half_width_samples
    if not np.any(mask):
        return float(profile[int(round(center_index)) % n])
    weights = np.exp(-0.5 * (distance[mask] / sigma) ** 2)
    return float(np.sum(profile[mask] * weights) / np.sum(weights))


def robust_cv_percent(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    denominator = max(abs(median), EPS)
    return 100.0 * 1.4826 * mad / denominator


def ring_uniformity_metrics(
    profile_raw: np.ndarray,
    harmonic: int,
    background_periods: float,
    sample_width_fraction: float,
) -> dict[str, float]:
    n = profile_raw.size
    period_samples = n / harmonic
    sigma = max(1.0, background_periods * period_samples)
    background = gaussian_filter1d(profile_raw, sigma=sigma, mode="wrap")
    profile = profile_raw - background

    theta = np.arange(n, dtype=np.float64) * (2.0 * np.pi / n)
    coefficient = np.mean(profile * np.exp(-1j * harmonic * theta))
    peak_angle = (-np.angle(coefficient) / harmonic) % (2.0 * np.pi / harmonic)
    peak_index0 = peak_angle * n / (2.0 * np.pi)
    half_width = max(0.5, 0.5 * sample_width_fraction * period_samples)

    peaks_raw = np.empty(harmonic, dtype=np.float64)
    peaks_detrended = np.empty(harmonic, dtype=np.float64)
    troughs_detrended = np.empty(harmonic, dtype=np.float64)
    for spoke in range(harmonic):
        peak_index = peak_index0 + spoke * period_samples
        trough_index = peak_index + 0.5 * period_samples
        peaks_raw[spoke] = circular_weighted_mean(profile_raw, peak_index, half_width)
        peaks_detrended[spoke] = circular_weighted_mean(profile, peak_index, half_width)
        troughs_detrended[spoke] = circular_weighted_mean(profile, trough_index, half_width)

    amplitudes = peaks_detrended - troughs_detrended
    if np.median(amplitudes) < 0:
        amplitudes = -amplitudes
        peaks_detrended, troughs_detrended = -troughs_detrended, -peaks_detrended

    amp_mean = float(np.mean(amplitudes))
    amp_std = float(np.std(amplitudes, ddof=1)) if harmonic > 1 else 0.0
    amp_median = float(np.median(amplitudes))
    amp_cv = 100.0 * amp_std / max(abs(amp_mean), EPS)
    amp_p05, amp_p95 = np.percentile(amplitudes, (5.0, 95.0))
    amp_spread = 100.0 * float(amp_p95 - amp_p05) / max(abs(amp_median), EPS)

    raw_dynamic = float(np.percentile(profile_raw, 99.0) - np.percentile(profile_raw, 1.0))
    peak_std = float(np.std(peaks_raw, ddof=1)) if harmonic > 1 else 0.0
    peak_spread_normalized = 100.0 * peak_std / max(abs(raw_dynamic), EPS)
    peak_mean = float(np.mean(peaks_raw))
    peak_cv_abs = 100.0 * peak_std / max(abs(peak_mean), EPS)

    fft_power = np.abs(np.fft.rfft(profile)) ** 2
    local_lo = max(1, harmonic - max(3, harmonic // 8))
    local_hi = min(fft_power.size - 1, harmonic + max(3, harmonic // 8))
    local_indices = np.arange(local_lo, local_hi + 1)
    exclude = np.abs(local_indices - harmonic) <= 1
    side_power = fft_power[local_indices[~exclude]]
    noise_power = float(np.median(side_power)) if side_power.size else EPS
    harmonic_snr_db = 10.0 * math.log10(max(float(fft_power[harmonic]), EPS) / max(noise_power, EPS))
    harmonic_fraction = float(fft_power[harmonic] / max(np.sum(fft_power[1:]), EPS))

    return {
        "spoke_count": float(harmonic),
        "spoke_amplitude_mean": amp_mean,
        "spoke_amplitude_std": amp_std,
        "spoke_amplitude_cv_percent": amp_cv,
        "spoke_amplitude_robust_cv_percent": robust_cv_percent(amplitudes),
        "spoke_amplitude_p95_p05_spread_percent": amp_spread,
        "spoke_peak_mean_raw": peak_mean,
        "spoke_peak_std_raw": peak_std,
        "spoke_peak_cv_abs_percent": peak_cv_abs,
        "spoke_peak_std_over_ring_dynamic_range_percent": peak_spread_normalized,
        "harmonic_snr_db": harmonic_snr_db,
        "harmonic_power_fraction": harmonic_fraction,
        "ring_mean_raw": float(np.mean(profile_raw)),
        "ring_std_raw": float(np.std(profile_raw)),
    }


def annular_profile(
    polar: np.ndarray,
    polar_radii: np.ndarray,
    radius: float,
    half_width: float,
) -> np.ndarray:
    mask = np.abs(polar_radii - radius) <= half_width
    if not np.any(mask):
        nearest = int(np.argmin(np.abs(polar_radii - radius)))
        return polar[nearest]
    return np.mean(polar[mask], axis=0)


def precompute_sector_model(
    angles: np.ndarray,
    center_angle: float,
    half_width: float,
    harmonic: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    delta = (angles - center_angle + np.pi) % (2.0 * np.pi) - np.pi
    mask = np.abs(delta) <= half_width
    theta = angles[mask]
    local = delta[mask]
    design = np.column_stack(
        (
            np.ones(theta.size, dtype=np.float64),
            local,
            np.cos(harmonic * theta),
            np.sin(harmonic * theta),
        )
    )
    pinv_t = np.linalg.pinv(design).T
    return mask, design, pinv_t


def fit_star_direction(
    polar: np.ndarray,
    radii_px: np.ndarray,
    angles: np.ndarray,
    harmonic: int,
    sector_centers: Sequence[float],
    half_width_deg: float,
    snr_threshold: float,
    amplitude_fraction_threshold: float,
    smooth_sigma: float,
    min_run: int,
) -> StarDirectionResult:
    amplitudes: list[np.ndarray] = []
    residuals: list[np.ndarray] = []
    half_width = np.deg2rad(half_width_deg)

    for center in sector_centers:
        mask, design, pinv_t = precompute_sector_model(angles, center, half_width, harmonic)
        data = polar[:, mask]
        coefficients = data @ pinv_t
        fitted = coefficients @ design.T
        residual = data - fitted
        amplitude = np.hypot(coefficients[:, 2], coefficients[:, 3])
        rms = np.sqrt(np.mean(residual * residual, axis=1))
        amplitudes.append(amplitude)
        residuals.append(rms)

    amplitude = np.median(np.stack(amplitudes, axis=0), axis=0)
    residual_rms = np.median(np.stack(residuals, axis=0), axis=0)
    amplitude_smooth = gaussian_filter1d(amplitude, sigma=max(smooth_sigma, 0.0), mode="nearest")
    residual_smooth = gaussian_filter1d(residual_rms, sigma=max(smooth_sigma, 0.0), mode="nearest")
    snr = amplitude_smooth / np.maximum(residual_smooth, EPS)

    outer_count = max(min_run, int(math.ceil(0.22 * radii_px.size)))
    outer_reference = float(np.median(amplitude_smooth[-outer_count:]))
    amplitude_fraction = amplitude_smooth / max(abs(outer_reference), EPS)
    valid = (snr >= snr_threshold) & (amplitude_fraction >= amplitude_fraction_threshold)

    cutoff_index: int | None = None
    run = max(1, int(min_run))
    for index in range(0, max(1, valid.size - run + 1)):
        if np.mean(valid[index : index + run]) >= 0.85:
            cutoff_index = index
            break

    if cutoff_index is None:
        cutoff_radius = float("nan")
        cutoff_frequency = float("nan")
        censored = False
    else:
        cutoff_radius = float(radii_px[cutoff_index])
        cutoff_frequency = float(harmonic / (2.0 * np.pi * cutoff_radius))
        censored = cutoff_index == 0

    return StarDirectionResult(
        radii_px=np.asarray(radii_px, dtype=np.float64),
        amplitude=amplitude_smooth,
        residual_rms=residual_smooth,
        snr=snr,
        amplitude_fraction=amplitude_fraction,
        cutoff_radius_px=cutoff_radius,
        cutoff_cyc_per_px=cutoff_frequency,
        cutoff_censored=censored,
    )


def frequency_resolution_metrics(prefix: str, cutoff: float, pixel_size: float | None, unit: str) -> dict[str, Any]:
    result: dict[str, Any] = {f"{prefix}_cutoff_cyc_per_px": cutoff}
    if np.isfinite(cutoff) and cutoff > 0:
        full_period = 1.0 / cutoff
        half_pitch = 0.5 / cutoff
    else:
        full_period = float("nan")
        half_pitch = float("nan")
    result[f"{prefix}_full_period_px"] = full_period
    result[f"{prefix}_half_pitch_px"] = half_pitch
    if pixel_size is not None:
        result[f"{prefix}_cutoff_cyc_per_{unit}"] = cutoff / pixel_size if np.isfinite(cutoff) else float("nan")
        result[f"{prefix}_full_period_{unit}"] = full_period * pixel_size
        result[f"{prefix}_half_pitch_{unit}"] = half_pitch * pixel_size
    return result


def prepare_psd_image(image: np.ndarray, y_slice: slice, x_slice: slice) -> np.ndarray:
    roi = np.asarray(image[y_slice, x_slice], dtype=np.float64)
    roi = roi - np.mean(roi)
    return roi


def one_dimensional_psd(image: np.ndarray, axis: int, window: str) -> tuple[np.ndarray, np.ndarray]:
    length = image.shape[axis]
    if axis == 1:
        data = image - image.mean(axis=1, keepdims=True)
        win = np.hanning(length) if window == "hann" else np.ones(length)
        transformed = np.fft.rfft(data * win[None, :], axis=1)
        power = np.mean(np.abs(transformed) ** 2, axis=0)
    elif axis == 0:
        data = image - image.mean(axis=0, keepdims=True)
        win = np.hanning(length) if window == "hann" else np.ones(length)
        transformed = np.fft.rfft(data * win[:, None], axis=0)
        power = np.mean(np.abs(transformed) ** 2, axis=1)
    else:
        raise ValueError(axis)
    normalization = max(float(np.sum(win * win)), EPS)
    power = power / normalization
    frequency = np.fft.rfftfreq(length, d=1.0)
    return frequency, power


def radial_2d_psd(image: np.ndarray, window: str) -> tuple[np.ndarray, np.ndarray]:
    h, w = image.shape
    if window == "hann":
        wy = np.hanning(h)
        wx = np.hanning(w)
        work = image * wy[:, None] * wx[None, :]
        normalization = max(float(np.sum(wy * wy) * np.sum(wx * wx)), EPS)
    else:
        work = image
        normalization = float(h * w)
    transformed = np.fft.fftshift(np.fft.fft2(work))
    power2d = np.abs(transformed) ** 2 / normalization
    fy = np.fft.fftshift(np.fft.fftfreq(h, d=1.0))
    fx = np.fft.fftshift(np.fft.fftfreq(w, d=1.0))
    fr = np.sqrt(fy[:, None] ** 2 + fx[None, :] ** 2)

    # Restrict to the inscribed Fourier circle (0..Nyquist) so the radial and
    # directional frequency ranges have the same physical interpretation.
    max_frequency = 0.5
    n_bins = max(32, min(h, w) // 2)
    edges = np.linspace(0.0, max_frequency, n_bins + 1)
    indices = np.digitize(fr.ravel(), edges) - 1
    valid = (indices >= 0) & (indices < n_bins)
    sums = np.bincount(indices[valid], weights=power2d.ravel()[valid], minlength=n_bins)
    counts = np.bincount(indices[valid], minlength=n_bins)
    radial = sums / np.maximum(counts, 1)
    frequency = 0.5 * (edges[:-1] + edges[1:])
    return frequency, radial


def estimate_noise_baseline(
    frequency: np.ndarray,
    power: np.ndarray,
    tail_range: RangeSpec,
    statistic: str,
) -> float:
    nyquist = 0.5
    mask = (frequency >= tail_range.start * nyquist) & (frequency <= tail_range.stop * nyquist)
    values = power[mask]
    values = values[np.isfinite(values)]
    if values.size < 3:
        raise ValueError("PSD noise-tail interval contains fewer than three bins")
    if statistic == "median":
        return float(np.median(values))
    return float(np.mean(values))


def highest_stable_frequency(
    frequency: np.ndarray,
    power_smooth: np.ndarray,
    threshold: float,
    search_stop: float,
    min_run: int,
) -> tuple[float, bool]:
    valid_domain = (frequency > 0.0) & (frequency <= search_stop)
    indices = np.flatnonzero(valid_domain)
    if indices.size == 0:
        return float("nan"), False
    above = power_smooth[indices] >= threshold
    run = max(1, int(min_run))
    kernel = np.ones(run, dtype=np.int32)
    stable = np.convolve(above.astype(np.int32), kernel, mode="same") >= max(1, int(math.ceil(0.8 * run)))
    candidates = indices[stable]
    if candidates.size == 0:
        return float("nan"), False
    index = int(candidates[-1])
    censored = index == indices[-1]

    # Linear interpolation with the next lower point, where available.
    cutoff = float(frequency[index])
    if index + 1 < frequency.size and power_smooth[index] >= threshold > power_smooth[index + 1]:
        x0, x1 = frequency[index], frequency[index + 1]
        y0, y1 = power_smooth[index], power_smooth[index + 1]
        if y1 != y0:
            cutoff = float(x0 + (threshold - y0) * (x1 - x0) / (y1 - y0))
    return cutoff, censored


def analyze_psd_curve(
    frequency: np.ndarray,
    power: np.ndarray,
    tail_range: RangeSpec,
    noise_stat: str,
    threshold_factor: float,
    smooth_sigma: float,
    min_run: int,
) -> PSDResult:
    power = np.asarray(power, dtype=np.float64)
    power_smooth = gaussian_filter1d(power, sigma=max(smooth_sigma, 0.0), mode="nearest")
    baseline = estimate_noise_baseline(frequency, power_smooth, tail_range, noise_stat)
    threshold = threshold_factor * baseline
    cutoff, censored = highest_stable_frequency(
        frequency,
        power_smooth,
        threshold,
        search_stop=tail_range.start * 0.5,
        min_run=min_run,
    )
    return PSDResult(
        frequency=frequency,
        power=power,
        power_smooth=power_smooth,
        noise_baseline=baseline,
        threshold=threshold,
        cutoff_cyc_per_px=cutoff,
        cutoff_censored=censored,
    )


def calculate_psd_metrics(
    image: np.ndarray,
    psd_slices: tuple[slice, slice],
    window: str,
    tail_range: RangeSpec,
    noise_stat: str,
    threshold_factor: float,
    smooth_sigma: float,
    min_run: int,
) -> tuple[dict[str, PSDResult], np.ndarray]:
    work = prepare_psd_image(image, *psd_slices)
    fx, px = one_dimensional_psd(work, axis=1, window=window)
    fy, py = one_dimensional_psd(work, axis=0, window=window)
    fr, pr = radial_2d_psd(work, window=window)
    results = {
        "x": analyze_psd_curve(fx, px, tail_range, noise_stat, threshold_factor, smooth_sigma, min_run),
        "y": analyze_psd_curve(fy, py, tail_range, noise_stat, threshold_factor, smooth_sigma, min_run),
        "radial": analyze_psd_curve(fr, pr, tail_range, noise_stat, threshold_factor, smooth_sigma, min_run),
    }
    return results, work


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        return
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fields.append(key)
                seen.add(key)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def add_relative_quality_scores(summary_rows: list[dict[str, Any]]) -> None:
    # Relative rank only: it is deliberately not interpreted as an absolute
    # metrological score.  Uniformity CV is lower-is-better; cutoff frequencies
    # and harmonic SNR are higher-is-better.
    higher_better = (
        "psd_x_cutoff_cyc_per_px",
        "psd_y_cutoff_cyc_per_px",
        "psd_radial_cutoff_cyc_per_px",
        "star_horizontal_sides_cutoff_cyc_per_px",
        "star_vertical_sides_cutoff_cyc_per_px",
        "ring_harmonic_snr_db_median",
    )
    lower_better = (
        "ring_spoke_amplitude_cv_percent_median",
        "ring_spoke_amplitude_cv_percent_worst",
    )

    keys = sorted({str(row["dataset"]) for row in summary_rows})
    for dataset in keys:
        indices = [i for i, row in enumerate(summary_rows) if row["dataset"] == dataset]
        component_scores: list[list[float]] = [[] for _ in indices]
        for metric in higher_better + lower_better:
            values = np.asarray([float(summary_rows[i].get(metric, np.nan)) for i in indices], dtype=np.float64)
            finite = np.isfinite(values)
            if np.sum(finite) < 2:
                continue
            ranks = np.full(values.shape, np.nan, dtype=np.float64)
            r = rankdata(values[finite], method="average")
            if metric in lower_better:
                r = (np.sum(finite) + 1.0) - r
            ranks[finite] = 100.0 * (r - 1.0) / max(np.sum(finite) - 1.0, 1.0)
            for local_index, score in enumerate(ranks):
                if np.isfinite(score):
                    component_scores[local_index].append(float(score))
        for local_index, row_index in enumerate(indices):
            scores = component_scores[local_index]
            summary_rows[row_index]["relative_quality_score_0_100"] = float(np.mean(scores)) if scores else float("nan")
            summary_rows[row_index]["relative_quality_score_components"] = len(scores)


def aggregate_all_channels(summary_rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    z1_values = sorted({float(row["z1"]) for row in summary_rows})
    output: list[dict[str, Any]] = []
    for z1 in z1_values:
        selected = [row for row in summary_rows if float(row["z1"]) == z1]
        scores = np.asarray([float(row.get("relative_quality_score_0_100", np.nan)) for row in selected])
        output.append(
            {
                "z1": z1,
                "datasets": ";".join(str(row["dataset"]) for row in selected),
                "relative_quality_score_mean_0_100": float(np.nanmean(scores)) if np.any(np.isfinite(scores)) else float("nan"),
                "relative_quality_score_min_0_100": float(np.nanmin(scores)) if np.any(np.isfinite(scores)) else float("nan"),
            }
        )
    return output


def import_matplotlib():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Circle
    except ImportError as exc:  # pragma: no cover
        raise SystemExit("Plotting requires Matplotlib; use --no-plots or install matplotlib.") from exc
    return plt, Circle


def plot_reference_geometry(
    output: Path,
    reference: np.ndarray,
    center_yx: tuple[float, float],
    ring_radii: np.ndarray,
    direction_half_width_deg: float,
) -> None:
    plt, Circle = import_matplotlib()
    cy, cx = center_yx
    low, high = np.percentile(reference, (1.0, 99.0))
    plt.figure(figsize=(8, 8))
    plt.imshow(reference, cmap="gray", vmin=low, vmax=high, origin="upper")
    axis = plt.gca()
    for radius in ring_radii:
        axis.add_patch(Circle((cx, cy), radius, fill=False, linewidth=1.0))
    plt.scatter([cx], [cy], marker="+")
    rline = float(np.max(ring_radii))
    half = np.deg2rad(direction_half_width_deg)
    for center in (0.0, 0.5 * np.pi, np.pi, 1.5 * np.pi):
        for angle in (center - half, center + half):
            plt.plot([cx, cx + rline * np.cos(angle)], [cy, cy + rline * np.sin(angle)], linewidth=0.8)
    plt.title("Reference image: fitted center, annuli, and directional sectors")
    plt.xlabel("ROI x [pixel]")
    plt.ylabel("ROI y [pixel]")
    plt.tight_layout()
    plt.savefig(output, dpi=180)
    plt.close()


def plot_harmonic_spectrum(
    output: Path,
    spectrum: np.ndarray,
    harmonic: int,
    search_range: tuple[int, int],
) -> None:
    plt, _ = import_matplotlib()
    lo, hi = search_range
    hi = min(hi, spectrum.size - 1)
    values = spectrum[lo : hi + 1]
    values = values / max(float(np.max(values)), EPS)
    plt.figure(figsize=(8, 5))
    plt.plot(np.arange(lo, hi + 1), values)
    plt.axvline(harmonic, linestyle="--", label=f"selected harmonic = {harmonic}")
    plt.xlabel("Angular harmonic / line pairs around 360°")
    plt.ylabel("Normalized averaged power")
    plt.title("Automatic spoke-harmonic estimate")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output, dpi=180)
    plt.close()


def rows_for_dataset(rows: Sequence[dict[str, Any]], dataset: str) -> list[dict[str, Any]]:
    return sorted((row for row in rows if row["dataset"] == dataset), key=lambda row: float(row["z1"]))


def plot_summary_metrics(
    output_dir: Path,
    summary_rows: Sequence[dict[str, Any]],
    ring_rows: Sequence[dict[str, Any]],
    pixel_size: float | None,
    pixel_unit: str,
) -> None:
    plt, _ = import_matplotlib()
    datasets = sorted({str(row["dataset"]) for row in summary_rows})
    for dataset in datasets:
        selected = rows_for_dataset(summary_rows, dataset)
        z1 = np.asarray([float(row["z1"]) for row in selected])

        plt.figure(figsize=(8, 5))
        plt.plot(z1, [row["ring_spoke_amplitude_cv_percent_median"] for row in selected], marker="o", label="median across annuli")
        plt.plot(z1, [row["ring_spoke_amplitude_cv_percent_worst"] for row in selected], marker="o", label="worst annulus")
        plt.xlabel("z1")
        plt.ylabel("Spoke amplitude CV [%] (lower is better)")
        plt.title(f"Annular spoke uniformity: {dataset}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / f"{dataset}_ring_uniformity_vs_z1.png", dpi=180)
        plt.close()

        plt.figure(figsize=(8, 5))
        for direction in ("x", "y", "radial"):
            plt.plot(
                z1,
                [row[f"psd_{direction}_full_period_px"] for row in selected],
                marker="o",
                label=direction,
            )
        plt.xlabel("z1")
        plt.ylabel("PSD cutoff full period [pixel] (lower is better)")
        plt.title(f"Modregger-adapted PSD resolution: {dataset}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / f"{dataset}_psd_resolution_vs_z1.png", dpi=180)
        plt.close()

        plt.figure(figsize=(8, 5))
        plt.plot(
            z1,
            [row["star_horizontal_sides_half_pitch_px"] for row in selected],
            marker="o",
            label="left/right sectors (tests tangential y)",
        )
        plt.plot(
            z1,
            [row["star_vertical_sides_half_pitch_px"] for row in selected],
            marker="o",
            label="top/bottom sectors (tests tangential x)",
        )
        plt.xlabel("z1")
        plt.ylabel("Star cutoff half-pitch [pixel] (lower is better)")
        plt.title(f"Directional radial-pattern resolution: {dataset}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / f"{dataset}_star_resolution_vs_z1.png", dpi=180)
        plt.close()

        plt.figure(figsize=(8, 5))
        plt.plot(z1, [row["relative_quality_score_0_100"] for row in selected], marker="o")
        plt.xlabel("z1")
        plt.ylabel("Relative quality score [0–100] (higher is better)")
        plt.title(f"Relative combined ranking: {dataset}")
        plt.tight_layout()
        plt.savefig(output_dir / f"{dataset}_relative_quality_score_vs_z1.png", dpi=180)
        plt.close()

        if pixel_size is not None:
            plt.figure(figsize=(8, 5))
            plt.plot(z1, [row[f"psd_x_full_period_{pixel_unit}"] for row in selected], marker="o", label="PSD x")
            plt.plot(z1, [row[f"psd_y_full_period_{pixel_unit}"] for row in selected], marker="o", label="PSD y")
            plt.plot(
                z1,
                [row[f"star_horizontal_sides_half_pitch_{pixel_unit}"] for row in selected],
                marker="o",
                label="star left/right half-pitch",
            )
            plt.plot(
                z1,
                [row[f"star_vertical_sides_half_pitch_{pixel_unit}"] for row in selected],
                marker="o",
                label="star top/bottom half-pitch",
            )
            plt.xlabel("z1")
            plt.ylabel(f"Resolution [{pixel_unit}]")
            plt.title(f"Physical resolution estimates: {dataset}")
            plt.legend()
            plt.tight_layout()
            plt.savefig(output_dir / f"{dataset}_physical_resolution_vs_z1.png", dpi=180)
            plt.close()

        selected_ring = [row for row in ring_rows if row["dataset"] == dataset]
        radii = sorted({float(row["radius_px"]) for row in selected_ring})
        matrix = np.full((len(radii), len(z1)), np.nan, dtype=np.float64)
        radius_index = {radius: i for i, radius in enumerate(radii)}
        z_index = {float(value): i for i, value in enumerate(z1)}
        for row in selected_ring:
            matrix[radius_index[float(row["radius_px"])], z_index[float(row["z1"])]] = float(
                row["spoke_amplitude_cv_percent"]
            )
        plt.figure(figsize=(9, 5))
        image = plt.imshow(matrix, aspect="auto", origin="lower")
        plt.colorbar(image, label="Spoke amplitude CV [%]")
        plt.yticks(np.arange(len(radii)), [f"{radius:.1f}" for radius in radii])
        tick_count = min(10, len(z1))
        tick_indices = np.unique(np.linspace(0, len(z1) - 1, tick_count).round().astype(int))
        plt.xticks(tick_indices, [f"{z1[i]:.6g}" for i in tick_indices], rotation=35, ha="right")
        plt.xlabel("z1")
        plt.ylabel("Annulus radius [pixel]")
        plt.title(f"Ring-by-ring nonuniformity: {dataset}")
        plt.tight_layout()
        plt.savefig(output_dir / f"{dataset}_ring_uniformity_heatmap.png", dpi=180)
        plt.close()


def plot_best_diagnostics(
    output_dir: Path,
    h5: h5py.File,
    summary_rows: Sequence[dict[str, Any]],
    z1_values: np.ndarray,
    polar_sampler: PolarSampler,
    harmonic: int,
    args: argparse.Namespace,
    psd_slices: tuple[slice, slice],
    psd_tail: RangeSpec,
) -> None:
    plt, _ = import_matplotlib()
    for dataset in sorted({str(row["dataset"]) for row in summary_rows}):
        candidates = rows_for_dataset(summary_rows, dataset)
        scores = np.asarray([float(row.get("relative_quality_score_0_100", np.nan)) for row in candidates])
        if not np.any(np.isfinite(scores)):
            continue
        best_row = candidates[int(np.nanargmax(scores))]
        best_z1 = float(best_row["z1"])
        best_index = int(np.argmin(np.abs(z1_values - best_z1)))
        image = read_image(h5[dataset], best_index)

        psd_results, _ = calculate_psd_metrics(
            image,
            psd_slices,
            args.psd_window,
            psd_tail,
            args.psd_noise_stat,
            args.psd_threshold_factor,
            args.psd_smooth_sigma,
            args.psd_min_run,
        )
        plt.figure(figsize=(8, 5))
        for direction, result in psd_results.items():
            plt.semilogy(
                result.frequency,
                result.power_smooth / max(result.noise_baseline, EPS),
                label=f"{direction}, cutoff={result.cutoff_cyc_per_px:.4g} cyc/px",
            )
        plt.axhline(args.psd_threshold_factor, linestyle="--", label="cutoff threshold")
        plt.xlabel("Spatial frequency [cycles/pixel]")
        plt.ylabel("Smoothed power / noise baseline")
        plt.title(f"PSD diagnostic at best-ranked z1={best_z1:.8g}: {dataset}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / f"{dataset}_best_z1_psd_diagnostic.png", dpi=180)
        plt.close()

        polar = polar_sampler.sample(image)
        horizontal = fit_star_direction(
            polar,
            polar_sampler.radii_px,
            polar_sampler.angles_rad,
            harmonic,
            (0.0, np.pi),
            args.direction_half_width_deg,
            args.star_snr_threshold,
            args.star_amplitude_fraction,
            args.star_smooth_sigma,
            args.star_min_run,
        )
        vertical = fit_star_direction(
            polar,
            polar_sampler.radii_px,
            polar_sampler.angles_rad,
            harmonic,
            (0.5 * np.pi, 1.5 * np.pi),
            args.direction_half_width_deg,
            args.star_snr_threshold,
            args.star_amplitude_fraction,
            args.star_smooth_sigma,
            args.star_min_run,
        )
        plt.figure(figsize=(8, 5))
        plt.plot(horizontal.radii_px, horizontal.snr, label="left/right sectors")
        plt.plot(vertical.radii_px, vertical.snr, label="top/bottom sectors")
        plt.axhline(args.star_snr_threshold, linestyle="--", label="SNR threshold")
        if np.isfinite(horizontal.cutoff_radius_px):
            plt.axvline(horizontal.cutoff_radius_px, linestyle=":")
        if np.isfinite(vertical.cutoff_radius_px):
            plt.axvline(vertical.cutoff_radius_px, linestyle=":")
        plt.xlabel("Radius [pixel]")
        plt.ylabel("Fitted spoke harmonic SNR")
        plt.title(f"Directional star diagnostic at best-ranked z1={best_z1:.8g}: {dataset}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / f"{dataset}_best_z1_star_diagnostic.png", dpi=180)
        plt.close()


def main() -> int:
    args = parse_args()
    input_path = args.input.expanduser().resolve()
    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else input_path.with_name(input_path.stem + "_roi_metrics")
    )
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Output directory is not empty: {output_dir}; pass --overwrite")
    output_dir.mkdir(parents=True, exist_ok=True)

    ring_radius_range = parse_float_range(args.ring_radius_range, "--ring-radius-range")
    harmonic_radius_range = parse_float_range(args.harmonic_radius_range, "--harmonic-radius-range")
    star_radius_range = parse_float_range(args.star_radius_range, "--star-radius-range")
    psd_tail = parse_float_range(args.psd_noise_tail, "--psd-noise-tail")
    harmonic_search = parse_int_range(args.harmonic_search, "--harmonic-search")
    if not (0.0 <= psd_tail.start < psd_tail.stop <= 1.0):
        raise ValueError("--psd-noise-tail must lie inside 0:1")
    if not (0.0 <= star_radius_range.start < star_radius_range.stop <= 1.0):
        raise ValueError("--star-radius-range must lie inside 0:1")
    if args.angular_samples < 64:
        raise ValueError("--angular-samples must be at least 64")

    ring_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []

    with h5py.File(input_path, "r") as h5:
        for key in args.keys:
            if key not in h5 or not isinstance(h5[key], h5py.Dataset):
                raise KeyError(f"Missing HDF5 dataset {key!r}")
        if args.reference_key not in h5 or not isinstance(h5[args.reference_key], h5py.Dataset):
            raise KeyError(f"Missing --reference-key dataset {args.reference_key!r}")

        shapes = [dataset_shape_nhw(h5[key]) for key in args.keys]
        if len(set(shapes)) != 1:
            raise ValueError(f"All --keys must have identical [N,H,W], got {dict(zip(args.keys, shapes))}")
        n, h, w = shapes[0]
        reference_shape = dataset_shape_nhw(h5[args.reference_key])
        if reference_shape != (n, h, w):
            raise ValueError(
                f"Reference dataset shape {reference_shape} differs from analyzed dataset shape {(n, h, w)}"
            )
        z1_values = load_z1_values(h5, n, args.z1_attr)
        roi_origin = parse_roi_origin(h5)

        print(f"Building reference image from {args.reference_key!r} ...")
        reference = build_reference(h5[args.reference_key], args.reference_index)

        explicit_center = parse_center(args.center)
        center_diagnostics: dict[str, float] = {}
        if explicit_center is None:
            print("Estimating pattern center from 180-degree centrosymmetry ...")
            center_yx, center_diagnostics = estimate_center(
                reference,
                args.center_search_radius,
                args.center_downsample,
                args.center_search_step,
            )
        else:
            center_yx = explicit_center
            if args.center_frame == "full":
                if roi_origin is None:
                    raise ValueError(
                        "--center-frame=full requires roi_y_python_slice and roi_x_python_slice attributes "
                        "in the merged HDF5 file"
                    )
                center_yx = (center_yx[0] - roi_origin[0], center_yx[1] - roi_origin[1])
        cy, cx = center_yx
        if not (0 <= cy < h and 0 <= cx < w):
            raise ValueError(f"Center {center_yx} is outside ROI shape {(h, w)}")

        rmax_complete = max_complete_radius((h, w), center_yx)
        rmax = rmax_complete - max(args.ring_half_width, 2.0) - 1.0
        if rmax <= 16:
            raise ValueError("Pattern center leaves too little complete circular support")
        ring_radii = choose_ring_radii(args.ring_radii, args.ring_count, ring_radius_range, rmax)

        star_rmin = max(2, int(math.ceil(star_radius_range.start * rmax)))
        star_rmax = min(int(math.floor(star_radius_range.stop * rmax)), int(math.floor(rmax)))
        if star_rmax - star_rmin < 20:
            raise ValueError("--star-radius-range is too narrow")
        polar_rmin = max(1, min(star_rmin, int(math.floor(float(np.min(ring_radii)) - args.ring_half_width - 1.0))))
        polar_rmax = min(
            int(math.floor(rmax)),
            max(star_rmax, int(math.ceil(float(np.max(ring_radii)) + args.ring_half_width + 1.0))),
        )
        polar_radii = np.arange(polar_rmin, polar_rmax + 1, dtype=np.float64)
        polar_sampler = PolarSampler.create(center_yx, polar_radii, args.angular_samples)
        polar_reference = polar_sampler.sample(reference)

        if args.spoke_harmonic.lower() == "auto":
            harmonic, harmonic_spectrum = estimate_spoke_harmonic(
                polar_reference,
                polar_radii,
                rmax,
                harmonic_search,
                harmonic_radius_range,
            )
        else:
            harmonic = int(args.spoke_harmonic)
            if harmonic < 1 or harmonic >= args.angular_samples // 2:
                raise ValueError("--spoke-harmonic is outside the supported angular frequency range")
            _, harmonic_spectrum = estimate_spoke_harmonic(
                polar_reference,
                polar_radii,
                rmax,
                harmonic_search,
                harmonic_radius_range,
            )

        psd_slices = parse_psd_roi(args.psd_roi, (h, w))
        print(
            f"Center (ROI y,x)=({cy:.3f},{cx:.3f}), complete radius={rmax_complete:.2f}px, "
            f"spoke harmonic={harmonic}, N={n}."
        )

        run_config: dict[str, Any] = {
            "input": str(input_path),
            "datasets": list(args.keys),
            "reference_key": args.reference_key,
            "reference_index": args.reference_index,
            "center_roi_yx": [cy, cx],
            "center_full_yx": [cy + roi_origin[0], cx + roi_origin[1]] if roi_origin is not None else None,
            "roi_origin_yx": list(roi_origin) if roi_origin is not None else None,
            "center_diagnostics": center_diagnostics,
            "complete_circle_radius_px": rmax_complete,
            "ring_radii_px": ring_radii.tolist(),
            "spoke_harmonic": harmonic,
            "pixel_size": args.pixel_size,
            "pixel_unit": args.pixel_unit,
            "method_note": (
                "PSD cutoff uses the maximum stable frequency where smoothed total power reaches "
                "psd_threshold_factor times the high-frequency noise baseline. With the default factor 2, "
                "the inferred signal power equals the inferred noise power at the cutoff under an additive model."
            ),
            "arguments": vars(args) | {"input": str(args.input), "output_dir": str(args.output_dir) if args.output_dir else None},
        }
        with (output_dir / "run_config.json").open("w", encoding="utf-8") as stream:
            json.dump(run_config, stream, indent=2, ensure_ascii=False)

        if not args.no_plots:
            plot_reference_geometry(
                output_dir / "reference_geometry.png",
                reference,
                center_yx,
                ring_radii,
                args.direction_half_width_deg,
            )
            plot_harmonic_spectrum(
                output_dir / "spoke_harmonic_spectrum.png",
                harmonic_spectrum,
                harmonic,
                harmonic_search,
            )

        for dataset_name in args.keys:
            dataset = h5[dataset_name]
            print(f"Analyzing {dataset_name!r} ...")
            for index, z1 in enumerate(z1_values):
                image = read_image(dataset, index)
                polar = polar_sampler.sample(image)

                current_ring_rows: list[dict[str, Any]] = []
                for radius in ring_radii:
                    profile = annular_profile(polar, polar_radii, float(radius), args.ring_half_width)
                    metrics = ring_uniformity_metrics(
                        profile,
                        harmonic,
                        args.angular_background_periods,
                        args.spoke_sample_width_fraction,
                    )
                    row: dict[str, Any] = {
                        "z1_index": index,
                        "z1": float(z1),
                        "dataset": dataset_name,
                        "radius_px": float(radius),
                    }
                    if args.pixel_size is not None:
                        row[f"radius_{args.pixel_unit}"] = float(radius * args.pixel_size)
                    row.update(metrics)
                    ring_rows.append(row)
                    current_ring_rows.append(row)

                horizontal = fit_star_direction(
                    polar,
                    polar_radii,
                    polar_sampler.angles_rad,
                    harmonic,
                    (0.0, np.pi),
                    args.direction_half_width_deg,
                    args.star_snr_threshold,
                    args.star_amplitude_fraction,
                    args.star_smooth_sigma,
                    args.star_min_run,
                )
                vertical = fit_star_direction(
                    polar,
                    polar_radii,
                    polar_sampler.angles_rad,
                    harmonic,
                    (0.5 * np.pi, 1.5 * np.pi),
                    args.direction_half_width_deg,
                    args.star_snr_threshold,
                    args.star_amplitude_fraction,
                    args.star_smooth_sigma,
                    args.star_min_run,
                )

                psd_results, psd_work = calculate_psd_metrics(
                    image,
                    psd_slices,
                    args.psd_window,
                    psd_tail,
                    args.psd_noise_stat,
                    args.psd_threshold_factor,
                    args.psd_smooth_sigma,
                    args.psd_min_run,
                )

                cvs = np.asarray([row["spoke_amplitude_cv_percent"] for row in current_ring_rows], dtype=float)
                harmonic_snr = np.asarray([row["harmonic_snr_db"] for row in current_ring_rows], dtype=float)
                summary: dict[str, Any] = {
                    "z1_index": index,
                    "z1": float(z1),
                    "dataset": dataset_name,
                    "spoke_harmonic": harmonic,
                    "center_y_roi_px": cy,
                    "center_x_roi_px": cx,
                    "ring_spoke_amplitude_cv_percent_median": float(np.nanmedian(cvs)),
                    "ring_spoke_amplitude_cv_percent_mean": float(np.nanmean(cvs)),
                    "ring_spoke_amplitude_cv_percent_worst": float(np.nanmax(cvs)),
                    "ring_harmonic_snr_db_median": float(np.nanmedian(harmonic_snr)),
                    "image_mean": float(np.mean(image)),
                    "image_std": float(np.std(image)),
                    "image_robust_dynamic_range_p99_p01": float(np.percentile(image, 99.0) - np.percentile(image, 1.0)),
                    "star_horizontal_sides_cutoff_radius_px": horizontal.cutoff_radius_px,
                    "star_horizontal_sides_cutoff_censored": horizontal.cutoff_censored,
                    "star_vertical_sides_cutoff_radius_px": vertical.cutoff_radius_px,
                    "star_vertical_sides_cutoff_censored": vertical.cutoff_censored,
                    "psd_roi_y": f"{psd_slices[0].start}:{psd_slices[0].stop}",
                    "psd_roi_x": f"{psd_slices[1].start}:{psd_slices[1].stop}",
                }
                summary.update(
                    frequency_resolution_metrics(
                        "star_horizontal_sides", horizontal.cutoff_cyc_per_px, args.pixel_size, args.pixel_unit
                    )
                )
                summary.update(
                    frequency_resolution_metrics(
                        "star_vertical_sides", vertical.cutoff_cyc_per_px, args.pixel_size, args.pixel_unit
                    )
                )
                for direction, result in psd_results.items():
                    summary.update(
                        frequency_resolution_metrics(
                            f"psd_{direction}", result.cutoff_cyc_per_px, args.pixel_size, args.pixel_unit
                        )
                    )
                    summary[f"psd_{direction}_noise_baseline"] = result.noise_baseline
                    summary[f"psd_{direction}_threshold"] = result.threshold
                    summary[f"psd_{direction}_cutoff_censored"] = result.cutoff_censored

                fx = float(summary["psd_x_cutoff_cyc_per_px"])
                fy = float(summary["psd_y_cutoff_cyc_per_px"])
                summary["psd_cutoff_x_over_y"] = fx / fy if np.isfinite(fx) and np.isfinite(fy) and fy > 0 else float("nan")
                hx = float(summary["star_horizontal_sides_half_pitch_px"])
                vy = float(summary["star_vertical_sides_half_pitch_px"])
                summary["star_vertical_over_horizontal_half_pitch_ratio"] = vy / hx if np.isfinite(hx) and np.isfinite(vy) and hx > 0 else float("nan")
                summary_rows.append(summary)

                print(
                    f"  [{index + 1:>3}/{n}] z1={z1:.8g}: median ring CV={summary['ring_spoke_amplitude_cv_percent_median']:.3g}% "
                    f"PSD period x/y={summary['psd_x_full_period_px']:.3g}/{summary['psd_y_full_period_px']:.3g}px "
                    f"star half-pitch LR/TB={summary['star_horizontal_sides_half_pitch_px']:.3g}/"
                    f"{summary['star_vertical_sides_half_pitch_px']:.3g}px"
                )

        add_relative_quality_scores(summary_rows)
        all_channel_rows = aggregate_all_channels(summary_rows)

        write_csv(output_dir / "ring_uniformity.csv", ring_rows)
        write_csv(output_dir / "resolution_and_summary_metrics.csv", summary_rows)
        write_csv(output_dir / "z1_combined_relative_ranking.csv", all_channel_rows)

        if not args.no_plots:
            plot_summary_metrics(output_dir, summary_rows, ring_rows, args.pixel_size, args.pixel_unit)
            plot_best_diagnostics(
                output_dir,
                h5,
                summary_rows,
                z1_values,
                polar_sampler,
                harmonic,
                args,
                psd_slices,
                psd_tail,
            )

    print(f"Done. Results written to: {output_dir}")
    print("Main tables:")
    print(f"  {output_dir / 'ring_uniformity.csv'}")
    print(f"  {output_dir / 'resolution_and_summary_metrics.csv'}")
    print(f"  {output_dir / 'z1_combined_relative_ranking.csv'}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
