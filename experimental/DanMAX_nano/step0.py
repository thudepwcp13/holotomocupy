#!/usr/bin/env python
"""DanMAX nano step 0: near-field ptychography probe calibration."""
from __future__ import annotations

import configparser
import os
import sys
from types import SimpleNamespace
from typing import Iterable, Tuple

import h5py
import numpy as np

try:
    import cupy as cp
    from mpi4py import MPI
    from holotomocupy.rec_nfp_mpi import RecNFP
    from holotomocupy.logger_config import logger, set_log_level
except Exception as exc:
    raise RuntimeError(
        "step0.py requires cupy, mpi4py, and holotomocupy installed with "
        "`pip install -e .`"
    ) from exc


class MaskedRecNFP(RecNFP):
    """RecNFP with a shared 2-D mask applied to loss, gradient, and Hessian."""

    def __init__(self, args):
        super().__init__(args)
        self.data_mask = cp.ones((self.nz, self.n), dtype="float32")

    def set_data_mask(self, mask: np.ndarray) -> None:
        mask = np.asarray(mask, dtype="float32")
        if mask.shape != (self.nz, self.n):
            raise ValueError(f"mask shape {mask.shape} != {(self.nz, self.n)}")
        if not np.all(np.isfinite(mask)) or np.any(mask < 0) or mask.sum() <= 0:
            raise ValueError(
                "mask must be finite, non-negative, and contain positive pixels"
            )
        self.data_mask = cp.asarray(mask)
        self.data_size = float(self.ntheta) * float(mask.sum())

    @staticmethod
    def _abs_safe(x):
        return cp.maximum(cp.abs(x), np.float32(1e-12))

    @staticmethod
    def _reprod(a, b):
        return cp.real(a) * cp.real(b) + cp.imag(a) * cp.imag(b)

    def F0(self, x, d):
        residual = cp.abs(x) - d
        return cp.sum(self.data_mask * residual * residual) / self.data_size

    def dF0(self, x, y, d, return_x=False):
        residual = self.data_mask * (x - d * x / self._abs_safe(x))
        return np.float32(2 / self.data_size) * cp.vdot(
            residual.view("float32"), y.view("float32")
        )

    def d2F_dF0(self, x, y, z, w, d):
        abs_x = self._abs_safe(x)
        l0 = x / abs_x
        d0 = d / abs_x
        value = (
            (1 - d0) * self._reprod(y, z)
            + d0 * self._reprod(l0, y) * self._reprod(l0, z)
        )
        if w is not None:
            value += self._reprod(x - d * l0, w)
        return np.float32(2 / self.data_size) * cp.sum(self.data_mask * value)

    def gF0(self, x, y):
        for idx in range(1, 4)[::-1]:
            x = self.F[idx](x)
        residual = x - y * x / self._abs_safe(x)
        return np.float32(2 / self.data_size) * self.data_mask * residual


def _bool(cfg, key: str, fallback: bool) -> bool:
    return cfg.get(key, fallback=str(fallback)).strip().lower() in {
        "1", "true", "yes", "y", "on"
    }


def _parse_args(path: str) -> SimpleNamespace:
    parser = configparser.ConfigParser(
        inline_comment_prefixes=("#",), interpolation=None
    )
    with open(path, "r", encoding="utf-8") as file:
        parser.read_string("[DEFAULT]\n" + file.read())
    cfg = parser["DEFAULT"]

    args = SimpleNamespace(
        dark_file=cfg.get("dark_file"),
        flat_file=cfg.get("flat_file"),
        sample_file=cfg.get("sample_file"),
        h5_out=cfg.get("h5_out"),
        path_out=cfg.get("path_out", fallback=None),
        detector_path=cfg.get("detector_path", fallback="/entry/measurement/orca"),
        x_path=cfg.get("x_path", fallback="/entry/measurement/tom_sam_x"),
        y_path=cfg.get("y_path", fallback="/entry/measurement/tom_y"),
        energy=cfg.getfloat("energy"),
        z1=cfg.getfloat("z1"),
        focustodetectordistance=cfg.getfloat("focustodetectordistance"),
        detector_pixelsize=cfg.getfloat("detector_pixelsize"),
        position_unit=cfg.get("position_unit", fallback="um").strip().lower(),
        pos_row_sign=cfg.getfloat("pos_row_sign", fallback=-1),
        pos_col_sign=cfg.getfloat("pos_col_sign", fallback=1),
        center_positions=_bool(cfg, "center_positions", True),
        frame_ids_spec=cfg.get("frame_ids", fallback="all").strip(),
        n=cfg.getint("n", fallback=2048),
        niter=cfg.getint("niter", fallback=129),
        nchunk=cfg.getint("nchunk", fallback=4),
        vis_step=cfg.getint("vis_step", fallback=32),
        err_step=cfg.getint("err_step", fallback=32),
        flat_correct=_bool(cfg, "flat_correct", True),
        use_valid_detector_mask=_bool(cfg, "use_valid_detector_mask", False),
        run_reconstruction=_bool(cfg, "run_reconstruction", False),
        write_corrected_preview=_bool(cfg, "write_corrected_preview", True),
        preview_count=cfg.getint("preview_count", fallback=8),
        write_position_bbox_plot=_bool(cfg, "write_position_bbox_plot", True),
        position_bbox_grid_size=cfg.getint("position_bbox_grid_size", fallback=5),
        log_level=cfg.get("log_level", fallback="INFO"),
    )
    args.rho = [
        float(value.strip())
        for value in cfg.get("rho", fallback="1,2,0.1").split(",")
        if value.strip()
    ]
    if len(args.rho) != 3:
        raise ValueError("rho must contain proj,probe,pos")
    if args.preview_count < 0:
        raise ValueError("preview_count must be >= 0")
    if args.position_bbox_grid_size < 1:
        raise ValueError("position_bbox_grid_size must be >= 1")
    return args


def _parse_frame_ids(spec: str, total_frames: int) -> np.ndarray:
    """Parse selected original frame IDs.

    Supported forms:
      frame_ids=all
      frame_ids=0,1,9,10
      frame_ids=0-24
      frame_ids=0-99:2       # inclusive range with step
      frame_ids=0:100:2      # Python slice syntax, stop is exclusive

    The specified order is preserved. Duplicate and out-of-range IDs are rejected
    so detector frames and motor positions cannot silently become misaligned.
    """
    if total_frames < 1:
        raise ValueError("sample dataset contains no frames")

    text = str(spec).strip()
    if not text or text.lower() in {"all", "*", "none"}:
        return np.arange(total_frames, dtype="int64")

    result: list[int] = []
    for raw_token in text.split(","):
        token = raw_token.strip()
        if not token:
            continue

        if ":" in token and "-" not in token:
            parts = token.split(":")
            if len(parts) not in {2, 3}:
                raise ValueError(f"Invalid frame slice token {token!r}")
            start = int(parts[0]) if parts[0] else 0
            stop = int(parts[1]) if parts[1] else total_frames
            step = int(parts[2]) if len(parts) == 3 and parts[2] else 1
            if step <= 0:
                raise ValueError(f"Frame slice step must be positive in {token!r}")
            result.extend(range(start, stop, step))
            continue

        range_part, separator, step_text = token.partition(":")
        step = int(step_text) if separator else 1
        if step <= 0:
            raise ValueError(f"Frame range step must be positive in {token!r}")
        if "-" in range_part:
            start_text, stop_text = range_part.split("-", 1)
            start, stop = int(start_text), int(stop_text)
            if start > stop:
                raise ValueError(f"Frame range start exceeds stop in {token!r}")
            result.extend(range(start, stop + 1, step))
        else:
            if separator:
                raise ValueError(f"Invalid frame token {token!r}")
            result.append(int(range_part))

    if not result:
        raise ValueError("frame_ids selected no frames")

    frame_ids = np.asarray(result, dtype="int64")
    invalid = frame_ids[(frame_ids < 0) | (frame_ids >= total_frames)]
    if invalid.size:
        raise ValueError(
            f"frame_ids contains out-of-range IDs {invalid.tolist()}; "
            f"valid range is 0..{total_frames - 1}"
        )
    unique_ids, counts = np.unique(frame_ids, return_counts=True)
    duplicates = unique_ids[counts > 1]
    if duplicates.size:
        raise ValueError(f"frame_ids contains duplicate IDs {duplicates.tolist()}")
    return frame_ids


def _format_frame_ids(frame_ids: np.ndarray, max_items: int = 20) -> str:
    values = np.asarray(frame_ids).reshape(-1)
    if values.size <= max_items:
        return ",".join(str(int(value)) for value in values)
    head_count = max_items // 2
    tail_count = max_items - head_count
    head = ",".join(str(int(value)) for value in values[:head_count])
    tail = ",".join(str(int(value)) for value in values[-tail_count:])
    return f"{head},...,{tail}"


def _dataset(fid: h5py.File, path: str) -> h5py.Dataset:
    if path not in fid or not isinstance(fid[path], h5py.Dataset):
        raise KeyError(f"HDF5 dataset {path!r} missing in {fid.filename}")
    return fid[path]


def _shape3(shape: Tuple[int, ...], file_name: str, path: str) -> Tuple[int, int, int]:
    if len(shape) == 2:
        return 1, int(shape[0]), int(shape[1])
    if len(shape) == 3:
        return tuple(map(int, shape))
    raise ValueError(f"Expected 2-D/3-D data at {path} in {file_name}, got {shape}")


def _axis_crop_pad(length: int, n: int):
    if n <= length:
        start = (length - n) // 2
        return slice(start, start + n), (0, 0)
    deficit = n - length
    return slice(0, length), (deficit // 2, deficit - deficit // 2)


def _center_crop_pad(ny: int, nx: int, n: int):
    n = max(ny, nx) if n <= 0 else n
    if n > max(ny, nx):
        raise ValueError(f"n={n} exceeds detector dimensions {(ny, nx)}")
    y_crop, y_pad = _axis_crop_pad(ny, n)
    x_crop, x_pad = _axis_crop_pad(nx, n)
    return y_crop, x_crop, (y_pad, x_pad), n


def _valid_mask(crop, pad, n: int) -> np.ndarray:
    mask = np.zeros((n, n), dtype="float32")
    height = crop[0].stop - crop[0].start
    width = crop[1].stop - crop[1].start
    y0, x0 = pad[0][0], pad[1][0]
    mask[y0:y0 + height, x0:x0 + width] = 1
    return mask


def _pad2(array: np.ndarray, pad, value: float) -> np.ndarray:
    if pad != ((0, 0), (0, 0)):
        array = np.pad(array, pad, mode="constant", constant_values=value)
    return np.asarray(array, dtype="float32")


def _pad3(array: np.ndarray, pad, value: float) -> np.ndarray:
    if pad != ((0, 0), (0, 0)):
        array = np.pad(
            array, ((0, 0), pad[0], pad[1]), mode="constant", constant_values=value
        )
    return np.asarray(array, dtype="float32")


def _mean_image(file_name: str, path: str, crop, pad) -> np.ndarray:
    with h5py.File(file_name, "r") as file:
        dataset = _dataset(file, path)
        nframes, _, _ = _shape3(dataset.shape, file_name, path)
        if dataset.ndim == 2:
            output = dataset[crop[0], crop[1]].astype("float64")
        else:
            output = np.zeros(
                (crop[0].stop - crop[0].start, crop[1].stop - crop[1].start),
                dtype="float64",
            )
            for index in range(nframes):
                output += dataset[index, crop[0], crop[1]]
            output /= max(nframes, 1)
    return _pad2(output, pad, float(np.nanmedian(output)))


def _selector_to_ids(selector: slice | Iterable[int] | np.ndarray, nframes: int) -> np.ndarray:
    if isinstance(selector, slice):
        return np.arange(nframes, dtype="int64")[selector]
    if isinstance(selector, np.ndarray):
        ids = selector
    else:
        ids = np.asarray(list(selector))
    return ids.astype("int64", copy=False).reshape(-1)


def _sample_frames(
    file_name: str,
    path: str,
    crop,
    pad,
    frame_ids: slice | Iterable[int] | np.ndarray,
) -> np.ndarray:
    """Read arbitrary original frame IDs while preserving requested order."""
    with h5py.File(file_name, "r") as file:
        dataset = _dataset(file, path)
        if dataset.ndim == 2:
            ids = _selector_to_ids(frame_ids, 1)
            if ids.size != 1 or int(ids[0]) != 0:
                raise ValueError("A 2-D detector dataset only supports frame_ids=0")
            output = dataset[crop[0], crop[1]][None]
        else:
            ids = _selector_to_ids(frame_ids, int(dataset.shape[0]))
            if ids.size == 0:
                height = crop[0].stop - crop[0].start
                width = crop[1].stop - crop[1].start
                output = np.empty((0, height, width), dtype=dataset.dtype)
            else:
                order = np.argsort(ids)
                sorted_ids = ids[order]
                sorted_output = dataset[sorted_ids, crop[0], crop[1]]
                inverse = np.empty_like(order)
                inverse[order] = np.arange(order.size)
                output = sorted_output[inverse]
    return _pad3(np.asarray(output, dtype="float32"), pad, np.nan)


def _positions(file_name: str, x_path: str, y_path: str):
    with h5py.File(file_name, "r") as file:
        x = np.asarray(_dataset(file, x_path)[()], "float64").reshape(-1)
        y = np.asarray(_dataset(file, y_path)[()], "float64").reshape(-1)
    if x.shape != y.shape:
        raise ValueError("x/y position arrays have different lengths")
    return x, y


def _unit_scale(unit: str, voxelsize: float) -> float:
    scales = {
        "m": 1, "meter": 1, "metre": 1, "mm": 1e-3, "um": 1e-6,
        "µm": 1e-6, "micron": 1e-6, "nm": 1e-9, "px": voxelsize,
        "pixel": voxelsize, "pixels": voxelsize,
    }
    if unit not in scales:
        raise ValueError(f"unsupported position_unit={unit}")
    return scales[unit]


def _prepare(raw: np.ndarray, dark: np.ndarray, flat_dark: np.ndarray, flat_correct: bool) -> np.ndarray:
    data = raw.astype("float32") - dark[None]
    data[data < 0] = 0
    if flat_correct:
        data /= flat_dark[None]
        data[~np.isfinite(data)] = 1
    else:
        data[~np.isfinite(raw)] = np.nan
    return data.astype("float32")


def _normalize(data: np.ndarray, comm=None):
    total = float(np.nansum(data))
    count = int(np.isfinite(data).sum())
    if comm is not None:
        total = comm.allreduce(total, op=MPI.SUM)
        count = comm.allreduce(count, op=MPI.SUM)
    mean = total / max(count, 1)
    data = data / max(mean, 1e-6)
    data[~np.isfinite(data)] = 1
    return data.astype("float32"), float(mean)


def _bbox_plot(
    path: str,
    pos: np.ndarray,
    frame_ids: np.ndarray,
    n: int,
    nobj: int,
    crop,
    pad,
    row_sign: float,
    col_sign: float,
    grid: int = 5,
):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    count = min(grid, 5)
    rows = np.linspace(pos[:, 0].min(), pos[:, 0].max(), count)
    cols = np.linspace(pos[:, 1].min(), pos[:, 1].max(), count)
    selected_local = []
    for row in rows:
        for col in cols:
            selected_local.append(
                int(np.argmin((pos[:, 0] - row) ** 2 + (pos[:, 1] - col) ** 2))
            )
    selected_local = np.asarray(selected_local, dtype="int32")
    selected_original = frame_ids[selected_local].astype("int64")

    center = nobj / 2
    height = crop[0].stop - crop[0].start
    width = crop[1].stop - crop[1].start
    boxes, valid_boxes = [], []
    figure, axes = plt.subplots(figsize=(10, 10))
    for order, (local_id, original_id) in enumerate(zip(selected_local, selected_original)):
        x0 = center - n / 2 - pos[local_id, 1]
        y0 = center - n / 2 - pos[local_id, 0]
        boxes.append((x0, y0, n, n))
        valid_boxes.append((x0 + pad[1][0], y0 + pad[0][0], width, height))
        color = plt.cm.viridis(order / max(len(selected_local) - 1, 1))
        axes.add_patch(Rectangle((x0, y0), n, n, fill=False, ls="--", lw=0.7, color=color, alpha=0.35))
        axes.add_patch(Rectangle((x0 + pad[1][0], y0 + pad[0][0]), width, height, fill=False, lw=1.1, color=color))
        axes.text(x0 + n / 2, y0 + n / 2, str(int(original_id)), fontsize=6, ha="center", color=color)

    origin = (0.22, 0.9)
    dx = -np.sign(col_sign) * 0.12
    dy = np.sign(row_sign) * 0.12
    axes.annotate("", (origin[0] + dx, origin[1]), origin, xycoords="axes fraction", arrowprops=dict(arrowstyle="->", lw=2.5))
    axes.annotate("", (origin[0], origin[1] + dy), origin, xycoords="axes fraction", arrowprops=dict(arrowstyle="->", lw=2.5))
    axes.text(origin[0], origin[1] + 0.02, "+motor x", transform=axes.transAxes, ha="center")
    axes.text(origin[0] + 0.02, origin[1], "+motor y", transform=axes.transAxes)
    axes.add_patch(Rectangle((0, 0), nobj, nobj, fill=False, lw=2, color="black"))
    axes.set(
        xlim=(0, nobj), ylim=(nobj, 0), aspect="equal",
        xlabel="global object column (pixel)",
        ylabel="global object row (pixel, downward)",
        title=(
            "NFP position/bbox sanity check\n"
            f"canvas={nobj}x{nobj}, solver={n}x{n}, valid={height}x{width}, "
            f"selected frames={len(frame_ids)}"
        ),
    )
    axes.grid(alpha=0.25)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    figure.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(figure)
    return {
        "indices": selected_local,
        "frame_ids": selected_original,
        "positions": pos[selected_local],
        "xywh": np.asarray(boxes, dtype="float32"),
        "valid_xywh": np.asarray(valid_boxes, dtype="float32"),
    }


def _write(file: h5py.File, name: str, value) -> None:
    if name in file:
        del file[name]
    file.create_dataset(name, data=value)


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python step0.py config_step0.conf")

    args = _parse_args(sys.argv[1])
    set_log_level(args.log_level)
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    gpu_count = cp.cuda.runtime.getDeviceCount()
    if gpu_count > 0:
        cp.cuda.Device(rank % gpu_count).use()

    with (
        h5py.File(args.dark_file, "r") as dark_file,
        h5py.File(args.flat_file, "r") as flat_file,
        h5py.File(args.sample_file, "r") as sample_file,
    ):
        dark_shape = _dataset(dark_file, args.detector_path).shape
        flat_shape = _dataset(flat_file, args.detector_path).shape
        sample_shape = _dataset(sample_file, args.detector_path).shape

    total_ntheta, ny, nx = _shape3(sample_shape, args.sample_file, args.detector_path)
    n_dark, dark_ny, dark_nx = _shape3(dark_shape, args.dark_file, args.detector_path)
    n_flat, flat_ny, flat_nx = _shape3(flat_shape, args.flat_file, args.detector_path)
    if (dark_ny, dark_nx) != (ny, nx) or (flat_ny, flat_nx) != (ny, nx):
        raise ValueError("dark/flat/sample detector dimensions differ")

    frame_ids = _parse_frame_ids(args.frame_ids_spec, total_ntheta)
    ntheta = int(frame_ids.size)

    y_crop, x_crop, pad, n = _center_crop_pad(ny, nx, args.n)
    mask = _valid_mask((y_crop, x_crop), pad, n)
    y_pad, x_pad = pad

    magnification = args.focustodetectordistance / args.z1
    voxelsize = args.detector_pixelsize / magnification
    wavelength = 1.24e-9 / args.energy

    x_raw_all, y_raw_all = _positions(args.sample_file, args.x_path, args.y_path)
    if len(x_raw_all) != total_ntheta:
        raise ValueError("number of positions does not match number of sample frames")

    x_raw = x_raw_all[frame_ids]
    y_raw = y_raw_all[frame_ids]
    x_position = x_raw.copy()
    y_position = y_raw.copy()
    if args.center_positions:
        x_position -= x_position.mean()
        y_position -= y_position.mean()

    scale = _unit_scale(args.position_unit, voxelsize)
    pos = np.stack(
        [
            args.pos_row_sign * y_position * scale / voxelsize,
            args.pos_col_sign * x_position * scale / voxelsize,
        ],
        axis=1,
    ).astype("float32")

    pos_range = int(np.ceil(np.abs(pos).max())) + 8
    nobj = int(np.ceil((n + 2 * pos_range) / 32)) * 32
    mode = "flat-corrected" if args.flat_correct else "dark-subtracted raw + global mean normalization"

    if rank == 0:
        logger.info("=== DanMAX nano step 0 sanity check ===")
        for key, value in (
            ("dark_file", args.dark_file),
            ("flat_file", args.flat_file),
            ("sample_file", args.sample_file),
            ("flat_correct", f"{args.flat_correct} ({mode})"),
            ("use_valid_detector_mask", args.use_valid_detector_mask),
            ("frame_ids spec", args.frame_ids_spec),
        ):
            logger.info(f"{key:24s}= {value}")
        logger.info(f"dark shape              = {dark_shape}  frames={n_dark}")
        logger.info(f"flat shape              = {flat_shape}  frames={n_flat}")
        logger.info(f"sample shape            = {sample_shape}  total frames={total_ntheta}")
        logger.info(f"selected frames         = {ntheta}/{total_ntheta}: {_format_frame_ids(frame_ids)}")
        logger.info(f"crop                    = rows[{y_crop.start}:{y_crop.stop}], cols[{x_crop.start}:{x_crop.stop}], n={n}")
        logger.info(f"padding                 = rows before/after={y_pad}, cols before/after={x_pad}")
        logger.info(f"valid detector pixels   = {int(mask.sum())}/{n*n} ({100 * mask.mean():.2f}%)")
        logger.info(f"energy                  = {args.energy:g} keV  wavelength={wavelength:.6e} m")
        logger.info(f"magnification           = {magnification:.6g}")
        logger.info(f"voxelsize               = {voxelsize:.6e} m ({voxelsize * 1e9:.3f} nm)")
        logger.info(f"positions pix row       = [{pos[:, 0].min():.3f}, {pos[:, 0].max():.3f}]")
        logger.info(f"positions pix col       = [{pos[:, 1].min():.3f}, {pos[:, 1].max():.3f}]")
        logger.info(f"nobj                    = {nobj}")

    bbox_data = None
    bbox_path = os.path.join(os.path.dirname(args.h5_out) or ".", "position_bbox_sanity.png")
    if rank == 0 and args.write_position_bbox_plot:
        bbox_data = _bbox_plot(
            bbox_path, pos, frame_ids, n, nobj, (y_crop, x_crop), pad,
            args.pos_row_sign, args.pos_col_sign, args.position_bbox_grid_size,
        )
        logger.info(f"Wrote position/bbox sanity plot to {bbox_path}")

    dark = _mean_image(args.dark_file, args.detector_path, (y_crop, x_crop), pad)
    flat = _mean_image(args.flat_file, args.detector_path, (y_crop, x_crop), pad)
    flat_dark = flat - dark
    epsilon = max(float(np.nanmedian(flat_dark)) * 1e-6, 1e-6)
    flat_dark = np.where(flat_dark > epsilon, flat_dark, epsilon).astype("float32")

    preview = None
    preview_mean = np.nan
    preview_frame_ids = frame_ids[: min(args.preview_count, ntheta)]
    if rank == 0 and preview_frame_ids.size > 0:
        raw_preview = _sample_frames(
            args.sample_file, args.detector_path, (y_crop, x_crop), pad,
            preview_frame_ids,
        )
        preview, preview_mean = _normalize(
            _prepare(raw_preview, dark, flat_dark, args.flat_correct)
        )
        logger.info(f"preview frame IDs       = {_format_frame_ids(preview_frame_ids)}")
        logger.info(f"flat-dark median        = {np.median(flat_dark):.6g}")
        logger.info(f"input normalization mean= {preview_mean:.6g}")
        logger.info(
            f"prepared preview p1/p99 = {np.percentile(preview, 1):.6g} / "
            f"{np.percentile(preview, 99):.6g}"
        )

    if rank == 0:
        os.makedirs(os.path.dirname(args.h5_out) or ".", exist_ok=True)
        with h5py.File(args.h5_out, "w") as output:
            for name, value in (
                ("dark_mean", dark),
                ("flat_mean", flat),
                ("flat_minus_dark", flat_dark),
                ("valid_detector_mask", mask),
                ("frame_ids", frame_ids.astype("int64")),
                ("pos", pos),
                ("tom_sam_x", x_raw.astype("float32")),
                ("tom_y", y_raw.astype("float32")),
            ):
                _write(output, name, value)
            if args.write_corrected_preview and preview is not None:
                _write(output, "corrected_preview", preview)
                _write(output, "preview_frame_ids", preview_frame_ids.astype("int64"))
            if bbox_data:
                for key, value in bbox_data.items():
                    _write(output, f"bbox_preview_{key}", value)

            output.attrs.update(
                flat_correct=bool(args.flat_correct),
                normalization_mode=mode,
                use_valid_detector_mask=bool(args.use_valid_detector_mask),
                valid_detector_fraction=float(mask.mean()),
                frame_ids_spec=args.frame_ids_spec,
                total_sample_frames=int(total_ntheta),
                selected_frame_count=int(ntheta),
                preview_normalization_mean=preview_mean,
                detector_path=args.detector_path,
                energy_keV=args.energy,
                wavelength_m=wavelength,
                z1_m=args.z1,
                focus_to_detector_distance_m=args.focustodetectordistance,
                detector_pixelsize_m=args.detector_pixelsize,
                magnification=magnification,
                voxelsize_m=voxelsize,
                crop_y_start=y_crop.start,
                crop_y_stop=y_crop.stop,
                crop_x_start=x_crop.start,
                crop_x_stop=x_crop.stop,
                pad_y_before=y_pad[0],
                pad_y_after=y_pad[1],
                pad_x_before=x_pad[0],
                pad_x_after=x_pad[1],
                n=n,
                nobj=nobj,
                position_bbox_plot=bbox_path if args.write_position_bbox_plot else "",
            )
        logger.info(f"Wrote sanity-check output to {args.h5_out}")

    comm.Barrier()
    if not args.run_reconstruction:
        if rank == 0:
            logger.info("run_reconstruction=false: stopping after sanity check.")
        return

    rec_args = SimpleNamespace(
        energy=args.energy,
        detector_pixelsize=args.detector_pixelsize,
        focustodetectordistance=args.focustodetectordistance,
        z1=args.z1,
        ntheta=ntheta,
        nz=n,
        n=n,
        nzobj=nobj,
        nobj=nobj,
        obj_dtype="complex64",
        rho=args.rho,
        niter=args.niter,
        nchunk=args.nchunk,
        vis_step=args.vis_step,
        err_step=args.err_step,
        start_iter=0,
        path_out=os.path.join(args.path_out, "nfp") if args.path_out else None,
        comm=comm,
    )
    reconstruction = MaskedRecNFP(rec_args) if args.use_valid_detector_mask else RecNFP(rec_args)
    if args.use_valid_detector_mask:
        reconstruction.set_data_mask(mask)

    local_frame_ids = frame_ids[reconstruction.st_theta:reconstruction.end_theta]
    raw = _sample_frames(
        args.sample_file, args.detector_path, (y_crop, x_crop), pad, local_frame_ids
    )
    prepared, global_mean = _normalize(
        _prepare(raw, dark, flat_dark, args.flat_correct), comm
    )
    if rank == 0:
        logger.info(f"NFP input mode          = {mode}")
        logger.info(
            f"NFP detector mask       = "
            f"{'valid detector only' if args.use_valid_detector_mask else 'disabled'}"
        )
        logger.info(f"NFP selected frames     = {ntheta}/{total_ntheta}")
        logger.info(f"NFP global mean before normalization = {global_mean:.6g}")

    reconstruction.data[:] = np.sqrt(np.abs(prepared)).astype("float32")
    reconstruction.vars["proj"][:] = 0
    reconstruction.vars["prb"][:] = 1
    reconstruction.vars["pos"][:] = cp.asarray(
        pos[reconstruction.st_theta:reconstruction.end_theta]
    )
    reconstruction.BH()

    pos_errors = comm.gather(
        reconstruction.vars["pos"].get() - reconstruction.pos_init.get(), root=0
    )
    probes = comm.gather(reconstruction.vars["prb"].get(), root=0)
    projects = comm.gather(reconstruction.vars["proj"].get(), root=0)

    if rank == 0:
        probe = probes[0]
        projection = np.concatenate(projects, axis=0)
        with h5py.File(args.h5_out, "a") as output:
            _write(output, "prb_amp", np.abs(probe).astype("float32"))
            _write(output, "prb_phase", np.angle(probe).astype("float32"))
            _write(output, "proj_delta", projection.real.astype("float32"))
            _write(output, "proj_beta", projection.imag.astype("float32"))
            _write(output, "pos_err", np.concatenate(pos_errors, axis=0).astype("float32"))
            output.attrs["nfp_input_global_mean"] = global_mean
        logger.info(f"Saved NFP reconstruction to {args.h5_out}")

    del reconstruction
    cp.get_default_memory_pool().free_all_blocks()


if __name__ == "__main__":
    main()
