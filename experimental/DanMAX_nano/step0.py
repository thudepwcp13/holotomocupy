#!/usr/bin/env python
"""DanMAX nano step 0: near-field ptychography probe calibration."""
from __future__ import annotations

import configparser
import os
import sys
from types import SimpleNamespace
from typing import Tuple

import h5py
import numpy as np

try:
    import cupy as cp
    from mpi4py import MPI
    from holotomocupy.rec_nfp_mpi import RecNFP
    from holotomocupy.logger_config import logger, set_log_level
except Exception as exc:
    raise RuntimeError(
        "step0.py requires cupy, mpi4py, and holotomocupy installed with `pip install -e .`"
    ) from exc


def _bool(cfg: configparser.SectionProxy, key: str, fallback: bool) -> bool:
    value = cfg.get(key, fallback=str(fallback))
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _parse_args(config_file: str) -> SimpleNamespace:
    parser = configparser.ConfigParser(inline_comment_prefixes=("#",), interpolation=None)
    with open(config_file, "r", encoding="utf-8") as f:
        parser.read_string("[DEFAULT]\n" + f.read())
    cfg = parser["DEFAULT"]

    args = SimpleNamespace()
    args.dark_file = cfg.get("dark_file")
    args.flat_file = cfg.get("flat_file")
    args.sample_file = cfg.get("sample_file")
    args.h5_out = cfg.get("h5_out")
    args.path_out = cfg.get("path_out", fallback=None)
    args.detector_path = cfg.get("detector_path", fallback="/entry/measurement/orca")
    args.x_path = cfg.get("x_path", fallback="/entry/measurement/tom_sam_x")
    args.y_path = cfg.get("y_path", fallback="/entry/measurement/tom_y")
    args.energy = cfg.getfloat("energy")
    args.z1 = cfg.getfloat("z1")
    args.focustodetectordistance = cfg.getfloat("focustodetectordistance")
    args.detector_pixelsize = cfg.getfloat("detector_pixelsize")
    args.position_unit = cfg.get("position_unit", fallback="um").strip().lower()
    args.pos_row_sign = cfg.getfloat("pos_row_sign", fallback=-1.0)
    args.pos_col_sign = cfg.getfloat("pos_col_sign", fallback=1.0)
    args.center_positions = _bool(cfg, "center_positions", True)
    args.n = cfg.getint("n", fallback=2048)
    args.niter = cfg.getint("niter", fallback=129)
    args.nchunk = cfg.getint("nchunk", fallback=4)
    args.vis_step = cfg.getint("vis_step", fallback=32)
    args.err_step = cfg.getint("err_step", fallback=32)
    args.rho = [float(x.strip()) for x in cfg.get("rho", fallback="1,2,0.1").split(",") if x.strip()]
    if len(args.rho) != 3:
        raise ValueError("rho must contain exactly three comma-separated values: proj,probe,pos")
    args.flat_correct = _bool(cfg, "flat_correct", True)
    args.run_reconstruction = _bool(cfg, "run_reconstruction", False)
    args.write_corrected_preview = _bool(cfg, "write_corrected_preview", True)
    args.preview_count = cfg.getint("preview_count", fallback=8)
    args.write_position_bbox_plot = _bool(cfg, "write_position_bbox_plot", True)
    args.position_bbox_grid_size = cfg.getint("position_bbox_grid_size", fallback=5)
    if args.position_bbox_grid_size < 1:
        raise ValueError("position_bbox_grid_size must be >= 1")
    args.log_level = cfg.get("log_level", fallback="INFO")
    return args


def _require_dataset(fid: h5py.File, path: str) -> h5py.Dataset:
    if path not in fid:
        raise KeyError(f"Required dataset {path!r} was not found in {fid.filename}")
    ds = fid[path]
    if not isinstance(ds, h5py.Dataset):
        raise TypeError(f"{path!r} in {fid.filename} is not an HDF5 dataset")
    return ds


def _as_3d_shape(shape: Tuple[int, ...], file_name: str, path: str) -> Tuple[int, int, int]:
    if len(shape) == 2:
        return 1, int(shape[0]), int(shape[1])
    if len(shape) == 3:
        return int(shape[0]), int(shape[1]), int(shape[2])
    raise ValueError(f"Expected 2-D or 3-D detector data at {path!r} in {file_name}, got {shape}")


def _axis_crop_pad(length: int, n: int) -> Tuple[slice, Tuple[int, int]]:
    if n <= length:
        start = (length - n) // 2
        return slice(start, start + n), (0, 0)
    deficit = n - length
    before = deficit // 2
    return slice(0, length), (before, deficit - before)


def _center_crop_pad(ny: int, nx: int, n: int):
    if n <= 0:
        n = max(ny, nx)
    if n > max(ny, nx):
        raise ValueError(f"Requested n={n} exceeds both detector dimensions {(ny, nx)}")
    yc, yp = _axis_crop_pad(ny, n)
    xc, xp = _axis_crop_pad(nx, n)
    return yc, xc, (yp, xp), n


def _pad_2d(arr: np.ndarray, pad, value: float) -> np.ndarray:
    if pad == ((0, 0), (0, 0)):
        return arr.astype("float32", copy=False)
    return np.pad(arr, pad, mode="constant", constant_values=value).astype("float32")


def _pad_stack(arr: np.ndarray, pad, value: float) -> np.ndarray:
    if pad == ((0, 0), (0, 0)):
        return arr.astype("float32", copy=False)
    return np.pad(arr, ((0, 0), pad[0], pad[1]), mode="constant", constant_values=value).astype("float32")


def _read_mean_image(file_name: str, path: str, crop, pad) -> np.ndarray:
    with h5py.File(file_name, "r") as f:
        ds = _require_dataset(f, path)
        nframes, _, _ = _as_3d_shape(ds.shape, file_name, path)
        acc = np.zeros(
            (crop[0].stop - crop[0].start, crop[1].stop - crop[1].start),
            dtype="float64",
        )
        if len(ds.shape) == 2:
            acc += ds[crop[0], crop[1]].astype("float64")
        else:
            for i in range(nframes):
                acc += ds[i, crop[0], crop[1]].astype("float64")
        mean_img = (acc / max(nframes, 1)).astype("float32")
    fill = float(np.nanmedian(mean_img)) if mean_img.size else 0.0
    return _pad_2d(mean_img, pad, fill)


def _read_sample_frames(file_name: str, path: str, crop, pad, frame_slice) -> np.ndarray:
    with h5py.File(file_name, "r") as f:
        ds = _require_dataset(f, path)
        raw = (
            ds[crop[0], crop[1]][None]
            if len(ds.shape) == 2
            else ds[frame_slice, crop[0], crop[1]]
        )
    return _pad_stack(raw.astype("float32"), pad, np.nan)


def _read_positions(sample_file: str, x_path: str, y_path: str):
    with h5py.File(sample_file, "r") as f:
        x = np.asarray(_require_dataset(f, x_path)[()], dtype="float64").reshape(-1)
        y = np.asarray(_require_dataset(f, y_path)[()], dtype="float64").reshape(-1)
    if x.shape != y.shape:
        raise ValueError(f"Position arrays have different lengths: {x.shape} and {y.shape}")
    return x, y


def _unit_scale_to_m(unit: str, voxelsize: float) -> float:
    scales = {
        "m": 1.0,
        "meter": 1.0,
        "metre": 1.0,
        "mm": 1e-3,
        "um": 1e-6,
        "µm": 1e-6,
        "micron": 1e-6,
        "nm": 1e-9,
        "px": voxelsize,
        "pixel": voxelsize,
        "pixels": voxelsize,
    }
    if unit not in scales:
        raise ValueError(f"Unsupported position_unit={unit!r}; use one of {sorted(scales)}")
    return scales[unit]


def _prepare_sample_chunk(
    raw: np.ndarray,
    dark: np.ndarray,
    flat_minus_dark: np.ndarray,
    flat_correct: bool,
) -> np.ndarray:
    """Prepare intensity data in either flat-corrected or raw-normalized mode.

    flat_correct=True:
        (sample-dark)/(flat-dark), with padded pixels mapped to intensity 1.
    flat_correct=False:
        sample-dark only. Padded pixels remain NaN until global normalization, then
        are replaced with normalized background intensity 1.
    """
    data = raw.astype("float32") - dark[None]
    data[data < 0] = 0
    if flat_correct:
        data = data / flat_minus_dark[None]
        data[~np.isfinite(data)] = 1.0
    else:
        data[~np.isfinite(raw)] = np.nan
    data[data < 0] = 0
    return data.astype("float32")


def _finite_sum_count(data: np.ndarray) -> Tuple[float, int]:
    finite = np.isfinite(data)
    return float(np.nansum(data)), int(finite.sum())


def _normalize_global(data: np.ndarray, comm) -> Tuple[np.ndarray, float]:
    local_sum, local_count = _finite_sum_count(data)
    total_sum = comm.allreduce(local_sum, op=MPI.SUM)
    total_count = comm.allreduce(local_count, op=MPI.SUM)
    global_mean = total_sum / max(total_count, 1)
    data = data / max(global_mean, 1e-6)
    data[~np.isfinite(data)] = 1.0
    return data.astype("float32"), float(global_mean)


def _normalize_preview(data: np.ndarray) -> Tuple[np.ndarray, float]:
    mean = float(np.nanmean(data))
    data = data / max(mean, 1e-6)
    data[~np.isfinite(data)] = 1.0
    return data.astype("float32"), mean


def _axis_scan_levels(values: np.ndarray, grid_size: int) -> np.ndarray:
    """Estimate scan levels and keep every other level when enough levels exist."""
    values = np.asarray(values, dtype="float64").reshape(-1)
    if values.size == 0:
        return np.empty(0, dtype="float64")

    # Regular NFP grids normally repeat the same row/column coordinates. Rounding
    # removes tiny readback jitter while retaining the motor-step separation.
    rounded = np.unique(np.round(values, decimals=3))
    max_regular_levels = max(2 * grid_size + 2, int(2 * np.sqrt(values.size)) + 2)
    if rounded.size <= max_regular_levels:
        levels = rounded.astype("float64")
    else:
        # Fallback for noisy/non-grid coordinates: estimate roughly sqrt(N) levels
        # from equal-population groups in sorted coordinate space.
        estimated_count = max(grid_size, int(round(np.sqrt(values.size))))
        groups = np.array_split(np.sort(values), estimated_count)
        levels = np.asarray([np.median(group) for group in groups if group.size], dtype="float64")

    levels = np.unique(levels)
    count = min(grid_size, levels.size)
    if count == 0:
        return np.empty(0, dtype="float64")

    # A 10x10 scan with grid_size=5 gives indices [0,2,4,6,8]: one scan
    # level is skipped between plotted boxes, as requested.
    if levels.size >= 2 * count:
        indices = np.arange(count, dtype=int) * 2
    else:
        indices = np.rint(np.linspace(0, levels.size - 1, count)).astype(int)
    return levels[indices]


def _select_position_grid(pos: np.ndarray, grid_size: int = 5):
    """Select measured frames nearest to a representative row x column grid."""
    row_levels = _axis_scan_levels(pos[:, 0], grid_size)
    col_levels = _axis_scan_levels(pos[:, 1], grid_size)
    if row_levels.size == 0 or col_levels.size == 0:
        raise ValueError("No positions are available for the bbox sanity plot")

    row_scale = max(float(np.ptp(pos[:, 0])), 1.0)
    col_scale = max(float(np.ptp(pos[:, 1])), 1.0)
    selected = np.empty((row_levels.size, col_levels.size), dtype="int32")
    used = set()

    for iy, row_target in enumerate(row_levels):
        for ix, col_target in enumerate(col_levels):
            distance = (
                ((pos[:, 0] - row_target) / row_scale) ** 2
                + ((pos[:, 1] - col_target) / col_scale) ** 2
            )
            for candidate in np.argsort(distance):
                candidate = int(candidate)
                if candidate not in used:
                    selected[iy, ix] = candidate
                    used.add(candidate)
                    break
            else:
                selected[iy, ix] = int(np.argmin(distance))

    return selected, row_levels, col_levels


def _write_position_bbox_plot(
    output_path: str,
    pos: np.ndarray,
    n: int,
    nobj: int,
    crop,
    pad,
    pos_row_sign: float,
    pos_col_sign: float,
    grid_size: int = 5,
):
    """Plot representative NFP object windows on the global object canvas.

    RecNFP's shift kernel samples object coordinates as

        object_col = local_col - pos_col + nobj/2
        object_row = local_row - pos_row + nobj/2

    so the n x n window top-left corner is canvas_center - n/2 - pos.
    The outer dashed box is the square solver window. The inner solid box
    is the real detector support after removing synthetic padding.
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
        from matplotlib.patches import Rectangle
    except ImportError as exc:
        raise RuntimeError(
            "The position bbox sanity plot requires matplotlib. "
            "Install it with `python -m pip install matplotlib`."
        ) from exc

    selected, row_levels, col_levels = _select_position_grid(pos, grid_size)
    flat_indices = selected.reshape(-1)
    selected_pos = pos[flat_indices].astype("float32")

    canvas_center = nobj / 2.0
    window_x0 = canvas_center - n / 2.0 - selected_pos[:, 1]
    window_y0 = canvas_center - n / 2.0 - selected_pos[:, 0]

    valid_h = int(crop[0].stop - crop[0].start)
    valid_w = int(crop[1].stop - crop[1].start)
    valid_x0 = window_x0 + int(pad[1][0])
    valid_y0 = window_y0 + int(pad[0][0])

    bbox_xywh = np.column_stack(
        [window_x0, window_y0, np.full_like(window_x0, n), np.full_like(window_y0, n)]
    ).astype("float32")
    valid_bbox_xywh = np.column_stack(
        [
            valid_x0,
            valid_y0,
            np.full_like(valid_x0, valid_w),
            np.full_like(valid_y0, valid_h),
        ]
    ).astype("float32")

    fig, ax = plt.subplots(figsize=(11, 11))
    cmap = plt.get_cmap("viridis")
    total = max(len(flat_indices) - 1, 1)

    for order, (frame_index, outer, inner) in enumerate(
        zip(flat_indices, bbox_xywh, valid_bbox_xywh)
    ):
        color = cmap(order / total)
        ax.add_patch(
            Rectangle(
                (outer[0], outer[1]),
                outer[2],
                outer[3],
                fill=False,
                edgecolor=color,
                linewidth=0.8,
                linestyle="--",
                alpha=0.35,
            )
        )
        ax.add_patch(
            Rectangle(
                (inner[0], inner[1]),
                inner[2],
                inner[3],
                fill=False,
                edgecolor=color,
                linewidth=1.2,
                alpha=0.75,
            )
        )
        center_x = outer[0] + outer[2] / 2.0
        center_y = outer[1] + outer[3] / 2.0
        ax.plot(center_x, center_y, marker=".", markersize=3, color=color)
        ax.text(
            center_x,
            center_y,
            str(int(frame_index)),
            fontsize=6,
            ha="center",
            va="center",
            color=color,
        )

    # Union of all solver windows, not only the selected 5x5 subset.
    all_x0 = canvas_center - n / 2.0 - pos[:, 1]
    all_y0 = canvas_center - n / 2.0 - pos[:, 0]
    union_x0 = float(all_x0.min())
    union_y0 = float(all_y0.min())
    union_x1 = float((all_x0 + n).max())
    union_y1 = float((all_y0 + n).max())
    ax.add_patch(
        Rectangle(
            (union_x0, union_y0),
            union_x1 - union_x0,
            union_y1 - union_y0,
            fill=False,
            linewidth=2.0,
            linestyle="-",
            edgecolor="black",
        )
    )

    # Arrows between increasing configured column/row scan levels. Because the
    # shift kernel uses "-pos", these arrows show the actual object-window motion.
    grid_centers_x = canvas_center - selected_pos[:, 1]
    grid_centers_y = canvas_center - selected_pos[:, 0]
    centers_x = grid_centers_x.reshape(selected.shape)
    centers_y = grid_centers_y.reshape(selected.shape)

    label_box = dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.85, edgecolor="none")
    if selected.shape[1] > 1:
        start = (centers_x[0, 0], centers_y[0, 0])
        end = (centers_x[0, 1], centers_y[0, 1])
        ax.annotate("", xy=end, xytext=start, arrowprops=dict(arrowstyle="->", linewidth=2.5))
        ax.text(
            0.5 * (start[0] + end[0]),
            0.5 * (start[1] + end[1]) - 70,
            "+ configured col",
            fontsize=9,
            ha="center",
            va="bottom",
            bbox=label_box,
        )
    if selected.shape[0] > 1:
        start = (centers_x[0, 0], centers_y[0, 0])
        end = (centers_x[1, 0], centers_y[1, 0])
        ax.annotate("", xy=end, xytext=start, arrowprops=dict(arrowstyle="->", linewidth=2.5))
        ax.text(
            0.5 * (start[0] + end[0]) + 70,
            0.5 * (start[1] + end[1]),
            "+ configured row",
            fontsize=9,
            ha="left",
            va="center",
            rotation=90,
            bbox=label_box,
        )

    # Explicit positive motor directions derived from pos_row_sign/pos_col_sign.
    # Axes-fraction coordinates keep this direction key readable above the boxes.
    motor_dx_axes = -np.sign(pos_col_sign) * 0.12
    motor_dy_axes = np.sign(pos_row_sign) * 0.12
    if motor_dx_axes == 0:
        motor_dx_axes = 0.12
    if motor_dy_axes == 0:
        motor_dy_axes = 0.12
    motor_origin = (0.22, 0.90)

    ax.annotate(
        "",
        xy=(motor_origin[0] + motor_dx_axes, motor_origin[1]),
        xytext=motor_origin,
        xycoords="axes fraction",
        textcoords="axes fraction",
        arrowprops=dict(arrowstyle="-|>", linewidth=3.0),
    )
    ax.text(
        motor_origin[0],
        motor_origin[1] + 0.025,
        "+motor x / tom_sam_x",
        transform=ax.transAxes,
        fontsize=10,
        ha="center",
        va="bottom",
        bbox=label_box,
    )
    ax.annotate(
        "",
        xy=(motor_origin[0], motor_origin[1] + motor_dy_axes),
        xytext=motor_origin,
        xycoords="axes fraction",
        textcoords="axes fraction",
        arrowprops=dict(arrowstyle="-|>", linewidth=3.0),
    )
    ax.text(
        motor_origin[0] + 0.02,
        motor_origin[1],
        "+motor y / tom_y",
        transform=ax.transAxes,
        fontsize=10,
        ha="left",
        va="center",
        bbox=label_box,
    )

    ax.add_patch(
        Rectangle((0, 0), nobj, nobj, fill=False, linewidth=2.0, edgecolor="black")
    )
    ax.set_xlim(0, nobj)
    ax.set_ylim(nobj, 0)  # image convention: row increases downward
    ax.set_aspect("equal")
    ax.set_xlabel("global object column (pixel)")
    ax.set_ylabel("global object row (pixel, increasing downward)")
    ax.set_title(
        f"NFP position/bbox sanity check\n"
        f"object canvas={nobj}x{nobj}, solver window={n}x{n}, "
        f"real detector support={valid_h}x{valid_w}, "
        f"shown={selected.shape[0]}x{selected.shape[1]} measured positions"
    )
    ax.grid(True, linewidth=0.3, alpha=0.35)
    ax.legend(
        handles=[
            Line2D([0], [0], linestyle="--", linewidth=1.0, label="square NFP solver window"),
            Line2D([0], [0], linestyle="-", linewidth=1.5, label="real detector support"),
            Line2D([0], [0], linestyle="-", linewidth=2.0, color="black", label="union of all 100 windows"),
        ],
        loc="lower right",
        fontsize=8,
    )

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    return {
        "indices": flat_indices.astype("int32"),
        "grid_indices": selected.astype("int32"),
        "positions": selected_pos,
        "bbox_xywh": bbox_xywh,
        "valid_bbox_xywh": valid_bbox_xywh,
        "row_levels": row_levels.astype("float32"),
        "col_levels": col_levels.astype("float32"),
    }


def _write_or_replace(fid: h5py.File, name: str, data: np.ndarray) -> None:
    if name in fid:
        del fid[name]
    fid.create_dataset(name, data=data)


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python step0.py config_step0.conf")
    args = _parse_args(sys.argv[1])
    set_log_level(args.log_level)
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    ngpus = cp.cuda.runtime.getDeviceCount()
    if ngpus > 0:
        cp.cuda.Device(rank % ngpus).use()

    with (
        h5py.File(args.dark_file, "r") as fd,
        h5py.File(args.flat_file, "r") as ff,
        h5py.File(args.sample_file, "r") as fs,
    ):
        dark_shape = _require_dataset(fd, args.detector_path).shape
        flat_shape = _require_dataset(ff, args.detector_path).shape
        sample_shape = _require_dataset(fs, args.detector_path).shape
        ntheta, ny, nx = _as_3d_shape(sample_shape, args.sample_file, args.detector_path)
        n_dark, dny, dnx = _as_3d_shape(dark_shape, args.dark_file, args.detector_path)
        n_flat, fny, fnx = _as_3d_shape(flat_shape, args.flat_file, args.detector_path)

    if (dny, dnx) != (ny, nx) or (fny, fnx) != (ny, nx):
        raise ValueError(
            f"Detector sizes differ: dark={(dny, dnx)}, flat={(fny, fnx)}, sample={(ny, nx)}"
        )

    y_crop, x_crop, pad, n = _center_crop_pad(ny, nx, args.n)
    y_pad, x_pad = pad
    magnification = args.focustodetectordistance / args.z1
    voxelsize = args.detector_pixelsize / magnification
    wavelength = 1.24e-9 / args.energy

    x_raw, y_raw = _read_positions(args.sample_file, args.x_path, args.y_path)
    if len(x_raw) != ntheta:
        raise ValueError(f"Positions ({len(x_raw)}) do not match sample frames ({ntheta})")

    x_pos, y_pos = x_raw.copy(), y_raw.copy()
    if args.center_positions:
        x_pos -= np.mean(x_pos)
        y_pos -= np.mean(y_pos)

    scale = _unit_scale_to_m(args.position_unit, voxelsize)
    pos = np.empty((ntheta, 2), dtype="float32")
    pos[:, 0] = args.pos_row_sign * y_pos * scale / voxelsize
    pos[:, 1] = args.pos_col_sign * x_pos * scale / voxelsize

    pos_range = int(np.ceil(np.abs(pos).max())) + 8
    nobj = int(np.ceil((n + 2 * pos_range) / 32)) * 32

    mode_name = (
        "flat-corrected"
        if args.flat_correct
        else "dark-subtracted raw + global mean normalization"
    )
    if rank == 0:
        logger.info("=== DanMAX nano step 0 sanity check ===")
        logger.info(f"dark_file               = {args.dark_file}")
        logger.info(f"flat_file               = {args.flat_file}")
        logger.info(f"sample_file             = {args.sample_file}")
        logger.info(f"flat_correct            = {args.flat_correct} ({mode_name})")
        logger.info(f"dark shape              = {dark_shape}  frames={n_dark}")
        logger.info(f"flat shape              = {flat_shape}  frames={n_flat}")
        logger.info(f"sample shape            = {sample_shape}  frames={ntheta}")
        logger.info(
            f"crop                    = rows[{y_crop.start}:{y_crop.stop}], "
            f"cols[{x_crop.start}:{x_crop.stop}], n={n}"
        )
        logger.info(
            f"padding                 = rows before/after={y_pad}, "
            f"cols before/after={x_pad}"
        )
        logger.info(
            f"energy                  = {args.energy:.6g} keV  "
            f"wavelength={wavelength:.6e} m"
        )
        logger.info(f"magnification           = {magnification:.6g}")
        logger.info(
            f"voxelsize               = {voxelsize:.6e} m ({voxelsize * 1e9:.3f} nm)"
        )
        logger.info(
            f"positions pix row       = [{pos[:, 0].min():.3f}, {pos[:, 0].max():.3f}]"
        )
        logger.info(
            f"positions pix col       = [{pos[:, 1].min():.3f}, {pos[:, 1].max():.3f}]"
        )
        logger.info(f"nobj                    = {nobj}")

    bbox_plot_data = None
    bbox_plot_path = os.path.join(
        os.path.dirname(args.h5_out) or ".", "position_bbox_sanity.png"
    )
    if rank == 0 and args.write_position_bbox_plot:
        bbox_plot_data = _write_position_bbox_plot(
            output_path=bbox_plot_path,
            pos=pos,
            n=n,
            nobj=nobj,
            crop=(y_crop, x_crop),
            pad=pad,
            pos_row_sign=args.pos_row_sign,
            pos_col_sign=args.pos_col_sign,
            grid_size=args.position_bbox_grid_size,
        )
        logger.info(f"Wrote position/bbox sanity plot to {bbox_plot_path}")

    dark = _read_mean_image(
        args.dark_file, args.detector_path, (y_crop, x_crop), pad
    )
    flat = _read_mean_image(
        args.flat_file, args.detector_path, (y_crop, x_crop), pad
    )
    flat_minus_dark = flat - dark
    eps = max(float(np.nanmedian(flat_minus_dark)) * 1e-6, 1e-6)
    flat_minus_dark = np.where(
        flat_minus_dark > eps, flat_minus_dark, eps
    ).astype("float32")

    preview_count = min(args.preview_count, ntheta)
    preview = None
    preview_norm = np.nan
    if rank == 0 and preview_count > 0:
        raw_preview = _read_sample_frames(
            args.sample_file,
            args.detector_path,
            (y_crop, x_crop),
            pad,
            slice(0, preview_count),
        )
        preview_raw = _prepare_sample_chunk(
            raw_preview, dark, flat_minus_dark, args.flat_correct
        )
        preview, preview_norm = _normalize_preview(preview_raw)
        logger.info(
            f"flat-dark median        = {float(np.median(flat_minus_dark)):.6g}"
        )
        logger.info(f"input normalization mean= {preview_norm:.6g}")
        logger.info(f"prepared preview mean   = {float(preview.mean()):.6g}")
        logger.info(
            f"prepared preview p1/p99 = {np.percentile(preview, 1):.6g} / "
            f"{np.percentile(preview, 99):.6g}"
        )

    if rank == 0:
        os.makedirs(os.path.dirname(args.h5_out) or ".", exist_ok=True)
        with h5py.File(args.h5_out, "w") as fout:
            _write_or_replace(fout, "dark_mean", dark)
            _write_or_replace(fout, "flat_mean", flat)
            _write_or_replace(fout, "flat_minus_dark", flat_minus_dark)
            _write_or_replace(fout, "pos", pos)
            _write_or_replace(fout, "tom_sam_x", x_raw.astype("float32"))
            _write_or_replace(fout, "tom_y", y_raw.astype("float32"))
            if args.write_corrected_preview and preview is not None:
                _write_or_replace(fout, "corrected_preview", preview)
            if bbox_plot_data is not None:
                _write_or_replace(
                    fout, "bbox_preview_indices", bbox_plot_data["indices"]
                )
                _write_or_replace(
                    fout, "bbox_preview_grid_indices", bbox_plot_data["grid_indices"]
                )
                _write_or_replace(
                    fout, "bbox_preview_positions", bbox_plot_data["positions"]
                )
                _write_or_replace(
                    fout, "bbox_preview_xywh", bbox_plot_data["bbox_xywh"]
                )
                _write_or_replace(
                    fout,
                    "bbox_preview_valid_xywh",
                    bbox_plot_data["valid_bbox_xywh"],
                )
                _write_or_replace(
                    fout, "bbox_preview_row_levels", bbox_plot_data["row_levels"]
                )
                _write_or_replace(
                    fout, "bbox_preview_col_levels", bbox_plot_data["col_levels"]
                )

            fout.attrs["flat_correct"] = bool(args.flat_correct)
            fout.attrs["normalization_mode"] = mode_name
            fout.attrs["preview_normalization_mean"] = preview_norm
            fout.attrs["detector_path"] = args.detector_path
            fout.attrs["energy_keV"] = args.energy
            fout.attrs["wavelength_m"] = wavelength
            fout.attrs["z1_m"] = args.z1
            fout.attrs["focus_to_detector_distance_m"] = (
                args.focustodetectordistance
            )
            fout.attrs["detector_pixelsize_m"] = args.detector_pixelsize
            fout.attrs["magnification"] = magnification
            fout.attrs["voxelsize_m"] = voxelsize
            fout.attrs["crop_y_start"] = y_crop.start
            fout.attrs["crop_y_stop"] = y_crop.stop
            fout.attrs["crop_x_start"] = x_crop.start
            fout.attrs["crop_x_stop"] = x_crop.stop
            fout.attrs["pad_y_before"] = y_pad[0]
            fout.attrs["pad_y_after"] = y_pad[1]
            fout.attrs["pad_x_before"] = x_pad[0]
            fout.attrs["pad_x_after"] = x_pad[1]
            fout.attrs["n"] = n
            fout.attrs["nobj"] = nobj
            fout.attrs["position_bbox_plot"] = (
                bbox_plot_path if args.write_position_bbox_plot else ""
            )
        logger.info(f"Wrote sanity-check output to {args.h5_out}")

    comm.Barrier()
    if not args.run_reconstruction:
        if rank == 0:
            logger.info(
                "run_reconstruction=false: stopping after sanity check."
            )
        return

    path_out = os.path.join(args.path_out, "nfp") if args.path_out else None
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
        path_out=path_out,
        comm=comm,
    )
    cl = RecNFP(rec_args)

    raw_slice = _read_sample_frames(
        args.sample_file,
        args.detector_path,
        (y_crop, x_crop),
        pad,
        slice(cl.st_theta, cl.end_theta),
    )
    prepared = _prepare_sample_chunk(
        raw_slice, dark, flat_minus_dark, args.flat_correct
    )
    prepared, global_mean = _normalize_global(prepared, comm)
    if rank == 0:
        logger.info(f"NFP input mode          = {mode_name}")
        logger.info(
            f"NFP global mean before normalization = {global_mean:.6g}"
        )

    cl.data[:] = np.sqrt(np.abs(prepared)).astype("float32")
    cl.vars["proj"][:] = 0
    cl.vars["prb"][:] = 1
    cl.vars["pos"][:] = cp.array(pos[cl.st_theta:cl.end_theta])
    cl.BH()

    pos_err_local = cl.vars["pos"].get() - cl.pos_init.get()
    all_pos_err = comm.gather(pos_err_local, root=0)
    all_prb = comm.gather(cl.vars["prb"].get(), root=0)
    all_proj = comm.gather(cl.vars["proj"].get(), root=0)

    if rank == 0:
        pos_err = np.concatenate(all_pos_err, axis=0)
        prb_np = all_prb[0]
        proj_np = np.concatenate(all_proj, axis=0)
        with h5py.File(args.h5_out, "a") as fout:
            _write_or_replace(
                fout, "prb_amp", np.abs(prb_np).astype("float32")
            )
            _write_or_replace(
                fout, "prb_phase", np.angle(prb_np).astype("float32")
            )
            _write_or_replace(
                fout, "proj_delta", proj_np.real.astype("float32")
            )
            _write_or_replace(
                fout, "proj_beta", proj_np.imag.astype("float32")
            )
            _write_or_replace(fout, "pos_err", pos_err.astype("float32"))
            fout.attrs["nfp_input_global_mean"] = global_mean
        logger.info(f"Saved NFP reconstruction to {args.h5_out}")

    del cl
    cp.get_default_memory_pool().free_all_blocks()


if __name__ == "__main__":
    main()
