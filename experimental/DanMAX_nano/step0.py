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
            raise ValueError("mask must be finite, non-negative, and contain positive pixels")
        self.data_mask = cp.asarray(mask)
        self.data_size = float(self.ntheta) * float(mask.sum())

    @staticmethod
    def _abs_safe(x):
        return cp.maximum(cp.abs(x), np.float32(1e-12))

    @staticmethod
    def _reprod(a, b):
        return cp.real(a) * cp.real(b) + cp.imag(a) * cp.imag(b)

    def F0(self, x, d):
        r = cp.abs(x) - d
        return cp.sum(self.data_mask * r * r) / self.data_size

    def dF0(self, x, y, d, return_x=False):
        r = self.data_mask * (x - d * x / self._abs_safe(x))
        return np.float32(2 / self.data_size) * cp.vdot(r.view("float32"), y.view("float32"))

    def d2F_dF0(self, x, y, z, w, d):
        ax = self._abs_safe(x)
        l0, d0 = x / ax, d / ax
        v = (1 - d0) * self._reprod(y, z) + d0 * self._reprod(l0, y) * self._reprod(l0, z)
        if w is not None:
            v += self._reprod(x - d * l0, w)
        return np.float32(2 / self.data_size) * cp.sum(self.data_mask * v)

    def gF0(self, x, y):
        for idx in range(1, 4)[::-1]:
            x = self.F[idx](x)
        r = x - y * x / self._abs_safe(x)
        return np.float32(2 / self.data_size) * self.data_mask * r


def _bool(cfg, key, fallback):
    return cfg.get(key, fallback=str(fallback)).strip().lower() in {"1", "true", "yes", "y", "on"}


def _parse_args(path: str) -> SimpleNamespace:
    parser = configparser.ConfigParser(inline_comment_prefixes=("#",), interpolation=None)
    with open(path, "r", encoding="utf-8") as f:
        parser.read_string("[DEFAULT]\n" + f.read())
    c = parser["DEFAULT"]
    a = SimpleNamespace(
        dark_file=c.get("dark_file"), flat_file=c.get("flat_file"), sample_file=c.get("sample_file"),
        h5_out=c.get("h5_out"), path_out=c.get("path_out", fallback=None),
        detector_path=c.get("detector_path", fallback="/entry/measurement/orca"),
        x_path=c.get("x_path", fallback="/entry/measurement/tom_sam_x"),
        y_path=c.get("y_path", fallback="/entry/measurement/tom_y"),
        energy=c.getfloat("energy"), z1=c.getfloat("z1"),
        focustodetectordistance=c.getfloat("focustodetectordistance"),
        detector_pixelsize=c.getfloat("detector_pixelsize"),
        position_unit=c.get("position_unit", fallback="um").strip().lower(),
        pos_row_sign=c.getfloat("pos_row_sign", fallback=-1),
        pos_col_sign=c.getfloat("pos_col_sign", fallback=1),
        center_positions=_bool(c, "center_positions", True), n=c.getint("n", fallback=2048),
        niter=c.getint("niter", fallback=129), nchunk=c.getint("nchunk", fallback=4),
        vis_step=c.getint("vis_step", fallback=32), err_step=c.getint("err_step", fallback=32),
        flat_correct=_bool(c, "flat_correct", True),
        use_valid_detector_mask=_bool(c, "use_valid_detector_mask", False),
        run_reconstruction=_bool(c, "run_reconstruction", False),
        write_corrected_preview=_bool(c, "write_corrected_preview", True),
        preview_count=c.getint("preview_count", fallback=8),
        write_position_bbox_plot=_bool(c, "write_position_bbox_plot", True),
        position_bbox_grid_size=c.getint("position_bbox_grid_size", fallback=5),
        log_level=c.get("log_level", fallback="INFO"),
    )
    a.rho = [float(v.strip()) for v in c.get("rho", fallback="1,2,0.1").split(",") if v.strip()]
    if len(a.rho) != 3:
        raise ValueError("rho must contain proj,probe,pos")
    return a


def _dataset(fid, path):
    if path not in fid or not isinstance(fid[path], h5py.Dataset):
        raise KeyError(f"HDF5 dataset {path!r} missing in {fid.filename}")
    return fid[path]


def _shape3(shape: Tuple[int, ...], file_name: str, path: str):
    if len(shape) == 2:
        return 1, int(shape[0]), int(shape[1])
    if len(shape) == 3:
        return tuple(map(int, shape))
    raise ValueError(f"Expected 2-D/3-D data at {path} in {file_name}, got {shape}")


def _axis_crop_pad(length, n):
    if n <= length:
        start = (length - n) // 2
        return slice(start, start + n), (0, 0)
    d = n - length
    return slice(0, length), (d // 2, d - d // 2)


def _center_crop_pad(ny, nx, n):
    n = max(ny, nx) if n <= 0 else n
    if n > max(ny, nx):
        raise ValueError(f"n={n} exceeds detector dimensions {(ny, nx)}")
    yc, yp = _axis_crop_pad(ny, n)
    xc, xp = _axis_crop_pad(nx, n)
    return yc, xc, (yp, xp), n


def _valid_mask(crop, pad, n):
    mask = np.zeros((n, n), dtype="float32")
    h = crop[0].stop - crop[0].start
    w = crop[1].stop - crop[1].start
    y0, x0 = pad[0][0], pad[1][0]
    mask[y0:y0 + h, x0:x0 + w] = 1
    return mask


def _pad2(a, pad, value):
    return (a if pad == ((0, 0), (0, 0)) else np.pad(a, pad, constant_values=value)).astype("float32")


def _pad3(a, pad, value):
    return (a if pad == ((0, 0), (0, 0)) else np.pad(a, ((0, 0), pad[0], pad[1]), constant_values=value)).astype("float32")


def _mean_image(file_name, path, crop, pad):
    with h5py.File(file_name, "r") as f:
        d = _dataset(f, path)
        nf, _, _ = _shape3(d.shape, file_name, path)
        if d.ndim == 2:
            out = d[crop[0], crop[1]].astype("float64")
        else:
            out = np.zeros((crop[0].stop-crop[0].start, crop[1].stop-crop[1].start), "float64")
            for i in range(nf):
                out += d[i, crop[0], crop[1]]
            out /= nf
    return _pad2(out, pad, float(np.nanmedian(out)))


def _sample_frames(file_name, path, crop, pad, frames):
    with h5py.File(file_name, "r") as f:
        d = _dataset(f, path)
        out = d[crop[0], crop[1]][None] if d.ndim == 2 else d[frames, crop[0], crop[1]]
    return _pad3(out, pad, np.nan)


def _positions(file_name, x_path, y_path):
    with h5py.File(file_name, "r") as f:
        x = np.asarray(_dataset(f, x_path)[()], "float64").reshape(-1)
        y = np.asarray(_dataset(f, y_path)[()], "float64").reshape(-1)
    if x.shape != y.shape:
        raise ValueError("x/y position arrays have different lengths")
    return x, y


def _unit_scale(unit, voxelsize):
    scales = {"m": 1, "meter": 1, "metre": 1, "mm": 1e-3, "um": 1e-6, "µm": 1e-6,
              "micron": 1e-6, "nm": 1e-9, "px": voxelsize, "pixel": voxelsize, "pixels": voxelsize}
    if unit not in scales:
        raise ValueError(f"unsupported position_unit={unit}")
    return scales[unit]


def _prepare(raw, dark, flat_dark, flat_correct):
    data = raw.astype("float32") - dark[None]
    data[data < 0] = 0
    if flat_correct:
        data /= flat_dark[None]
        data[~np.isfinite(data)] = 1
    else:
        data[~np.isfinite(raw)] = np.nan
    return data.astype("float32")


def _normalize(data, comm=None):
    total, count = float(np.nansum(data)), int(np.isfinite(data).sum())
    if comm is not None:
        total = comm.allreduce(total, op=MPI.SUM)
        count = comm.allreduce(count, op=MPI.SUM)
    mean = total / max(count, 1)
    data = data / max(mean, 1e-6)
    data[~np.isfinite(data)] = 1
    return data.astype("float32"), float(mean)


def _bbox_plot(path, pos, n, nobj, crop, pad, row_sign, col_sign, grid=5):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    rows = np.linspace(pos[:, 0].min(), pos[:, 0].max(), min(grid, 5))
    cols = np.linspace(pos[:, 1].min(), pos[:, 1].max(), min(grid, 5))
    selected = []
    for r in rows:
        for c in cols:
            selected.append(int(np.argmin((pos[:, 0]-r)**2 + (pos[:, 1]-c)**2)))
    selected = np.asarray(selected, "int32")
    center = nobj / 2
    h, w = crop[0].stop-crop[0].start, crop[1].stop-crop[1].start
    boxes, valid_boxes = [], []
    fig, ax = plt.subplots(figsize=(10, 10))
    for order, idx in enumerate(selected):
        x0, y0 = center-n/2-pos[idx, 1], center-n/2-pos[idx, 0]
        boxes.append((x0, y0, n, n))
        valid_boxes.append((x0+pad[1][0], y0+pad[0][0], w, h))
        color = plt.cm.viridis(order / max(len(selected)-1, 1))
        ax.add_patch(Rectangle((x0, y0), n, n, fill=False, ls="--", lw=.7, color=color, alpha=.35))
        ax.add_patch(Rectangle((x0+pad[1][0], y0+pad[0][0]), w, h, fill=False, lw=1.1, color=color))
        ax.text(x0+n/2, y0+n/2, str(idx), fontsize=6, ha="center", color=color)
    origin = (.22, .9)
    dx, dy = -np.sign(col_sign)*.12, np.sign(row_sign)*.12
    ax.annotate("", (origin[0]+dx, origin[1]), origin, xycoords="axes fraction", arrowprops=dict(arrowstyle="->", lw=2.5))
    ax.annotate("", (origin[0], origin[1]+dy), origin, xycoords="axes fraction", arrowprops=dict(arrowstyle="->", lw=2.5))
    ax.text(origin[0], origin[1]+.02, "+motor x", transform=ax.transAxes, ha="center")
    ax.text(origin[0]+.02, origin[1], "+motor y", transform=ax.transAxes)
    ax.add_patch(Rectangle((0, 0), nobj, nobj, fill=False, lw=2, color="black"))
    ax.set(xlim=(0, nobj), ylim=(nobj, 0), aspect="equal",
           xlabel="global object column (pixel)", ylabel="global object row (pixel, downward)",
           title=f"NFP position/bbox sanity check\ncanvas={nobj}x{nobj}, solver={n}x{n}, valid={h}x{w}")
    ax.grid(alpha=.25)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return {"indices": selected, "positions": pos[selected], "xywh": np.asarray(boxes, "float32"),
            "valid_xywh": np.asarray(valid_boxes, "float32")}


def _write(f, name, value):
    if name in f:
        del f[name]
    f.create_dataset(name, data=value)


def main():
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python step0.py config_step0.conf")
    a = _parse_args(sys.argv[1])
    set_log_level(a.log_level)
    comm, rank = MPI.COMM_WORLD, MPI.COMM_WORLD.Get_rank()
    if cp.cuda.runtime.getDeviceCount() > 0:
        cp.cuda.Device(rank % cp.cuda.runtime.getDeviceCount()).use()

    with h5py.File(a.dark_file, "r") as fd, h5py.File(a.flat_file, "r") as ff, h5py.File(a.sample_file, "r") as fs:
        dark_shape, flat_shape, sample_shape = _dataset(fd, a.detector_path).shape, _dataset(ff, a.detector_path).shape, _dataset(fs, a.detector_path).shape
    ntheta, ny, nx = _shape3(sample_shape, a.sample_file, a.detector_path)
    n_dark, dny, dnx = _shape3(dark_shape, a.dark_file, a.detector_path)
    n_flat, fny, fnx = _shape3(flat_shape, a.flat_file, a.detector_path)
    if (dny, dnx) != (ny, nx) or (fny, fnx) != (ny, nx):
        raise ValueError("dark/flat/sample detector dimensions differ")

    yc, xc, pad, n = _center_crop_pad(ny, nx, a.n)
    mask = _valid_mask((yc, xc), pad, n)
    ypad, xpad = pad
    mag = a.focustodetectordistance / a.z1
    voxelsize = a.detector_pixelsize / mag
    wavelength = 1.24e-9 / a.energy
    xr, yr = _positions(a.sample_file, a.x_path, a.y_path)
    if len(xr) != ntheta:
        raise ValueError("number of positions does not match number of frames")
    x, y = xr.copy(), yr.copy()
    if a.center_positions:
        x -= x.mean(); y -= y.mean()
    scale = _unit_scale(a.position_unit, voxelsize)
    pos = np.stack([a.pos_row_sign*y*scale/voxelsize, a.pos_col_sign*x*scale/voxelsize], axis=1).astype("float32")
    pos_range = int(np.ceil(np.abs(pos).max())) + 8
    nobj = int(np.ceil((n + 2*pos_range)/32))*32
    mode = "flat-corrected" if a.flat_correct else "dark-subtracted raw + global mean normalization"

    if rank == 0:
        logger.info("=== DanMAX nano step 0 sanity check ===")
        for key, value in (("dark_file", a.dark_file), ("flat_file", a.flat_file), ("sample_file", a.sample_file),
                           ("flat_correct", f"{a.flat_correct} ({mode})"), ("use_valid_detector_mask", a.use_valid_detector_mask)):
            logger.info(f"{key:24s}= {value}")
        logger.info(f"dark shape              = {dark_shape}  frames={n_dark}")
        logger.info(f"flat shape              = {flat_shape}  frames={n_flat}")
        logger.info(f"sample shape            = {sample_shape}  frames={ntheta}")
        logger.info(f"crop                    = rows[{yc.start}:{yc.stop}], cols[{xc.start}:{xc.stop}], n={n}")
        logger.info(f"padding                 = rows before/after={ypad}, cols before/after={xpad}")
        logger.info(f"valid detector pixels   = {int(mask.sum())}/{n*n} ({100*mask.mean():.2f}%)")
        logger.info(f"energy                  = {a.energy:g} keV  wavelength={wavelength:.6e} m")
        logger.info(f"magnification           = {mag:.6g}")
        logger.info(f"voxelsize               = {voxelsize:.6e} m ({voxelsize*1e9:.3f} nm)")
        logger.info(f"positions pix row       = [{pos[:,0].min():.3f}, {pos[:,0].max():.3f}]")
        logger.info(f"positions pix col       = [{pos[:,1].min():.3f}, {pos[:,1].max():.3f}]")
        logger.info(f"nobj                    = {nobj}")

    bbox_data = None
    bbox_path = os.path.join(os.path.dirname(a.h5_out) or ".", "position_bbox_sanity.png")
    if rank == 0 and a.write_position_bbox_plot:
        bbox_data = _bbox_plot(bbox_path, pos, n, nobj, (yc, xc), pad, a.pos_row_sign, a.pos_col_sign, a.position_bbox_grid_size)
        logger.info(f"Wrote position/bbox sanity plot to {bbox_path}")

    dark = _mean_image(a.dark_file, a.detector_path, (yc, xc), pad)
    flat = _mean_image(a.flat_file, a.detector_path, (yc, xc), pad)
    flat_dark = flat-dark
    eps = max(float(np.nanmedian(flat_dark))*1e-6, 1e-6)
    flat_dark = np.where(flat_dark > eps, flat_dark, eps).astype("float32")
    preview, preview_mean = None, np.nan
    if rank == 0 and a.preview_count > 0:
        raw = _sample_frames(a.sample_file, a.detector_path, (yc, xc), pad, slice(0, min(a.preview_count, ntheta)))
        preview, preview_mean = _normalize(_prepare(raw, dark, flat_dark, a.flat_correct))
        logger.info(f"flat-dark median        = {np.median(flat_dark):.6g}")
        logger.info(f"input normalization mean= {preview_mean:.6g}")
        logger.info(f"prepared preview p1/p99 = {np.percentile(preview,1):.6g} / {np.percentile(preview,99):.6g}")

    if rank == 0:
        os.makedirs(os.path.dirname(a.h5_out) or ".", exist_ok=True)
        with h5py.File(a.h5_out, "w") as f:
            for name, value in (("dark_mean", dark), ("flat_mean", flat), ("flat_minus_dark", flat_dark),
                                ("valid_detector_mask", mask), ("pos", pos), ("tom_sam_x", xr.astype("float32")),
                                ("tom_y", yr.astype("float32"))):
                _write(f, name, value)
            if a.write_corrected_preview and preview is not None:
                _write(f, "corrected_preview", preview)
            if bbox_data:
                for key, value in bbox_data.items():
                    _write(f, f"bbox_preview_{key}", value)
            f.attrs.update(flat_correct=bool(a.flat_correct), normalization_mode=mode,
                           use_valid_detector_mask=bool(a.use_valid_detector_mask), valid_detector_fraction=float(mask.mean()),
                           preview_normalization_mean=preview_mean, detector_path=a.detector_path,
                           energy_keV=a.energy, wavelength_m=wavelength, z1_m=a.z1,
                           focus_to_detector_distance_m=a.focustodetectordistance,
                           detector_pixelsize_m=a.detector_pixelsize, magnification=mag, voxelsize_m=voxelsize,
                           crop_y_start=yc.start, crop_y_stop=yc.stop, crop_x_start=xc.start, crop_x_stop=xc.stop,
                           pad_y_before=ypad[0], pad_y_after=ypad[1], pad_x_before=xpad[0], pad_x_after=xpad[1],
                           n=n, nobj=nobj, position_bbox_plot=bbox_path if a.write_position_bbox_plot else "")
        logger.info(f"Wrote sanity-check output to {a.h5_out}")

    comm.Barrier()
    if not a.run_reconstruction:
        if rank == 0:
            logger.info("run_reconstruction=false: stopping after sanity check.")
        return

    rec_args = SimpleNamespace(energy=a.energy, detector_pixelsize=a.detector_pixelsize,
        focustodetectordistance=a.focustodetectordistance, z1=a.z1, ntheta=ntheta,
        nz=n, n=n, nzobj=nobj, nobj=nobj, obj_dtype="complex64", rho=a.rho,
        niter=a.niter, nchunk=a.nchunk, vis_step=a.vis_step, err_step=a.err_step,
        start_iter=0, path_out=os.path.join(a.path_out, "nfp") if a.path_out else None, comm=comm)
    cl = MaskedRecNFP(rec_args) if a.use_valid_detector_mask else RecNFP(rec_args)
    if a.use_valid_detector_mask:
        cl.set_data_mask(mask)

    raw = _sample_frames(a.sample_file, a.detector_path, (yc, xc), pad, slice(cl.st_theta, cl.end_theta))
    prepared, global_mean = _normalize(_prepare(raw, dark, flat_dark, a.flat_correct), comm)
    if rank == 0:
        logger.info(f"NFP input mode          = {mode}")
        logger.info(f"NFP detector mask       = {'valid detector only' if a.use_valid_detector_mask else 'disabled'}")
        logger.info(f"NFP global mean before normalization = {global_mean:.6g}")
    cl.data[:] = np.sqrt(np.abs(prepared)).astype("float32")
    cl.vars["proj"][:] = 0; cl.vars["prb"][:] = 1
    cl.vars["pos"][:] = cp.asarray(pos[cl.st_theta:cl.end_theta])
    cl.BH()

    pos_err = comm.gather(cl.vars["pos"].get()-cl.pos_init.get(), root=0)
    probes = comm.gather(cl.vars["prb"].get(), root=0)
    projects = comm.gather(cl.vars["proj"].get(), root=0)
    if rank == 0:
        prb, proj = probes[0], np.concatenate(projects, axis=0)
        with h5py.File(a.h5_out, "a") as f:
            _write(f, "prb_amp", np.abs(prb).astype("float32")); _write(f, "prb_phase", np.angle(prb).astype("float32"))
            _write(f, "proj_delta", proj.real.astype("float32")); _write(f, "proj_beta", proj.imag.astype("float32"))
            _write(f, "pos_err", np.concatenate(pos_err, axis=0).astype("float32"))
            f.attrs["nfp_input_global_mean"] = global_mean
        logger.info(f"Saved NFP reconstruction to {a.h5_out}")
    del cl
    cp.get_default_memory_pool().free_all_blocks()


if __name__ == "__main__":
    main()
