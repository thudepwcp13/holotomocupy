#!/usr/bin/env python
"""ID16B NFP step 0 for scan-dependent HDF5 keys."""
from __future__ import annotations

import configparser
import os
import sys
from types import SimpleNamespace

import cupy as cp
import h5py
import numpy as np
from holotomocupy.logger_config import logger, set_log_level
from holotomocupy.rec_nfp_mpi import RecNFP
from mpi4py import MPI


def _bool(cfg, key, fallback):
    return cfg.get(key, fallback=str(fallback)).strip().lower() in {
        "1", "true", "yes", "y", "on"
    }


def _clean(value):
    return str(value).strip().strip("\"'“”‘’")


def _parse_ids(spec, name):
    """Inclusive scan-ID parser: 1:10, 1:10:2, 1-10, or 1,5,8."""
    values = []
    for raw in _clean(spec).split(","):
        token = raw.strip()
        if not token:
            continue
        if ":" in token:
            parts = token.split(":")
            if len(parts) not in (2, 3):
                raise ValueError(f"Invalid {name} token: {token}")
            start, stop = int(parts[0]), int(parts[1])
            step = int(parts[2]) if len(parts) == 3 and parts[2] else 1
            if step <= 0 or start > stop:
                raise ValueError(f"Invalid {name} range: {token}")
            values.extend(range(start, stop + 1, step))
        elif "-" in token:
            start, stop = map(int, token.split("-", 1))
            if start > stop:
                raise ValueError(f"Invalid {name} range: {token}")
            values.extend(range(start, stop + 1))
        else:
            values.append(int(token))
    if not values:
        raise ValueError(f"{name} selected no IDs")
    ids = np.asarray(values, dtype="int64")
    if np.unique(ids).size != ids.size:
        raise ValueError(f"{name} contains duplicate IDs")
    return ids


def _fmt_ids(ids, limit=20):
    ids = np.asarray(ids).reshape(-1)
    if len(ids) <= limit:
        return ",".join(map(str, ids.tolist()))
    half = limit // 2
    return f"{','.join(map(str, ids[:half]))},...,{','.join(map(str, ids[-half:]))}"


def _key(template, scan_id):
    """Expand a scan-dependent HDF5 key template.

    Preferred forms are ``/${n}.1/...`` and ``/{n}.1/...``. A backslash-
    escaped ``${n}`` placeholder is also accepted. For robustness, this also
    repairs the shell-mangled form ``/${n.1/...}``, which can be produced when
    Bash interprets nested parameter-expansion braces in a launcher script.
    """
    original = _clean(template)
    template = original.replace(r"\${n}", "${n}")
    value = str(int(scan_id))

    if "${n}" in template:
        expanded = template.replace("${n}", value)
    elif "{n}" in template:
        expanded = template.replace("{n}", value)
    else:
        # Recover a common malformed form such as:
        #   /${n.1/measurement/pco1}
        # as:
        #   /<scan_id>.1/measurement/pco1
        start = template.find("${n")
        end = template.find("}", start + 3) if start >= 0 else -1
        if start < 0 or end < 0:
            raise ValueError(
                "Key template must contain ${n} or {n}; "
                f"received {original!r}"
            )
        inside_suffix = template[start + 3:end]
        expanded = template[:start] + value + inside_suffix + template[end + 1:]
        logger.warning(
            "Recovered malformed HDF5 key template %r as %r. "
            "Prefer an explicit ${n} or {n} placeholder.",
            original,
            expanded,
        )

    if "${n" in expanded or "{n}" in expanded:
        raise ValueError(
            f"Unresolved scan placeholder after expanding {original!r}: {expanded!r}"
        )
    if not expanded.startswith("/"):
        raise ValueError(f"Expanded HDF5 key must start with '/': {expanded!r}")
    return expanded


def _args(path):
    parser = configparser.ConfigParser(
        inline_comment_prefixes=("#",), interpolation=None
    )
    with open(path, "r", encoding="utf-8") as stream:
        parser.read_string("[DEFAULT]\n" + stream.read())
    c = parser["DEFAULT"]
    a = SimpleNamespace(
        dark_file=_clean(c.get("dark_file")),
        dark_key=_clean(c.get("dark_key")),
        dark_nframes=c.getint("dark_nframes", fallback=51),
        flat_file=_clean(c.get("flat_file")),
        flat_key=_clean(c.get("flat_key")),
        flat_scan_ids=_parse_ids(c.get("flat_scan_ids"), "flat_scan_ids"),
        sample_file=_clean(c.get("sample_file")),
        sample_key=_clean(c.get("sample_key")),
        sample_scan_ids=_parse_ids(c.get("sample_scan_ids"), "sample_scan_ids"),
        frame_ids_spec=_clean(c.get("frame_ids", fallback="all")),
        motor_x_key=_clean(c.get("motor_x_key")),
        motor_y_key=_clean(c.get("motor_y_key")),
        h5_out=_clean(c.get("h5_out")),
        path_out=_clean(c.get("path_out", fallback="")) or None,
        energy=c.getfloat("energy"),
        z1=c.getfloat("z1"),
        focustodetectordistance=c.getfloat("focustodetectordistance"),
        detector_pixelsize=c.getfloat("detector_pixelsize"),
        position_unit=_clean(c.get("position_unit", fallback="mm")).lower(),
        pos_row_sign=c.getfloat("pos_row_sign", fallback=-1),
        pos_col_sign=c.getfloat("pos_col_sign", fallback=1),
        center_positions=_bool(c, "center_positions", True),
        n=c.getint("n", fallback=2048),
        niter=c.getint("niter", fallback=10),
        nchunk=c.getint("nchunk", fallback=4),
        vis_step=c.getint("vis_step", fallback=1),
        err_step=c.getint("err_step", fallback=1),
        flat_correct=_bool(c, "flat_correct", False),
        run_reconstruction=_bool(c, "run_reconstruction", False),
        write_corrected_preview=_bool(c, "write_corrected_preview", True),
        preview_count=c.getint("preview_count", fallback=8),
        write_position_bbox_plot=_bool(c, "write_position_bbox_plot", True),
        position_bbox_grid_size=c.getint("position_bbox_grid_size", fallback=5),
        log_level=_clean(c.get("log_level", fallback="INFO")),
    )
    a.rho = [
        float(value.strip())
        for value in c.get("rho", fallback="1,2,0.00001").split(",")
    ]
    if len(a.rho) != 3:
        raise ValueError("rho must contain proj,probe,pos")

    if a.frame_ids_spec.lower() in ("", "all", "*"):
        a.selected_scan_ids = a.sample_scan_ids.copy()
    else:
        a.selected_scan_ids = _parse_ids(a.frame_ids_spec, "frame_ids")
        missing = a.selected_scan_ids[
            ~np.isin(a.selected_scan_ids, a.sample_scan_ids)
        ]
        if len(missing):
            raise ValueError(f"frame_ids not in sample_scan_ids: {missing.tolist()}")

    # Validate all key templates before opening large image stacks.
    first_flat_id = int(a.flat_scan_ids[0])
    first_sample_id = int(a.selected_scan_ids[0])
    _key(a.flat_key, first_flat_id)
    _key(a.sample_key, first_sample_id)
    _key(a.motor_x_key, first_sample_id)
    _key(a.motor_y_key, first_sample_id)
    return a


def _dataset(file, key):
    if key not in file or not isinstance(file[key], h5py.Dataset):
        raise KeyError(f"Missing HDF5 dataset {key!r} in {file.filename}")
    return file[key]


def _image(file, key, shape):
    ds = _dataset(file, key)
    if ds.ndim == 2:
        out = np.asarray(ds[()], dtype="float32")
    elif ds.ndim == 3 and ds.shape[0] == 1:
        out = np.asarray(ds[0], dtype="float32")
    else:
        raise ValueError(f"{key} must be HxW or 1xHxW, got {ds.shape}")
    if out.shape != shape:
        raise ValueError(f"{key} has shape {out.shape}, expected {shape}")
    return out


def _scalar(file, key):
    value = np.asarray(_dataset(file, key)[()])
    if value.size != 1:
        raise ValueError(f"{key} must contain one motor value, got {value.shape}")
    return float(value.reshape(-1)[0])


def _dark_mean(a, shape):
    with h5py.File(a.dark_file, "r") as file:
        ds = _dataset(file, a.dark_key)
        if ds.ndim != 3:
            raise ValueError(f"dark dataset must be 3-D, got {ds.shape}")
        if a.dark_nframes > ds.shape[0]:
            raise ValueError(
                f"dark_nframes={a.dark_nframes} > available {ds.shape[0]}"
            )
        out = (
            np.asarray(ds[:a.dark_nframes], dtype="float64")
            .mean(0)
            .astype("float32")
        )
    if out.shape != shape:
        raise ValueError(f"dark mean shape {out.shape}, expected {shape}")
    return out


def _flat_mean(a, shape):
    out = np.zeros(shape, dtype="float64")
    with h5py.File(a.flat_file, "r") as file:
        for scan_id in a.flat_scan_ids:
            out += _image(file, _key(a.flat_key, scan_id), shape)
    return (out / len(a.flat_scan_ids)).astype("float32")


def _metadata(a, shape):
    x = np.empty(len(a.selected_scan_ids), dtype="float64")
    y = np.empty_like(x)
    with h5py.File(a.sample_file, "r") as file:
        for index, scan_id in enumerate(a.selected_scan_ids):
            image_key = _key(a.sample_key, scan_id)
            ds = _dataset(file, image_key)
            valid = ds.shape == shape or (
                ds.ndim == 3 and ds.shape == (1, *shape)
            )
            if not valid:
                raise ValueError(
                    f"{image_key} has shape {ds.shape}, "
                    f"expected {shape} or {(1, *shape)}"
                )
            x[index] = _scalar(file, _key(a.motor_x_key, scan_id))
            y[index] = _scalar(file, _key(a.motor_y_key, scan_id))
    return x, y


def _unit_scale(unit, voxelsize):
    scales = {
        "m": 1,
        "mm": 1e-3,
        "um": 1e-6,
        "µm": 1e-6,
        "nm": 1e-9,
        "px": voxelsize,
        "pixel": voxelsize,
    }
    if unit not in scales:
        raise ValueError(f"Unsupported position_unit={unit}")
    return scales[unit]


def _prepare(raw, dark, flat_dark, flat_correct):
    data = raw.astype("float32") - dark
    np.maximum(data, 0, out=data)
    if flat_correct:
        data /= flat_dark
    data[~np.isfinite(data)] = 0
    return data


def _global_mean(a, local_ids, dark, flat_dark, shape, comm):
    total, count = 0.0, 0
    with h5py.File(a.sample_file, "r") as file:
        for scan_id in local_ids:
            data = _prepare(
                _image(file, _key(a.sample_key, scan_id), shape),
                dark,
                flat_dark,
                a.flat_correct,
            )
            total += float(data.sum(dtype="float64"))
            count += data.size
    total = comm.allreduce(total, op=MPI.SUM)
    count = comm.allreduce(count, op=MPI.SUM)
    return total / max(count, 1)


def _load_data(out, a, local_ids, dark, flat_dark, mean, shape):
    with h5py.File(a.sample_file, "r") as file:
        for index, scan_id in enumerate(local_ids):
            data = _prepare(
                _image(file, _key(a.sample_key, scan_id), shape),
                dark,
                flat_dark,
                a.flat_correct,
            )
            out[index] = np.sqrt(
                np.abs(data / max(mean, 1e-12))
            ).astype("float32")


def _bbox(path, pos, scan_ids, n, nobj, grid):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    rows = np.linspace(pos[:, 0].min(), pos[:, 0].max(), min(grid, len(pos)))
    cols = np.linspace(pos[:, 1].min(), pos[:, 1].max(), min(grid, len(pos)))
    selected = []
    for row in rows:
        for col in cols:
            selected.append(
                int(np.argmin((pos[:, 0] - row) ** 2 + (pos[:, 1] - col) ** 2))
            )
    selected = np.asarray(list(dict.fromkeys(selected)), dtype="int32")

    center = nobj / 2
    boxes = []
    fig, ax = plt.subplots(figsize=(10, 10))
    for order, index in enumerate(selected):
        x0 = center - n / 2 - pos[index, 1]
        y0 = center - n / 2 - pos[index, 0]
        boxes.append((x0, y0, n, n))
        color = plt.cm.viridis(order / max(len(selected) - 1, 1))
        ax.add_patch(Rectangle((x0, y0), n, n, fill=False, color=color))
        ax.text(
            x0 + n / 2,
            y0 + n / 2,
            str(int(scan_ids[index])),
            fontsize=6,
            ha="center",
            color=color,
        )
    ax.add_patch(Rectangle((0, 0), nobj, nobj, fill=False, color="black", lw=2))
    ax.set(
        xlim=(0, nobj),
        ylim=(nobj, 0),
        aspect="equal",
        xlabel="global object column",
        ylabel="global object row",
        title=f"ID16B NFP bbox: detector={n}x{n}, selected scans={len(scan_ids)}",
    )
    ax.grid(alpha=0.25)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return {
        "local_indices": selected,
        "scan_ids": scan_ids[selected],
        "positions": pos[selected],
        "xywh": np.asarray(boxes, dtype="float32"),
    }


def _write(file, name, value):
    if name in file:
        del file[name]
    file.create_dataset(name, data=value)


def main():
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python step0.py config_step0.conf")
    a = _args(sys.argv[1])
    set_log_level(a.log_level)

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    if cp.cuda.runtime.getDeviceCount():
        cp.cuda.Device(rank % cp.cuda.runtime.getDeviceCount()).use()

    shape = (a.n, a.n)
    ntheta = len(a.selected_scan_ids)
    if rank == 0:
        dark = _dark_mean(a, shape)
        flat = _flat_mean(a, shape)
        motor_x, motor_y = _metadata(a, shape)
    else:
        dark = np.empty(shape, "float32")
        flat = np.empty(shape, "float32")
        motor_x = np.empty(ntheta, "float64")
        motor_y = np.empty(ntheta, "float64")
    comm.Bcast(dark, root=0)
    comm.Bcast(flat, root=0)
    comm.Bcast(motor_x, root=0)
    comm.Bcast(motor_y, root=0)

    flat_dark = flat - dark
    eps = max(float(np.median(flat_dark)) * 1e-6, 1e-6)
    flat_dark = np.where(flat_dark > eps, flat_dark, eps).astype("float32")

    magnification = a.focustodetectordistance / a.z1
    voxelsize = a.detector_pixelsize / magnification
    x = motor_x.copy()
    y = motor_y.copy()
    if a.center_positions:
        x -= x.mean()
        y -= y.mean()
    scale = _unit_scale(a.position_unit, voxelsize)
    pos = np.stack(
        [
            a.pos_row_sign * y * scale / voxelsize,
            a.pos_col_sign * x * scale / voxelsize,
        ],
        axis=1,
    ).astype("float32")
    pos_range = int(np.ceil(np.abs(pos).max())) + 8
    nobj = int(np.ceil((a.n + 2 * pos_range) / 32)) * 32
    mode = "flat-corrected" if a.flat_correct else "dark-subtracted raw"

    if rank == 0:
        logger.info("=== ID16B NFP step 0 ===")
        logger.info(f"dark key                = {a.dark_key}, frames={a.dark_nframes}")
        logger.info(f"flat key example        = {_key(a.flat_key, a.flat_scan_ids[0])}")
        logger.info(f"sample key example      = {_key(a.sample_key, a.selected_scan_ids[0])}")
        logger.info(f"flat scans              = {_fmt_ids(a.flat_scan_ids)}")
        logger.info(
            f"sample scans            = {ntheta}/{len(a.sample_scan_ids)}: "
            f"{_fmt_ids(a.selected_scan_ids)}"
        )
        logger.info(f"image shape             = {shape} (no crop/padding)")
        logger.info(f"input mode              = {mode}")
        logger.info(f"magnification           = {magnification:.6g}")
        logger.info(f"voxelsize               = {voxelsize * 1e9:.3f} nm")
        logger.info(
            f"position row            = [{pos[:, 0].min():.3f}, "
            f"{pos[:, 0].max():.3f}] px"
        )
        logger.info(
            f"position col            = [{pos[:, 1].min():.3f}, "
            f"{pos[:, 1].max():.3f}] px"
        )
        logger.info(f"nobj                    = {nobj}")

    bbox_data = None
    bbox_path = os.path.join(
        os.path.dirname(a.h5_out) or ".", "position_bbox_sanity.png"
    )
    if rank == 0 and a.write_position_bbox_plot:
        bbox_data = _bbox(
            bbox_path,
            pos,
            a.selected_scan_ids,
            a.n,
            nobj,
            a.position_bbox_grid_size,
        )

    preview = None
    preview_ids = a.selected_scan_ids[: min(a.preview_count, ntheta)]
    if rank == 0 and len(preview_ids):
        preview = np.empty((len(preview_ids), *shape), "float32")
        with h5py.File(a.sample_file, "r") as file:
            for index, scan_id in enumerate(preview_ids):
                preview[index] = _prepare(
                    _image(file, _key(a.sample_key, scan_id), shape),
                    dark,
                    flat_dark,
                    a.flat_correct,
                )
        preview /= max(float(preview.mean(dtype="float64")), 1e-12)

    if rank == 0:
        os.makedirs(os.path.dirname(a.h5_out) or ".", exist_ok=True)
        with h5py.File(a.h5_out, "w") as out:
            for name, value in (
                ("dark_mean", dark),
                ("flat_mean", flat),
                ("flat_minus_dark", flat_dark),
                ("flat_scan_ids", a.flat_scan_ids),
                ("sample_scan_ids", a.sample_scan_ids),
                ("selected_scan_ids", a.selected_scan_ids),
                ("frame_ids", a.selected_scan_ids),
                ("motor_x", motor_x),
                ("motor_y", motor_y),
                ("pos", pos),
            ):
                _write(out, name, value)
            if a.write_corrected_preview and preview is not None:
                _write(out, "corrected_preview", preview)
                _write(out, "preview_scan_ids", preview_ids)
            if bbox_data:
                for key, value in bbox_data.items():
                    _write(out, f"bbox_preview_{key}", value)
            out.attrs.update(
                energy_keV=a.energy,
                z1_m=a.z1,
                focus_to_detector_distance_m=a.focustodetectordistance,
                detector_pixelsize_m=a.detector_pixelsize,
                magnification=magnification,
                voxelsize_m=voxelsize,
                n=a.n,
                nobj=nobj,
                frame_ids_spec=a.frame_ids_spec,
                normalization_mode=mode,
            )
        logger.info(f"Wrote sanity output to {a.h5_out}")

    comm.Barrier()
    if not a.run_reconstruction:
        return

    rec_args = SimpleNamespace(
        energy=a.energy,
        detector_pixelsize=a.detector_pixelsize,
        focustodetectordistance=a.focustodetectordistance,
        z1=a.z1,
        ntheta=ntheta,
        nz=a.n,
        n=a.n,
        nzobj=nobj,
        nobj=nobj,
        obj_dtype="complex64",
        rho=a.rho,
        niter=a.niter,
        nchunk=a.nchunk,
        vis_step=a.vis_step,
        err_step=a.err_step,
        start_iter=0,
        path_out=os.path.join(a.path_out, "nfp") if a.path_out else None,
        comm=comm,
    )
    rec = RecNFP(rec_args)
    local_ids = a.selected_scan_ids[rec.st_theta:rec.end_theta]
    mean = _global_mean(a, local_ids, dark, flat_dark, shape, comm)
    _load_data(rec.data, a, local_ids, dark, flat_dark, mean, shape)
    rec.vars["proj"][:] = 0
    rec.vars["prb"][:] = 1
    rec.vars["pos"][:] = cp.asarray(pos[rec.st_theta:rec.end_theta])
    rec.BH()

    local_pos_err = rec.vars["pos"].get() - rec.pos_init.get()
    pos_errs = comm.gather(local_pos_err, root=0)

    # Probe/projection are replicated MPI variables; write the rank-0 copy only.
    if rank == 0:
        probe = rec.vars["prb"].get()
        projection = rec.vars["proj"].get()
        with h5py.File(a.h5_out, "a") as out:
            _write(out, "prb_amp", np.abs(probe).astype("float32"))
            _write(out, "prb_phase", np.angle(probe).astype("float32"))
            _write(out, "proj_delta", projection.real.astype("float32"))
            _write(out, "proj_beta", projection.imag.astype("float32"))
            _write(out, "pos_err", np.concatenate(pos_errs).astype("float32"))
            out.attrs["nfp_input_global_mean"] = mean
        logger.info(f"Saved NFP result to {a.h5_out}")

    del rec
    cp.get_default_memory_pool().free_all_blocks()


if __name__ == "__main__":
    main()
