#!/usr/bin/env python
"""ID16B NFP step 0 v2: reconstruction with a weighted flat-field constraint.

The optimized objective is

    L = L_sample + flat_loss_weight * L_flat

with amplitude-domain mean-squared errors

    L_sample = mean((|D(P * exp(i S_pos(O)))| - A_sample)^2)
    L_flat   = mean((|D(P)|                    - A_flat)^2).

Both terms are normalized by their own detector-pixel counts, so a weight of 1
gives equal weight to their average per-pixel residuals. The flat term constrains
only the probe; the sample term continues to update projection, probe, and
positions.

Important: when ``flat_loss_weight > 0``, use ``flat_correct=false``. Dividing
the sample images by the flat before reconstruction removes the probe-amplitude
structure that the additional flat term is intended to fit.
"""
from __future__ import annotations

import configparser
import os
import sys
import time
from types import SimpleNamespace

import h5py
import numpy as np
import pandas as pd

import step0 as base


class FlatConstrainedRecNFP(base.RecNFP):
    """RecNFP with an additional weighted empty-beam amplitude loss."""

    def __init__(self, args):
        super().__init__(args)
        self.flat_amplitude = base.cp.ones((self.nz, self.n), dtype="float32")
        self.flat_loss_weight = np.float32(0.0)
        self.flat_size = float(self.nz * self.n)
        self.table = pd.DataFrame(
            columns=[
                "iter",
                "sample_err",
                "flat_err",
                "flat_weighted_err",
                "err",
                "time",
            ]
        )

    @staticmethod
    def _abs_safe(x):
        return base.cp.maximum(base.cp.abs(x), np.float32(1e-12))

    @staticmethod
    def _reprod(a, b):
        return base.cp.real(a) * base.cp.real(b) + base.cp.imag(a) * base.cp.imag(b)

    def set_flat_constraint(self, flat_amplitude: np.ndarray, weight: float) -> None:
        flat_amplitude = np.asarray(flat_amplitude, dtype="float32")
        if flat_amplitude.shape != (self.nz, self.n):
            raise ValueError(
                f"flat amplitude shape {flat_amplitude.shape} != {(self.nz, self.n)}"
            )
        if not np.all(np.isfinite(flat_amplitude)) or np.any(flat_amplitude < 0):
            raise ValueError("flat amplitude must be finite and non-negative")
        if not np.isfinite(weight) or weight < 0:
            raise ValueError("flat_loss_weight must be finite and >= 0")
        self.flat_amplitude = base.cp.asarray(flat_amplitude)
        self.flat_loss_weight = np.float32(weight)
        self.flat_size = float(self.nz * self.n)

    def _flat_forward(self, probe):
        return self.cl_prop.D(probe, 0)

    def _flat_loss(self, probe) -> float:
        predicted = self._flat_forward(probe)
        residual = base.cp.abs(predicted) - self.flat_amplitude
        value = base.cp.sum(residual * residual) / self.flat_size
        return float(value.get())

    def _flat_gradient(self, probe):
        predicted = self._flat_forward(probe)
        residual = predicted - (
            self.flat_amplitude * predicted / self._abs_safe(predicted)
        )
        scale = np.float32(2.0 * float(self.flat_loss_weight) / self.flat_size)
        return scale * self.cl_prop.DT(residual, 0)

    def _flat_hessian(self, probe, direction_y, direction_z) -> float:
        predicted = self._flat_forward(probe)
        propagated_y = self.cl_prop.D(direction_y, 0)
        propagated_z = self.cl_prop.D(direction_z, 0)
        abs_predicted = self._abs_safe(predicted)
        unit = predicted / abs_predicted
        data_ratio = self.flat_amplitude / abs_predicted
        value = (
            (1 - data_ratio) * self._reprod(propagated_y, propagated_z)
            + data_ratio
            * self._reprod(unit, propagated_y)
            * self._reprod(unit, propagated_z)
        )
        scale = np.float32(2.0 * float(self.flat_loss_weight) / self.flat_size)
        return float((scale * base.cp.sum(value)).get())

    def gradients(self, vars, grads):
        super().gradients(vars, grads)
        if self.flat_loss_weight > 0:
            grads["prb"][:] += self._flat_gradient(vars["prb"])

    def hessian(self, vars, grads, etas):
        value = super().hessian(vars, grads, etas)
        if self.flat_loss_weight > 0 and self.rank == 0:
            value += self._flat_hessian(
                vars["prb"], grads["prb"], etas["prb"]
            )
        return value

    def _sample_loss(self, vars) -> float:
        return super().min(vars["prb"], vars["proj"], vars["pos"])

    def loss_terms(self, vars):
        sample_err = self._sample_loss(vars)
        flat_err = self._flat_loss(vars["prb"])
        weighted_flat_err = float(self.flat_loss_weight) * flat_err
        return sample_err, flat_err, weighted_flat_err, sample_err + weighted_flat_err

    def min(self, prb, proj, pos):
        sample_err = super().min(prb, proj, pos)
        flat_err = self._flat_loss(prb)
        return sample_err + float(self.flat_loss_weight) * flat_err

    def error_debug(self, vars, iteration):
        if not (iteration % self.err_step == 0 and self.err_step != -1):
            return
        sample_err, flat_err, weighted_flat_err, total_err = self.loss_terms(vars)
        if self.rank != 0:
            return
        elapsed = 0.0 if iteration == -1 else time.time() - self.time_start
        prefix = "Initial" if iteration == -1 else f"iter={iteration}"
        base.logger.warning(
            f"{prefix}: sample={sample_err:1.5e} "
            f"flat={flat_err:1.5e} "
            f"weighted_flat={weighted_flat_err:1.5e} "
            f"total={total_err:1.5e}"
        )
        self.table.loc[len(self.table)] = [
            iteration,
            sample_err,
            flat_err,
            weighted_flat_err,
            total_err,
            elapsed,
        ]
        if iteration != -1:
            self.time_start = time.time()
        if getattr(self, "path_out", None):
            name = os.path.join(self.path_out, "conv_nfp_flat.csv")
            os.makedirs(os.path.dirname(name), exist_ok=True)
            self.table.to_csv(name, index=False)


def _parse_v2_args(path: str) -> SimpleNamespace:
    args = base._args(path)
    parser = configparser.ConfigParser(
        inline_comment_prefixes=("#",), interpolation=None
    )
    with open(path, "r", encoding="utf-8") as file:
        parser.read_string("[DEFAULT]\n" + file.read())
    cfg = parser["DEFAULT"]
    args.flat_loss_weight = cfg.getfloat("flat_loss_weight", fallback=1.0)
    args.save_flat_diagnostics = base._bool(
        cfg, "save_flat_diagnostics", True
    )
    if not np.isfinite(args.flat_loss_weight) or args.flat_loss_weight < 0:
        raise ValueError("flat_loss_weight must be finite and >= 0")
    if args.flat_loss_weight > 0 and args.flat_correct:
        raise ValueError(
            "flat_loss_weight > 0 requires flat_correct=false. Otherwise the "
            "sample data have already been divided by the flat while the forward "
            "model still multiplies by the recovered probe."
        )
    return args


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python step0_v2.py config_step0.conf")

    a = _parse_v2_args(sys.argv[1])
    base.set_log_level(a.log_level)
    comm = base.MPI.COMM_WORLD
    rank = comm.Get_rank()
    gpu_count = base.cp.cuda.runtime.getDeviceCount()
    if gpu_count:
        base.cp.cuda.Device(rank % gpu_count).use()

    shape = (a.n, a.n)
    ntheta = len(a.selected_scan_ids)
    if rank == 0:
        dark = base._dark_mean(a, shape)
        flat = base._flat_mean(a, shape)
        motor_x, motor_y = base._metadata(a, shape)
    else:
        dark = np.empty(shape, dtype="float32")
        flat = np.empty(shape, dtype="float32")
        motor_x = np.empty(ntheta, dtype="float64")
        motor_y = np.empty(ntheta, dtype="float64")
    comm.Bcast(dark, root=0)
    comm.Bcast(flat, root=0)
    comm.Bcast(motor_x, root=0)
    comm.Bcast(motor_y, root=0)

    flat_dark = flat - dark
    epsilon = max(float(np.median(flat_dark)) * 1e-6, 1e-6)
    flat_dark = np.where(flat_dark > epsilon, flat_dark, epsilon).astype(
        "float32"
    )

    magnification = a.focustodetectordistance / a.z1
    voxelsize = a.detector_pixelsize / magnification
    x = motor_x.copy()
    y = motor_y.copy()
    if a.center_positions:
        x -= x.mean()
        y -= y.mean()
    scale = base._unit_scale(a.position_unit, voxelsize)
    pos = np.stack(
        [
            a.pos_row_sign * y * scale / voxelsize,
            a.pos_col_sign * x * scale / voxelsize,
        ],
        axis=1,
    ).astype("float32")
    pos_range = int(np.ceil(np.abs(pos).max())) + 8
    nobj = int(np.ceil((a.n + 2 * pos_range) / 32)) * 32
    mode = (
        "flat-corrected"
        if a.flat_correct
        else "dark-subtracted raw + shared sample/flat intensity scale"
    )

    if rank == 0:
        base.logger.info("=== ID16B NFP step 0 v2: flat-constrained NFP ===")
        base.logger.info(
            f"dark key                = {a.dark_key}, frames={a.dark_nframes}"
        )
        base.logger.info(
            f"flat key example        = "
            f"{base._key(a.flat_key, a.flat_scan_ids[0])}"
        )
        base.logger.info(
            f"sample key example      = "
            f"{base._key(a.sample_key, a.selected_scan_ids[0])}"
        )
        base.logger.info(
            f"flat scans              = {base._fmt_ids(a.flat_scan_ids)}"
        )
        base.logger.info(
            f"sample scans            = {ntheta}/{len(a.sample_scan_ids)}: "
            f"{base._fmt_ids(a.selected_scan_ids)}"
        )
        base.logger.info(f"image shape             = {shape} (no crop/padding)")
        base.logger.info(f"input mode              = {mode}")
        base.logger.info(f"flat_loss_weight        = {a.flat_loss_weight:g}")
        base.logger.info(f"magnification           = {magnification:.6g}")
        base.logger.info(f"voxelsize               = {voxelsize * 1e9:.3f} nm")
        base.logger.info(
            f"position row            = [{pos[:, 0].min():.3f}, "
            f"{pos[:, 0].max():.3f}] px"
        )
        base.logger.info(
            f"position col            = [{pos[:, 1].min():.3f}, "
            f"{pos[:, 1].max():.3f}] px"
        )
        base.logger.info(f"nobj                    = {nobj}")

    bbox_data = None
    bbox_path = os.path.join(
        os.path.dirname(a.h5_out) or ".", "position_bbox_sanity_v2.png"
    )
    if rank == 0 and a.write_position_bbox_plot:
        bbox_data = base._bbox(
            bbox_path,
            pos,
            a.selected_scan_ids,
            a.n,
            nobj,
            a.position_bbox_grid_size,
        )
        base.logger.info(f"Wrote position/bbox sanity plot to {bbox_path}")

    preview = None
    preview_ids = a.selected_scan_ids[: min(a.preview_count, ntheta)]
    if rank == 0 and len(preview_ids):
        preview = np.empty((len(preview_ids), *shape), dtype="float32")
        with h5py.File(a.sample_file, "r") as file:
            for index, scan_id in enumerate(preview_ids):
                preview[index] = base._prepare(
                    base._image(
                        file, base._key(a.sample_key, scan_id), shape
                    ),
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
                base._write(out, name, value)
            if a.write_corrected_preview and preview is not None:
                base._write(out, "corrected_preview", preview)
                base._write(out, "preview_scan_ids", preview_ids)
            if bbox_data:
                for key, value in bbox_data.items():
                    base._write(out, f"bbox_preview_{key}", value)
            out.attrs.update(
                algorithm="step0_v2_flat_constrained_nfp",
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
                flat_correct=bool(a.flat_correct),
                flat_loss_weight=float(a.flat_loss_weight),
                save_flat_diagnostics=bool(a.save_flat_diagnostics),
            )
        base.logger.info(f"Wrote sanity output to {a.h5_out}")

    comm.Barrier()
    if not a.run_reconstruction:
        if rank == 0:
            base.logger.info(
                "run_reconstruction=false: stopping after sanity check."
            )
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
        path_out=(
            os.path.join(a.path_out, "nfp_v2") if a.path_out else None
        ),
        comm=comm,
    )
    rec = (
        FlatConstrainedRecNFP(rec_args)
        if a.flat_loss_weight > 0
        else base.RecNFP(rec_args)
    )

    local_ids = a.selected_scan_ids[rec.st_theta : rec.end_theta]
    mean = base._global_mean(a, local_ids, dark, flat_dark, shape, comm)
    base._load_data(rec.data, a, local_ids, dark, flat_dark, mean, shape)

    flat_normalized = flat_dark / max(mean, 1e-12)
    flat_normalized[~np.isfinite(flat_normalized)] = 0
    flat_normalized = np.maximum(flat_normalized, 0).astype("float32")
    flat_amplitude = np.sqrt(flat_normalized).astype("float32")
    if isinstance(rec, FlatConstrainedRecNFP):
        rec.set_flat_constraint(flat_amplitude, a.flat_loss_weight)

    if rank == 0:
        base.logger.info(f"NFP input mode          = {mode}")
        base.logger.info(
            f"NFP selected scans      = {ntheta}/{len(a.sample_scan_ids)}"
        )
        base.logger.info(
            f"shared intensity scale  = sample global mean {mean:.6g}"
        )
        base.logger.info(
            f"flat amplitude p1/p99   = "
            f"{np.percentile(flat_amplitude, 1):.6g} / "
            f"{np.percentile(flat_amplitude, 99):.6g}"
        )

    rec.vars["proj"][:] = 0
    rec.vars["prb"][:] = 1
    rec.vars["pos"][:] = base.cp.asarray(
        pos[rec.st_theta : rec.end_theta]
    )
    rec.BH()

    final_losses = None
    flat_pred_amplitude = None
    if isinstance(rec, FlatConstrainedRecNFP):
        final_losses = rec.loss_terms(rec.vars)
        if rank == 0 and a.save_flat_diagnostics:
            flat_pred_amplitude = base.cp.asnumpy(
                base.cp.abs(rec.cl_prop.D(rec.vars["prb"], 0))
            ).astype("float32")

    local_pos_err = rec.vars["pos"].get() - rec.pos_init.get()
    pos_errs = comm.gather(local_pos_err, root=0)

    if rank == 0:
        probe = rec.vars["prb"].get()
        projection = rec.vars["proj"].get()
        with h5py.File(a.h5_out, "a") as out:
            base._write(out, "prb_amp", np.abs(probe).astype("float32"))
            base._write(out, "prb_phase", np.angle(probe).astype("float32"))
            base._write(
                out, "proj_delta", projection.real.astype("float32")
            )
            base._write(
                out, "proj_beta", projection.imag.astype("float32")
            )
            base._write(
                out,
                "pos_err",
                np.concatenate(pos_errs).astype("float32"),
            )
            base._write(out, "flat_target_amplitude", flat_amplitude)
            if flat_pred_amplitude is not None:
                base._write(
                    out, "flat_pred_amplitude", flat_pred_amplitude
                )
                base._write(
                    out,
                    "flat_amplitude_residual",
                    flat_pred_amplitude - flat_amplitude,
                )
            out.attrs["nfp_input_global_mean"] = mean
            if final_losses is not None:
                sample_err, flat_err, weighted_flat_err, total_err = final_losses
                out.attrs["final_sample_amplitude_mse"] = sample_err
                out.attrs["final_flat_amplitude_mse"] = flat_err
                out.attrs["final_weighted_flat_amplitude_mse"] = (
                    weighted_flat_err
                )
                out.attrs["final_total_loss"] = total_err
        base.logger.info(
            f"Saved flat-constrained NFP result to {a.h5_out}"
        )

    del rec
    base.cp.get_default_memory_pool().free_all_blocks()


if __name__ == "__main__":
    main()
