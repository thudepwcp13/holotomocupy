#!/usr/bin/env python
"""DanMAX nano step 0 v2: NFP reconstruction with a flat-field constraint.

The optimized objective is

    L = L_sample + flat_loss_weight * L_flat

with amplitude-domain mean-squared errors

    L_sample = mean_mask((|D(P * exp(i S_pos(O)))| - A_sample)^2)
    L_flat   = mean_mask((|D(P)|                    - A_flat)^2).

Both terms are normalized by their own number of valid detector pixels, so a
weight of 1 gives equal weight to their average per-pixel residuals.  The flat
term constrains only the probe; the sample term continues to update projection,
probe, and positions.

Important: when ``flat_loss_weight > 0``, use ``flat_correct=false``.  Dividing
sample images by the flat before reconstruction would remove the probe-amplitude
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


class FlatConstrainedRecNFP(base.MaskedRecNFP):
    """RecNFP with an additional weighted empty-beam amplitude loss.

    The flat branch is linear in the probe,

        probe -> D(probe) -> amplitude loss,

    and therefore contributes only to the probe gradient and to the probe-probe
    block of the Hessian used by the existing BH/CG optimizer.
    """

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

    def set_data_mask(self, mask: np.ndarray) -> None:
        super().set_data_mask(mask)
        self.flat_size = float(np.asarray(mask, dtype="float32").sum())

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
        self.flat_size = float(base.cp.asnumpy(base.cp.sum(self.data_mask)))
        if self.flat_size <= 0:
            raise ValueError("flat constraint mask contains no valid pixels")

    def _flat_forward(self, probe):
        return self.cl_prop.D(probe, 0)

    def _flat_loss(self, probe) -> float:
        predicted = self._flat_forward(probe)
        residual = base.cp.abs(predicted) - self.flat_amplitude
        value = base.cp.sum(self.data_mask * residual * residual) / self.flat_size
        return float(value.get())

    def _flat_gradient(self, probe):
        predicted = self._flat_forward(probe)
        residual = self.data_mask * (
            predicted
            - self.flat_amplitude * predicted / self._abs_safe(predicted)
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
        return float((scale * base.cp.sum(self.data_mask * value)).get())

    def gradients(self, vars, grads):
        # The parent method computes and MPI-reduces the sample-data gradient.
        super().gradients(vars, grads)
        if self.flat_loss_weight > 0:
            # The flat target and probe are replicated, so every rank adds the
            # same already-global flat gradient without another all-reduce.
            grads["prb"][:] += self._flat_gradient(vars["prb"])

    def hessian(self, vars, grads, etas):
        value = super().hessian(vars, grads, etas)
        if self.flat_loss_weight > 0 and self.rank == 0:
            # BH all-reduces this scalar immediately afterwards.  Add the
            # replicated flat contribution on rank 0 only to avoid multiplying
            # it by the MPI world size.
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
        if not (
            iteration % self.err_step == 0 and self.err_step != -1
        ):
            return

        sample_err, flat_err, flat_weighted_err, total_err = self.loss_terms(vars)
        if self.rank != 0:
            return

        elapsed = 0.0 if iteration == -1 else time.time() - self.time_start
        prefix = "Initial" if iteration == -1 else f"iter={iteration}"
        base.logger.warning(
            f"{prefix}: sample={sample_err:1.5e} "
            f"flat={flat_err:1.5e} "
            f"weighted_flat={flat_weighted_err:1.5e} "
            f"total={total_err:1.5e}"
        )
        self.table.loc[len(self.table)] = [
            iteration,
            sample_err,
            flat_err,
            flat_weighted_err,
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
    args = base._parse_args(path)
    parser = configparser.ConfigParser(
        inline_comment_prefixes=("#",), interpolation=None
    )
    with open(path, "r", encoding="utf-8") as file:
        parser.read_string("[DEFAULT]\n" + file.read())
    cfg = parser["DEFAULT"]

    args.flat_loss_weight = cfg.getfloat("flat_loss_weight", fallback=1.0)
    args.save_flat_diagnostics = base._bool(cfg, "save_flat_diagnostics", True)
    if not np.isfinite(args.flat_loss_weight) or args.flat_loss_weight < 0:
        raise ValueError("flat_loss_weight must be finite and >= 0")
    if args.flat_loss_weight > 0 and args.flat_correct:
        raise ValueError(
            "flat_loss_weight > 0 requires flat_correct=false.  Otherwise the "
            "sample data have already been divided by the flat while the forward "
            "model still multiplies by the recovered probe."
        )
    return args


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python step0_v2.py config_step0.conf")

    args = _parse_v2_args(sys.argv[1])
    base.set_log_level(args.log_level)
    comm = base.MPI.COMM_WORLD
    rank = comm.Get_rank()

    gpu_count = base.cp.cuda.runtime.getDeviceCount()
    if gpu_count > 0:
        base.cp.cuda.Device(rank % gpu_count).use()

    with (
        h5py.File(args.dark_file, "r") as dark_file,
        h5py.File(args.flat_file, "r") as flat_file,
        h5py.File(args.sample_file, "r") as sample_file,
    ):
        dark_shape = base._dataset(dark_file, args.detector_path).shape
        flat_shape = base._dataset(flat_file, args.detector_path).shape
        sample_shape = base._dataset(sample_file, args.detector_path).shape

    total_ntheta, ny, nx = base._shape3(
        sample_shape, args.sample_file, args.detector_path
    )
    n_dark, dark_ny, dark_nx = base._shape3(
        dark_shape, args.dark_file, args.detector_path
    )
    n_flat, flat_ny, flat_nx = base._shape3(
        flat_shape, args.flat_file, args.detector_path
    )
    if (dark_ny, dark_nx) != (ny, nx) or (flat_ny, flat_nx) != (ny, nx):
        raise ValueError("dark/flat/sample detector dimensions differ")

    frame_ids = base._parse_frame_ids(args.frame_ids_spec, total_ntheta)
    ntheta = int(frame_ids.size)
    y_crop, x_crop, pad, n = base._center_crop_pad(ny, nx, args.n)
    mask = base._valid_mask((y_crop, x_crop), pad, n)
    y_pad, x_pad = pad

    magnification = args.focustodetectordistance / args.z1
    voxelsize = args.detector_pixelsize / magnification
    wavelength = 1.24e-9 / args.energy

    x_raw_all, y_raw_all = base._positions(
        args.sample_file, args.x_path, args.y_path
    )
    if len(x_raw_all) != total_ntheta:
        raise ValueError("number of positions does not match number of sample frames")

    x_raw = x_raw_all[frame_ids]
    y_raw = y_raw_all[frame_ids]
    x_position = x_raw.copy()
    y_position = y_raw.copy()
    if args.center_positions:
        x_position -= x_position.mean()
        y_position -= y_position.mean()

    scale = base._unit_scale(args.position_unit, voxelsize)
    pos = np.stack(
        [
            args.pos_row_sign * y_position * scale / voxelsize,
            args.pos_col_sign * x_position * scale / voxelsize,
        ],
        axis=1,
    ).astype("float32")

    pos_range = int(np.ceil(np.abs(pos).max())) + 8
    nobj = int(np.ceil((n + 2 * pos_range) / 32)) * 32
    mode = (
        "flat-corrected"
        if args.flat_correct
        else "dark-subtracted raw + shared sample/flat intensity scale"
    )

    if rank == 0:
        base.logger.info("=== DanMAX nano step 0 v2: flat-constrained NFP ===")
        for key, value in (
            ("dark_file", args.dark_file),
            ("flat_file", args.flat_file),
            ("sample_file", args.sample_file),
            ("flat_correct", f"{args.flat_correct} ({mode})"),
            ("flat_loss_weight", args.flat_loss_weight),
            ("use_valid_detector_mask", args.use_valid_detector_mask),
            ("frame_ids spec", args.frame_ids_spec),
        ):
            base.logger.info(f"{key:24s}= {value}")
        base.logger.info(f"dark shape              = {dark_shape}  frames={n_dark}")
        base.logger.info(f"flat shape              = {flat_shape}  frames={n_flat}")
        base.logger.info(
            f"sample shape            = {sample_shape}  total frames={total_ntheta}"
        )
        base.logger.info(
            f"selected frames         = {ntheta}/{total_ntheta}: "
            f"{base._format_frame_ids(frame_ids)}"
        )
        base.logger.info(
            f"crop                    = rows[{y_crop.start}:{y_crop.stop}], "
            f"cols[{x_crop.start}:{x_crop.stop}], n={n}"
        )
        base.logger.info(
            f"padding                 = rows before/after={y_pad}, "
            f"cols before/after={x_pad}"
        )
        base.logger.info(
            f"valid detector pixels   = {int(mask.sum())}/{n*n} "
            f"({100 * mask.mean():.2f}%)"
        )
        base.logger.info(
            f"energy                  = {args.energy:g} keV  "
            f"wavelength={wavelength:.6e} m"
        )
        base.logger.info(f"magnification           = {magnification:.6g}")
        base.logger.info(
            f"voxelsize               = {voxelsize:.6e} m "
            f"({voxelsize * 1e9:.3f} nm)"
        )
        base.logger.info(
            f"positions pix row       = [{pos[:, 0].min():.3f}, {pos[:, 0].max():.3f}]"
        )
        base.logger.info(
            f"positions pix col       = [{pos[:, 1].min():.3f}, {pos[:, 1].max():.3f}]"
        )
        base.logger.info(f"nobj                    = {nobj}")

    bbox_data = None
    bbox_path = os.path.join(
        os.path.dirname(args.h5_out) or ".", "position_bbox_sanity_v2.png"
    )
    if rank == 0 and args.write_position_bbox_plot:
        bbox_data = base._bbox_plot(
            bbox_path,
            pos,
            frame_ids,
            n,
            nobj,
            (y_crop, x_crop),
            pad,
            args.pos_row_sign,
            args.pos_col_sign,
            args.position_bbox_grid_size,
        )
        base.logger.info(f"Wrote position/bbox sanity plot to {bbox_path}")

    dark = base._mean_image(
        args.dark_file, args.detector_path, (y_crop, x_crop), pad
    )
    flat = base._mean_image(
        args.flat_file, args.detector_path, (y_crop, x_crop), pad
    )
    flat_dark = flat - dark
    epsilon = max(float(np.nanmedian(flat_dark)) * 1e-6, 1e-6)
    flat_dark = np.where(flat_dark > epsilon, flat_dark, epsilon).astype("float32")

    preview = None
    preview_mean = np.nan
    preview_frame_ids = frame_ids[: min(args.preview_count, ntheta)]
    if rank == 0 and preview_frame_ids.size > 0:
        raw_preview = base._sample_frames(
            args.sample_file,
            args.detector_path,
            (y_crop, x_crop),
            pad,
            preview_frame_ids,
        )
        preview, preview_mean = base._normalize(
            base._prepare(raw_preview, dark, flat_dark, args.flat_correct)
        )
        base.logger.info(
            f"preview frame IDs       = {base._format_frame_ids(preview_frame_ids)}"
        )
        base.logger.info(f"flat-dark median        = {np.median(flat_dark):.6g}")
        base.logger.info(f"preview normalization mean = {preview_mean:.6g}")

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
                base._write(output, name, value)
            if args.write_corrected_preview and preview is not None:
                base._write(output, "corrected_preview", preview)
                base._write(
                    output, "preview_frame_ids", preview_frame_ids.astype("int64")
                )
            if bbox_data:
                for key, value in bbox_data.items():
                    base._write(output, f"bbox_preview_{key}", value)

            output.attrs.update(
                algorithm="step0_v2_flat_constrained_nfp",
                flat_correct=bool(args.flat_correct),
                flat_loss_weight=float(args.flat_loss_weight),
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
                position_bbox_plot=(
                    bbox_path if args.write_position_bbox_plot else ""
                ),
            )
        base.logger.info(f"Wrote sanity-check output to {args.h5_out}")

    comm.Barrier()
    if not args.run_reconstruction:
        if rank == 0:
            base.logger.info("run_reconstruction=false: stopping after sanity check.")
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
        path_out=(os.path.join(args.path_out, "nfp_v2") if args.path_out else None),
        comm=comm,
    )

    if args.flat_loss_weight > 0:
        reconstruction = FlatConstrainedRecNFP(rec_args)
    elif args.use_valid_detector_mask:
        reconstruction = base.MaskedRecNFP(rec_args)
    else:
        reconstruction = base.RecNFP(rec_args)

    if args.use_valid_detector_mask:
        reconstruction.set_data_mask(mask)

    local_frame_ids = frame_ids[
        reconstruction.st_theta : reconstruction.end_theta
    ]
    raw = base._sample_frames(
        args.sample_file,
        args.detector_path,
        (y_crop, x_crop),
        pad,
        local_frame_ids,
    )
    prepared, global_mean = base._normalize(
        base._prepare(raw, dark, flat_dark, args.flat_correct), comm
    )

    # Use exactly the same intensity scale for sample and empty-beam targets.
    # This preserves the physical relation sample ~= |D(P*T)|^2 and
    # flat ~= |D(P)|^2 instead of normalizing the two terms independently.
    flat_normalized = flat_dark / max(global_mean, 1e-6)
    flat_normalized[~np.isfinite(flat_normalized)] = 0
    flat_normalized = np.maximum(flat_normalized, 0).astype("float32")
    flat_amplitude = np.sqrt(flat_normalized).astype("float32")

    if isinstance(reconstruction, FlatConstrainedRecNFP):
        reconstruction.set_flat_constraint(
            flat_amplitude, args.flat_loss_weight
        )

    if rank == 0:
        base.logger.info(f"NFP input mode          = {mode}")
        base.logger.info(
            "NFP detector mask       = "
            + (
                "valid detector only"
                if args.use_valid_detector_mask
                else "disabled"
            )
        )
        base.logger.info(
            f"NFP selected frames     = {ntheta}/{total_ntheta}"
        )
        base.logger.info(
            f"shared intensity scale  = sample global mean {global_mean:.6g}"
        )
        base.logger.info(
            f"flat amplitude p1/p99   = {np.percentile(flat_amplitude, 1):.6g} / "
            f"{np.percentile(flat_amplitude, 99):.6g}"
        )

    reconstruction.data[:] = np.sqrt(np.abs(prepared)).astype("float32")
    reconstruction.vars["proj"][:] = 0
    reconstruction.vars["prb"][:] = 1
    reconstruction.vars["pos"][:] = base.cp.asarray(
        pos[reconstruction.st_theta : reconstruction.end_theta]
    )
    reconstruction.BH()

    final_losses = None
    flat_pred_amplitude = None
    if isinstance(reconstruction, FlatConstrainedRecNFP):
        final_losses = reconstruction.loss_terms(reconstruction.vars)
        if rank == 0 and args.save_flat_diagnostics:
            flat_pred_amplitude = base.cp.asnumpy(
                base.cp.abs(reconstruction.cl_prop.D(reconstruction.vars["prb"], 0))
            ).astype("float32")

    pos_errors = comm.gather(
        reconstruction.vars["pos"].get() - reconstruction.pos_init.get(), root=0
    )
    probes = comm.gather(reconstruction.vars["prb"].get(), root=0)
    projects = comm.gather(reconstruction.vars["proj"].get(), root=0)

    if rank == 0:
        probe = probes[0]
        # Preserve the output convention of step0.py for an isolated v1/v2
        # comparison.
        projection = np.concatenate(projects, axis=0)
        with h5py.File(args.h5_out, "a") as output:
            base._write(output, "prb_amp", np.abs(probe).astype("float32"))
            base._write(output, "prb_phase", np.angle(probe).astype("float32"))
            base._write(
                output, "proj_delta", projection.real.astype("float32")
            )
            base._write(
                output, "proj_beta", projection.imag.astype("float32")
            )
            base._write(
                output,
                "pos_err",
                np.concatenate(pos_errors, axis=0).astype("float32"),
            )
            base._write(output, "flat_target_amplitude", flat_amplitude)
            if flat_pred_amplitude is not None:
                base._write(output, "flat_pred_amplitude", flat_pred_amplitude)
                base._write(
                    output,
                    "flat_amplitude_residual",
                    flat_pred_amplitude - flat_amplitude,
                )
            output.attrs["nfp_input_global_mean"] = global_mean
            if final_losses is not None:
                sample_err, flat_err, weighted_flat_err, total_err = final_losses
                output.attrs["final_sample_amplitude_mse"] = sample_err
                output.attrs["final_flat_amplitude_mse"] = flat_err
                output.attrs["final_weighted_flat_amplitude_mse"] = (
                    weighted_flat_err
                )
                output.attrs["final_total_loss"] = total_err
        base.logger.info(f"Saved flat-constrained NFP reconstruction to {args.h5_out}")

    del reconstruction
    base.cp.get_default_memory_pool().free_all_blocks()


if __name__ == "__main__":
    main()
