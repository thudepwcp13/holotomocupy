"""
RecDelta: cascade-based parameterization u = delta * (1 + i * bd)

Adds a NEW cascade level F4: (prb, proj, bd, pos) -> (prb, proj*(1+i*bd), pos).
The Radon transform R: delta -> proj stays OUTSIDE the cascade (chunked over z),
analogous to how parent Rec keeps the Radon outside; what was the parent's `gF4` (RT for grad_obj)
is renamed `gF5` here.

Variables:
  vars['obj']         = delta       (real 3D, float32)
  vars['proj']   = R(delta)    (complex64 with imag=0; complex for MPI redist compatibility)
  vars['bd']  = scalar      (cupy shape (1,) float32)
  vars['prb']         = probe       (complex)
  vars['pos']         = positions   (real)
  (no more vars['proj'])

Cascade variables at level 4: (prb, proj, bd, pos)
Cascade variables at level 3: (prb, proj_complex, pos)        -- after F4

args.rho is length 4: [obj, prb, pos, bd]
"""

import numpy as np
import cupy as cp
import nvtx

from .rec_mpi import Rec
from .utils import make_pinned, redot, mshow, mshow_polar, mshow_pos, logger, time
from .mpi_functions import *


class RecDelta(Rec):
    def __init__(self, args):
        if args.obj_dtype != 'complex64':
            raise ValueError(
                f"RecDelta requires args.obj_dtype='complex64' "
                f"(proj/shift/MPI machinery is complex); got {args.obj_dtype!r}. "
                f"obj-side arrays are reallocated as float32 internally."
            )
        if not hasattr(args, 'rho') or len(args.rho) != 4:
            raise ValueError(
                f"RecDelta requires args.rho of length 4 "
                f"[obj, prb, pos, bd]; got {getattr(args, 'rho', None)}"
            )
        super().__init__(args)
        # Extend cascade lists with F4 (innermost). Parent gF0..gF3 already handle len(self.F)
        # via the patched range(_, len(self.F)) loops.
        self.F      = [self.F0, self.F1, self.F2, self.F3, self.F4]
        self.gF     = [self.gF0, self.gF1, self.gF2, self.gF3, self.gF4]
        self.dF     = [self.dF0, self.dF1, self.dF2, self.dF3, self.dF4]
        self.d2F_dF = [self.d2F_dF0, self.d2F_dF1, self.d2F_dF2, self.d2F_dF3, self.d2F_dF4]
        self.rho_sq['bd'] = float(args.rho[3]) ** 2

    # ------------------------------------------------------------------ alloc
    def alloc_arrays(self):
        super().alloc_arrays()

        # obj (= delta) is real float32
        self.u_pad = make_pinned([self.local_nzobj + 4, self.nobj, self.nobj], dtype='float32')
        self.e_pad = make_pinned([self.local_nzobj + 4, self.nobj, self.nobj], dtype='float32')
        self.u_pad[:] = 0
        self.e_pad[:] = 0
        self.vars['obj']  = self.u_pad[2:-2]
        self.etas['obj']  = self.e_pad[2:-2]
        self.grads['obj'] = make_pinned([self.local_nzobj, self.nobj, self.nobj], dtype='float32')

        # proj cache: complex64 (imag=0) so MPI redist (locked to complex64) accepts it.
        self.vars['proj']  = make_pinned([self.local_ntheta, self.nzobj, self.nobj], dtype='complex64')
        self.grads['proj'] = make_pinned([self.local_ntheta, self.nzobj, self.nobj], dtype='complex64')
        self.etas['proj']  = make_pinned([self.local_ntheta, self.nzobj, self.nobj], dtype='complex64')

        # bd scalar
        self.vars['bd']  = cp.zeros((1,), dtype='float32')
        self.grads['bd'] = cp.zeros((1,), dtype='float32')
        self.etas['bd']  = cp.zeros((1,), dtype='float32')

        # tmp pinned buffer for fwd_tomo on real obj
        self.proj_tmp_real = make_pinned([self.ntheta, self.local_nzobj, self.nobj], dtype='complex64')
        # complex tmp for RT(grads['proj']) -> grads['obj'] (gF5)
        self._gF5_tmp = make_pinned([self.local_nzobj, self.nobj, self.nobj], dtype='complex64')

    # ============================================================ NEW F4 layer
    @nvtx.annotate("F4", color="green")
    def F4(self, x):
        """In: (prb, proj, bd, pos)
        Out: (prb, proj * (1 + i*bd), pos)   -- shape matches level 3 input"""
        prb, proj, bd, pos = x
        proj_complex = proj * (1.0 + 1j * bd)
        return [prb, proj_complex, pos]

    @nvtx.annotate("dF4", color="green")
    def dF4(self, x, y, return_x=True):
        """In:  x=(prb, proj, bd, pos),
               y=(yprb, yproj, ybd, ypos)
        Out:   ([prb, F4_proj, pos], [yprb, dF4_y, ypos])  if return_x
               else just the second list."""
        prb, proj, bd, pos = x
        yprb, yproj, ybd, ypos = y
        # dF4: differential of (proj * (1+i*bd))
        yproj_complex = (1.0 + 1j * bd) * yproj + 1j * proj * ybd
        if return_x:
            xproj_complex = proj * (1.0 + 1j * bd)
            return ([prb, xproj_complex, pos], [yprb, yproj_complex, ypos])
        return [yprb, yproj_complex, ypos]

    @nvtx.annotate("d2F_dF4", color="purple")
    def d2F_dF4(self, x, y, z, w):
        """Cascade composition rule contribution at level 4.
        d2F4(y, z) per element: only cross term (proj <-> bd) is nonzero
            -> yproj_complex_part = i * (yproj * zbd + zproj * ybd)
        Plus propagation of accumulator w via dF4 (if w not None, but at innermost level w==[None]*4).
        Returns 3-element [yprb, yproj_complex, ypos]  (matches level 3 input arity)."""
        prb, proj, bd, pos = x
        _, yproj, ybd, _ = y
        _, zproj, zbd, _ = z

        # second-derivative cross term in proj_complex slot
        yproj_complex = 1j * (yproj * zbd + zproj * ybd)

        # propagate w through dF4 (skip when w is the [None]*4 init from hessian_cascade)
        wprb_out, wpos_out = None, None
        if w is not None and w[1] is not None:
            w_prb, w_proj, w_bd, w_pos = w
            yproj_complex += (1.0 + 1j * bd) * w_proj + 1j * proj * w_bd
            wprb_out = w_prb
            wpos_out = w_pos

        # F4 doesn't touch pos, but parent's d2F_dF3 assumes w[1] and w[2] come paired
        # (both None or both arrays). Since we emit a non-None proj contribution, also
        # emit a zero pos contribution to maintain that invariant.
        if wpos_out is None:
            wpos_out = cp.zeros_like(pos)

        return [wprb_out, yproj_complex, wpos_out]

    @nvtx.annotate("gF4", color="green")
    def gF4(self, x, y):
        """Adjoint of dF4s.
        In:  x=(prb, proj, bd, pos)  at level 4,
             y=(yprb, yproj_complex, ypos) at level 3 (cotangent).
        Out: 4-element cotangent at level 4: [yprb, yproj_out, ybd_out, ypos]
             ybd_out is a scalar (per-chunk sum; will be accumulated across chunks via += in
             the gpu_batch wrapper)."""
        prb, proj, bd_arr, pos = x
        yprb, yproj_complex, ypos = y
        bd = bd_arr.reshape(())  # scalar broadcast

        yproj_out = yproj_complex.real + bd * yproj_complex.imag
        ybd_out = (cp.sum(proj.real * yproj_complex.imag)).reshape(1)

        return [yprb, yproj_out, ybd_out, ypos]

    # ======================================================== gF5 (was parent gF4: outside-cascade RT)
    @nvtx.annotate("gF5", color="green")
    def gF5(self, gradobj, gradproj):
        """Adjoint Radon (chunked over z): grads['obj'] (real) = Re[ RT(grads['proj']) ].
        proj has imag==0 by construction, so the imag part of RT is numerically negligible."""
        @self.gpu_batch(axis_out=0, axis_inp=1, nout=1)
        def _gF5(self, gradobj, gradproj):
            tmp = self.cl_tomo.RT(gradproj)   # complex64 in/out
            gradobj[:] = tmp.real
        _gF5(self, gradobj, gradproj)

    # ============================================================ helpers
    def _refresh_proj(self, vars):
        """vars['proj'] = R(vars['obj']) (real, written into complex64 buffer)."""
        self.fwd_tomo(vars['obj'], out=self.proj_tmp_real)
        self.cl_mpi.redist(self.proj_tmp_real, vars['proj'])

    # ============================================================ BH
    def BH(self, writer=None):
        vars = self.vars
        grads = self.grads
        etas = self.etas

        # shrinkage (constant after init; kept here for parity with parent)
        self.eff_demagnifications[:] = (1 + self.shrink_nd) / cp.array(self.norm_magnifications[None, :])

        # normalize obj for normal operators
        vars["obj"] /= self.norm_const
        if self.start_iter == 0:
            vars["obj"] *= self.cl_tomo.mask

        self.pos_init = vars['pos'].copy()

        # initial proj = R(delta)
        self._refresh_proj(vars)

        # initial error
        self.error_debug(vars, -1)

        self.time_start = time.time()
        for i in range(self.start_iter, self.niter):
            nvtx.push_range("::BH:" + str(i))

            # ---- gradients (writes grads['obj'] real, grads['proj'], grads['bd'] scalar)
            nvtx.push_range("gradients")
            self.gradients(vars, grads)
            nvtx.pop_range()

            # ---- propagate grads['obj'] through R to drive grads['proj']
            #      (analogous to parent line 173: fwd_tomo(grads['obj']) -> grads['proj'])
            nvtx.push_range(":::BH:fwd_tomo")
            self.fwd_tomo(grads["obj"], out=self.proj_tmp_real)
            nvtx.pop_range()
            nvtx.push_range(":::BH:redist", color='red')
            self.cl_mpi.redist(self.proj_tmp_real, grads['proj'])
            nvtx.pop_range()

            # ---- CG direction beta from Hessian-weighted inner products
            if i == self.start_iter:
                beta = 0
            else:
                nvtx.push_range(":::BH:calc beta")
                top = self.hessian(vars, grads, etas)
                bottom = self.hessian(vars, etas, etas)
                top, bottom = self.allreduce2(top, bottom)
                beta = top / bottom
                nvtx.pop_range()

            # ---- step size alpha
            nvtx.push_range(":::BH:calc_alpha")
            top = 0
            for v in ["obj", "pos", "bd"]:
                top -= self.linear_redot_batch(etas[v], grads[v], beta, -1) / self.rho_sq[v]
            dot_prb = self.linear_redot_batch(etas['prb'], grads['prb'], beta, -1)
            if self.rank == 0:
                top -= dot_prb / self.rho_sq['prb']
            # update etas['proj'] in lockstep (drift variable, no rho_sq term in numerator)
            self.linear_batch(etas['proj'], grads['proj'], beta, -1)

            bottom = self.hessian(vars, etas, etas)
            top, bottom = self.allreduce2(top, bottom)
            alpha = top / bottom
            nvtx.pop_range()

            # ---- update variables: var += alpha * eta
            for v in ["obj", "prb", "pos", "proj", "bd"]:
                self.linear_batch(vars[v], etas[v], 1, alpha)

            # ---- refresh proj = R(delta) for next iter
            self._refresh_proj(vars)

            # ---- error / vis
            nvtx.push_range(":::BH:calc error", color='gray')
            self.error_debug(vars, i)
            nvtx.pop_range()
            nvtx.push_range(":::BH:vis_debug", color='gray')
            self.vis_debug(vars, i, writer)
            nvtx.pop_range()
            nvtx.pop_range()  # ::BH:i

        vars["obj"] *= self.norm_const
        return vars

    # ============================================================ gradients (5-level cascade)
    def gradients_cascade(self, vars, grads):
        """Cascade gradient over the 5-level cascade.
        Outputs (with rho_sq scaling applied here for uniformity):
          grads['proj'] = y[1] * rho_sq['obj']
          grads['pos']       = y[3] * rho_sq['pos']
          grads['prb']       += y[0] * rho_sq['prb']  (accumulated across chunks)
          grads['bd'] += y[2] * rho_sq['bd']  (accumulated; final allreduce in self.gradients)
        """
        grads['prb'][:] = 0
        grads['bd'][:] = 0

        @self.gpu_batch(axis_out=0, axis_inp=0, nout=4)
        def _gradients_cascade(self,
                               gradproj, gradpos, gradprb, gradbd,
                               d, eff_demag, proj, pos, prb, bd):
            self._eff_demag_chunk = eff_demag
            x = [prb, proj, bd, pos]
            y = d
            for id in range(len(self.gF)):  # 0, 1, 2, 3, 4
                y = self.gF[id](x, y)
            # y is now 4-element at level 4: [yprb, yproj, ybd, ypos]
            gradprb[:]  += y[0] * self.rho_sq['prb']
            gradproj[:] = y[1] * self.rho_sq['obj']
            gradbd[:]   += y[2] * self.rho_sq['bd']
            gradpos[:]  = y[3] * self.rho_sq['pos']

        # IMPORTANT: gpu_batch needs proper inputs first (sliced over axis_inp=0), then nonproper.
        # proj, pos: proper (chunked over theta).  prb, bd: nonproper (replicated).
        _gradients_cascade(self,
                           grads['proj'], grads['pos'], grads['prb'], grads['bd'],
                           self.data, self.eff_demagnifications,
                           vars['proj'], vars['pos'],
                           vars['prb'], vars['bd'])

    def gradients(self, vars, grads):
        """Full gradient: cascade -> proj, then gF5 (RT) -> obj, plus regularization + allreduces."""
        # 1. cascade -> grads['proj'], grads['pos'], grads['prb'], grads['bd']
        self.gradients_cascade(vars, grads)

        # 2. gF5: RT applied chunked over z slices  -> grads['obj'] (real)
        nvtx.push_range(":::BH:redist back", color='red')
        self.cl_mpi.redist(grads['proj'], self.proj_tmp_real, direction='backward')
        nvtx.pop_range()
        self.gF5(grads['obj'], self.proj_tmp_real)

        # 3. biharmonic regularization on obj (in-place)
        self.gradient_laplacian(grads['obj'])

        # 4. probe-fit gradient (rank 0) + allreduce probe gradient across ranks
        if self.rank == 0:
            self.gradient_prbfit(grads["prb"], vars["prb"])
        grads['prb'][:] = cp.array(self.allreduce(grads['prb'].get()))

        # 5. allreduce scalar bd gradient across MPI ranks
        grads['bd'][:] = cp.array(self.allreduce(grads['bd'].get()))

    # ============================================================ hessian (5-level cascade)
    @timer
    def hessian_cascade(self, vars, grads, etas):
        """Cascade Hessian-weighted inner product <H · grads, etas> over 5 levels.
        Returns a single scalar (host)."""
        out = cp.zeros(1, dtype="float32")

        @self.gpu_batch(axis_out=0, axis_inp=0, nout=1)
        def _hessian_cascade(
            self, out, d, eff_demag,
            x_pos, y_pos, z_pos,
            x_proj, y_proj, z_proj,
            x_prb, y_prb, z_prb,
            x_bd,  y_bd,  z_bd,
        ):
            self._eff_demag_chunk = eff_demag
            x = [x_prb, x_proj, x_bd, x_pos]
            y = [y_prb, y_proj, y_bd, y_pos]
            z = [z_prb, z_proj, z_bd, z_pos]
            w = [None, None, None, None]

            y_is_z = y[0] is z[0]

            for id in range(1, len(self.F))[::-1]:  # 4, 3, 2, 1
                w = self.d2F_dF[id](x, y, z, w)
                fx, y = self.dF[id](x, y)
                if y_is_z:
                    z = y
                else:
                    z = self.dF[id](x, z, return_x=False)
                x = fx

            out[:] += self.d2F_dF[0](x, y, z, w, d)

        # proper inputs first (data, eff_demag, pos triple, proj triple),
        # then nonproper (prb triple, bd triple).
        _hessian_cascade(
            self, out, self.data, self.eff_demagnifications,
            vars['pos'], grads['pos'],  etas['pos'],
            vars['proj'], grads['proj'], etas['proj'],
            vars['prb'], grads['prb'], etas['prb'],
            vars['bd'], grads['bd'], etas['bd'],
        )

        return out[0].get()

    # ============================================================ min (5-level cascade)
    @timer
    def min(self, prb, obj, pos, proj, bd):
        """Loss evaluation. Override of parent.
        Signature shadows parent's `min(prb, obj, pos, proj)` but takes proj + bd
        instead of the complex proj."""
        out = cp.zeros(1, dtype="float32")

        @self.gpu_batch(axis_out=0, axis_inp=0, nout=1)
        def _min(self, out, proj, pos, data, eff_demag, prb, bd):
            self._eff_demag_chunk = eff_demag
            x = [prb, proj, bd, pos]
            y = x
            for id in range(1, len(self.F))[::-1]:  # 4, 3, 2, 1
                y = self.F[id](y)
            out[:] += self.F0(y, data)

        _min(self, out, proj, pos, self.data, self.eff_demagnifications, prb, bd)
        out = out[0]

        if self.rank == 0:
            for j in range(self.ndist):
                Dprb = self.cl_prop.D(prb[j : j + 1], j)[0]
                out += self.lam_prbfit / self.prb_size * cp.linalg.norm(cp.abs(Dprb) - self.ref[j]) ** 2
        return self.allreduce(np.array(out.get() + self._lap_energy_local(), dtype='float32'))

    # ============================================================ synthetic data
    def gen_sqrt_data(self, vars, out):
        """Generate synthetic |F(vars)| from real delta + scalar bd."""
        self.eff_demagnifications[:] = (1 + self.shrink_nd) / cp.array(self.norm_magnifications[None, :])
        vars["obj"] /= self.norm_const
        self._refresh_proj(vars)

        @self.gpu_batch(axis_out=0, axis_inp=0, nout=1)
        def _gen_data(self, out, proj, pos, eff_demag, prb, bd):
            self._eff_demag_chunk = eff_demag
            x = [prb, proj, bd, pos]
            y = x
            for id in range(1, len(self.F))[::-1]:
                y = self.F[id](y)
            out[:] = cp.abs(y)

        _gen_data(self, out,
                  vars['proj'], vars['pos'], self.eff_demagnifications,
                  vars['prb'], vars['bd'])
        vars["obj"] *= self.norm_const

    def compute_residual(self, vars):
        """Return float32 numpy [local_ntheta, ndist, nz, n]: |F(vars)| - sqrt(data)."""
        self._refresh_proj(vars)
        res = np.empty([self.local_ntheta, self.ndist, self.nz, self.n], dtype='float32')
        for theta_st in range(0, self.local_ntheta, self.nchunk):
            theta_end = min(theta_st + self.nchunk, self.local_ntheta)
            self._eff_demag_chunk = self.eff_demagnifications[theta_st:theta_end]
            proj_ch = cp.array(vars['proj'][theta_st:theta_end])
            pos_ch       = vars['pos'][theta_st:theta_end]
            x = [vars['prb'], proj_ch, vars['bd'], pos_ch]
            for id in range(1, len(self.F))[::-1]:
                x = self.F[id](x)
            res[theta_st:theta_end] = cp.asnumpy(cp.abs(x)) - self.data[theta_st:theta_end]
        return res

    # ============================================================ logging / vis
    def error_debug(self, vars, i):
        """Same as parent + log bd and 1/bd. Override min-call with new signature."""
        if not (i % self.error_step == 0 and self.error_step != -1):
            return
        err = self.min(vars["prb"], vars["obj"], vars["pos"], vars["proj"], vars["bd"])
        if self.rank == 0:
            bd = float(vars['bd'][0])
            inv_bd = 1.0 / bd if bd != 0 else float('inf')
            if i == -1:
                logger.warning(f"Initial {err=:1.5e}  delta/beta={inv_bd:.1f}")
                self.table.loc[len(self.table)] = [i, err, 0]
            else:
                ittime = time.time() - self.time_start
                logger.warning(f"iter={i}: {ittime:.4f}sec {err=:1.5e}  delta/beta={inv_bd:.1f}")
                self.table.loc[len(self.table)] = [i, err, ittime]
            self.time_start = time.time()
            if hasattr(self, 'path_out'):
                import os as _os
                name = f"{self.path_out}/conv.csv"
                _os.makedirs(_os.path.dirname(name), exist_ok=True)
                self.table.to_csv(name, index=False)

    def vis_debug(self, vars, i, writer=None):
        """Inline display: real delta slice + log scalar bd."""
        if not (i % self.checkpoint_step == 0 and self.checkpoint_step != -1):
            return
        if writer is not None:
            if i > self.start_iter:
                residual = self.compute_residual(vars)
                writer.write_checkpoint(vars, i, self.norm_const, residual=residual)
            # if self.rank == 0:
                # logger.warning(f"iter={i}: bd={float(vars['bd'][0]):.6e}")
            return
        zmid = self.local_nzobj // 2
        mshow(vars['obj'][zmid], True, figsize=(8, 3))
        # mshow_polar(vars['prb'][0], True, figsize=(8, 3))
        # mshow_pos(vars['pos'] - self.pos_init, True, figsize=(8, 3))
        # if self.rank == 0:
            # logger.warning(f"iter={i}: bd={float(vars['bd'][0]):.6e}")
