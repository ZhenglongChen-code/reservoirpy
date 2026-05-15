"""
JAX-accelerated two-phase IMPES solver for 2D structured grids.

Matrix-free five-point FVM stencil with:
  - Corey relative permeability model
  - Implicit pressure (CG) + explicit saturation (upstream weighting)
  - CFL-adaptive time stepping
  - Peaceman well model (BHP-controlled injection / production)
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np

try:
    import jax

    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
except ImportError as exc:
    raise ImportError(
        "JAX is required. Install with `pip install jax` or `pip install -e .[jax]`."
    ) from exc

from reservoirpy.utils.units import uc


@dataclass(frozen=True)
class JaxCGInfo:
    iterations: int
    residual_norm: float


def _as_2d(value: Any, ny: int, nx: int, name: str) -> np.ndarray:
    if isinstance(value, (int, float)):
        return np.full((ny, nx), float(value), dtype=np.float64)
    arr = np.asarray(value, dtype=np.float64)
    if arr.shape == (ny, nx):
        return arr
    if arr.shape == (1, ny, nx):
        return arr.reshape(ny, nx)
    if arr.size == ny * nx:
        return arr.reshape(ny, nx)
    raise ValueError(f"{name} must be scalar or reshapeable to ({ny}, {nx})")


def _peaceman_wi(k_m2, dx, dy, dz, mu, rw=0.1, skin=0.0):
    r_eq = 0.2 * np.sqrt(dx ** 2 + dy ** 2)
    return 2.0 * np.pi * k_m2 * dz / (mu * (np.log(r_eq / rw) + skin))


class JaxTwoPhaseIMPES:

    def __init__(
        self,
        nx: int,
        ny: int,
        dx: float,
        dy: float,
        dz: float,
        permeability_mD: Any,
        porosity: Any,
        mu_o: float,
        mu_w: float,
        compressibility: float,
        kro_params: Dict[str, float],
        krw_params: Dict[str, float],
        wells_config: Iterable[Dict[str, Any]],
        cg_tolerance: float = 1e-8,
        cg_maxiter: int = 2000,
        cfl_factor: float = 0.8,
    ):
        self.nx, self.ny = nx, ny
        self.dx, self.dy, self.dz = dx, dy, dz
        self.mu_o = mu_o
        self.mu_w = mu_w
        self.compressibility = compressibility
        self.kro_params = kro_params
        self.krw_params = krw_params
        self.cg_tolerance = cg_tolerance
        self.cg_maxiter = cg_maxiter
        self.cfl_factor = cfl_factor

        perm = _as_2d(permeability_mD, ny, nx, "permeability_mD")
        self.perm_m2 = uc.md_to_m2(perm)
        phi = _as_2d(porosity, ny, nx, "porosity")
        self.porosity = jnp.array(phi)
        self.volumes = jnp.full((ny, nx), dx * dy * dz)

        kx = self.perm_m2[:, :-1]
        ky = self.perm_m2[:, 1:]
        k_harm_x = 2.0 * kx * ky / (kx + ky + 1e-30)
        self.trans_x = jnp.array(k_harm_x * dy * dz / (mu_w * dx))

        kx = self.perm_m2[:-1, :]
        ky = self.perm_m2[1:, :]
        k_harm_y = 2.0 * kx * ky / (kx + ky + 1e-30)
        self.trans_y = jnp.array(k_harm_y * dx * dz / (mu_w * dy))

        wells = list(wells_config)
        self.n_wells = len(wells)
        well_y, well_x, well_wi, well_bhp, well_is_inj = [], [], [], [], []
        for w in wells:
            z, y, x = w["location"]
            k = self.perm_m2[y, x]
            rw_val = w.get("rw", 0.1)
            skin = w.get("skin_factor", 0.0)
            wi = _peaceman_wi(k, dx, dy, dz, mu_w, rw_val, skin)
            well_y.append(y)
            well_x.append(x)
            well_wi.append(wi)
            well_bhp.append(float(w["value"]))
            well_is_inj.append(1.0 if w["value"] > 0 else 0.0)

        self.well_y = jnp.array(well_y, dtype=jnp.int32)
        self.well_x = jnp.array(well_x, dtype=jnp.int32)
        self.well_wi = jnp.array(well_wi)
        self.well_bhp = jnp.array(well_bhp)
        self.well_is_inj = jnp.array(well_is_inj)

    def initialize(self, initial_pressure_Pa: float, initial_Sw: float):
        p = jnp.full((self.ny, self.nx), initial_pressure_Pa)
        sw = jnp.full((self.ny, self.nx), initial_Sw)
        return p, sw

    @partial(jax.jit, static_argnums=(0,))
    def compute_mobility(self, Sw):
        S_or = self.kro_params["S_or"]
        S_wr = self.krw_params["S_wr"]
        n_o = self.kro_params["n_o"]
        n_w = self.krw_params["n_w"]

        S_o_norm = jnp.clip((1.0 - Sw - S_or) / (1.0 - S_wr - S_or), 0.0, 1.0)
        S_w_norm = jnp.clip((Sw - S_wr) / (1.0 - S_wr - S_or), 0.0, 1.0)

        kro = jnp.where(Sw <= S_wr, 1.0, jnp.where(Sw >= 1.0 - S_or, 0.0, S_o_norm ** n_o))
        krw = jnp.where(Sw <= S_wr, 0.0, jnp.where(Sw >= 1.0 - S_or, 1.0, S_w_norm ** n_w))

        lambda_w = krw / self.mu_w
        lambda_o = kro / self.mu_o
        lambda_t = lambda_w + lambda_o
        f_w = lambda_w / (lambda_t + 1e-30)
        return lambda_w, lambda_o, lambda_t, f_w

    @partial(jax.jit, static_argnums=(0,))
    def solve_pressure(self, pressure, saturation, dt):
        _, _, lambda_t, f_w = self.compute_mobility(saturation)
        ms = self.mu_w * lambda_t

        ms_x0 = ms[:, :-1]
        ms_x1 = ms[:, 1:]
        ms_face_x = 2.0 * ms_x0 * ms_x1 / (ms_x0 + ms_x1 + 1e-30)
        T_x = self.trans_x * ms_face_x

        ms_y0 = ms[:-1, :]
        ms_y1 = ms[1:, :]
        ms_face_y = 2.0 * ms_y0 * ms_y1 / (ms_y0 + ms_y1 + 1e-30)
        T_y = self.trans_y * ms_face_y

        acc = self.volumes * self.porosity * self.compressibility / dt

        well_ms = ms[self.well_y, self.well_x]
        eff_wi = self.well_wi * well_ms

        b = acc * pressure
        for k in range(self.n_wells):
            b = b.at[self.well_y[k], self.well_x[k]].add(eff_wi[k] * self.well_bhp[k])

        def matvec(p):
            out = acc * p
            out = out.at[:, :-1].add(T_x * (p[:, :-1] - p[:, 1:]))
            out = out.at[:, 1:].add(T_x * (p[:, 1:] - p[:, :-1]))
            out = out.at[:-1, :].add(T_y * (p[:-1, :] - p[1:, :]))
            out = out.at[1:, :].add(T_y * (p[1:, :] - p[:-1, :]))
            for k in range(self.n_wells):
                out = out.at[self.well_y[k], self.well_x[k]].add(
                    eff_wi[k] * p[self.well_y[k], self.well_x[k]])
            return out

        b_flat = b.ravel()
        x0_flat = pressure.ravel()

        def cg_cond(state):
            _, r_norm, i = state
            return (r_norm > self.cg_tolerance) & (i < self.cg_maxiter)

        def cg_body(state):
            x, r_norm_sq, i = state
            r = b_flat - matvec(x.reshape(self.ny, self.nx)).ravel()
            r_norm_sq = jnp.dot(r, r)
            p_dir = r
            Ap = matvec(p_dir.reshape(self.ny, self.nx)).ravel()
            pAp = jnp.dot(p_dir, Ap)
            alpha = r_norm_sq / (pAp + 1e-30)
            x = x + alpha * p_dir
            r_new = b_flat - matvec(x.reshape(self.ny, self.nx)).ravel()
            r_new_sq = jnp.dot(r_new, r_new)
            beta = r_new_sq / (r_norm_sq + 1e-30)
            return x, r_new_sq, i + 1

        init_r = b_flat - matvec(x0_flat.reshape(self.ny, self.nx)).ravel()
        init_r_sq = jnp.dot(init_r, init_r)
        x_final, final_r_sq, iters = jax.lax.while_loop(
            cg_cond, cg_body, (x0_flat, init_r_sq, 0))

        new_p = x_final.reshape(self.ny, self.nx)
        return new_p, ms, f_w, eff_wi

    @partial(jax.jit, static_argnums=(0,))
    def update_saturation(self, pressure_old, pressure_new, saturation, dt, ms, f_w, eff_wi):
        ms_x0 = ms[:, :-1]
        ms_x1 = ms[:, 1:]
        ms_face_x = 2.0 * ms_x0 * ms_x1 / (ms_x0 + ms_x1 + 1e-30)
        T_x = self.trans_x * ms_face_x

        ms_y0 = ms[:-1, :]
        ms_y1 = ms[1:, :]
        ms_face_y = 2.0 * ms_y0 * ms_y1 / (ms_y0 + ms_y1 + 1e-30)
        T_y = self.trans_y * ms_face_y

        dp_x = pressure_new[:, 1:] - pressure_new[:, :-1]
        dp_y = pressure_new[1:, :] - pressure_new[:-1, :]

        up_x = dp_x >= 0
        fw_face_x = jnp.where(up_x, f_w[:, 1:], f_w[:, :-1])
        T_w_x = T_x * fw_face_x
        flux_x = T_w_x * dp_x

        up_y = dp_y >= 0
        fw_face_y = jnp.where(up_y, f_w[1:, :], f_w[:-1, :])
        T_w_y = T_y * fw_face_y
        flux_y = T_w_y * dp_y

        dSw = jnp.zeros((self.ny, self.nx))
        dSw = dSw.at[:, :-1].add(flux_x)
        dSw = dSw.at[:, 1:].add(-flux_x)
        dSw = dSw.at[:-1, :].add(flux_y)
        dSw = dSw.at[1:, :].add(-flux_y)

        for k in range(self.n_wells):
            wy, wx = self.well_y[k], self.well_x[k]
            q_total = eff_wi[k] * (pressure_new[wy, wx] - self.well_bhp[k])
            fw_cell = f_w[wy, wx]
            dSw_well = jnp.where(q_total < 0, jnp.abs(q_total), fw_cell * q_total)
            dSw = dSw.at[wy, wx].add(dSw_well)

        Sw_new = saturation + dt * dSw / (self.volumes * self.porosity + 1e-30)
        Sw_new = jnp.clip(Sw_new, 0.0, 1.0)
        return Sw_new

    @partial(jax.jit, static_argnums=(0,))
    def compute_cfl_dt(self, pressure, saturation):
        _, _, lambda_t, f_w = self.compute_mobility(saturation)
        ms = self.mu_w * lambda_t

        ms_x0 = ms[:, :-1]
        ms_x1 = ms[:, 1:]
        ms_face_x = 2.0 * ms_x0 * ms_x1 / (ms_x0 + ms_x1 + 1e-30)
        T_x = self.trans_x * ms_face_x

        ms_y0 = ms[:-1, :]
        ms_y1 = ms[1:, :]
        ms_face_y = 2.0 * ms_y0 * ms_y1 / (ms_y0 + ms_y1 + 1e-30)
        T_y = self.trans_y * ms_face_y

        dp_x = jnp.abs(pressure[:, 1:] - pressure[:, :-1])
        dp_y = jnp.abs(pressure[1:, :] - pressure[:-1, :])
        v_x = T_x * dp_x
        v_y = T_y * dp_y

        dSw_fd = 0.01
        S_or = self.kro_params["S_or"]
        S_wr = self.krw_params["S_wr"]
        n_o = self.kro_params["n_o"]
        n_w = self.krw_params["n_w"]

        def _fw(s):
            S_w_norm = jnp.clip((s - S_wr) / (1.0 - S_wr - S_or), 0.0, 1.0)
            krw = jnp.where(s <= S_wr, 0.0, jnp.where(s >= 1.0 - S_or, 1.0, S_w_norm ** n_w))
            S_o_norm = jnp.clip((1.0 - s - S_or) / (1.0 - S_wr - S_or), 0.0, 1.0)
            kro = jnp.where(s <= S_wr, 1.0, jnp.where(s >= 1.0 - S_or, 0.0, S_o_norm ** n_o))
            lw = krw / self.mu_w
            lo = kro / self.mu_o
            return lw / (lw + lo + 1e-30)

        Sw_x = saturation[:, :-1]
        dfw_x = (_fw(Sw_x + dSw_fd) - _fw(Sw_x - dSw_fd)) / (2 * dSw_fd)
        active_x = v_x * jnp.abs(dfw_x) > 1e-30
        dt_x = jnp.where(active_x,
                         self.porosity[:, :-1] * self.volumes[:, :-1] / (v_x * jnp.abs(dfw_x) + 1e-30),
                         jnp.inf)

        Sw_y = saturation[:-1, :]
        dfw_y = (_fw(Sw_y + dSw_fd) - _fw(Sw_y - dSw_fd)) / (2 * dSw_fd)
        active_y = v_y * jnp.abs(dfw_y) > 1e-30
        dt_y = jnp.where(active_y,
                         self.porosity[:-1, :] * self.volumes[:-1, :] / (v_y * jnp.abs(dfw_y) + 1e-30),
                         jnp.inf)

        dt_min = jnp.minimum(jnp.min(dt_x), jnp.min(dt_y))
        return dt_min * self.cfl_factor

    def solve_timestep(self, pressure, saturation, dt):
        new_p, ms, f_w, eff_wi = self.solve_pressure(pressure, saturation, dt)
        new_sw = self.update_saturation(pressure, new_p, saturation, dt, ms, f_w, eff_wi)
        return new_p, new_sw

    def run(self, initial_pressure_Pa, initial_Sw, total_time, n_snapshots=100,
            return_history=True):
        snap_fracs = np.concatenate([
            np.geomspace(0.001, 0.25, n_snapshots * 2 // 3, endpoint=False),
            np.linspace(0.25, 1.0, n_snapshots - n_snapshots * 2 // 3),
        ])
        snap_fracs = np.unique(np.round(snap_fracs, 6))[:n_snapshots]
        snap_times = snap_fracs * total_time

        p, sw = self.initialize(initial_pressure_Pa, initial_Sw)
        p = jax.device_get(p)
        sw = jax.device_get(sw)

        p_history = [p.copy()]
        sw_history = [sw.copy()]

        current_time = 0.0
        next_snap = 1
        max_sub = 5000
        sub = 0

        while current_time < total_time and sub < max_sub:
            target = snap_times[next_snap] if next_snap < n_snapshots else total_time
            remaining = target - current_time
            if remaining <= 0:
                next_snap += 1
                continue

            cfl_dt = float(self.compute_cfl_dt(p, sw))
            if cfl_dt < 1.0:
                cfl_dt = 1.0
            dt = min(cfl_dt, remaining)

            new_p, new_sw = self.solve_timestep(p, sw, dt)
            p = jax.device_get(new_p)
            sw = jax.device_get(new_sw)
            current_time += dt
            sub += 1

            while next_snap < n_snapshots and current_time >= snap_times[next_snap] - 1.0:
                p_history.append(p.copy())
                sw_history.append(sw.copy())
                next_snap += 1

        while len(p_history) < n_snapshots:
            p_history.append(p_history[-1])
            sw_history.append(sw_history[-1])

        result = {
            "pressure": np.stack(p_history),
            "saturation": np.stack(sw_history),
        }
        return result
