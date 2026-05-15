"""
Minimal JAX prototype for 2D single-phase pressure solves.

The solver uses a matrix-free five-point finite-volume stencil and a JAX CG
iteration, so no SciPy sparse matrix is assembled on the hot path.
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
except ImportError as exc:  # pragma: no cover - exercised only without jax
    raise ImportError(
        "JAX is required for reservoirpy.models.jax_single_phase. "
        "Install it with `pip install jax` or `pip install -e .[jax]`."
    ) from exc

from reservoirpy.utils.units import uc


@dataclass(frozen=True)
class JaxCGInfo:
    """Convergence information returned by the matrix-free CG solve."""

    iterations: int
    residual_norm: float


def _as_2d_field(value: Any, ny: int, nx: int, name: str) -> np.ndarray:
    if isinstance(value, (int, float)):
        return np.full((ny, nx), float(value), dtype=np.float64)

    array = np.asarray(value, dtype=np.float64)
    if array.shape == (ny, nx):
        return array
    if array.shape == (1, ny, nx):
        return array.reshape(ny, nx)
    if array.size == ny * nx:
        return array.reshape(ny, nx)
    raise ValueError(f"{name} must be scalar or reshapeable to ({ny}, {nx})")


def _compute_well_index(
    permeability_m2: float,
    dx: float,
    dy: float,
    dz: float,
    viscosity: float,
    rw: float,
    skin_factor: float,
) -> float:
    re = 0.14 * np.sqrt(dx**2 + dy**2)
    return (
        2.0
        * np.pi
        * float(permeability_m2)
        * float(dz)
        / (float(viscosity) * (np.log(re / rw) + skin_factor))
    )


def _build_well_arrays(
    wells_config: Iterable[Dict[str, Any]],
    permeability_m2: np.ndarray,
    nx: int,
    ny: int,
    dx: float,
    dy: float,
    dz: float,
    viscosity: float,
) -> Tuple[np.ndarray, np.ndarray]:
    well_diag = np.zeros((ny, nx), dtype=np.float64)
    well_rhs = np.zeros((ny, nx), dtype=np.float64)

    for well in wells_config:
        if well.get("control_type") != "bhp":
            raise ValueError("JaxSinglePhaseCG prototype supports only BHP wells")

        location = well["location"]
        if len(location) != 3:
            raise ValueError("well location must be [z, y, x]")

        z, y, x = location
        if z != 0:
            raise ValueError("JaxSinglePhaseCG prototype supports only nz=1")
        if not (0 <= y < ny and 0 <= x < nx):
            raise ValueError(f"well location out of range: {location}")

        wi = _compute_well_index(
            permeability_m2[y, x],
            dx,
            dy,
            dz,
            viscosity,
            well.get("rw", 0.05),
            well.get("skin_factor", 0.0),
        )
        well_diag[y, x] += wi
        well_rhs[y, x] += wi * float(well["value"])

    return well_diag, well_rhs


@jax.jit
def _transmissibility_faces(
    permeability_m2: jnp.ndarray,
    dx: float,
    dy: float,
    dz: float,
    viscosity: float,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    k_left = permeability_m2[:, :-1]
    k_right = permeability_m2[:, 1:]
    tx = jnp.where(
        (k_left + k_right) > 0.0,
        2.0 * k_left * k_right / (k_left + k_right),
        0.0,
    )
    tx = tx * (dy * dz) / (viscosity * dx)

    k_bottom = permeability_m2[:-1, :]
    k_top = permeability_m2[1:, :]
    ty = jnp.where(
        (k_bottom + k_top) > 0.0,
        2.0 * k_bottom * k_top / (k_bottom + k_top),
        0.0,
    )
    ty = ty * (dx * dz) / (viscosity * dy)
    return tx, ty


def _make_matvec(
    tx: jnp.ndarray,
    ty: jnp.ndarray,
    accumulation: jnp.ndarray,
    well_diag: jnp.ndarray,
):
    def matvec(pressure: jnp.ndarray) -> jnp.ndarray:
        out = (accumulation + well_diag) * pressure
        out = out.at[:, :-1].add(tx * (pressure[:, :-1] - pressure[:, 1:]))
        out = out.at[:, 1:].add(tx * (pressure[:, 1:] - pressure[:, :-1]))
        out = out.at[:-1, :].add(ty * (pressure[:-1, :] - pressure[1:, :]))
        out = out.at[1:, :].add(ty * (pressure[1:, :] - pressure[:-1, :]))
        return out

    return matvec


def _cg_solve(matvec, b: jnp.ndarray, x0: jnp.ndarray, tolerance: float, maxiter: int):
    r0 = b - matvec(x0)
    p0 = r0
    rs0 = jnp.sum(r0 * r0)
    b_norm2 = jnp.maximum(jnp.sum(b * b), 1.0)
    target = (tolerance**2) * b_norm2

    def cond_fun(state):
        iteration, _, _, _, rs_old = state
        return (iteration < maxiter) & (rs_old > target)

    def body_fun(state):
        iteration, x, r, p, rs_old = state
        ap = matvec(p)
        alpha = rs_old / (jnp.sum(p * ap) + 1e-300)
        x_new = x + alpha * p
        r_new = r - alpha * ap
        rs_new = jnp.sum(r_new * r_new)
        beta = rs_new / (rs_old + 1e-300)
        p_new = r_new + beta * p
        return iteration + 1, x_new, r_new, p_new, rs_new

    init_state = (jnp.array(0), x0, r0, p0, rs0)
    iterations, x, _, _, rs_final = jax.lax.while_loop(cond_fun, body_fun, init_state)
    return x, iterations, jnp.sqrt(rs_final)


@partial(jax.jit, static_argnames=("maxiter",))
def _solve_pressure_step(
    pressure_old: jnp.ndarray,
    permeability_m2: jnp.ndarray,
    porosity: jnp.ndarray,
    well_diag: jnp.ndarray,
    well_rhs: jnp.ndarray,
    dx: float,
    dy: float,
    dz: float,
    viscosity: float,
    compressibility: float,
    dt: float,
    tolerance: float,
    maxiter: int,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    tx, ty = _transmissibility_faces(permeability_m2, dx, dy, dz, viscosity)
    accumulation = (dx * dy * dz) * porosity * compressibility / dt
    rhs = accumulation * pressure_old + well_rhs
    matvec = _make_matvec(tx, ty, accumulation, well_diag)
    pressure_new, iterations, residual_norm = _cg_solve(
        matvec, rhs, pressure_old, tolerance, maxiter
    )
    return pressure_new, iterations, residual_norm


def resample_perm_field(
    perm: Any,
    target_nx: int,
    target_ny: int,
) -> np.ndarray:
    """Downsample a permeability field via geometric-mean block averaging.

    Each coarse cell covers a block of fine cells.  The coarse value is the
    geometric mean of the fine cells inside the block, which is equivalent to
    the arithmetic mean of log-permeability — the standard upscaling approach
    in reservoir engineering.

    Args:
        perm: fine-scale permeability (ny_src, nx_src) or (1, ny_src, nx_src),
              any unit (mD or m²) — the function is unit-agnostic.
        target_nx: target x-dimension.
        target_ny: target y-dimension.

    Returns:
        Coarse permeability array with the same ndim as *perm*.
    """
    squeeze = False
    arr = np.asarray(perm, dtype=np.float64)
    if arr.ndim == 3:
        arr = arr.reshape(arr.shape[1], arr.shape[2])
        squeeze = True
    ny_src, nx_src = arr.shape

    if ny_src % target_ny != 0:
        raise ValueError(f"ny_src={ny_src} must be divisible by target_ny={target_ny}")
    if nx_src % target_nx != 0:
        raise ValueError(f"nx_src={nx_src} must be divisible by target_nx={target_nx}")

    by = ny_src // target_ny
    bx = nx_src // target_nx

    log_k = np.log(np.maximum(arr, 1e-30))
    log_k_coarse = log_k.reshape(target_ny, by, target_nx, bx).mean(axis=(1, 3))
    coarse = np.exp(log_k_coarse)

    if squeeze:
        coarse = coarse[np.newaxis, :, :]
    return coarse


def remap_well_positions(
    wells: Iterable[Dict[str, Any]],
    nx_src: int,
    ny_src: int,
    nx_dst: int,
    ny_dst: int,
) -> List[Dict[str, Any]]:
    """Remap well grid indices from a source resolution to a destination resolution.

    Conversion path:  fine-grid (y, x) → normalised (y_frac, x_frac) ∈ [0, 1)
    → coarse-grid (y', x').

    All other well keys (control_type, value, rw, skin_factor, …) are preserved.
    """
    remapped = []
    for w in wells:
        z, y, x = w["location"]
        y_frac = (y + 0.5) / ny_src
        x_frac = (x + 0.5) / nx_src
        y_new = min(int(y_frac * ny_dst), ny_dst - 1)
        x_new = min(int(x_frac * nx_dst), nx_dst - 1)
        w_new = dict(w)
        w_new["location"] = [z, y_new, x_new]
        remapped.append(w_new)
    return remapped


def interpolate_to_fine(
    field_coarse: np.ndarray,
    nx_fine: int,
    ny_fine: int,
) -> np.ndarray:
    """Bilinear interpolation of a coarse 2D field to a fine grid.

    Convenience wrapper around ``scipy.ndimage.zoom`` so that multi-resolution
    pressure fields can be visually compared on the same grid.
    """
    from scipy.ndimage import zoom

    ny_c, nx_c = field_coarse.shape[-2], field_coarse.shape[-1]
    factors = (ny_fine / ny_c, nx_fine / nx_c)
    if field_coarse.ndim == 3:
        factors = (1.0,) + factors
    return zoom(field_coarse, factors, order=1)


def run_multiresolution(
    perm_fine: Any,
    wells_fine: Iterable[Dict[str, Any]],
    Lx: float,
    Ly: float,
    dz: float,
    resolutions: Iterable[Tuple[int, int]],
    dt: float,
    n_steps: int,
    initial_pressure: float,
    viscosity: float,
    compressibility: float,
    porosity: Any,
    cg_tolerance: float = 1e-8,
    cg_maxiter: int = 1000,
) -> Dict[Tuple[int, int], Dict[str, Any]]:
    """Run single-phase simulation at multiple grid resolutions.

    The physical domain (Lx × Ly) is identical across resolutions; only the
    grid spacing changes.  Permeability is upscaled with :func:`resample_perm_field`
    and well positions are remapped with :func:`remap_well_positions`.

    Args:
        perm_fine: fine-scale permeability (1, ny_fine, nx_fine) or
                   (ny_fine, nx_fine), in mD.
        wells_fine: well list with fine-grid ``location`` indices.
        Lx, Ly: physical domain size (m).
        dz: cell thickness (m).
        resolutions: iterable of ``(nx, ny)`` pairs, e.g.
                     ``[(16, 16), (32, 32), (64, 64)]``.
        dt: time step (s).
        n_steps: number of time steps.
        initial_pressure: initial pressure (Pa).
        viscosity: fluid viscosity (Pa·s).
        compressibility: rock compressibility (1/Pa).
        porosity: porosity (scalar or array).
        cg_tolerance: CG convergence tolerance.
        cg_maxiter: CG maximum iterations.

    Returns:
        ``{(nx, ny): result_dict}`` — one entry per resolution, each with the
        same keys as :meth:`JaxSinglePhaseCG.run`.
    """
    perm_arr = np.asarray(perm_fine, dtype=np.float64)
    if perm_arr.ndim == 3:
        ny_fine, nx_fine = perm_arr.shape[1], perm_arr.shape[2]
    else:
        ny_fine, nx_fine = perm_arr.shape

    results: Dict[Tuple[int, int], Dict[str, Any]] = {}

    for nx, ny in resolutions:
        dx = Lx / nx
        dy = Ly / ny

        if nx == nx_fine and ny == ny_fine:
            perm = perm_fine
            wells = list(wells_fine)
        else:
            perm = resample_perm_field(perm_fine, nx, ny)
            wells = remap_well_positions(wells_fine, nx_fine, ny_fine, nx, ny)

        solver = JaxSinglePhaseCG(
            nx=nx,
            ny=ny,
            dx=dx,
            dy=dy,
            dz=dz,
            permeability_mD=perm,
            porosity=porosity,
            viscosity=viscosity,
            compressibility=compressibility,
            wells_config=wells,
            cg_tolerance=cg_tolerance,
            cg_maxiter=cg_maxiter,
        )

        result = solver.run(
            initial_pressure=initial_pressure,
            dt=dt,
            n_steps=n_steps,
            return_history=True,
        )
        results[(nx, ny)] = result

    return results


class JaxSinglePhaseCG:
    """
    Minimal 2D single-phase JAX prototype.

    Scope:
    - structured 2D grids, including the intended 64x64 case
    - five-point no-flow-boundary finite-volume stencil
    - BHP wells only
    - fixed time step
    - matrix-free conjugate-gradient pressure solve
    """

    def __init__(
        self,
        nx: int = 64,
        ny: int = 64,
        dx: float = 50.0,
        dy: float = 50.0,
        dz: float = 10.0,
        permeability_mD: Any = 100.0,
        porosity: Any = 0.2,
        viscosity: float = 1e-3,
        compressibility: float = 1e-9,
        wells_config: List[Dict[str, Any]] | None = None,
        cg_tolerance: float = 1e-8,
        cg_maxiter: int = 1000,
    ):
        self.nx = int(nx)
        self.ny = int(ny)
        self.dx = float(dx)
        self.dy = float(dy)
        self.dz = float(dz)
        self.viscosity = float(viscosity)
        self.compressibility = float(compressibility)
        self.cg_tolerance = float(cg_tolerance)
        self.cg_maxiter = int(cg_maxiter)

        perm_mD = _as_2d_field(permeability_mD, self.ny, self.nx, "permeability_mD")
        poro = _as_2d_field(porosity, self.ny, self.nx, "porosity")
        perm_m2 = uc.md_to_m2(perm_mD)

        well_diag, well_rhs = _build_well_arrays(
            wells_config or [],
            perm_m2,
            self.nx,
            self.ny,
            self.dx,
            self.dy,
            self.dz,
            self.viscosity,
        )

        self.permeability_mD = perm_mD
        self.porosity = poro
        self.well_diag = well_diag
        self.well_rhs = well_rhs

        self._permeability_m2_jax = jnp.asarray(perm_m2)
        self._porosity_jax = jnp.asarray(poro)
        self._well_diag_jax = jnp.asarray(well_diag)
        self._well_rhs_jax = jnp.asarray(well_rhs)

    def initialize_pressure(self, initial_pressure: float = 30e6) -> np.ndarray:
        return np.full((self.ny, self.nx), float(initial_pressure), dtype=np.float64)

    def solve_timestep(
        self,
        pressure: np.ndarray,
        dt: float,
    ) -> Tuple[np.ndarray, JaxCGInfo]:
        pressure_new, iterations, residual_norm = _solve_pressure_step(
            jnp.asarray(pressure, dtype=jnp.float64).reshape(self.ny, self.nx),
            self._permeability_m2_jax,
            self._porosity_jax,
            self._well_diag_jax,
            self._well_rhs_jax,
            self.dx,
            self.dy,
            self.dz,
            self.viscosity,
            self.compressibility,
            float(dt),
            self.cg_tolerance,
            self.cg_maxiter,
        )
        return np.asarray(pressure_new), JaxCGInfo(
            iterations=int(iterations),
            residual_norm=float(residual_norm),
        )

    def run(
        self,
        initial_pressure: float = 30e6,
        dt: float = 86400.0,
        n_steps: int = 1,
        return_history: bool = True,
    ) -> Dict[str, Any]:
        pressure = self.initialize_pressure(initial_pressure)
        history = [pressure.copy()] if return_history else []
        cg_info = []

        for _ in range(int(n_steps)):
            pressure, info = self.solve_timestep(pressure, dt)
            cg_info.append(info)
            if return_history:
                history.append(pressure.copy())

        result = {
            "pressure": pressure,
            "cg_info": cg_info,
            "grid": {"nx": self.nx, "ny": self.ny, "dx": self.dx, "dy": self.dy, "dz": self.dz},
        }
        if return_history:
            result["pressure_history"] = np.stack(history)
        return result
