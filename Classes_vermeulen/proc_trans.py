## Fichier d'orchestration :
"""
Pipeline complète de traitement d'un transect ADCP répété.
Créé à partir de ProcTrans.m. 
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

import numpy as np

from .Bathymetry import ConstantWaterLevel, VaryingWaterLevel
from .LocationBasedVelocitySolver import LocationBasedVelocitySolver
from .SigmaZetaMeshFromVMADCP import SigmaZetaMeshFromVMADCP
from .TimeBasedVelocitySolver import TimeBasedVelocitySolver
from .VMADCP import VMADCP
from .XSection import XSection
from .get_gpsvel import get_gpsvel


# @dataclass
# class ProcTransSection:
#     row_index: int
#     crossing_ids: np.ndarray
#     ensembles: np.ndarray
#     eta: np.ndarray
#     maxeta: np.ndarray
#     mineta: np.ndarray
#     time: np.ndarray
#     Tvec: np.ndarray
#     Nvec: np.ndarray
#     Pm: np.ndarray
#     mesh: Any
#     solver: Any
#     vel: np.ndarray
#     cov_vel: np.ndarray
#     nb_vel: np.ndarray
#     r2: np.ndarray
#     r_sig: np.ndarray
#     vel_sn: np.ndarray | None = None
#     cov_vel_sn: np.ndarray | None = None

@dataclass
class ProcTransSection:
    row_index: int
    crossing_ids: np.ndarray
    ensembles: np.ndarray
    eta: np.ndarray
    maxeta: np.ndarray
    mineta: np.ndarray
    time: np.ndarray
    Tvec: np.ndarray
    Nvec: np.ndarray
    Pm: np.ndarray
    mesh: Any
    solver: Any
    vel: np.ndarray
    cov_vel: np.ndarray
    nb_vel: np.ndarray
    r2: np.ndarray
    r_sig: np.ndarray
    vel_sn: np.ndarray | None = None
    cov_vel_sn: np.ndarray | None = None
    # Per-crossing results (None si non calculés)
    vel_cross: np.ndarray | None = None
    cov_vel_cross: np.ndarray | None = None
    nb_vel_cross: np.ndarray | None = None
    r2_cross: np.ndarray | None = None
    r_sig_cross: np.ndarray | None = None
    vel_sn_cross: np.ndarray | None = None
    cov_vel_sn_cross: np.ndarray | None = None


@dataclass
class ProcTransResult:
    sections: list[ProcTransSection] = field(default_factory=list)
    params: dict[str, Any] = field(default_factory=dict)


def _tid_to_2d(tid: np.ndarray) -> np.ndarray:
    tid_arr = np.asarray(tid, dtype=int)
    if tid_arr.ndim != 2:
        raise ValueError("tid must be a 2D array shaped (n_sections, n_ensembles)")
    return tid_arr


def _to_numeric_time(time: Any) -> np.ndarray:
    if time is None:
        return np.empty((0,), dtype=float)

    arr = np.asarray(time)
    if arr.size == 0:
        return np.empty((0,), dtype=float)

    if np.issubdtype(arr.dtype, np.datetime64):
        arr_ns = arr.astype("datetime64[ns]").astype("int64")
        return arr_ns.astype(float) / 1e9

    try:
        return arr.astype(float).reshape(-1)
    except Exception:
        return np.arange(arr.size, dtype=float)


def _slice_axis_1(value: Any, mask: np.ndarray) -> Any:
    if value is None:
        return None

    arr = np.asarray(value)
    if arr.ndim == 0:
        return arr.item()

    if arr.ndim >= 2 and arr.shape[1] == mask.size:
        return arr[:, mask, ...]
    if arr.ndim == 1 and arr.size == mask.size:
        return arr[mask]

    return value

# Helper pour slicer des vitesses de navigation par ensemble :
# (On découpe des tableaux de forme (N,3) ou (N,) sur l'axe temps/ensemble.)
def _slice_axis_0(value: Any, mask: np.ndarray) -> Any:
    if value is None:
        return None

    arr = np.asarray(value)
    if arr.ndim == 0:
        return arr.item()

    if arr.ndim >= 1 and arr.shape[0] == mask.size:
        return arr[mask, ...]

    return value


def _build_water_level_object(time_numeric: np.ndarray, eta: float | np.ndarray | None):
    if eta is None:
        return None

    eta_arr = np.asarray(eta, dtype=float).reshape(-1)
    if eta_arr.size == 0:
        return ConstantWaterLevel(0.0)

    if eta_arr.size == 1:
        return ConstantWaterLevel(float(eta_arr[0]))

    if time_numeric.size != eta_arr.size:
        time_numeric = np.arange(eta_arr.size, dtype=float)

    return VaryingWaterLevel(time_numeric, eta_arr)

# Helper pour normaliser ship_reference : 
def _normalize_ship_reference(ship_reference: str) -> str:
    ref = (ship_reference or "bt").strip().lower()

    if ref in ("bt", "bt_vel", "bottom_track", "bottomtrack"):
        return "bt_vel"
    if ref in ("gps", "gps_vel", "gpsvelocity"):
        return "gps_vel"

    raise ValueError(
        f"Unsupported ship_reference={ship_reference!r}. "
        "Expected one of: bt, bt_vel, gps, gps_vel."
    )


def _clone_section_source(source: Any, mask: np.ndarray, eta: float | np.ndarray | None, depth_transducer: float):
    section = SimpleNamespace()
    section._raw = getattr(source, "_raw", None)

    for name in (
        "depth_cell_position",
        "water_velocity",
        "xform",
        "horizontal_position",
        "vertical_position",
        "bed_position",
        "water_level",
        "time",
    ):
        if hasattr(source, name):
            value = getattr(source, name)
            sliced = _slice_axis_1(value, mask)
            setattr(section, name, sliced)

    section.time = _to_numeric_time(getattr(section, "time", None))
    section.n_ensembles = int(np.count_nonzero(mask))
    section.filters = list(getattr(source, "filters", []))
    section.beam_angle_deg = float(getattr(source, "beam_angle_deg", 20.0))

    if hasattr(source, "transducer"):
        section.transducer = getattr(source, "transducer")
    else:
        section.transducer = SimpleNamespace()

    if not hasattr(section.transducer, "depth"):
        section.transducer.depth = float(depth_transducer)
    else:
        section.transducer.depth = float(depth_transducer)

    wl_obj = _build_water_level_object(section.time, eta)
    if wl_obj is not None:
        section.water_level_object = wl_obj
        section.water_level = np.asarray(wl_obj.get_water_level(section.time), dtype=float).reshape(-1)
    elif hasattr(source, "water_level_object"):
        section.water_level_object = getattr(source, "water_level_object")
        if hasattr(source, "water_level"):
            section.water_level = np.asarray(_slice_axis_1(getattr(source, "water_level"), mask), dtype=float).reshape(-1)
        else:
            section.water_level = np.zeros((section.n_ensembles,), dtype=float)
    else:
        section.water_level_object = ConstantWaterLevel(0.0)
        section.water_level = np.zeros((section.n_ensembles,), dtype=float)

    return section


def _section_geometry(section_source: Any, pusr=None) -> XSection:
    xs = XSection(section_source)
    if pusr is not None and np.asarray(pusr).size == 2:
        xs.origin = np.asarray(pusr, dtype=float).reshape(2)
    return xs


# def _empty_section(row_index: int) -> ProcTransSection:
#     empty = np.empty((0,), dtype=float)
#     return ProcTransSection(
#         row_index=row_index,
#         crossing_ids=np.empty((0,), dtype=int),
#         ensembles=np.empty((0,), dtype=int),
#         eta=empty,
#         maxeta=empty,
#         mineta=empty,
#         time=empty,
#         Tvec=np.array([np.nan, np.nan], dtype=float),
#         Nvec=np.array([np.nan, np.nan], dtype=float),
#         Pm=np.array([np.nan, np.nan], dtype=float),
#         mesh=None,
#         solver=None,
#         vel=np.empty((0, 3), dtype=float),
#         cov_vel=np.empty((0, 3, 3), dtype=float),
#         nb_vel=np.empty((0,), dtype=int),
#         r2=np.empty((0,), dtype=float),
#         r_sig=np.empty((0,), dtype=float),
#         vel_sn=None,
#         cov_vel_sn=None,
#     )

def _empty_section(row_index: int) -> ProcTransSection:
    empty = np.empty((0,), dtype=float)
    return ProcTransSection(
        row_index=row_index,
        crossing_ids=np.empty((0,), dtype=int),
        ensembles=np.empty((0,), dtype=int),
        eta=empty,
        maxeta=empty,
        mineta=empty,
        time=empty,
        Tvec=np.array([np.nan, np.nan], dtype=float),
        Nvec=np.array([np.nan, np.nan], dtype=float),
        Pm=np.array([np.nan, np.nan], dtype=float),
        mesh=None,
        solver=None,
        vel=np.empty((0, 3), dtype=float),
        cov_vel=np.empty((0, 3, 3), dtype=float),
        nb_vel=np.empty((0,), dtype=int),
        r2=np.empty((0,), dtype=float),
        r_sig=np.empty((0,), dtype=float),
        vel_sn=None,
        cov_vel_sn=None,
        vel_cross=None,
        cov_vel_cross=None,
        nb_vel_cross=None,
        r2_cross=None,
        r_sig_cross=None,
        vel_sn_cross=None,
        cov_vel_sn_cross=None,
    )


def proc_trans(
    adcp: Any,
    tid: np.ndarray,
    *,
    depth_transducer: float = 0.3,
    delta_n: float = 5.0,
    delta_z: float = 1.0,
    eta: float | np.ndarray | None = 0.0,
    minimum_sigma: float = 0.06,
    cumulate_crossings: bool = False,
    conventional_processing: bool = False,
    top_mesh_lowest_eta: bool = False,
    constant_zeta_mesh: bool = False,
    remove_outliers: float = 0.0,
    proximity: float = 0.0,
    std_filtering: float = 6.0,
    ship_reference: str = "bt",
    pusr=None,
    rotate_pars: bool = True,
    model=None,
    known=None,
    bathymetry=None,
    get_velocity=None,
    use_time_solver: bool = False,
    process_crossings: bool = False,
) -> ProcTransResult:
    """
    Python orchestration layer for the MATLAB procTrans workflow.

    Notes
    -----
    - This reuses the existing Python mesh/solver classes.
    - The MATLAB custom hooks 'Model', 'Known' and 'GetVelocity' are accepted
      for API parity but are not yet wired to a custom model pipeline.
    """
    tid_arr = _tid_to_2d(tid)
    nav_ref = _normalize_ship_reference(ship_reference)
    
    bt_velocity_all = None
    gps_velocity_all = None

    if not isinstance(adcp, VMADCP):
        try:
            bt_velocity_all = get_gpsvel(adcp, nav_ref="bt_vel")
        except Exception:
            bt_velocity_all = None

        try:
            gps_velocity_all = get_gpsvel(adcp, nav_ref="gps_vel")
        except Exception:
            gps_velocity_all = None

    # Création de l'objet VMADCP :
    if isinstance(adcp, VMADCP):
        vmadcp = adcp
    else:
        vmadcp = VMADCP(adcp, nav_ref=nav_ref)

    n_ensembles = int(getattr(vmadcp, "nensembles", tid_arr.shape[1]))
    if tid_arr.shape[1] != n_ensembles:
        raise ValueError(
            f"tid has {tid_arr.shape[1]} ensembles but adcp exposes {n_ensembles}"
        )

    result = ProcTransResult(
        params={
            "depth_transducer": depth_transducer,
            "delta_n": delta_n,
            "delta_z": delta_z,
            "eta": eta,
            "minimum_sigma": minimum_sigma,
            "cumulate_crossings": cumulate_crossings,
            "conventional_processing": conventional_processing,
            "top_mesh_lowest_eta": top_mesh_lowest_eta,
            "constant_zeta_mesh": constant_zeta_mesh,
            "remove_outliers": remove_outliers,
            "proximity": proximity,
            "std_filtering": std_filtering,
            "ship_reference": ship_reference,
            "pusr": pusr,
            "rotate_pars": rotate_pars,
            "model": model,
            "known": known,
            "get_velocity": get_velocity,
            "use_time_solver": use_time_solver,
            "process_crossings": process_crossings
        }
    )

    solver_cls = TimeBasedVelocitySolver if use_time_solver else LocationBasedVelocitySolver

    for row_index in range(tid_arr.shape[0]):
        mask = tid_arr[row_index] > 0
        if not np.any(mask):
            result.sections.append(_empty_section(row_index))
            continue

        section_source = _clone_section_source(
            vmadcp,
            mask=mask,
            eta=eta,
            depth_transducer=depth_transducer,
        )

        section_source.ship_reference = nav_ref

        section_bt_velocity = _slice_axis_0(bt_velocity_all, mask) if bt_velocity_all is not None else None
        section_gps_velocity = _slice_axis_0(gps_velocity_all, mask) if gps_velocity_all is not None else None

        section_source.bt_velocity = section_bt_velocity
        section_source.gps_velocity = section_gps_velocity

        if nav_ref == "gps_vel":
            section_source.ship_velocity = section_gps_velocity
        elif nav_ref == "bt_vel":
            section_source.ship_velocity = section_bt_velocity
        else:
            section_source.ship_velocity = section_bt_velocity if section_bt_velocity is not None else section_gps_velocity

        if section_source.ship_velocity is None:
            section_source.ship_velocity = np.zeros((section_source.n_ensembles, 3), dtype=float)

        try:
            xs = _section_geometry(section_source, pusr=pusr)
        except Exception as exc:
            raise RuntimeError(f"Cannot build XSection for row {row_index}") from exc

    
        mesh_builder = SigmaZetaMeshFromVMADCP(
            section_source,
            bathymetry=bathymetry,
            xs=xs,
            deltan=delta_n,
            deltaz=delta_z,
            time=getattr(section_source, "time", None),
        )
        mesh = mesh_builder.get_mesh()

        solver = solver_cls(
            adcp=section_source,
            mesh=mesh,
            xs=xs,
        )

        vel, cov_vel, nb_vel, r2, r_sig = solver.get_velocity(
            f_vitesse_z=10.0,
            f_direction_fixe=0.97,
            f_direction_pond=0.0002,
            pond_vitesses=1.0,
        )

        if rotate_pars:
            vel_sn, cov_vel_sn = solver.rotate_to_xs(vel, cov_vel)
        else:
            vel_sn, cov_vel_sn = vel, cov_vel

        
        # --- Optionnel : traitement par crossing pour parité MATLAB ---
        tid_row_masked = tid_arr[row_index, mask]
        crossing_ids = np.unique(tid_row_masked).astype(int)

        vel_cross = None
        cov_vel_cross = None
        nb_vel_cross = None
        r2_cross = None
        r_sig_cross = None
        vel_sn_cross = None
        cov_vel_sn_cross = None

        if process_crossings and crossing_ids.size > 0:
            ncells = int(getattr(mesh, "ncells", None) or getattr(mesh, "cell_to_mat", None).size)
            n_cross = crossing_ids.size

            vel_cross = np.full((ncells, n_cross, 3), np.nan)
            cov_vel_cross = np.full((ncells, n_cross, 3, 3), np.nan)
            nb_vel_cross = np.zeros((ncells, n_cross), dtype=int)
            r2_cross = np.full((ncells, n_cross), np.nan)
            r_sig_cross = np.full((ncells, n_cross), np.nan)
            vel_sn_cross = np.full((ncells, n_cross, 3), np.nan)
            cov_vel_sn_cross = np.full((ncells, n_cross, 3, 3), np.nan)

            for ic, cid in enumerate(crossing_ids):
                # mask over original ensembles for this crossing
                cross_mask_global = (tid_arr[row_index] == int(cid))
                if not np.any(cross_mask_global):
                    continue

                # build a per-crossing adcp slice (re-uses same mesh & xs)
                section_cross = _clone_section_source(
                    vmadcp,
                    mask=cross_mask_global,
                    eta=eta,
                    depth_transducer=depth_transducer,
                )

                solver_cross = solver_cls(adcp=section_cross, mesh=mesh, xs=xs)

                try:
                    v_c, cov_c, nb_c, r2_c, r_sig_c = solver_cross.get_velocity(
                        f_vitesse_z=10.0,
                        f_direction_fixe=0.97,
                        f_direction_pond=0.0002,
                        pond_vitesses=1.0,
                    )
                except Exception:
                    continue

                v_c = np.asarray(v_c, dtype=float)
                cov_c = np.asarray(cov_c, dtype=float)
                nb_c = np.asarray(nb_c, dtype=int)
                r2_c = np.asarray(r2_c, dtype=float)
                r_sig_c = np.asarray(r_sig_c, dtype=float)

                # store
                if v_c.shape[0] == ncells:
                    vel_cross[:, ic, :] = v_c
                else:
                    # try to broadcast per-cell results if solver returns shape mismatch
                    try:
                        vel_cross[: v_c.shape[0], ic, :] = v_c
                    except Exception:
                        pass

                if cov_c.shape[0] == ncells:
                    cov_vel_cross[:, ic, :, :] = cov_c

                nb_vel_cross[:, ic] = nb_c
                r2_cross[:, ic] = r2_c
                r_sig_cross[:, ic] = r_sig_c

                # rotate per-crossing if demandé
                if rotate_pars:
                    try:
                        v_sn_c, cov_sn_c = solver_cross.rotate_to_xs(v_c, cov_c)
                        if v_sn_c.shape[0] == ncells:
                            vel_sn_cross[:, ic, :] = v_sn_c
                        if cov_sn_c is not None and cov_sn_c.shape[0] == ncells:
                            cov_vel_sn_cross[:, ic, :, :] = cov_sn_c
                    except Exception:
                        pass
        # --- fin per-crossing ---

        section_time = np.asarray(getattr(section_source, "time", np.array([], dtype=float)))
        if section_time.size == 0:
            eta_series = np.asarray([], dtype=float)
        else:
            wl_obj = getattr(section_source, "water_level_object", ConstantWaterLevel(0.0))
            try:
                eta_series = np.asarray(wl_obj.get_water_level(section_time), dtype=float).reshape(-1)
            except Exception:
                eta_series = np.asarray(getattr(section_source, "water_level", np.zeros((section_source.n_ensembles,))), dtype=float).reshape(-1)

        if eta_series.size == 0:
            eta_mean = np.array([], dtype=float)
            eta_max = np.array([], dtype=float)
            eta_min = np.array([], dtype=float)
        else:
            eta_mean = np.array([np.nanmean(eta_series)], dtype=float)
            eta_max = np.array([np.nanmax(eta_series)], dtype=float)
            eta_min = np.array([np.nanmin(eta_series)], dtype=float)

        hp = np.asarray(getattr(section_source, "horizontal_position", np.empty((2, 0))), dtype=float)
        if hp.ndim == 2 and hp.shape[1] > 0 and np.any(np.isfinite(hp)):
            finite = np.all(np.isfinite(hp), axis=0)
            pts = hp[:, finite]
            if pts.shape[1] >= 2:
                xs_origin = np.asarray(xs.origin, dtype=float).reshape(2)
                direction = np.asarray(xs.direction, dtype=float).reshape(2)
            else:
                xs_origin = np.nanmean(hp, axis=1)
                direction = np.array([1.0, 0.0], dtype=float)
        else:
            xs_origin = np.array([np.nan, np.nan], dtype=float)
            direction = np.array([1.0, 0.0], dtype=float)

        Tvec = np.asarray(direction, dtype=float).reshape(2)
        Nvec = np.asarray(xs.direction_orthogonal, dtype=float).reshape(2)
        Pm = np.asarray(xs_origin, dtype=float).reshape(2)


        result.sections.append(
            ProcTransSection(
                row_index=row_index,
                crossing_ids=crossing_ids,
                ensembles=np.flatnonzero(mask).astype(int),
                eta=eta_mean,
                maxeta=eta_max,
                mineta=eta_min,
                time=section_time,
                Tvec=Tvec,
                Nvec=Nvec,
                Pm=Pm,
                mesh=mesh,
                solver=solver,
                vel=np.asarray(vel, dtype=float),
                cov_vel=np.asarray(cov_vel, dtype=float),
                nb_vel=np.asarray(nb_vel, dtype=int),
                r2=np.asarray(r2, dtype=float),
                r_sig=np.asarray(r_sig, dtype=float),
                vel_sn=np.asarray(vel_sn, dtype=float) if vel_sn is not None else None,
                cov_vel_sn=np.asarray(cov_vel_sn, dtype=float) if cov_vel_sn is not None else None,
                vel_cross=vel_cross,
                cov_vel_cross=cov_vel_cross,
                nb_vel_cross=nb_vel_cross,
                r2_cross=r2_cross,
                r_sig_cross=r_sig_cross,
                vel_sn_cross=vel_sn_cross,
                cov_vel_sn_cross=cov_vel_sn_cross,
            )
        )

    return result


procTrans = proc_trans