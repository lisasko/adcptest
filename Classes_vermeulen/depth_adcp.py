from __future__ import annotations

import math
import numpy as np


def build_hpr_matrix(heading_deg, pitch_deg=0.0, roll_deg=0.0):
    h = np.radians(np.asarray(heading_deg, dtype=float))
    p = np.radians(np.asarray(pitch_deg, dtype=float))
    r = np.radians(np.asarray(roll_deg, dtype=float))

    ch, sh = np.cos(h), np.sin(h)
    cp, sp = np.cos(p), np.sin(p)
    cr, sr = np.cos(r), np.sin(r)

    shape = np.broadcast(ch, sh, cp, sp, cr, sr).shape
    out = np.zeros(shape + (4, 4), dtype=float)

    out[..., 0, 0] = ch * cr + sh * sp * sr
    out[..., 0, 1] = sh * cp
    out[..., 0, 2] = ch * sr - sh * sp * cr

    out[..., 1, 0] = -sh * cr + ch * sp * sr
    out[..., 1, 1] = ch * cp
    out[..., 1, 2] = -sh * sr - ch * sp * cr

    out[..., 2, 0] = -cp * sr
    out[..., 2, 1] = sp
    out[..., 2, 2] = cp * cr

    out[..., 3, 3] = 1.0
    return out


def depth_adcp(
    heading_deg,
    pitch_deg,
    roll_deg,
    beam_angle_deg,
    btrange_cm,
    transducer_depth_m=0.0,
    use_ext_heading=False,
    beam3_misalign_deg=0.0,
    external_heading_deg=None,
):
    heading_deg = np.asarray(heading_deg, dtype=float)
    pitch_deg = np.asarray(pitch_deg, dtype=float)
    roll_deg = np.asarray(roll_deg, dtype=float)
    btrange_cm = np.asarray(btrange_cm, dtype=float)

    if use_ext_heading:
        if external_heading_deg is None:
            raise ValueError("external_heading_deg is required when use_ext_heading=True")
        heading_deg = np.asarray(external_heading_deg, dtype=float) + float(beam3_misalign_deg)

    heading = np.radians(heading_deg)
    pitch = np.radians(pitch_deg)
    roll = np.radians(roll_deg)

    pitch = np.arctan(np.tan(pitch) * np.cos(roll))

    bangle = math.radians(float(beam_angle_deg))
    tbangle = math.tan(bangle)
    vecmagn = math.sqrt(tbangle * tbangle + 1.0)

    if btrange_cm.ndim == 1:
        btrange_cm = btrange_cm.reshape(4, -1)
    elif btrange_cm.shape[0] != 4 and btrange_cm.shape[-1] == 4:
        btrange_cm = np.moveaxis(btrange_cm, -1, 0)

    if btrange_cm.shape[0] != 4:
        raise ValueError("btrange_cm must have 4 beams")

    n_ens = btrange_cm.shape[1]
    if heading.ndim == 0:
        heading = np.full((n_ens,), float(heading))
    if pitch.ndim == 0:
        pitch = np.full((n_ens,), float(pitch))
    if roll.ndim == 0:
        roll = np.full((n_ens,), float(roll))

    if heading.shape[0] != n_ens:
        heading = np.broadcast_to(heading.reshape(-1)[0], (n_ens,))
    if pitch.shape[0] != n_ens:
        pitch = np.broadcast_to(pitch.reshape(-1)[0], (n_ens,))
    if roll.shape[0] != n_ens:
        roll = np.broadcast_to(roll.reshape(-1)[0], (n_ens,))

    zz = -np.ones((4, n_ens), dtype=float)
    xx = np.zeros((4, n_ens), dtype=float)
    yy = np.zeros((4, n_ens), dtype=float)

    xx[0, :] = zz[0, :] * tbangle
    xx[1, :] = -zz[1, :] * tbangle
    yy[2, :] = -zz[2, :] * tbangle
    yy[3, :] = zz[3, :] * tbangle

    ch, sh = np.cos(heading), np.sin(heading)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cr, sr = np.cos(roll), np.sin(roll)

    m11 = ch * cr + sh * sp * sr
    m12 = sh * cp
    m13 = ch * sr - sh * sp * cr

    m21 = -sh * cr + ch * sp * sr
    m22 = ch * cp
    m23 = -sh * sr - ch * sp * cr

    m31 = -cp * sr
    m32 = sp
    m33 = cp * cr

    xxt = xx * m11 + yy * m12 + zz * m13
    yyt = xx * m21 + yy * m22 + zz * m23
    zzt = xx * m31 + yy * m32 + zz * m33

    D = btrange_cm / 100.0 / math.cos(bangle)
    xxt = D / vecmagn * xxt
    yyt = D / vecmagn * yyt
    zzt = D / vecmagn * zzt
    zzt = zzt - float(transducer_depth_m)

    return np.stack(
        [
            xxt,
            yyt,
            zzt,
        ],
        axis=-1,
    )


def cor_adcp_simple(
    vel_beam,
    heading_deg,
    pitch_deg,
    roll_deg,
    t_matrix,
    own_matrix=None,
):
    vel_beam = np.asarray(vel_beam, dtype=float)
    t_matrix = np.asarray(t_matrix, dtype=float)

    if own_matrix is not None:
        own_matrix = np.asarray(own_matrix, dtype=float)
        return own_matrix @ vel_beam

    hpr = build_hpr_matrix(heading_deg, pitch_deg, roll_deg)
    if hpr.ndim != 2:
        raise ValueError("This helper expects scalar heading/pitch/roll")
    return hpr[:3, :3] @ t_matrix[:3, :3] @ vel_beam