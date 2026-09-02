from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree


def _resolve_k(span: float, n: int) -> int:
    if span <= 1.0:
        k = int(round(span * n))
    else:
        k = int(round(span))
    return max(3, min(n, k))


def _poly_features(dx: np.ndarray, order: int) -> np.ndarray:
    # dx: (k, d)
    k, d = dx.shape
    cols = [np.ones((k, 1), dtype=float), dx]
    if order == 2:
        quad = []
        for i in range(d):
            for j in range(i, d):
                quad.append((dx[:, i] * dx[:, j])[:, None])
        if quad:
            cols.append(np.hstack(quad))
    return np.hstack(cols)


def _tricube(dist: np.ndarray) -> np.ndarray:
    dmax = np.nanmax(dist)
    if not np.isfinite(dmax) or dmax <= 0:
        return np.ones_like(dist, dtype=float)
    u = np.clip(dist / dmax, 0.0, 1.0)
    return (1.0 - u**3) ** 1.5


def _fit_at_points(
    x_data: np.ndarray,
    y_data: np.ndarray,
    x_query: np.ndarray,
    k: int,
    order: int,
    robust_w: np.ndarray,
    nthreads: int | None,
) -> np.ndarray:
    n = x_data.shape[0]
    m = x_query.shape[0]
    out = np.full((m,), np.nan, dtype=float)
    tree = cKDTree(x_data)

    kk = min(k, n)
    try:
        _, idx = tree.query(x_query, k=kk, workers=(1 if nthreads is None else int(nthreads)))
    except TypeError:
        _, idx = tree.query(x_query, k=kk)

    if kk == 1:
        idx = idx[:, None]

    for i in range(m):
        neigh = idx[i]
        xn = x_data[neigh]
        yn = y_data[neigh]

        dx = xn - x_query[i]
        dist = np.linalg.norm(dx, axis=1)
        w = _tricube(dist) * robust_w[neigh]

        good = np.isfinite(yn) & np.isfinite(w) & (w > 0)
        if np.count_nonzero(good) < 3:
            continue

        A = _poly_features(dx[good], order)
        b = yn[good]
        # sw = np.sqrt(w[good])

        # Aw = A * sw[:, None]
        # bw = b * sw

        ## 03/08
        Aw = A * w[good, None]
        bw = b * w[good]

        try:
            coef, *_ = np.linalg.lstsq(Aw, bw, rcond=None)

            ##
            fitted = coef[0]
            # b_min = np.min(b)
            # b_max = np.max(b)

            # out[i] = float(np.clip(fitted, b_min, b_max))
            ##
            out[i] = coef[0]
        except np.linalg.LinAlgError:
            continue

    return out


def loess_nd(
    x: np.ndarray,
    v: np.ndarray,
    xi: np.ndarray,
    span: float,
    niter: int = 0,
    order: int = 1,
    nthreads: int | None = None,
) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    v = np.asarray(v, dtype=float).reshape(-1)
    xi = np.asarray(xi, dtype=float)

    if x.ndim != 2 or xi.ndim != 2:
        raise ValueError("x and xi must be 2D arrays")
    if x.shape[0] != v.size:
        raise ValueError("x and v size mismatch")
    if x.shape[1] != xi.shape[1]:
        raise ValueError("x and xi must have same number of columns")
    if order not in (1, 2):
        raise ValueError("order must be 1 or 2")
    if span <= 0:
        raise ValueError("span must be > 0")

    finite = np.all(np.isfinite(x), axis=1) & np.isfinite(v)
    x_data = x[finite]
    y_data = v[finite]

    if x_data.shape[0] == 0:
        return np.full((xi.shape[0],), np.nan, dtype=float)

    k = _resolve_k(float(span), x_data.shape[0])
    robust_w = np.ones((x_data.shape[0],), dtype=float)

    # Robust loops on training points only
    for _ in range(max(0, int(niter))):
        y_hat = _fit_at_points(
            x_data=x_data,
            y_data=y_data,
            x_query=x_data,
            k=k,
            order=order,
            robust_w=robust_w,
            nthreads=nthreads,
        )
        resid = y_data - y_hat
        med = np.nanmedian(np.abs(resid))
        s = max(6.0 * med, np.finfo(float).eps)
        u = resid / s
        robust_w = np.where(np.abs(u) < 1.0, 1.0 - u**2, 0.0)
        robust_w[~np.isfinite(robust_w)] = 0.0

    return _fit_at_points(
        x_data=x_data,
        y_data=y_data,
        x_query=xi,
        k=k,
        order=order,
        robust_w=robust_w,
        nthreads=nthreads,
    )

