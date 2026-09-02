# -*- coding: utf-8 -*-
"""
Calcul du débit total pour la méthode Vermeulen.
"""

import numpy as np


def polygon_area(n_coords, z_coords):
    """
    Aire d'un polygone (cellule du maillage SigmaZeta).

    Parameters :
        n_coords, z_coords: array-like
            Coordonnées des sommets du polygone, dans l'ordre (peu importe le
            sens de parcours, la formule renvoie une valeur positive).

    Returns :
        float : aire du polygone (m²), 0.0 si moins de 3 sommets valides.
    """
    n_coords = np.asarray(n_coords, dtype=float)
    z_coords = np.asarray(z_coords, dtype=float)
    valid = np.isfinite(n_coords) & np.isfinite(z_coords)
    n_coords, z_coords = n_coords[valid], z_coords[valid]
    if n_coords.size < 3:
        return 0.0
    return 0.5 * abs(
        np.dot(n_coords, np.roll(z_coords, -1)) - np.dot(z_coords, np.roll(n_coords, -1))
    )


def compute_discharge_vermeulen(mesh, vel_sn):
    """
    Parameters :
        mesh: SigmaZetaMesh
            Maillage Vermeulen déjà résolu (utilise mesh.n_patch, mesh.z_patch,
            mesh.ncells).
        vel_sn: np.ndarray (ncells, 3)
            Vitesse par cellule en repère section (sortie de
            VelocitySolver.rotate_to_xs -> vel_sn[0] dans main2.py). Colonne 0 =
            vitesse streamwise.

    Returns :
        dict :
            cell_area: np.ndarray (ncells,)       aire de chaque cellule (m²)
            cell_discharge: np.ndarray (ncells,)  débit de chaque cellule (m³/s)
            total_discharge: float                débit total (m³/s)
    """
    vel_sn = np.asarray(vel_sn, dtype=float)
    ncells = mesh.ncells

    cell_area = np.full(ncells, np.nan)
    for cc in range(ncells):
        cell_area[cc] = polygon_area(mesh.n_patch[:, cc], mesh.z_patch[:, cc])

    streamwise = vel_sn[:, 0]
    cell_discharge = cell_area * streamwise
    total_discharge = float(np.nansum(cell_discharge))

    print(
        f"DEBUG discharge Vermeulen : aire mouillée totale (maillage) = "
        f"{np.nansum(cell_area):.2f} m2, débit total = {total_discharge:.4f} m3/s"
    )

    return {
        "cell_area": cell_area,
        "cell_discharge": cell_discharge,
        "total_discharge": total_discharge,
    }