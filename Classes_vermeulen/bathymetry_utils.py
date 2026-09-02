import numpy as np

def extract_bed_position_from_meas(meas):
    """
    Extrait les coordonnées x, y, z de la bathymétrie depuis les transects de meas.

    Args:
        meas: Objet contenant les transects ADCP.

    Returns:
        bed_position: Tableau NumPy de forme (1, N, 1, 3) contenant les coordonnées x, y, z.
    """
    bed_x = []
    bed_y = []
    bed_z = []

    for transect in meas.transects:
        # Récupérer les coordonnées UTM du bateau (x, y)
        if hasattr(transect, 'gps') and transect.gps is not None:
            gps = transect.gps
            if hasattr(gps, 'utm_ens_m') and gps.utm_ens_m is not None:
                utm_ens_m = np.asarray(gps.utm_ens_m, dtype=float)  # Forme (2, N_ensembles)
                if utm_ens_m.ndim == 2 and utm_ens_m.shape[0] == 2:
                    x_ens = utm_ens_m[0, :]  # Coordonnées x
                    y_ens = utm_ens_m[1, :]  # Coordonnées y

        # Récupérer les profondeurs (z) depuis bt_depths
        if hasattr(transect, 'depths') and transect.depths is not None:
            depths = transect.depths
            if hasattr(depths, 'bt_depths') and depths.bt_depths is not None:
                bt_depths = depths.bt_depths
                if hasattr(bt_depths, 'depth_processed_m') and bt_depths.depth_processed_m is not None:
                    z_ens = np.asarray(bt_depths.depth_processed_m, dtype=float)  # Profondeurs (z)

                    # Ajouter les coordonnées (x, y, z) à bed_x, bed_y, bed_z
                    bed_x.extend(x_ens)
                    bed_y.extend(y_ens)
                    bed_z.extend(z_ens)

    # Convertir en tableaux NumPy
    bed_x = np.array(bed_x, dtype=float)
    bed_y = np.array(bed_y, dtype=float)
    bed_z = np.array(bed_z, dtype=float)

    print(f"\n--- Données de bathymétrie extraites ---")
    print(f"Nombre de points: {len(bed_x)}")
    print(f"Plage x: [{np.nanmin(bed_x):.2f}, {np.nanmax(bed_x):.2f}]")
    print(f"Plage y: [{np.nanmin(bed_y):.2f}, {np.nanmax(bed_y):.2f}]")
    print(f"Plage z: [{np.nanmin(bed_z):.2f}, {np.nanmax(bed_z):.2f}]")

    # Convertir en tableau NumPy avec la forme (1, N, 1, 3)
    bed_position = np.array([bed_x, bed_y, bed_z]).T.reshape(1, -1, 1, 3)
    return bed_position