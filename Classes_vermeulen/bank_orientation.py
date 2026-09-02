import numpy as np

def get_bank_orientation(meas_or_vmadcp):
    """
    Détermine si le transect commence par la rive droite (retourne True) ou gauche (retourne False).
    Returns:
        bool: True si start_edge = 'Right', False si start_edge = 'Left'
    """
    # Cas 1: Objet Measurement (directement)
    if hasattr(meas_or_vmadcp, 'transects') and len(meas_or_vmadcp.transects) > 0:
        transect = meas_or_vmadcp.transects[0]
    # Cas 2: Objet VMADCP (accéder à source)
    elif hasattr(meas_or_vmadcp, 'source') and hasattr(meas_or_vmadcp.source, 'transects'):
        transect = meas_or_vmadcp.source.transects[0]
    else:
        return False  # Valeur par défaut (Left)

    # Récupérer start_edge ou orig_start_edge
    if hasattr(transect, 'start_edge') and transect.start_edge is not None:
        start_edge = str(transect.start_edge).lower()
    elif hasattr(transect, 'orig_start_edge') and transect.orig_start_edge is not None:
        start_edge = str(transect.orig_start_edge).lower()
    else:
        return False  # Valeur par défaut (Left)

    # Retourne True si start_edge = 'right', False sinon
    return start_edge == 'right'

def _is_right_start_edge(start_edge):
    """Vérifie si start_edge correspond à 'Right'."""
    if not start_edge:
        return False
    if isinstance(start_edge, (list, tuple, np.ndarray)):
        if len(start_edge) == 0:
            return False
        start_edge = start_edge[0]
    return str(start_edge).lower() == 'right'

def normalize_transects_orientation(meas):
    """
    Normalise l'orientation de tous les transects pour qu'ils aillent 
    SYSTÉMATIQUEMENT de la Rive Gauche vers la Rive Droite.
    
    Si start_edge == 'Right', les données du transect sont inversées (flipped)
    le long de l'axe des ensembles.
    """
    print("\n--- Normalisation de l'orientation des transects (RG -> RD) ---")
    
    for idx, transect in enumerate(meas.transects):
        start_edge = getattr(transect, 'start_edge', 'Left')
        if isinstance(start_edge, str):
            start_edge_clean = start_edge.strip().capitalize()
        else:
            start_edge_clean = 'Left'
            
        # Détection si départ en Rive Droite ('Right' ou 'R')
        is_right = start_edge_clean.startswith('R')
        
        if is_right:
            print(f"Transect [{idx}] : Démarré en Rive Droite ('{start_edge}'). -> INVERSION DES DONNÉES ENSEMBLE PAR ENSEMBLE.")
            
            # 1. Traitement des données d'ensembles (GPS, Nav, Sensors, Bottom Track, Depths)
            if hasattr(transect, 'gps') and transect.gps is not None:
                for attr in ['gga_lat_ens_deg', 'gga_lon_ens_deg', 'vtg_kph']:
                    if hasattr(transect.gps, attr) and getattr(transect.gps, attr) is not None:
                        val = getattr(transect.gps, attr)
                        setattr(transect.gps, attr, np.flip(val, axis=-1))

            if hasattr(transect, 'boat_vel') and transect.boat_vel is not None:
                for attr in ['bt_vel_mps', 'gga_vel_mps', 'vtg_vel_mps']:
                    if hasattr(transect.boat_vel, attr) and getattr(transect.boat_vel, attr) is not None:
                        val = getattr(transect.boat_vel, attr)
                        setattr(transect.boat_vel, attr, np.flip(val, axis=-1))

            if hasattr(transect, 'sensors') and transect.sensors is not None:
                for sens in ['heading', 'roll', 'pitch']:
                    if hasattr(transect.sensors, sens):
                        obj = getattr(transect.sensors, sens)
                        if hasattr(obj, 'data') and obj.data is not None:
                            obj.data = np.flip(obj.data, axis=-1)

            if hasattr(transect, 'depths') and transect.depths is not None:
                if hasattr(transect.depths, 'bt_depths') and transect.depths.bt_depths is not None:
                    if hasattr(transect.depths.bt_depths, 'depth_beams_m') and transect.depths.bt_depths.depth_beams_m is not None:
                        # Matrice 2D [4, n_ensembles]
                        transect.depths.bt_depths.depth_beams_m = np.flip(transect.depths.bt_depths.depth_beams_m, axis=-1)
                    if hasattr(transect.depths.bt_depths, 'depth_processed_m') and transect.depths.bt_depths.depth_processed_m is not None:
                        transect.depths.bt_depths.depth_processed_m = np.flip(transect.depths.bt_depths.depth_processed_m, axis=-1)

            # 2. Traitement des vitesses d'eau (Water Velocity / w_vel)
            if hasattr(transect, 'w_vel') and transect.w_vel is not None:
                if hasattr(transect.w_vel, 'vel_raw_mps') and transect.w_vel.vel_raw_mps is not None:
                    # Matrice 3D [4, n_cells, n_ensembles] ou 2D
                    transect.w_vel.vel_raw_mps = np.flip(transect.w_vel.vel_raw_mps, axis=-1)
            
            # Une fois inversé, le transect va désormais de RG -> RD
            transect.start_edge = 'Left_normalized'
            
        else:
            print(f"Transect [{idx}] : Démarré en Rive Gauche ('{start_edge}'). -> Conservation du sens d'origine.")

# import numpy as np

# def get_bank_orientation(meas_or_vmadcp):
#     """
#     Détermine si le transect commence par la rive gauche ou droite.
#     Returns:
#         str: 'left' ou 'right'
#     """
#     # Cas 1: Objet Measurement (directement)
#     if hasattr(meas_or_vmadcp, 'transects') and len(meas_or_vmadcp.transects) > 0:
#         transect = meas_or_vmadcp.transects[0]
#     # Cas 2: Objet VMADCP (accéder à source)
#     elif hasattr(meas_or_vmadcp, 'source') and hasattr(meas_or_vmadcp.source, 'transects'):
#         transect = meas_or_vmadcp.source.transects[0]
#     else:
#         return 'left'  # Valeur par défaut

#     # Récupérer start_edge ou orig_start_edge
#     if hasattr(transect, 'start_edge') and transect.start_edge is not None:
#         start_edge = transect.start_edge.lower()
#         if start_edge in ['left', 'right']:
#             return start_edge
#     elif hasattr(transect, 'orig_start_edge') and transect.orig_start_edge is not None:
#         start_edge = transect.orig_start_edge.lower()
#         if start_edge in ['left', 'right']:
#             return start_edge

#     return 'left' 

# def invert_data_if_right(start_edge, nvec, zvec, known_n, known_z, bathy_known):
#     """
#     Inverse les données si start_edge = 'right' pour toujours avoir RG à gauche et RD à droite.

#     Args:
#         start_edge: 'left' ou 'right'
#         nvec: Coordonnées "n" du maillage
#         zvec: Élévations du lit
#         known_n: Coordonnées "n" des points de bathymétrie
#         known_z: Élévations des points de bathymétrie
#         bathy_known: Tableau (3, N) des points de bathymétrie (x, y, z)

#     Returns:
#         nvec_inverted, zvec_inverted, known_n_inverted, known_z_inverted, bathy_known_inverted
#     """
#     if start_edge == 'right':
#         print("Inversion des données pour avoir RG à gauche et RD à droite.")

#         # Inverser nvec et zvec
#         nvec_inverted = -nvec[::-1]  # Inverser l'ordre et le signe
#         zvec_inverted = zvec[::-1]   # Inverser l'ordre uniquement

#         # Inverser known_n et known_z
#         known_n_inverted = -known_n[::-1]
#         known_z_inverted = known_z[::-1]

#         # Inverser bathy.known (coordonnées x et y)
#         bathy_known_inverted = bathy_known.copy()
#         bathy_known_inverted[0, :] = -bathy_known[0, ::-1]  # Inverser x
#         bathy_known_inverted[1, :] = -bathy_known[1, ::-1]  # Inverser y
#         # z reste inchangé (car lié à la profondeur)

#         return nvec_inverted, zvec_inverted, known_n_inverted, known_z_inverted, bathy_known_inverted
#     else:
#         return nvec, zvec, known_n, known_z, bathy_known