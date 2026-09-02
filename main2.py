#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script de lancement pour le traitement des données ADCP montées sur bateau (VMADCP).
"""

import os
import sys

import numpy as np
import pandas as pd
import traceback
import matplotlib.pyplot as plt
from simplekml import Kml
from pyproj import Transformer
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from scipy.spatial import Delaunay

current_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(current_dir)

sys.path.insert(0, os.path.join(parent_dir, 'qrevint_21_03'))
sys.path.insert(0, os.path.join(parent_dir, 'Classes_vermeulen'))

from file_selector import select_measurement, load_measurement
from parametres import get_clean_parameters
from Classes_vermeulen.VMADCP import VMADCP
from Classes_vermeulen.EnsembleFilter import EnsembleFilter
from Classes_vermeulen.XSection import XSection
from Classes_vermeulen.Bathymetry import BathymetryScatteredPoints
from Classes_vermeulen.SigmaZetaMeshFromVMADCP import SigmaZetaMeshFromVMADCP
from Classes_vermeulen.LocationBasedVelocitySolver import LocationBasedVelocitySolver
from get_utm_zone import get_utm_zone
from Classes_vermeulen.plot_mesh_bathy import plot_bathymetry_and_mesh
from Classes_vermeulen.LocationBasedVelocitySolver import LocationBasedVelocitySolver
from Classes_vermeulen.plot_velocity import plot_velocity_cross_section
from plot_MAP import run_map_comparison, plot_comparison, map_profile_has_result, plot_comparison_info_count
from Classes_vermeulen.QVermeulen import compute_discharge_vermeulen
from plot_Q import plot_discharge_summary
from sensitivity_analysis import run_sensitivity_analysis


def main():

    # Sélection du fichier de mesure
    print("Sélectionnez une mesure ADCP (.mmt ou .mat)...")
    path_meas, type_meas, name_meas, selected_transects = select_measurement()

    # Chemin parent (3 niveaux au-dessus)
    if isinstance(path_meas, list):
        path_parent = os.path.dirname(path_meas[0])  
    else:
        path_parent = os.path.dirname(path_meas)
    for _ in range(3):
        path_parent = os.path.dirname(path_parent)

    # Chemin pour sauvegarder les résultats
    path_results = os.path.join(path_parent, 'Results')
    if not os.path.exists(path_results):
        os.makedirs(path_results)

    # Chargement des données
    print(f"\nChargement des données pour la mesure {name_meas}...")

    try:
        if type_meas == 'TRDI':
            path_file_str = path_meas[0] if isinstance(path_meas, list) else path_meas
            meas = load_measurement(path_file_str, type_meas, selected_transects)
        else:  # SonTek
            meas = load_measurement(path_meas, type_meas, selected_transects)
    except Exception as e:
        print(f"Erreur lors du chargement initial : {e}")
        traceback.print_exc()
        sys.exit(1)
    

    # Rechargement forcé si les transects ne sont pas chargés par défaut quand on met un .mmt en entrée
    if len(meas.transects) == 0:
        print("Attention: Aucun transect n'a été chargé via les filtres initiaux.")
        print("Tentative de chargement forcé de TOUS les transects du fichier...")
        try : 
            if type_meas == 'TRDI':
                meas.load_trdi(path_file_str, checked=False)
            else:  # SonTek : la sélection des transects est faite au moment de la sélection des fichiers .mat
                print("Impossible de charger les transects. Vérifiez les fichiers .mat et la sélection des transects.")
                sys.exit(1)
            print(f" Rechargement réussi : {len(meas.transects)} transects chargés.")
            if len(meas.transects) == 0:
                print("Erreur critique : Impossible de charger les transects de ce fichier.")
                sys.exit(1)
    
        except Exception as e:
            print(f"Échec du rechargement forcé : {e}")
            sys.exit(1)            

    print(f"\nNombre de transects chargés et prêts : {len(meas.transects)}")

    # Test de la valeur donnée de profondeur du transducteur : 
    
    print("\nVérification de la profondeur du transducteur...")

    first_transect = meas.transects[0]

    if hasattr(first_transect, 'depths') and first_transect.depths is not None:
        ref = first_transect.depths.selected
        if ref and hasattr(first_transect.depths, ref):
            depth_obj = getattr(first_transect.depths, ref)
            if hasattr(depth_obj, 'draft_use_m'):
                print(f"Valeur fournie par les données brutes ({ref}) : {depth_obj.draft_use_m} m")


    # Mise à jour des paramètres utilisateur

    print(f"\nMise à jour des paramètres utilisateur...")
    params = get_clean_parameters()

    err_heading = params["err_heading"]
    err_roll = params["err_roll"]
    err_pitch = params["err_pitch"]
    filtre_vitesse_z = params["filtre_vitesse_z"]
    filtre_direction_fixe = params["filtre_direction_fixe"]
    filtre_direction_pond = params["filtre_direction_pond"]
    ponderation_vitesses = params["ponderation_vitesses"]
    nbr_cell_hor = params["nbr_cell_hor"]
    nbr_cell_vert = params["nbr_cell_vert"]

    param_data = {
        "Paramètre": [
            "Erreur cap", "Erreur roulis", "Erreur tangage",
            "Filtre vitesse", "Filtre direction max", "Pondération filtre direction",
            "Pondération sur les vitesses", "Nombre cellules horizontales", "Nombre cellules verticales"
        ],
        "Valeur": [
            err_heading, err_roll, err_pitch,
            filtre_vitesse_z, filtre_direction_fixe, filtre_direction_pond,
            ponderation_vitesses, nbr_cell_hor, nbr_cell_vert
        ]
    }


    # Application de ces paramètres à l'objet de mesure

    print("Application des corrections d'attitude sur les transects...")

    num_transects_modifies = 0

    for transect in meas.transects:
        
        if hasattr(transect, 'sensors') and transect.sensors is not None:
            sensors = transect.sensors
            
            if err_heading != 0:
                corr_heading_deg = err_heading / 100.0
                if hasattr(sensors, 'heading') and sensors.heading is not None:
                    sensors.heading.data += corr_heading_deg
                    if hasattr(sensors.heading, 'avg_value') and sensors.heading.avg_value is not None:
                        sensors.heading.avg_value += corr_heading_deg
            
            if err_roll != 0:
                if hasattr(sensors, 'roll') and sensors.roll is not None:
                    sensors.roll.data += err_roll
                    if hasattr(sensors.roll, 'avg_value') and sensors.roll.avg_value is not None:
                        sensors.roll.avg_value += err_roll
            
            if err_pitch != 0:
                if hasattr(sensors, 'pitch') and sensors.pitch is not None:
                    sensors.pitch.data += err_pitch
                    if hasattr(sensors.pitch, 'avg_value') and sensors.pitch.avg_value is not None:
                        sensors.pitch.avg_value += err_pitch
                        
            num_transects_modifies += 1

    print(f"Corrections appliquées avec succès sur {num_transects_modifies} transect(s).")
    
    df_params = pd.DataFrame(param_data)
    excel_path = os.path.join(path_results, "param.xlsx")
    df_params.to_excel(excel_path, index=False)
    print(f"Paramètres sauvegardés dans {excel_path}")


    # Initaliation de l'objet VMADCP avec les transects chargés

    print("\nCréation de l'objet VMADCP...")

    reference_navigation = "gps_vel" # ou "bt_vel" mais la méthode vermeulen nécessite une navigation gps pour positionner les mesures dans le maillage. 

    try :
        vmadcp_obj = VMADCP(
            source=meas,
            transect_idx=0,
            nav_ref=reference_navigation,
            use_raw_bt_beam_bathy=True
        )

        print(f"Objet VMADCP initialisé avec succès.")
        print(f"- Nombre d'ensembles extraits : {vmadcp_obj.nensembles}")
        print(f"- Dimension des cellules de vitesse : {vmadcp_obj.water_velocity.shape}")

    except Exception as e:
        print(f"Erreur lors de l'initialisation de l'objet VMADCP : {e}")
        print("\n--- TRACEBACK COMPLETE DETECTEE ---")
        traceback.print_exc()  
        print("-----------------------------------")
        sys.exit(1)

    ## 03/08 : DEBUG
    print(f"\nDEBUG adcp.time : size={vmadcp_obj.time.size}, dtype={vmadcp_obj.time.dtype}")
    print(f"DEBUG adcp.time sample: {vmadcp_obj.time[:3]}")

    bp = np.asarray(vmadcp_obj.bed_position, dtype=float)  # (1, n_ens, 4, 3)

    x_ok = np.isfinite(bp[..., 0])
    y_ok = np.isfinite(bp[..., 1])
    z_ok = np.isfinite(bp[..., 2])

    print(f"\nDEBUG points avec z fini          : {int(np.sum(z_ok))}")
    print(f"DEBUG points avec x,y finis       : {int(np.sum(x_ok & y_ok))}")
    print(f"DEBUG points avec x,y,z finis     : {int(np.sum(x_ok & y_ok & z_ok))}")
    print(f"DEBUG points z fini MAIS x/y NaN  : {int(np.sum(z_ok & ~(x_ok & y_ok)))}")

    hp = vmadcp_obj._horizontal_position  # (2, n_ens)
    hp_bad = ~(np.isfinite(hp[0]) & np.isfinite(hp[1]))
    print(f"DEBUG ensembles horizontal_position NaN : {int(np.sum(hp_bad))} / {hp.shape[1]}")
    ##


    # Define cross sections

    print("\nDéfinition des sections transversales...")

    xs = XSection(vmadcp_obj)  
    
    print("\nExport des transects et de la XSection en KML...")
    print(f"XSection: origin={xs.origin}, direction={xs.direction}, scale={xs.scale}")

    kml = Kml()

    for i, transect in enumerate(meas.transects):
        if hasattr(transect, "gps") and transect.gps is not None:
            lat = np.asarray(transect.gps.gga_lat_ens_deg, dtype=float)
            lon = np.asarray(transect.gps.gga_lon_ens_deg, dtype=float)

            linestring = kml.newlinestring(name=f"Transect {i}")
            valid_mask = ~(np.isnan(lat) | np.isnan(lon))
            linestring.coords = list(zip(lon[valid_mask], lat[valid_mask]))  
            linestring.style.linestyle.color = "ff0000ff"  # Rouge
            linestring.style.linestyle.width = 2


    xsection_line = kml.newlinestring(name="XSection")
    xsection_origin_utm = xs.origin
    xsection_direction = xs.direction
    xsection_scale = xs.scale

    start_utm = (
        xsection_origin_utm[0] - xsection_direction[0] * xsection_scale / 2,
        xsection_origin_utm[1] - xsection_direction[1] * xsection_scale / 2,
    )
    end_utm = (
        xsection_origin_utm[0] + xsection_direction[0] * xsection_scale / 2,
        xsection_origin_utm[1] + xsection_direction[1] * xsection_scale / 2,
    )

    zone_meas = get_utm_zone(np.nanmean(vmadcp_obj._longitude))
    
    transformer_utm_to_wgs84 = Transformer.from_crs(
        f"+proj=utm +zone={zone_meas} +ellps=intl +towgs84=-87,-98,-121,0,0,0,0 +units=m +no_defs",  
        "+proj=longlat +ellps=intl +towgs84=-87,-98,-121,0,0,0,0 +no_defs"  
    )
    
    # transformer_utm_to_wgs84 = Transformer.from_crs(
    #     "EPSG:4326",
    #     f"+proj=utm +zone={zone_meas} +datum=WGS84 +units=m +no_defs",
    #     always_xy=True,
    # )

    start_lon, start_lat = transformer_utm_to_wgs84.transform(start_utm[0], start_utm[1])
    end_lon, end_lat = transformer_utm_to_wgs84.transform(end_utm[0], end_utm[1])

    xsection_line.coords = [(start_lon, start_lat), (end_lon, end_lat)]
    xsection_line.style.linestyle.color = "ffff0000"  # Bleu
    xsection_line.style.linestyle.width = 3

    origin_lon, origin_lat = transformer_utm_to_wgs84.transform(xs.origin[0], xs.origin[1])
    point = kml.newpoint(name="XSection Origin", coords=[(origin_lon, origin_lat)])
    point.style.iconstyle.color = "ff0000ff"  # Bleu
    point.style.iconstyle.scale = 1.5
    point.style.iconstyle.icon.href = "http://maps.google.com/mapfiles/kml/shapes/star.png"

    # Sauvegarde du fichier KML

    kml_path = os.path.join(path_results, f"{name_meas}_transects_and_xsection.kml")
    kml.save(kml_path)
    print(f"Fichier KML sauvegardé : {kml_path}")


    # Bathymétrie 

    print("\nCréation de la bathymétrie...")

    ensemble_filter = EnsembleFilter(vmadcp_obj) 

    bathy = BathymetryScatteredPoints(vmadcp_obj, ensemble_filter)

    print(f"\n[DIAG] Nombre de points bruts retenus pour la bathymétrie : {bathy.known.shape[1]}")
    print(f"[DIAG] span utilisé par l'interpolateur : {bathy.interpolator.span}")
    print(f"[DIAG] k voisins effectif (span<=1 => round(span*n)) : "
        f"{max(3, min(bathy.known.shape[1], round(bathy.interpolator.span * bathy.known.shape[1])))}")

    print("\n--- Comparaison BathymetryScatteredPoints ---")
    print(f"  Nombre de points connus: {bathy.known.shape[1]}")
    print(f"  Plage x: [{np.min(bathy.known[0, :]):.2f}, {np.max(bathy.known[0, :]):.2f}]")
    print(f"  Plage y: [{np.min(bathy.known[1, :]):.2f}, {np.max(bathy.known[1, :]):.2f}]")
    print(f"  Plage z: [{np.min(bathy.known[2, :]):.2f}, {np.max(bathy.known[2, :]):.2f}]")

    bad = ensemble_filter.all_cells_bad(vmadcp_obj)
    print("\n--- Comparaison EnsembleFilter ---")
    print(f"Python - EnsembleFilter:")
    print(f"  Nombre de cellules filtrées: {np.sum(bad)}")

    # Paramètre de lissage 
    ## Explication Matlab : lower the degree of smoothing (neighborhood is now 0.5 % of all depth points)
    # bathy.interpolator.span = 0.005

    n_known = bathy.known.shape[1]
    min_neighbors_effective = 20 
    span_adapte = max(0.005, (min_neighbors_effective + 1) / max(n_known, 1))
    bathy.interpolator.span = span_adapte
    print(
        f"DEBUG span interpolateur bathymétrie : {span_adapte:.4f} "
        f"({n_known} points connus, cible ~{min_neighbors_effective} voisins effectifs "
        f"après perte du voisin le plus lointain à poids nul)"
    )

    print(f"Nombre de points connus pour la bathymétrie : {bathy.known.shape[1]}")
    if bathy.known.shape[1] == 0:
        raise ValueError("Aucun point connu pour la bathymétrie. Vérifiez vmadcp_obj.bed_position.")


    # Profondeur maximale et longueur de la section
    known_x_utm = bathy.known[0, :]
    known_y_utm = bathy.known[1, :]
    known_z_raw = bathy.known[2, :]

    ## 04/08
    valid_known = np.isfinite(known_x_utm) & np.isfinite(known_y_utm) & np.isfinite(known_z_raw)
    x_valid = known_x_utm[valid_known]
    y_valid = known_y_utm[valid_known]
    z_valid = known_z_raw[valid_known]

    d_sec = float(np.sqrt(
        (np.max(y_valid) - np.min(y_valid)) ** 2 +
        (np.max(x_valid) - np.min(x_valid)) ** 2
    ))
    prof = float(np.abs(np.min(z_valid)))

    print(f"\nDEBUG d_sec (diagonale x/y, formule MATLAB) : {d_sec:.2f} m")
    print(f"DEBUG prof (max profondeur)                 : {prof:.2f} m")
    print(f"DEBUG x span : {np.max(x_valid) - np.min(x_valid):.2f} m")
    print(f"DEBUG y span : {np.max(y_valid) - np.min(y_valid):.2f} m")
    ##

    print(f"Profondeur maximale de la section : {prof:.2f} m")
    print(f"Longueur de la section : {d_sec:.2f} m")
    print(f"Mise à jour de XSection.scale : {xs.scale:.2f} m")


    # Maillage

    print("\nCréation du maillage...")

    # Générateur de maillage

    mmaker = SigmaZetaMeshFromVMADCP(
        vmadcp_obj,  
        bathy,      
        xs,       
        ensemble_filter 
    )


    # Configuration des résolutions horizontale et verticale (calcul dynamique)
    mmaker.deltan = d_sec / (nbr_cell_hor * 25/21)
    mmaker.deltaz = prof / (nbr_cell_vert * 4/3)

    # # Valeurs par défaut (utilisées en Matlab)
    # mmaker.deltan = 5.0 
    # mmaker.deltaz = 1.0 

    try:
        mesh = mmaker.get_mesh()

        print("Maillage généré avec succès.")
        print(f"Nombre de cellules dans le maillage : {mesh.ncells}")

        print(f"Mesh.nb_all: min={np.min(mesh.nb_all):.2f}, max={np.max(mesh.nb_all):.2f}")
        print(f"Mesh.zb_all: min={np.min(mesh.zb_all):.2f}, max={np.max(mesh.zb_all):.2f}")

        print(f"Mesh.water_level: {mesh.water_level:.2f} m")

    except ImportError as e:

        if "alphashape" in str(e) or "shapely" in str(e):
            print("Erreur : Les packages 'alphashape' et 'shapely' sont requis pour générer le maillage.")
            print("Installez-les avec : pip install alphashape shapely")
        else:
            print(f"Erreur inattendue : {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Erreur lors de la génération du maillage : {e}")
        print("\n--- TRACEBACK COMPLETE DETECTEE ---")
        traceback.print_exc()
        print("-----------------------------------")
        sys.exit(1)


    # Visualisation de la bathymétrie et du maillage

    print("\nGénération des visualisations...")

    plot_bathymetry_and_mesh(bathy, mesh, xs, vmadcp=vmadcp_obj)

    plt.show(block=False)
    plt.pause(0.1)


    # Résolution du champ de vitesses

    print("\nRésolution du champ de vitesses (LocationBasedVelocitySolver)...")

    vsolver = LocationBasedVelocitySolver(
        vmadcp_obj,
        bathy,
        xs,
        ensemble_filter,
        mesh,
    )

    vel, cov_vel, nb_vel, r2, r_sig = vsolver.get_velocity(
        f_vitesse=filtre_vitesse_z,
        f_direction_fixe=filtre_direction_fixe,
        f_direction_pond=filtre_direction_pond,
        pond_vitesses=ponderation_vitesses,
    )

    print(f"vel[0] (cellule 0..4): {np.asarray(vel[0])[:5]}")

    vel_sn, cov_vel_sn = vsolver.rotate_to_xs(vel, cov_vel)
    vel_rozovskii = vsolver.compute_rozovskii(vel)

    n_valid_cells = int(np.sum(np.isfinite(vel_sn[0][:, 0])))
    print(f"Nombre de cellules avec vitesse streamwise calculée : {n_valid_cells} / {mesh.ncells}")

    vel_sn_arr = np.asarray(vel_sn[0], dtype=float)
    streamwise = vel_sn_arr[:, 0]
    print(
        f"\nDEBUG vel_sn streamwise (repere section, avant affichage) : "
        f"min={np.nanmin(streamwise):.3f} m/s, max={np.nanmax(streamwise):.3f} m/s, "
        f"mean={np.nanmean(streamwise):.3f} m/s, std={np.nanstd(streamwise):.3f} m/s, "
        f"NaN count={int(np.sum(np.isnan(streamwise)))}"
    )


    print("\nGénération de la visualisation du champ de vitesses...")

    _, _ = plot_velocity_cross_section(
        mesh, vel_sn[0], vmadcp=vmadcp_obj,
        component="streamwise",
        clim=None,
    )

    plt.show(block=False)
    plt.pause(0.1)


    print("\nCalcul de la méthode classique (MAP) pour comparaison...")

    map_results = run_map_comparison(
        meas,
        navigation_reference_user='GPS',   
        nbr_cell_hor=nbr_cell_hor,
        nbr_cell_vert=nbr_cell_vert,
        methods=('streamwise', 'roz'),           # options : 'streamwise' et 'roz' (Rozovskii)
    )

    map_profile_sw = map_results['streamwise']
    map_profile_roz = map_results['roz']

    # if map_profile_has_result(map_profile_sw):
    #     fig_cmp_sw, _ = plot_comparison(
    #         mesh, vel_sn[0], vmadcp_obj, map_profile_sw,
    #         method_label="streamwise",
    #     )
    #     plt.show(block=False)
    #     plt.pause(0.1)
    # else:
    #     print("Comparaison MAP streamwise indisponible (voir logs ci-dessus).")

    # if map_profile_has_result(map_profile_roz):
    #     fig_cmp_roz, _ = plot_comparison(
    #         mesh, vel_rozovskii[0], vmadcp_obj, map_profile_roz,8
    #         method_label="Rozovskii",
    #     )
    #     plt.show(block=False)
    #     plt.pause(0.1)
    # else:
    #     print("Comparaison MAP Rozovskii indisponible (voir logs ci-dessus).")

    shared_x_alignment = None  # sera rempli par le premier appel réussi (streamwise)

    if map_profile_has_result(map_profile_sw):
        fig_cmp_sw, _, shared_x_alignment = plot_comparison(
            mesh, vel_sn[0], vmadcp_obj, map_profile_sw,
            method_label="streamwise",
        )
        plt.show(block=False)
        plt.pause(0.1)
    else:
        print("Comparaison MAP streamwise indisponible (voir logs ci-dessus).")

    if map_profile_has_result(map_profile_roz):
        fig_cmp_roz, _, _ = plot_comparison(
            mesh, vel_rozovskii[0], vmadcp_obj, map_profile_roz,
            method_label="Rozovskii",
            x_alignment=shared_x_alignment,  # None si streamwise a échoué -> recalcul de secours
        )
        plt.show(block=False)
        plt.pause(0.1)
    else:
        print("Comparaison MAP Rozovskii indisponible (voir logs ci-dessus).")


    # Nombre d'informations par cellules
    print("\nGénération de la comparaison du nombre de mesures brutes par cellule...")

    if map_profile_sw is not None and map_profile_sw.info_cell is not None:
        fig_cmp_info, _ = plot_comparison_info_count(
            mesh, nb_vel[0], vmadcp_obj, map_profile_sw,
        )
        plt.show(block=False)
        plt.pause(0.1)
    else:
        print("Comparaison du nombre de mesures brutes indisponible (MAP streamwise manquant).")


    # Calcul du débit

    map_results_discharge = run_map_comparison(
        meas, navigation_reference_user='GPS',
        nbr_cell_hor=nbr_cell_hor, nbr_cell_vert=nbr_cell_vert,
        extrap_option=True,   # <-- important pour un débit total réaliste
        methods=('streamwise',),
    )


    discharge_verm = compute_discharge_vermeulen(mesh, vel_sn[0])
    q_vermeulen = discharge_verm["total_discharge"]

    ## 27/08
    q_map_ref = map_results['streamwise'].total_discharge if map_results.get('streamwise') else None
    if q_map_ref is not None and np.sign(q_vermeulen) != np.sign(q_map_ref):
        print(
            f"ATTENTION : signe du débit Vermeulen ({q_vermeulen:+.2f} m3/s) opposé "
            f"à celui de MAP ({q_map_ref:+.2f} m3/s) -- inversion de la convention "
            f"de signe streamwise du Vermeulen (vecteur propre ACP sans signe "
            f"canonique) pour cohérence physique."
        )
        vel_sn[0][:, 0] *= -1.0   # inverse la composante streamwise affichée
        discharge_verm = compute_discharge_vermeulen(mesh, vel_sn[0])
        q_vermeulen = discharge_verm["total_discharge"]
    ##

    print("\n Comparaison des débits:")
    print(f"Débit Vermeulen         : {q_vermeulen:.4f} m3/s")

    if map_results.get('streamwise') is not None:
        q_map = map_results['streamwise'].total_discharge
        print(f"Débit MAP (streamwise)  : {q_map:.4f} m3/s")
        if q_map:
            ecart = 100 * (q_vermeulen - q_map) / q_map
            print(f"Écart relatif Vermeulen vs MAP : {ecart:+.2f} %")

    if map_results.get('roz') is not None:
        q_roz = map_results['roz'].total_discharge
        print(f"Débit MAP (Rozovskii)   : {q_roz:.4f} m3/s")

    q_map_extrap = map_results_discharge['streamwise'].total_discharge
    print(f"Débit MAP (streamwise, avec extrapolation) : {q_map_extrap:.4f} m3/s")

    ecart_extrap = 100 * (q_vermeulen - q_map_extrap) / q_map_extrap
    print(f"Écart relatif Vermeulen vs MAP extrapolé : {ecart_extrap:+.2f} %")

    rows = [
        {"label": "Aire mouillée (Vermeulen)", "value": f"{discharge_verm['cell_area'].sum():.2f} m²"},
        {"label": "", "value": ""},
        {"label": "Débit Vermeulen", "value": f"{q_vermeulen:.4f} m³/s"},
        {"label": "Débit MAP (streamwise)", "value": f"{q_map:.4f} m³/s"},
        {"label": "Débit MAP (Rozovskii)", "value": f"{q_roz:.4f} m³/s"},
        {"label": "Débit MAP (streamwise, extrapolé)", "value": f"{q_map_extrap:.4f} m³/s"},
        {"label": "", "value": ""},
        {"label": "Écart Vermeulen vs MAP (streamwise)", "value": f"{ecart:+.2f} %"},
        {"label": "Écart Vermeulen vs MAP (extrapolé)", "value": f"{ecart_extrap:+.2f} %"},
    ]

    fig_q, ax_q = plot_discharge_summary(rows, title=f"Synthèse des débits -- {name_meas}")
    plt.show(block=False)
    plt.pause(0.1)

    # Analyses de sensibilité

    RUN_SENSITIVITY_ANALYSIS = True

    if RUN_SENSITIVITY_ANALYSIS:
        print("\nAnalyse de sensibilité géométrique (PDOP)...")
        run_sensitivity_analysis(mesh, vsolver, nb_vel, vmadcp=vmadcp_obj)



    print("\nAppuie sur Entrée pour continuer...")
    input()
    
   

if __name__ == "__main__":

    ## Phase de correction : 
    
    if os.environ.get("STRICT_DEPRECATION") == "1":
        from debug import run_with_deprecation_report
        run_with_deprecation_report(main)

    elif os.environ.get("QREVINT_WARNINGS_SUMMARY") == "1":
        from debug import summarize_qrevint_warnings
        summarize_qrevint_warnings(main)
        
    else:
        main()

    ##

    # main()