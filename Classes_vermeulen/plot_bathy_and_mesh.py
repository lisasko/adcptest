# from mpl_toolkits.mplot3d import Axes3D
# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib import cm
# from scipy.spatial import Delaunay
# import pandas as pd
# from matplotlib.collections import PolyCollection

# def plot_bathymetry_and_mesh(bathy, mesh, xs):
#     """
#     Trace la bathymétrie et le maillage en 3D (comme en MATLAB).
#     """
#     fig = plt.figure(figsize=(12, 8))
#     ax = fig.add_subplot(111, projection='3d')

#     # Tracer les points de bathymétrie
#     hp = ax.scatter(
#         bathy.known[0, :],
#         bathy.known[1, :],
#         bathy.known[2, :],
#         c='k',
#         s=2,
#         label='Points de bathymétrie',
#         alpha=0.5
#     )

#     # Triangulation pour la surface de bathymétrie
#     try:
#         tri = Delaunay(np.vstack((bathy.known[0, :], bathy.known[1, :])).T)
#         z_interp = bathy.get_bed_elev(bathy.known[0, :], bathy.known[1, :])

#         # Tracer la surface triangulée SANS légende (pour éviter l'erreur)
#         ht = ax.plot_trisurf(
#             bathy.known[0, :],
#             bathy.known[1, :],
#             z_interp,
#             cmap=cm.viridis,
#             alpha=0.7,
#             linewidth=0.0,
#             label=''  # ✅ Supprimer le label pour éviter l'erreur
#         )
#         fig.colorbar(ht, ax=ax, label='Élévation du lit (m)')
#     except Exception as e:
#         print(f"Warning: Impossible de tracer la surface triangulée: {e}")

#     # Tracer les bords du maillage
#     try:
#         n_left = mesh.n_left
#         n_right = mesh.n_right
#         zb_left = mesh.zb_left
#         zb_right = mesh.zb_right

#         x_left, y_left = xs.sn2xy(np.zeros_like(n_left), n_left)
#         x_right, y_right = xs.sn2xy(np.zeros_like(n_right), n_right)

#         ax.plot(
#             x_left, y_left, zb_left,
#             'b-', linewidth=1, label='Bord gauche du maillage'
#         )
#         ax.plot(
#             x_right, y_right, zb_right,
#             'r-', linewidth=1, label='Bord droit du maillage'
#         )
#     except Exception as e:
#         print(f"Warning: Impossible de tracer le maillage: {e}")

#     # Configuration du graphique
#     ax.set_xlabel('UTM x (m)')
#     ax.set_ylabel('UTM y (m)')
#     ax.set_zlabel('Élévation (m)')
#     ax.set_title('Bathymétrie et maillage')

#     # ✅ Ajouter une légende manuelle pour les lignes (pas pour plot_trisurf)
#     handles, labels = ax.get_legend_handles_labels()
#     if handles:  # ✅ Vérifier qu'il y a des handles avant d'ajouter la légende
#         ax.legend(handles, labels)

#     ax.set_box_aspect([5, 5, 1])  # Ratio comme en MATLAB

#     plt.tight_layout()
#     plt.show(block=False)
#     plt.pause(0.1)


# # def plot_mesh_2d(mesh, bathy, xs, n_points_raw=None, n_start_rg=None, n_end_rd=None):
# #     """
# #     Trace le profil 2D du maillage et de la bathymétrie (comme en MATLAB).
# #     """
# #     fig, ax = plt.subplots(figsize=(12, 6))

# #     # Bathymetry
# #     known_x = bathy.known[0, :]
# #     known_y = bathy.known[1, :]
# #     known_z = bathy.known[2, :]

# #     valid_mask = np.isfinite(known_x) & np.isfinite(known_y) & np.isfinite(known_z)
# #     known_x = known_x[valid_mask]
# #     known_y = known_y[valid_mask]
# #     known_z = known_z[valid_mask]

# #     _, n_known = xs.xy2sn(known_x, known_y)
# #     valid_mask = np.isfinite(n_known)
# #     n_known = n_known[valid_mask]
# #     known_z = known_z[valid_mask]

# #     n_unique = np.linspace(np.min(n_known), np.max(n_known), 200)  # 200 points pour lisser
# #     z_mean = np.zeros_like(n_unique)

# #     for i, n_val in enumerate(n_unique):
# #         mask = np.abs(n_known - n_val) < 0.5  # Fenêtre de 0.5m autour de n_val
# #         if np.any(mask):
# #             z_mean[i] = np.nanmean(known_z[mask])
# #         else:
# #             z_mean[i] = np.nan

# #     z_mean = pd.Series(z_mean).interpolate().values

# #     # Tracer le fond interpolé (ligne noire)
# #     ax.plot(n_unique, -z_mean, 'k-', linewidth=2, label='Bathymétrie moyenne') 

# #     # Tracer le niveau d'eau (ligne bleue)
# #     wl = mesh.water_level
# #     ax.axhline(y=-wl, color='b', linestyle='-', linewidth=2, label='Niveau d\'eau')

# #     # Tracer les bords du maillage (nw, zb_all)
# #     # Bord gauche et droit du maillage
# #     ax.plot(mesh.nb_all, -mesh.zb_all, 'k-', linewidth=1, label='Bords du maillage')

# #     # Tracer la ligne du niveau d'eau pour nw
# #     ax.plot(mesh.nw, np.full_like(mesh.nw, -wl), 'b-', linewidth=1)

# #     # Tracer les cellules du maillage (patchs)
# #     if hasattr(mesh, 'n_patch') and hasattr(mesh, 'z_patch'):
# #         # Tracer les patchs (remplissage des cellules)
# #         for i in range(mesh.n_patch.shape[1]):
# #             n_patch_i = mesh.n_patch[:, i]
# #             z_patch_i = mesh.z_patch[:, i]
# #             # Filtrer les NaN
# #             valid_patch = np.isfinite(n_patch_i) & np.isfinite(z_patch_i)
# #             if np.sum(valid_patch) >= 2:  # Au moins 2 points valides pour tracer
# #                 ax.fill(
# #                     n_patch_i[valid_patch],
# #                     -z_patch_i[valid_patch],
# #                     'g-',
# #                     alpha=0.2,
# #                     edgecolor='k',
# #                     linewidth=0.5
# #                 )

# #     # Configuration du graphique
# #     ax.set_xlabel('Distance le long de la section (n) [m]')
# #     ax.set_ylabel('Profondeur (m)')
# #     ax.set_title('Profil 2D : Bathymétrie et maillage')
# #     ax.legend()
# #     ax.grid(True, linestyle='--', alpha=0.7)
# #     ax.invert_yaxis()  # Profondeur vers le bas (comme en MATLAB)

# #     # Inverser l'axe x si nécessaire (comme en MATLAB)
# #     # ax.invert_xaxis()

# #     # Vérification de l'orientation de la section transversale (rive droite / rive gauche)

# #     n_min, n_max = np.min(n_points_raw), np.max(n_points_raw)
    
# #     if n_start_rg < n_end_rd:
# #         print(" Repère XSection aligné avec RG à gauche.")
# #         ax.set_xlim(left=n_min - 2, right=n_max + 2)
# #     else:
# #         print("Inversion du repère graphique pour mettre la Rive Gauche à gauche.")
# #         ax.set_xlim(left=n_max + 2, right=n_min - 2)
        
# #     # Ajouter les labels RG/RD
# #     ax.text(0.02, 0.95, 'RG', transform=ax.transAxes, 
# #         ha='left', va='top', fontweight='bold', fontsize=8, 
# #         bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='gray', alpha=0.9))
            
# #     ax.text(0.98, 0.95, 'RD', transform=ax.transAxes, 
# #         ha='right', va='top', fontweight='bold', fontsize=8, 
# #         bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='gray', alpha=0.9))

# #     # Affichage des valeurs des points au survol 
# #     coord_box = ax.annotate(
# #         'n = -- m\nz = -- m',
# #         xy=(1.0, 0.0),                 
# #         xycoords='axes fraction',
# #         xytext=(0, -12),                 
# #         textcoords='offset points',
# #         ha='right', va='top',
# #         fontsize=8.5, fontweight='normal', color='#333333',
# #         bbox=dict(
# #             boxstyle='round,pad=0.25', 
# #             facecolor='#ffffd1',         
# #             edgecolor='#cccccc',         
# #             alpha=0.9
# #         ),
# #         zorder=10
# #     )

# #     def onclick(event):
# #         if event.inaxes == ax:
# #             n_click = event.xdata
# #             z_click = event.ydata
# #             coord_box.set_text(f"n = {n_click:.2f} m\nz = {z_click:.2f} m")
# #             fig.canvas.draw_idle()  

# #     fig.canvas.mpl_connect('button_press_event', onclick)

# #     plt.tight_layout()
# #     plt.show(block=False)
# #     plt.pause(0.1)


# def plot_mesh_2d(mesh, bathy=None, xs=None, n_points_raw=None, n_start_rg=None, n_end_rd=None):
#     fig, ax = plt.subplots(figsize=(12, 6))
#     wl = mesh.water_level

#     # =========================================================================
#     # 1. BATHYMÉTRIE MOYENNE (LIGNE NOIRE)
#     # =========================================================================
#     if bathy is not None and xs is not None:
#         known_x = bathy.known[0, :]
#         known_y = bathy.known[1, :]
#         known_z = bathy.known[2, :]

#         valid_mask = np.isfinite(known_x) & np.isfinite(known_y) & np.isfinite(known_z)
#         known_x, known_y, known_z = known_x[valid_mask], known_y[valid_mask], known_z[valid_mask]

#         _, n_known = xs.xy2sn(known_x, known_y)
#         valid_mask = np.isfinite(n_known)
#         n_known, known_z = n_known[valid_mask], known_z[valid_mask]

#         if len(n_known) > 0:
#             n_unique = np.linspace(np.min(n_known), np.max(n_known), 200)
#             z_mean = np.zeros_like(n_unique)

#             for i, n_val in enumerate(n_unique):
#                 mask = np.abs(n_known - n_val) < 0.5
#                 if np.any(mask):
#                     z_mean[i] = np.nanmean(known_z[mask])
#                 else:
#                     z_mean[i] = np.nan

#             z_mean = pd.Series(z_mean).interpolate().bfill().ffill().values
#             ax.plot(n_unique, z_mean, 'k-', linewidth=2.0, label='Bathymétrie moyenne', zorder=4)

#     # =========================================================================
#     # 2. CONSTRUCTION ET TRACÉ DU MAILLAGE (POLYCOLLECTION)
#     # =========================================================================
#     if hasattr(mesh, 'col_to_cell') and len(mesh.col_to_cell) > 0:
#         polygons = []
#         for i in range(len(mesh.col_to_cell)):
#             c = mesh.col_to_cell[i]
            
#             nl = mesh.n_left[c]
#             nr = mesh.n_right[c]
            
#             z_bot_l = mesh.z_bottom_left[i]
#             z_top_l = mesh.z_top_left[i]
#             z_bot_r = mesh.z_bottom_right[i]
#             z_top_r = mesh.z_top_right[i]
            
#             # Contour rectangulaire/trapézoïdal de la cellule
#             poly = [
#                 (nl, z_bot_l),
#                 (nl, z_top_l),
#                 (nr, z_top_r),
#                 (nr, z_bot_r)
#             ]
#             polygons.append(poly)

#         coll = PolyCollection(
#             polygons,
#             facecolors='none',      # Fond transparent (plus de pavé jaune)
#             edgecolors='#444444',   # Grille fine sombre
#             linewidths=0.4,
#             zorder=2
#         )
#         ax.add_collection(coll)

#     # =========================================================================
#     # 3. NIVEAU D'EAU
#     # =========================================================================
#     if hasattr(mesh, 'nw') and mesh.nw.size >= 2:
#         ax.plot(mesh.nw.reshape(-1), np.full_like(mesh.nw.reshape(-1), wl), 'b-', linewidth=2.5, label='Niveau d\'eau', zorder=5)
#     else:
#         ax.axhline(y=wl, color='b', linestyle='-', linewidth=2.5, label='Niveau d\'eau', zorder=5)

#     # =========================================================================
#     # 4. CONFIGURATION DE L'AXE & ORIENTATION RG / RD
#     # =========================================================================
#     ax.set_xlabel('Distance le long de la section (n) [m]')
#     ax.set_ylabel('Élévation / Profondeur [m]')
#     ax.grid(True, linestyle=':', alpha=0.6)
    
#     # Inversion Y classique pour garder la surface de l'eau vers le haut
#     ax.invert_yaxis()

#     # Orientation RG / RD (Inversion dynamique de l'axe X)
#     if n_points_raw is not None and n_start_rg is not None and n_end_rd is not None:
#         n_min, n_max = np.min(n_points_raw), np.max(n_points_raw)
#         if n_start_rg > n_end_rd:
#             ax.set_xlim(left=n_min - 0.5, right=n_max + 0.5)
#         else:
#             ax.set_xlim(left=n_max + 0.5, right=n_min - 0.5)

#     # Labels RG / RD
#     ax.text(0.02, 0.95, 'RG', transform=ax.transAxes, ha='left', va='top', 
#             fontweight='bold', fontsize=10, bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='gray'))
#     ax.text(0.98, 0.95, 'RD', transform=ax.transAxes, ha='right', va='top', 
#             fontweight='bold', fontsize=10, bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='gray'))

#     # =========================================================================
#     # 5. BOÎTE D'AFFICHAGE DYNAMIQUE (N, Z, X_UTM, Y_UTM) AU CLIC
#     # =========================================================================
#     coord_box = ax.annotate(
#         'Cliquez sur le graphique\nn = -- m | z = -- m\nX = -- | Y = --',
#         xy=(0.5, 0.02),                 
#         xycoords='axes fraction',
#         ha='center', va='bottom',
#         fontsize=9, fontweight='normal', color='#222222',
#         bbox=dict(boxstyle='round,pad=0.4', facecolor='#ffffd1', edgecolor='#cccccc', alpha=0.95),
#         zorder=10
#     )

#     def onclick(event):
#         if event.inaxes == ax and event.xdata is not None and event.ydata is not None:
#             n_click = event.xdata
#             z_click = event.ydata
            
#             # Calcul des coordonnées UTM si xs est disponible
#             if xs is not None:
#                 try:
#                     x_utm, y_utm = xs.sn2xy(np.array([0.0]), np.array([n_click]))
#                     x_str = f"{x_utm[0]:.2f}"
#                     y_str = f"{y_utm[0]:.2f}"
#                 except Exception:
#                     x_str, y_str = "--", "--"
#             else:
#                 x_str, y_str = "--", "--"

#             coord_box.set_text(f"n = {n_click:.2f} m | z = {z_click:.2f} m\nX UTM = {x_str} m | Y UTM = {y_str} m")
#             fig.canvas.draw_idle()

#     fig.canvas.mpl_connect('button_press_event', onclick)

#     ax.legend(loc='lower right')
#     plt.tight_layout()
#     plt.show(block=False)