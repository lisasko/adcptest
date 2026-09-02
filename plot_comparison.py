# -*- coding: utf-8 -*-
"""
plot_comparison.py
==================
Comparaison graphique des résultats MAP_streamwise / MAP_roz (méthode classique)
vs MAP_vermeulen (nouvelle méthode Location-Based).

Figures produites :
  - Comparaison streamwise (classique vs Vermeulen) sur le même axe de couleur
  - Différence streamwise (Vermeulen - Classique)
  - Comparaison nb_vel (classique vs Vermeulen)
  - Comparaison r2 (Vermeulen uniquement)
  - Comparaison r_sig (Vermeulen uniquement)

Usage dans run_MAP.py :
    from plot_comparison import plot_all_comparisons
    plot_all_comparisons(average_profile, average_profile_verm, name_meas)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import scipy as sc
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from math import isnan
import os


# ============================================================
# Utilitaires internes
# ============================================================

def _jet(x):
    """Colormap jet : scalaire [0,1] → [R, G, B]."""
    if not isinstance(x, float) or isnan(x):
        return [1, 1, 1]
    if x < 0:
        return [0, 0, 0.5]
    elif x < 0.125:
        return [0, 0, 0.5 + 4 * x]
    elif x < 0.375:
        return [0, 4 * (x - 0.125), 1]
    elif x < 0.625:
        return [4 * (x - 0.375), 1, -4 * (x - 0.625)]
    elif x < 0.875:
        return [1, -4 * (x - 0.875), 0]
    elif x < 1:
        return [-4 * (x - 1.125), 0, 0]
    else:
        return [0.5, 0, 0]


def _build_patches(profile, flexible_mesh=False):
    """Construit les polygones matplotlib depuis un objet MAP_*.

    Returns
    -------
    patches : list[Polygon]
    vect_st : list[float]   vitesse streamwise par cellule
    vect_tr : list[float]   vitesse transverse par cellule
    vect_w  : list[float]   vitesse verticale par cellule
    mid_dist : np.ndarray   centres horizontaux des verticales
    depths_plt : list[float] profondeur par verticale
    x_fill, depth_fill : arrays pour masquer sous le fond
    """
    borders  = profile.borders_ens
    dcb      = profile.depth_cells_border
    vel_st   = profile.streamwise_velocity
    vel_tr   = profile.transverse_velocity
    vel_w    = profile.vertical_velocity

    n_vert, n_hor = vel_st.shape
    mid_dist = (borders[1:] + borders[:-1]) / 2

    patches   = []
    vect_st   = []
    vect_tr   = []
    vect_w    = []

    for j in range(n_hor):
        for i in range(n_vert):
            if np.isnan(dcb[i + 1, j]):
                continue
            x0, x1 = borders[j], borders[j + 1]
            y0_l, y1_l = dcb[i, j], dcb[i + 1, j]
            if flexible_mesh and j < n_hor - 1:
                y0_r, y1_r = dcb[i, j + 1], dcb[i + 1, j + 1]
            else:
                y0_r, y1_r = y0_l, y1_l

            if any(np.isnan([x0, x1, y0_l, y1_l, y0_r, y1_r])):
                continue
            patches.append(Polygon([[x0, y0_l], [x1, y0_r], [x1, y1_r], [x0, y1_l]], closed=True))
            vect_st.append(vel_st[i, j])
            vect_tr.append(vel_tr[i, j] if vel_tr is not None else np.nan)
            vect_w.append(vel_w[i, j]   if vel_w  is not None else np.nan)

    depths_plt = [profile.depths[j] if not np.isnan(profile.depths[j]) else 0
                  for j in range(n_hor)]
    x_fill     = np.concatenate(([mid_dist[0] - 1], mid_dist, [mid_dist[-1] + 1]))
    depth_fill = np.concatenate(([0], depths_plt, [0]))

    return patches, vect_st, vect_tr, vect_w, mid_dist, depths_plt, x_fill, depth_fill


def _vmin_vmax(vect):
    """Calcule vmin/vmax robuste avec z-score (même logique que MAP_streamwise)."""
    v0 = [0 if (v is None or isnan(float(v))) else float(v) for v in vect]
    if not v0:
        return 0.0, 1.0
    zs   = np.abs(sc.stats.zscore(v0))
    vmax = vmin = float(np.mean(v0))
    for i, v in enumerate(vect):
        if v is None or isnan(float(v)):
            continue
        v = float(v)
        if zs[i] < 3:
            if v > vmax:
                vmax = v
            elif v < vmin:
                vmin = v
    sign_min = np.sign(vmin) if vmin != 0 else 1
    sign_max = np.sign(vmax) if vmax != 0 else 1
    return vmin * (1 - 0.1 * sign_min), vmax * (1 + 0.1 * sign_max)


def _draw_section(ax, patches, vect, vmin, vmax,
                  mid_dist, depths_plt, x_fill, depth_fill,
                  title, cbar_label, fig,
                  quiver_x=None, quiver_y=None, quiver_scale=None):
    """Dessine une section colorée sur ax."""
    norm = [(float(v) - vmin) / (vmax - vmin) if (v is not None and not isnan(float(v))) else np.nan
            for v in vect]
    colors = [_jet(float(n)) if (n is not None and not isnan(float(n))) else [1,1,1] for n in norm]

    pc = PatchCollection(patches, facecolors='white')
    pc.set_color(colors)
    ax.add_collection(pc)
    ax.autoscale()
    ax.invert_yaxis()

    max_depth = max(d for d in depths_plt if d > 0) if any(d > 0 for d in depths_plt) else 1
    ax.fill_between(x_fill, max_depth + 2, depth_fill, color='w')
    ax.plot(mid_dist, depths_plt, color='k', linewidth=1.5)
    ax.plot(mid_dist, [0]*len(depths_plt), color='b', linewidth=2)

    if quiver_x is not None and quiver_scale is not None and quiver_scale > 0:
        n_patches = len(patches)
        if len(quiver_x) == n_patches and len(quiver_y) == n_patches:
            mid_x_q = [np.mean([v[0] for v in p.get_xy()]) for p in patches]
            mid_y_q = [np.mean([v[1] for v in p.get_xy()]) for p in patches]
            ax.quiver(mid_x_q, mid_y_q, quiver_x, quiver_y,
                      units='xy', scale=quiver_scale,
                      width=0.015 * max_depth, color='k')

    # colorbar via scatter fantôme
    X_sc = list(range(len(vect)))
    Y_sc = [v + max_depth * 3 for v in X_sc]
    sc_plot = ax.scatter(X_sc, Y_sc,
                         c=[float(v) if v is not None and not isnan(float(v)) else np.nan for v in vect],
                         cmap='jet', vmin=vmin, vmax=vmax, s=0)
    cb = fig.colorbar(sc_plot, ax=ax)
    cb.ax.set_ylabel(cbar_label, fontsize=11)

    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xlabel('Distance (m)', fontsize=11)
    ax.set_ylabel('Profondeur (m)', fontsize=11)
    ax.set_xlim(mid_dist[0] - 0.5, mid_dist[-1] + 0.5)
    ax.set_ylim(top=-0.1, bottom=max_depth * 1.05)
    ax.tick_params(axis='both', direction='in',
                   bottom=True, top=True, left=True, right=True)

def _add_bank_labels(ax, left_label='RG', right_label='RD'):
    """Ajoute 'RG' à gauche et 'RD' à droite sur un axe (coordonnées axes).

    Les labels sont placés hors de la zone de tracé via `transform=ax.transAxes`.
    """
    try:
        ax.text(0.02, -0.08, left_label,
                transform=ax.transAxes, ha='left', va='top', fontweight='bold', clip_on=False)
        ax.text(0.98, -0.08, right_label,
                transform=ax.transAxes, ha='right', va='top', fontweight='bold', clip_on=False)
    except Exception:
        # Ne doit pas interrompre la génération des figures
        pass


def _start_edge_is_right(profile):
    start_edge = getattr(profile, 'orig_start_edge', None)
    if not start_edge:
        start_edge = getattr(profile, 'start_edge', None)
    if isinstance(start_edge, (list, tuple, np.ndarray)):
        if len(start_edge) == 0:
            return False
        start_edge = start_edge[0]
    return str(start_edge).lower() == 'right'


def _apply_bank_orientation(ax, profile):
    if _start_edge_is_right(profile):
        ax.invert_xaxis()


# ============================================================
# Figures de comparaison
# ============================================================

def plot_streamwise_comparison(profile_classic, profile_verm,
                                name_meas='', save_dir='Images_ppt'):
    """Figure 2×1 : streamwise classique (gauche) vs Vermeulen (droite).

    Même échelle de couleur pour les deux panneaux.
    """
    os.makedirs(save_dir, exist_ok=True)

    (p_c, st_c, tr_c, w_c,
     md_c, dp_c, xf_c, df_c) = _build_patches(profile_classic, flexible_mesh=False)
    (p_v, st_v, tr_v, w_v,
     md_v, dp_v, xf_v, df_v) = _build_patches(profile_verm, flexible_mesh=True)

    # Échelle commune
    vmin, vmax = _vmin_vmax(st_c + st_v)

    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    fig.suptitle(f'Comparaison Streamwise velocity — {name_meas}',
                 fontsize=14, fontweight='bold')

    _draw_section(axes[0], p_c, st_c, vmin, vmax,
                  md_c, dp_c, xf_c, df_c,
                  'Méthode Classique', 'Vitesse streamwise (m/s)', fig,
                  quiver_x=tr_c, quiver_y=w_c,
                  quiver_scale=profile_classic.quiver_scale)

    _draw_section(axes[1], p_v, st_v, vmin, vmax,
                  md_v, dp_v, xf_v, df_v,
                  'Méthode Vermeulen', 'Vitesse streamwise (m/s)', fig,
                  quiver_x=tr_v, quiver_y=w_v,
                  quiver_scale=profile_verm.quiver_scale)

    _apply_bank_orientation(axes[0], profile_classic)
    _apply_bank_orientation(axes[1], profile_verm)

    # Bank labels (Rive Gauche / Rive Droite)
    try:
        _add_bank_labels(axes[0])
        _add_bank_labels(axes[1])
    except Exception:
        pass

    plt.tight_layout()
    fname = os.path.join(save_dir, f'Comparaison_Streamwise_{name_meas}.png')
    plt.savefig(fname, bbox_inches='tight', dpi=200)
    plt.show()
    print(f"  → Sauvegardé : {fname}")


def plot_difference(profile_classic, profile_verm,
                    name_meas='', save_dir='Images_ppt'):
    """Figure : différence Vermeulen - Classique (même maillage requis)."""
    os.makedirs(save_dir, exist_ok=True)

    # Les deux grilles doivent avoir la même forme pour la différence directe.
    # Si elles diffèrent, on interpole sur la grille Vermeulen.
    st_c = profile_classic.streamwise_velocity  # (n_vert, n_hor)
    st_v = profile_verm.streamwise_velocity

    if st_c.shape == st_v.shape:
        diff = st_v - st_c
    else:
        # Interpolation bilinéaire de la grille classique sur la grille Vermeulen
        import scipy.interpolate as sinterp
        md_c = (profile_classic.borders_ens[1:] + profile_classic.borders_ens[:-1]) / 2
        md_v = (profile_verm.borders_ens[1:] + profile_verm.borders_ens[:-1]) / 2
        n_vert_c, n_hor_c = st_c.shape
        n_vert_v, n_hor_v = st_v.shape
        dcb_c = profile_classic.depth_cells_border
        dcb_v = profile_verm.depth_cells_border

        # Centres des cellules classique
        centers_c_x = np.repeat(md_c, n_vert_c)
        centers_c_y = np.tile(
            [(dcb_c[i, :].mean() + dcb_c[i+1, :].mean()) / 2
             for i in range(n_vert_c)], n_hor_c)
        vals_c = st_c.T.flatten()

        # Centres des cellules Vermeulen (points cibles)
        centers_v_x = np.repeat(md_v, n_vert_v)
        centers_v_y = np.tile(
            [(dcb_v[i, :].mean() + dcb_v[i+1, :].mean()) / 2
             for i in range(n_vert_v)], n_hor_v)

        valid_src = np.isfinite(vals_c) & np.isfinite(centers_c_x) & np.isfinite(centers_c_y)
        st_c_interp = np.full(centers_v_x.shape, np.nan, dtype=float)
        valid_tgt = np.isfinite(centers_v_x) & np.isfinite(centers_v_y)

        if np.count_nonzero(valid_src) >= 3 and np.count_nonzero(valid_tgt) > 0:
            st_c_interp[valid_tgt] = sinterp.griddata(
                np.c_[centers_c_x[valid_src], centers_c_y[valid_src]],
                vals_c[valid_src],
                np.c_[centers_v_x[valid_tgt], centers_v_y[valid_tgt]],
                method='linear')

        diff = st_v - st_c_interp.reshape(n_hor_v, n_vert_v).T

    # Construire les patches sur la grille Vermeulen
    (p_v, _, _, _, md_v, dp_v, xf_v, df_v) = _build_patches(profile_verm, flexible_mesh=True)

    # Vecteur différence aligné sur les patches valides
    borders_v = profile_verm.borders_ens
    dcb_v     = profile_verm.depth_cells_border
    n_vert_v, n_hor_v = profile_verm.streamwise_velocity.shape
    diff_vect = []
    for j in range(n_hor_v):
        for i in range(n_vert_v):
            if np.isnan(dcb_v[i+1, j]):
                continue
            if any(np.isnan([borders_v[j], borders_v[j+1],
                             dcb_v[i,j], dcb_v[i+1,j]])):
                continue
            diff_vect.append(diff[i, j])

    # Échelle centrée sur 0
    diff_finite = [v for v in diff_vect if v is not None and np.isfinite(v)]
    if diff_finite:
        abs_max = np.percentile(np.abs(diff_finite), 95) * 1.1
    else:
        abs_max = 0.1
    vmin_d, vmax_d = -abs_max, abs_max

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.suptitle(f'Différence Streamwise (Vermeulen − Classique) — {name_meas}',
                 fontsize=13, fontweight='bold')
    _draw_section(ax, p_v, diff_vect, vmin_d, vmax_d,
                  md_v, dp_v, xf_v, df_v,
                  'Vermeulen − Classique',
                  'Différence de vitesse (m/s)', fig)

    _apply_bank_orientation(ax, profile_verm)

    # Bank labels
    try:
        _add_bank_labels(ax)
    except Exception:
        pass

    plt.tight_layout()
    fname = os.path.join(save_dir, f'Difference_Streamwise_{name_meas}.png')
    plt.savefig(fname, bbox_inches='tight', dpi=200)
    plt.show()
    print(f"  → Sauvegardé : {fname}")


def plot_nb_vel_comparison(profile_classic, profile_verm,
                            name_meas='', save_dir='Images_ppt'):
    """Figure 2×1 : nb infos classique vs Vermeulen (même échelle)."""
    os.makedirs(save_dir, exist_ok=True)

    (p_c, _, _, _, md_c, dp_c, xf_c, df_c) = _build_patches(profile_classic, flexible_mesh=False)
    (p_v, _, _, _, md_v, dp_v, xf_v, df_v) = _build_patches(profile_verm, flexible_mesh=True)

    # Nb infos classique
    n_vert_c, n_hor_c = profile_classic.streamwise_velocity.shape
    borders_c = profile_classic.borders_ens
    dcb_c     = profile_classic.depth_cells_border
    ic_c = profile_classic.info_cell
    nb_c = []
    for j in range(n_hor_c):
        for i in range(n_vert_c):
            if np.isnan(dcb_c[i+1, j]):
                continue
            if any(np.isnan([borders_c[j], borders_c[j+1], dcb_c[i,j], dcb_c[i+1,j]])):
                continue
            nb_c.append(ic_c[i, j] if ic_c is not None else np.nan)

    # Nb infos Vermeulen
    n_vert_v, n_hor_v = profile_verm.streamwise_velocity.shape
    borders_v = profile_verm.borders_ens
    dcb_v     = profile_verm.depth_cells_border
    nb_v_arr  = profile_verm.nb_vel
    nb_v = []
    for j in range(n_hor_v):
        for i in range(n_vert_v):
            if np.isnan(dcb_v[i+1, j]):
                continue
            if any(np.isnan([borders_v[j], borders_v[j+1], dcb_v[i,j], dcb_v[i+1,j]])):
                continue
            nb_v.append(nb_v_arr[i, j] if nb_v_arr is not None else np.nan)

    nb_max = max(
        np.nanmax([v for v in nb_c if v is not None and np.isfinite(v)] or [1]),
        np.nanmax([v for v in nb_v if v is not None and np.isfinite(v)] or [1]))

    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    fig.suptitle(f"Nombre d'informations par cellule — {name_meas}",
                 fontsize=14, fontweight='bold')

    _draw_section(axes[0], p_c, nb_c, 0, nb_max,
                  md_c, dp_c, xf_c, df_c,
                  'Méthode Classique', "Nb infos", fig)
    _draw_section(axes[1], p_v, nb_v, 0, nb_max,
                  md_v, dp_v, xf_v, df_v,
                  'Méthode Vermeulen (mesures radiales)', "Nb infos", fig)

    _apply_bank_orientation(axes[0], profile_classic)
    _apply_bank_orientation(axes[1], profile_verm)

    # Bank labels
    try:
        _add_bank_labels(axes[0])
        _add_bank_labels(axes[1])
    except Exception:
        pass

    plt.tight_layout()
    fname = os.path.join(save_dir, f'Comparaison_NbInfos_{name_meas}.png')
    plt.savefig(fname, bbox_inches='tight', dpi=200)
    plt.show()
    print(f"  → Sauvegardé : {fname}")


def plot_quality_vermeulen(profile_verm, name_meas='', save_dir='Images_ppt'):
    """Figure 1×2 : r² et σ résidu de la méthode Vermeulen."""
    os.makedirs(save_dir, exist_ok=True)

    (p_v, _, _, _, md_v, dp_v, xf_v, df_v) = _build_patches(profile_verm, flexible_mesh=True)

    n_vert_v, n_hor_v = profile_verm.streamwise_velocity.shape
    borders_v = profile_verm.borders_ens
    dcb_v     = profile_verm.depth_cells_border

    r2_vect   = []
    rsig_vect = []
    for j in range(n_hor_v):
        for i in range(n_vert_v):
            if np.isnan(dcb_v[i+1, j]):
                continue
            if any(np.isnan([borders_v[j], borders_v[j+1], dcb_v[i,j], dcb_v[i+1,j]])):
                continue
            r2_vect.append(
                profile_verm.r2[i, j] if profile_verm.r2 is not None else np.nan)
            rsig_vect.append(
                profile_verm.r_sig[i, j] if profile_verm.r_sig is not None else np.nan)

    r2_max   = np.nanpercentile([v for v in r2_vect   if np.isfinite(v)] or [1], 95)
    rsig_max = np.nanpercentile([v for v in rsig_vect if np.isfinite(v)] or [1], 95)

    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    fig.suptitle(f'Qualité de la résolution Vermeulen — {name_meas}',
                 fontsize=14, fontweight='bold')

    _draw_section(axes[0], p_v, r2_vect, 0, r2_max,
                  md_v, dp_v, xf_v, df_v,
                  'Erreur quadratique R²', 'R² (m²/s²)', fig)
    _draw_section(axes[1], p_v, rsig_vect, 0, rsig_max,
                  md_v, dp_v, xf_v, df_v,
                  'Écart-type résidu σ', 'σ (m/s)', fig)

    _apply_bank_orientation(axes[0], profile_verm)
    _apply_bank_orientation(axes[1], profile_verm)

    # Bank labels
    try:
        _add_bank_labels(axes[0])
        _add_bank_labels(axes[1])
    except Exception:
        pass

    plt.tight_layout()
    fname = os.path.join(save_dir, f'Qualite_Vermeulen_{name_meas}.png')
    plt.savefig(fname, bbox_inches='tight', dpi=200)
    plt.show()
    print(f"  → Sauvegardé : {fname}")


def plot_vertical_profiles(profile_classic, profile_verm,
                            name_meas='', save_dir='Images_ppt',
                            n_profiles=5):
    """Profils verticaux de vitesse streamwise : classique vs Vermeulen.

    Compare n_profiles verticales réparties uniformément sur la section.
    """
    os.makedirs(save_dir, exist_ok=True)

    st_c  = profile_classic.streamwise_velocity
    st_v  = profile_verm.streamwise_velocity
    dcb_c = profile_classic.depth_cells_border
    dcb_v = profile_verm.depth_cells_border
    md_c  = (profile_classic.borders_ens[1:] + profile_classic.borders_ens[:-1]) / 2
    md_v  = (profile_verm.borders_ens[1:]    + profile_verm.borders_ens[:-1])    / 2

    # Sélection de n_profiles verticales sur la grille classique
    idx_c = np.linspace(0, len(md_c) - 1, n_profiles).astype(int)

    fig, axes = plt.subplots(1, n_profiles, figsize=(4 * n_profiles, 6),
                              sharey=True)
    fig.suptitle(f'Profils verticaux streamwise — {name_meas}',
                 fontsize=14, fontweight='bold')

    for k, ic in enumerate(idx_c):
        ax = axes[k]
        x_pos_c = md_c[ic]

        # Profil classique
        dcc_c = (dcb_c[1:, ic] + dcb_c[:-1, ic]) / 2
        vel_c = st_c[:, ic]
        valid_c = np.isfinite(vel_c) & np.isfinite(dcc_c)
        ax.plot(vel_c[valid_c], dcc_c[valid_c],
                'b-o', markersize=4, linewidth=1.5, label='Classique')

        # Profil Vermeulen au point le plus proche
        iv = np.argmin(np.abs(md_v - x_pos_c))
        dcc_v = (dcb_v[1:, iv] + dcb_v[:-1, iv]) / 2
        vel_v = st_v[:, iv]
        valid_v = np.isfinite(vel_v) & np.isfinite(dcc_v)
        ax.plot(vel_v[valid_v], dcc_v[valid_v],
                'r--s', markersize=4, linewidth=1.5, label='Vermeulen')

        ax.invert_yaxis()
        ax.set_xlabel('V streamwise (m/s)', fontsize=10)
        ax.set_title(f'x ≈ {x_pos_c:.1f} m', fontsize=10)
        ax.axvline(0, color='k', linewidth=0.8, linestyle=':')
        ax.grid(True, alpha=0.3)
        if k == 0:
            ax.set_ylabel('Profondeur (m)', fontsize=10)
            ax.legend(fontsize=9)

    plt.tight_layout()
    fname = os.path.join(save_dir, f'Profils_Verticaux_{name_meas}.png')
    plt.savefig(fname, bbox_inches='tight', dpi=200)
    plt.show()
    print(f"  → Sauvegardé : {fname}")


# ============================================================
# Fonction principale
# ============================================================

def plot_all_comparisons(profile_classic, profile_verm,
                          name_meas='', save_dir='Images_ppt',
                          profile_roz=None, profile_roz_verm=None):
    """Lance toutes les comparaisons graphiques.

    Parameters
    ----------
    profile_classic : MAP_streamwise
        Résultats de la méthode classique (streamwise).
    profile_verm : MAP_vermeulen
        Résultats de la méthode Vermeulen.
    name_meas : str
        Nom de la mesure (pour les titres et noms de fichiers).
    save_dir : str
        Dossier de sauvegarde des figures.
    profile_roz : MAP_roz, optional
        Résultats Rozovskii classique (si disponible).
    profile_roz_verm : MAP_roz, optional
        Résultats Rozovskii Vermeulen (si disponible, futur).
    """
    print("\n=== Génération des figures de comparaison ===")
    os.makedirs(save_dir, exist_ok=True)

    print("1/4 Comparaison streamwise (classique vs Vermeulen)...")
    plot_streamwise_comparison(profile_classic, profile_verm,
                                name_meas, save_dir)

    print("2/4 Différence streamwise...")
    plot_difference(profile_classic, profile_verm, name_meas, save_dir)

    print("3/4 Nombre d'informations par cellule...")
    plot_nb_vel_comparison(profile_classic, profile_verm, name_meas, save_dir)

    print("4/4 Qualité Vermeulen (R², σ)...")
    plot_quality_vermeulen(profile_verm, name_meas, save_dir)

    print("5/5 Profils verticaux streamwise...")
    plot_vertical_profiles(profile_classic, profile_verm, name_meas, save_dir)

    print(f"\nToutes les figures ont été sauvegardées dans '{save_dir}/'")