# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from PyQt5 import QtWidgets


FILL_OPTIONS = {
    "streamwise": "Vitesse streamwise (axe fixe, section moyenne)",
    "primaire": "Vitesse primaire (Rozovskii)",
}
ARROW_OPTIONS = {
    "transverse": "Courant transverse ",
    "secondaire": "Courant secondaire (Rozovskii)",
}


def select_velocity_view_mode(parent=None):

    app = QtWidgets.QApplication.instance()
    owns_app = app is None
    if owns_app:
        app = QtWidgets.QApplication([])

    dialog = QtWidgets.QDialog(parent)
    dialog.setWindowTitle("Choix du mode d'affichage des vitesses")
    dialog.setMinimumWidth(420)
    layout = QtWidgets.QVBoxLayout(dialog)

    layout.addWidget(QtWidgets.QLabel("Champ principal :"))
    fill_combo = QtWidgets.QComboBox()
    for key, label in FILL_OPTIONS.items():
        fill_combo.addItem(label, userData=key)
    layout.addWidget(fill_combo)

    layout.addWidget(QtWidgets.QLabel("Courant secondaire :"))
    arrow_combo = QtWidgets.QComboBox()
    for key, label in ARROW_OPTIONS.items():
        arrow_combo.addItem(label, userData=key)
    layout.addWidget(arrow_combo)

    btn_row = QtWidgets.QHBoxLayout()
    ok_btn = QtWidgets.QPushButton("OK")
    cancel_btn = QtWidgets.QPushButton("Annuler")
    ok_btn.clicked.connect(dialog.accept)
    cancel_btn.clicked.connect(dialog.reject)
    btn_row.addStretch()
    btn_row.addWidget(cancel_btn)
    btn_row.addWidget(ok_btn)
    layout.addLayout(btn_row)

    result = dialog.exec_()
    if owns_app:
        app.quit()

    if result != QtWidgets.QDialog.Accepted:
        return None, None

    return fill_combo.currentData(), arrow_combo.currentData()


def plot_velocity_view(mesh, fields, fill_mode, arrow_mode, vmadcp=None, clim=None, ax=None,
                       cmap="jet"):

    from Classes_vermeulen.plot_mesh_bathy import _rg_left_orientation

    required = {
        "streamwise": ["streamwise"],
        "primaire": ["primaire"],
    }[fill_mode]
    for key in required:
        if key not in fields:
            raise KeyError(f"Champ manquant '{key}' pour fill_mode='{fill_mode}' -- vérifie le dict `fields`.")

    if fill_mode == "streamwise":
        values = np.asarray(fields["streamwise"], dtype=float)
        fill_label = "Vitesse streamwise (m/s)"
    else:
        values = np.asarray(fields["primaire"], dtype=float)
        fill_label = "Vitesse primaire -- Rozovskii (m/s)"

    if arrow_mode == "transverse":
        arrow_n = np.asarray(fields["transverse"], dtype=float)
        arrow_z = np.asarray(fields["vertical"], dtype=float)
        arrow_label = "Courant transverse (axe fixe)"
    else:
        arrow_n = np.asarray(fields["secondaire_n"], dtype=float)
        arrow_z = np.asarray(fields["secondaire_z"], dtype=float)
        arrow_label = "Courant secondaire (Rozovskii)"

    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))
    else:
        fig = ax.figure

    if mesh.nb_all.size > 0:
        ax.plot(mesh.nb_all, mesh.zb_all, "k-", linewidth=2.0, label="Lit (maillage)", zorder=4)

    if mesh.ncells > 0:
        polygons = [np.c_[mesh.n_patch[:, cc], mesh.z_patch[:, cc]] for cc in range(mesh.ncells)]
        coll = PolyCollection(polygons, cmap=cmap, edgecolors="#444444", linewidths=0.3, zorder=2)
        coll.set_array(values)
        if clim is not None:
            coll.set_clim(*clim)
        else:
            finite_vals = values[np.isfinite(values)]
            if finite_vals.size > 0:
                coll.set_clim(*np.nanpercentile(finite_vals, [5, 95]))
        ax.add_collection(coll)
        cb = fig.colorbar(coll, ax=ax)
        cb.set_label(fill_label)

        # Flèches des courants secondaires
        q_norm = np.sqrt(arrow_n**2 + arrow_z**2)
        q_norm[np.isnan(q_norm)] = 0.0
        mean_norm = float(np.mean(q_norm)) if q_norm.size else 0.0
        if mean_norm > 0:
            outliers = q_norm > 4.0 * mean_norm
            arrow_n = np.where(outliers, 0.0, arrow_n)
            arrow_z = np.where(outliers, 0.0, arrow_z)
            ax.quiver(mesh.n_center, mesh.z_center, arrow_n, arrow_z,
                     color="k", angles="xy", scale_units="xy", scale=1.0, zorder=5)
            x0, y0, arrow_len = 0.06, 0.04, 0.05
            ax.annotate("", xy=(x0 + arrow_len, y0), xytext=(x0, y0),
                       xycoords="axes fraction", textcoords="axes fraction",
                       arrowprops=dict(arrowstyle="-|>", color="k", lw=1.5))
            ax.text(x0 + arrow_len + 0.01, y0, "1 m/s", transform=ax.transAxes, fontsize=8, va="center")

    wl = float(mesh.water_level)
    if mesh.nw.size > 0:
        nw_flat = mesh.nw.reshape(-1)
        ax.plot(nw_flat, np.full_like(nw_flat, wl), "b-", linewidth=2.5, label="Niveau d'eau", zorder=6)

    ax.set_xlabel("Distance le long de la section (n) [m]")
    ax.set_ylabel("Élévation [m]")
    ax.set_title(f"{fill_label.split(' (')[0]} + {arrow_label}")
    ax.grid(True, linestyle=":", alpha=0.6)

    if vmadcp is not None and hasattr(mesh, "xs") and mesh.nb_all.size > 0:
        start_is_left = _rg_left_orientation(mesh.xs, vmadcp)
        n_min, n_max = float(np.min(mesh.nb_all)), float(np.max(mesh.nb_all))
        margin = 0.05 * max(n_max - n_min, 1e-6)
        if start_is_left:
            ax.set_xlim(left=n_min - margin, right=n_max + margin)
        else:
            ax.set_xlim(left=n_max + margin, right=n_min - margin)
        ax.text(0.02, 0.95, "RG", transform=ax.transAxes, ha="left", va="top", fontweight="bold",
               bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="gray"))
        ax.text(0.98, 0.95, "RD", transform=ax.transAxes, ha="right", va="top", fontweight="bold",
               bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="gray"))

    ax.legend(loc="lower right")
    return ax