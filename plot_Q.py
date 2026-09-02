# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt

def plot_discharge_summary(rows, title="Résumé des débits", figsize=(8, None)):
    """
    Returns :
        fig, ax
    """

    width, height = figsize
    if height is None:
        height = 0.5 + 0.35 * len(rows)

    fig, ax = plt.subplots(figsize=(width, height))
    ax.axis("off")

    table_data = [[r["label"], r["value"]] for r in rows]
    tbl = ax.table(
        cellText=table_data,
        colLabels=["Indicateur", "Valeur"],
        cellLoc="left",
        colLoc="left",
        loc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1, 1.6)

    # Mise en forme :
    for (row_idx, col_idx), cell in tbl.get_celld().items():
        if row_idx == 0:
            cell.set_text_props(fontweight="bold")
            cell.set_facecolor("#e0e0e0")
        cell.set_edgecolor("#bbbbbb")

    ax.set_title(title, fontsize=13, pad=15)
    fig.tight_layout()
    return fig, ax