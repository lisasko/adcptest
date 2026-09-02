import numpy as np
from PyQt5 import QtWidgets, QtCore
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import requests
from io import BytesIO
from PIL import Image

from Classes_vermeulen.VMADCP import VMADCP
from get_utm_zone import get_utm_zone


def _compute_zone_meas(meas):
    lons = []
    for t in meas.transects:
        if hasattr(t, "gps") and t.gps is not None:
            lon = np.asarray(t.gps.gga_lon_ens_deg, dtype=float)
            lons.append(lon[np.isfinite(lon)])
    if lons:
        return get_utm_zone(float(np.nanmean(np.concatenate(lons))))
    return 31



class TransectSelectorDialog(QtWidgets.QDialog):

    def __init__(self, meas, zone_meas=None, title="Sélection des transects", parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.resize(1000, 650)

        self.meas = meas
        self.n_transects = len(meas.transects)
        self.zone_meas = zone_meas if zone_meas is not None else _compute_zone_meas(meas)
        self.labels = [f"Transect {i} ({getattr(t, 'file_name', '?')})" for i, t in enumerate(meas.transects)]
        self.tracks = self._extract_tracks()
        self.selected = [True] * self.n_transects
        self.cancelled = False

        self._build_ui()
        self._draw_tracks()

    def _extract_tracks(self):
        tracks = []
        for t in self.meas.transects:
            try:
                x, y = VMADCP._get_track(t, nav_ref="bt_vel", zone_meas=self.zone_meas)
            except Exception as e:
                print(f"ATTENTION : trajectoire indisponible pour un transect ({e})")
                x, y = np.array([]), np.array([])
            tracks.append((np.asarray(x, dtype=float), np.asarray(y, dtype=float)))
        return tracks

    def _build_ui(self):
        layout = QtWidgets.QVBoxLayout(self)
        content = QtWidgets.QHBoxLayout()
        layout.addLayout(content)

        # Graphique visualisation des transects
        self.fig = Figure(figsize=(6, 6))
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvas(self.fig)
        content.addWidget(self.canvas, stretch=3)

        # Liste à cocher
        right_panel = QtWidgets.QVBoxLayout()
        content.addLayout(right_panel, stretch=1)

        right_panel.addWidget(QtWidgets.QLabel(f"Transects disponibles ({self.n_transects}) :"))
        self.list_widget = QtWidgets.QListWidget()
        for label in self.labels:
            item = QtWidgets.QListWidgetItem(label)
            item.setFlags(item.flags() | QtCore.Qt.ItemIsUserCheckable)
            item.setCheckState(QtCore.Qt.Checked)
            self.list_widget.addItem(item)
        self.list_widget.itemChanged.connect(self._on_item_changed)
        right_panel.addWidget(self.list_widget)

        btn_row = QtWidgets.QHBoxLayout()
        btn_all = QtWidgets.QPushButton("Tout sélectionner")
        btn_none = QtWidgets.QPushButton("Tout désélectionner")
        btn_all.clicked.connect(self._select_all)
        btn_none.clicked.connect(self._select_none)
        btn_row.addWidget(btn_all)
        btn_row.addWidget(btn_none)
        right_panel.addLayout(btn_row)

        # Bouton de validation
        bottom_row = QtWidgets.QHBoxLayout()
        btn_ok = QtWidgets.QPushButton("OK")
        btn_cancel = QtWidgets.QPushButton("Annuler")
        btn_ok.clicked.connect(self.accept)
        btn_cancel.clicked.connect(self._on_cancel)
        bottom_row.addStretch()
        bottom_row.addWidget(btn_ok)
        bottom_row.addWidget(btn_cancel)
        layout.addLayout(bottom_row)



    def _draw_tracks(self):
        self.ax.clear()
        self.ax.set_title("Trajectoires des transects")
        self.ax.set_xlabel("Easting (m)")
        self.ax.set_ylabel("Northing (m)")
        self.ax.set_aspect("equal", adjustable="datalim")

        self.ax.grid(True, linestyle=":", alpha=0.5)

        ## 02/09
        def _utm_epsg_from_zone(zone, northern=True):
            return (32600 if northern else 32700) + int(zone)
        
        all_x = np.concatenate([x for x, y in self.tracks if x.size > 0]) if self.tracks else np.array([])
        all_y = np.concatenate([y for x, y in self.tracks if y.size > 0]) if self.tracks else np.array([])

        if all_x.size > 0 and all_y.size > 0:
            xmin, xmax = float(np.min(all_x)), float(np.max(all_x))
            ymin, ymax = float(np.min(all_y)), float(np.max(all_y))
            margin_x = 0.15 * max(xmax - xmin, 1.0)
            margin_y = 0.15 * max(ymax - ymin, 1.0)
            bbox = (xmin - margin_x, xmax + margin_x, ymin - margin_y, ymax + margin_y)

            utm_epsg = _utm_epsg_from_zone(self.zone_meas, northern=True)
            basemap = get_basemap(bbox, crs_box=utm_epsg, crs_map=utm_epsg)

            if basemap is not None and basemap["data"] is not None:
                bx0, by0, bx1, by1 = basemap["extent"]
                self.ax.imshow(
                    basemap["data"], extent=(bx0, bx1, by0, by1),
                    origin="upper", zorder=0, interpolation="bilinear",
                )
        ##

        # self.colors_cycle = self.fig.gca()._get_lines.prop_cycler

        self.lines = []
        for i, (x, y) in enumerate(self.tracks):
            (line,) = self.ax.plot(x, y, "-", linewidth=2.0, label=self.labels[i])
            self.lines.append(line)
        self.base_colors = [line.get_color() for line in self.lines]
        self.canvas.draw_idle()

    def _refresh_line_colors(self):
        for i, line in enumerate(self.lines):
            if self.selected[i]:
                line.set_color(self.base_colors[i])
                line.set_linewidth(2.0)
                line.set_alpha(1.0)
            else:
                line.set_color("red")
                line.set_linewidth(1.0)
                line.set_alpha(0.35)
        self.canvas.draw_idle()

    def _on_item_changed(self, item):
        idx = self.list_widget.row(item)
        self.selected[idx] = (item.checkState() == QtCore.Qt.Checked)
        self._refresh_line_colors()

    def _select_all(self):
        for i in range(self.list_widget.count()):
            self.list_widget.item(i).setCheckState(QtCore.Qt.Checked)

    def _select_none(self):
        for i in range(self.list_widget.count()):
            self.list_widget.item(i).setCheckState(QtCore.Qt.Unchecked)

    def _on_cancel(self):
        self.cancelled = True
        self.reject()

    def get_selected_indices(self):
        return [i for i in range(self.n_transects) if self.selected[i]]



def select_transects_interactive(meas, zone_meas=None, title="Sélection des transects"):

    app = QtWidgets.QApplication.instance()
    owns_app = app is None
    if owns_app:
        app = QtWidgets.QApplication([])

    dialog = TransectSelectorDialog(meas, zone_meas=zone_meas, title=title)
    result = dialog.exec_()

    if dialog.cancelled or result == QtWidgets.QDialog.Rejected:
        print("Sélection annulée -- tous les transects sont conservés par défaut.")
        kept_indices = list(range(len(meas.transects)))
    else:
        kept_indices = dialog.get_selected_indices()
        if len(kept_indices) == 0:
            raise ValueError("Aucun transect sélectionné -- au moins un transect est requis.")
        print(f"Transects retenus : {kept_indices} / {len(meas.transects)}")

    if owns_app:
        app.quit()

    return kept_indices

def _fetch_arcgis_export_image(base_url, bbox, crs_box, crs_map, timeout, verify, proxies):

    xmin, xmax, ymin, ymax = bbox
    params = {
        "bbox": f"{xmin},{ymin},{xmax},{ymax}",
        "bboxSR": crs_box,
        "imageSR": crs_map,
        "format": "png",
        "transparent": "true",
        "f": "json",
    }
    resp = requests.get(base_url, params=params, timeout=timeout, verify=verify, proxies=proxies)
    resp.raise_for_status()
    meta = resp.json()

    if "href" not in meta:
        raise ValueError(f"Pas de champ 'href' dans la réponse : {meta}")

    img_resp = requests.get(meta["href"], timeout=timeout, verify=verify, proxies=proxies)
    img_resp.raise_for_status()
    img = Image.open(BytesIO(img_resp.content)).convert("RGBA")
    data = np.asarray(img)

    extent_raw = meta.get("extent", {})
    extent = [extent_raw.get("xmin"), extent_raw.get("ymin"),
             extent_raw.get("xmax"), extent_raw.get("ymax")]
    return {"data": data, "extent": extent, "size": (meta.get("width"), meta.get("height"))}


def get_ortho_stinger(bbox, crs_box=2154, crs_map=2154, timeout=15, verify=True, proxies=None):

    base_url = "https://recette-stinger.edf.fr/server2/rest/services/EXT/BDORTHO_France/MapServer/export"
    return _fetch_arcgis_export_image(base_url, bbox, crs_box, crs_map, timeout, verify, proxies)


def get_ortho_esri_public(bbox, crs_box=2154, crs_map=2154, timeout=15):

    base_url = "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/export"
    return _fetch_arcgis_export_image(base_url, bbox, crs_box, crs_map, timeout, verify=True, proxies=None)


def get_basemap(bbox, crs_box=2154, crs_map=2154, timeout=15, verify=True, proxies=None):

    try:
        return get_ortho_stinger(bbox, crs_box=crs_box, crs_map=crs_map,
                                 timeout=timeout, verify=verify, proxies=proxies)
    except Exception as e:
        print(f"ATTENTION : STINGER indisponible ({e}), tuilisation du fond public Esri World Imagery.")
        try:
            return get_ortho_esri_public(bbox, crs_box=crs_box, crs_map=crs_map, timeout=timeout)
        except Exception as e2:
            print(f"ATTENTION : fond de carte public également indisponible ({e2}), affichage sans fond de carte.")
            return None