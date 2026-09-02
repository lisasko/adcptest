import os
import sys
from typing import List, Tuple, Union
from PyQt5 import QtWidgets
import scipy.io as sio

current_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, os.path.join(parent_dir, 'qrevint_21_03'))
sys.path.insert(0, os.path.join(parent_dir, 'Classes_vermeulen'))

from Classes_vermeulen.OpenMeasurementDialog import OpenMeasurementDialog
from Classes.Measurement import Measurement
from transect_selector_gui import select_transects_interactive

_app = None

def get_qapplication() :
    global _app
    if _app is None : 
        _app = QtWidgets.QApplication(sys.argv)
    return _app
 

""""
    File Selection.

    Return :
        path_meas: Chemin du fichier sélectionné (str ou list).
        type_meas: Type de mesure ('TRDI', 'SonTek', ou 'QRev').
        name_meas: Nom de la station.
        checked: Booléen indiquant si seuls les transects "checked" doivent être chargés (pour TRDI).
    
"""


def select_file() -> Tuple[Union[str, List[str]], str, str, bool]:

    app = get_qapplication()

    if app is None:
        app = QtWidgets.QApplication(sys.argv)

    dialog = OpenMeasurementDialog(None)
    dialog.exec_()

    print(dialog.fullName)

    if not dialog.fullName:
        sys.exit(0)

    path_meas = dialog.fullName
    type_meas = dialog.type
    name_meas = dialog.fileName[0] if isinstance(dialog.fileName, list) else dialog.fileName
    checked = dialog.checked

    return path_meas, type_meas, name_meas, checked


"""
    Transects Selection.

    Args:
        mmt_path: Chemin vers le fichier .mmt. (TRDI)
        mat_path: Chemin vers le fichier .mat. (SonTek)

    Returns:
        List[str]: Liste des chemins des fichiers .pd0 associés. (TRDI)
        List[int]: Liste des indices de transects (0, 1, 2, ...). (SonTek)
       
"""

def get_transects_from_mmt(mmt_path: str) -> List[str]:

    mmt_dir = os.path.dirname(mmt_path)
    pd0_files = [
        os.path.join(mmt_dir, f)
        for f in os.listdir(mmt_dir)
        if f.lower().endswith('.pd0') and os.path.isfile(os.path.join(mmt_dir, f))
    ]
    pd0_files.sort()
    return pd0_files

def get_transects_from_mat(mat_path: str) -> List[int]:

    try:
        import scipy.io as sio
        mat_data = sio.loadmat(mat_path, simplify_cells=True)
        if 'transects' in mat_data:
            return list(range(len(mat_data['transects'])))
        else:
            return [0]
    except Exception:
        return [0]


## A SUPPR (plus utilisé car sélection via visualisation graphique) :
def select_transects(transects: List[Union[str, int]], type_meas: str) -> List[Union[str, int]]:

    if not transects:
        QtWidgets.QMessageBox.warning(None, "Avertissement", "Aucun transect trouvé.")
        return transects

    app = get_qapplication()
    if app is None:
        app = QtWidgets.QApplication(sys.argv)

    dialog = QtWidgets.QDialog()
    dialog.setWindowTitle("Sélection des transects")
    dialog.setMinimumWidth(400)

    layout = QtWidgets.QVBoxLayout(dialog)
    label = QtWidgets.QLabel(f"Sélectionnez les transects à traiter ({len(transects)} disponibles):")
    layout.addWidget(label)

    checkboxes = []
    scroll_area = QtWidgets.QScrollArea()
    scroll_content = QtWidgets.QWidget()
    scroll_layout = QtWidgets.QVBoxLayout(scroll_content)

    for i, transect in enumerate(transects):
        if type_meas == 'TRDI':
            transect_name = os.path.basename(transect)
        else:
            transect_name = f"Transect {i}"

        checkbox = QtWidgets.QCheckBox(transect_name)
        checkbox.setChecked(True)
        scroll_layout.addWidget(checkbox)
        checkboxes.append((checkbox, transect))

    scroll_area.setWidget(scroll_content)
    scroll_area.setWidgetResizable(True)
    layout.addWidget(scroll_area)

    button_layout = QtWidgets.QHBoxLayout()
    select_all_btn = QtWidgets.QPushButton("Tout sélectionner")
    deselect_all_btn = QtWidgets.QPushButton("Tout désélectionner")
    ok_btn = QtWidgets.QPushButton("OK")
    cancel_btn = QtWidgets.QPushButton("Annuler")

    button_layout.addWidget(select_all_btn)
    button_layout.addWidget(deselect_all_btn)
    button_layout.addStretch()
    button_layout.addWidget(cancel_btn)
    button_layout.addWidget(ok_btn)
    layout.addLayout(button_layout)

    select_all_btn.clicked.connect(lambda: [cb.setChecked(True) for cb, _ in checkboxes])
    deselect_all_btn.clicked.connect(lambda: [cb.setChecked(False) for cb, _ in checkboxes])
    ok_btn.clicked.connect(dialog.accept)
    cancel_btn.clicked.connect(dialog.reject)

    if dialog.exec_() == QtWidgets.QDialog.Accepted:
        selected_transects = [
            transect for checkbox, transect in checkboxes if checkbox.isChecked()
        ]
        if not selected_transects:
            QtWidgets.QMessageBox.warning(
                None, "Avertissement",
                "Aucun transect sélectionné. Tous les transects seront traités."
            )
            selected_transects = transects
        return selected_transects
    else:
        return transects


def select_measurement() -> Tuple[Union[str, List[str]], str, str, List[Union[str, int]]]:

    path_meas, type_meas, name_meas, checked = select_file()

    if isinstance(path_meas, list):
        path_parent = os.path.dirname(path_meas[0])
    else:
        path_parent = os.path.dirname(path_meas)
    for _ in range(3):
        path_parent = os.path.dirname(path_parent)

    os.makedirs(os.path.join(current_dir, 'qrevint_21_03'), exist_ok=True)
    with open(os.path.join(current_dir, 'qrevint_21_03', 'path_file.txt'), 'w') as f:
        f.write(path_parent)

    # if type_meas == 'TRDI': # Pour TRDI, path_meas est une liste de fichiers : [mmt_file, pd0_file1, ...]
    #     mmt_path = path_meas[0] if isinstance(path_meas, list) else path_meas
    #     transects = get_transects_from_mmt(mmt_path)
    #     selected_transects = select_transects(transects, type_meas)
    # else:  # SonTek 
    #     selected_transects = [0]

    selected_transects = []

    return path_meas, type_meas, name_meas, selected_transects



def load_measurement(
    path_meas: Union[str, List[str]],
    type_meas: str,
    selected_transects: List[Union[str, int]],
    checked: bool = False,
    interactive: bool = True,
):

    if type_meas == 'TRDI':

        mmt_path = path_meas[0] if isinstance(path_meas, list) else path_meas
        meas = Measurement(mmt_path, source='TRDI', checked=checked)

        ##
        selected_transects = []

        if selected_transects:
            selected_indices = [
                i for i, transect in enumerate(meas.transects)
                if os.path.basename(transect.file_name) in [os.path.basename(f) for f in selected_transects]
            ]
            meas.transects = [meas.transects[i] for i in selected_indices]
            meas.checked_transect_idx = list(range(len(meas.transects)))
        elif interactive:
            kept_indices = select_transects_interactive(meas, title="Sélection des transects")
            meas.transects = [meas.transects[i] for i in kept_indices]
            meas.checked_transect_idx = list(range(len(meas.transects)))

        # if interactive :
        #     kept_indices = select_transects_interactive(meas, title="Sélection des transects")
        #     meas.transects = [meas.transects[i] for i in kept_indices]
        #     meas.checked_transect_idx = list(range(len(meas.transects)))

        return meas

    else:  # SonTek 
        mat_paths = [os.path.abspath(f) for f in path_meas] if isinstance(path_meas, list) else [os.path.abspath(path_meas)]
        meas = Measurement(mat_paths, source='SonTek')

        if interactive:
            kept_indices = select_transects_interactive(meas, title="Sélection des transects")

            if len(kept_indices) < len(mat_paths):
                kept_paths = [mat_paths[i] for i in kept_indices]
                print(f"DEBUG SonTek : reconstruction de Measurement avec {len(kept_paths)}/{len(mat_paths)} fichiers .mat retenus")
                meas = Measurement(kept_paths, source='SonTek')
            else:
                print("DEBUG SonTek : tous les transects conservés, pas de reconstruction nécessaire")

            
        return meas

