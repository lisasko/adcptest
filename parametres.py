#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from PyQt5 import QtWidgets, QtCore

class ParamSimulationDialog(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Paramètres utilisateurs de la mesure")
        self.setMinimumWidth(450)
        
        # Configuration du layout principal
        layout = QtWidgets.QVBoxLayout(self)
        
        # Formulaire pour aligner les labels et les champs de saisie
        form_layout = QtWidgets.QFormLayout()
        
        # --- Section 1: Erreurs d'Attitude ---
        form_layout.addRow(QtWidgets.QLabel("<b>Erreurs de rotation :</b>"))
        self.spin_heading = QtWidgets.QDoubleSpinBox()
        self.spin_heading.setRange(-36000, 36000)
        self.spin_heading.setValue(0.0)
        form_layout.addRow("Erreur cap (centième de degré) :", self.spin_heading)
        
        self.spin_roll = QtWidgets.QDoubleSpinBox()
        self.spin_roll.setRange(-180.0, 180.0)
        self.spin_roll.setValue(0.0)
        form_layout.addRow("Erreur roulis (degré) :", self.spin_roll)
        
        self.spin_pitch = QtWidgets.QDoubleSpinBox()
        self.spin_pitch.setRange(-180.0, 180.0)
        self.spin_pitch.setValue(0.0)
        form_layout.addRow("Erreur tangage (degré) :", self.spin_pitch)
        
        form_layout.addRow(QtWidgets.QLabel("<br><b>Filtres de vitesse & direction :</b>"))
        
        # --- Section 2: Filtres ---
        self.spin_filtre_z = QtWidgets.QDoubleSpinBox()
        self.spin_filtre_z.setRange(1.0, 4.0)  # Restriction Matlab cf. VelocitySolver Line 354
        self.spin_filtre_z.setValue(10.0)     # Valeur de ta simulation par défaut (out of bounds ? mis à 10 comme ton script)
        self.spin_filtre_z.setRange(0.0, 50.0) # Augmenté pour accepter ton "10" initial
        form_layout.addRow("Filtre vitesse Z :", self.spin_filtre_z)
        
        self.spin_dir_fixe = QtWidgets.QDoubleSpinBox()
        self.spin_dir_fixe.setRange(0.94, 1.0)
        self.spin_dir_fixe.setDecimals(4)
        self.spin_dir_fixe.setValue(0.97)
        form_layout.addRow("Filtre direction max :", self.spin_dir_fixe)
        
        self.spin_dir_pond = QtWidgets.QDoubleSpinBox()
        self.spin_dir_pond.setRange(0.0, 0.01)
        self.spin_dir_pond.setDecimals(6)
        self.spin_dir_pond.setValue(0.0002)
        form_layout.addRow("Pondération filtre direction :", self.spin_dir_pond)
        
        self.spin_pond_vitesses = QtWidgets.QDoubleSpinBox()
        self.spin_pond_vitesses.setRange(0.001, 100.0)
        self.spin_pond_vitesses.setValue(1.0)
        form_layout.addRow("Pondération sur les vitesses :", self.spin_pond_vitesses)
        
        form_layout.addRow(QtWidgets.QLabel("<br><b>Résolution du maillage :</b>"))
        
        # --- Section 3: Maillage (Entiers) ---
        self.spin_cell_hor = QtWidgets.QSpinBox()
        self.spin_cell_hor.setRange(1, 500)
        self.spin_cell_hor.setValue(42)
        form_layout.addRow("Nombre de cellules horizontales :", self.spin_cell_hor)
        
        self.spin_cell_vert = QtWidgets.QSpinBox()
        self.spin_cell_vert.setRange(1, 500)
        self.spin_cell_vert.setValue(24)
        form_layout.addRow("Nombre de cellules verticales :", self.spin_cell_vert)
        
        layout.addLayout(form_layout)
        
        # --- Boutons OK / Annuler ---
        button_layout = QtWidgets.QHBoxLayout()
        self.btn_ok = QtWidgets.QPushButton("Valider les paramètres")
        self.btn_cancel = QtWidgets.QPushButton("Annuler")
        
        button_layout.addStretch()
        button_layout.addWidget(self.btn_cancel)
        button_layout.addWidget(self.btn_ok)
        layout.addLayout(button_layout)
        
        # Connexions des boutons
        self.btn_ok.clicked.connect(self.accept)
        self.btn_cancel.clicked.connect(self.reject)

    def get_values(self):
        """ Retourne les paramètres saisis sous forme de dictionnaire """
        return {
            "err_heading": self.spin_heading.value(),
            "err_roll": self.spin_roll.value(),
            "err_pitch": self.spin_pitch.value(),
            "filtre_vitesse_z": self.spin_filtre_z.value(),
            "filtre_direction_fixe": self.spin_dir_fixe.value(),
            "filtre_direction_pond": self.spin_dir_pond.value(),
            "ponderation_vitesses": self.spin_pond_vitesses.value(),
            "nbr_cell_hor": self.spin_cell_hor.value(),
            "nbr_cell_vert": self.spin_cell_vert.value()
        }

def get_clean_parameters() -> dict:
    """ Fonction utilitaire pour instancier et afficher le dialogue """
    from file_selector import get_qapplication
    app = get_qapplication() # Réutilise l'application globale déjà créée
    
    dialog = ParamSimulationDialog()
    if dialog.exec_() == QtWidgets.QDialog.Accepted:
        return dialog.get_values()
    else:
        print("❌ Fenêtre fermée ou annulée par l'utilisateur. Sortie.")
        import sys
        sys.exit(0)