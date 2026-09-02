import sys
import os
import scipy.io as sio
from PyQt5 import QtWidgets, QtCore

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'qrevint_21_03'))
from Classes.stickysettings import StickySettings as SSet


"""

    Dialog to allow users to select measurement files for processing.

    Attributes
    ----------
    settings: dict
        Dictionary used to store user defined settings.
    fullName: list
        Full name of files including path.
    fileName: list
        List of one or more fileNames to be processed.
    pathName: str
        Path to folder containing files.
    type: str
        Type of file (SonTek, TRDI, QRev).
    checked: bool
        Switch for TRDI files (True: load only checked, False: load all).

"""


class OpenMeasurementDialog(QtWidgets.QDialog):

    def __init__(self, parent=None):

        super(OpenMeasurementDialog, self).__init__(parent)

        if parent is not None and hasattr(parent, 'settingsFile'):
            self.settings = SSet(parent.settingsFile)
        else:
            self.settings = SSet("qrevint_settings")  # Nom arbitraire

        self.fullName = []
        self.fileName = []
        self.pathName = []
        self.type = ''
        self.checked = False
        self.setup_ui()

    
    def setup_ui(self):
        """Construit l'interface manuellement."""
        self.setWindowTitle("Sélectionnez un fichier ADCP")
        self.setMinimumWidth(350)

        # Layout principal
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(15, 15, 15, 15)
        layout.setSpacing(10)

        self.label = QtWidgets.QLabel(
            "<b>Pour SonTek :</b> Sélectionner directement "
            "les fichiers <i>.mat</i> correspondant aux transects voulus (multi-sélection)."
        )
        self.label.setWordWrap(True)
        self.label.setAlignment(QtCore.Qt.AlignCenter)
        layout.addWidget(self.label)

        # Boutons
        self.pbSonTek = QtWidgets.QPushButton("SonTek Matlab File (*.mat)")
        self.pbTRDI = QtWidgets.QPushButton("TRDI mmt File (*.mmt)")
        # self.pbQRev = QtWidgets.QPushButton("QRev File (*_QRev.mat)")
        self.pbCancel = QtWidgets.QPushButton("Cancel")

        layout.addWidget(self.pbSonTek)
        layout.addWidget(self.pbTRDI)
        # layout.addWidget(self.pbQRev)
        layout.addWidget(self.pbCancel)

        # Connexions
        self.pbSonTek.clicked.connect(self.select_sontek)
        self.pbTRDI.clicked.connect(self.select_trdi)
        # self.pbQRev.clicked.connect(self.select_qrev)
        self.pbCancel.clicked.connect(self.cancel)


        
    def select_sontek(self):
        """Sélectionne un fichier SonTek (.mat)."""
        folder = self.default_folder()
        self.fullName = QtWidgets.QFileDialog.getOpenFileNames(
            self, "Open SonTek File", folder, "SonTek Matlab File (*.mat)"
        )[0]
        if self.fullName:
            self.process_names()
            self.type = "SonTek"
        self.close()

    
    def select_trdi(self):
        """Sélectionne un fichier TRDI (.mmt)."""
        folder = self.default_folder()
        self.fullName = QtWidgets.QFileDialog.getOpenFileNames(
            self, "Open TRDI File", folder, "TRDI mmt File (*.mmt)"
        )[0]
        if self.fullName:
            self.process_names()
            self.type = "TRDI"
        self.close()

    def cancel(self):
        """Ferme la fenêtre sans sélection."""
        self.close()


    def default_folder(self):
        """Retourne le dossier par défaut (depuis les paramètres)."""
        try:
            folder = self.settings.get("Folder")
            if not folder:
                folder = os.getcwd()
        except:
            folder = os.getcwd()
        return folder

    def process_names(self):
        """Traite les noms de fichiers sélectionnés."""
        if isinstance(self.fullName, str):
            self.pathName, self.fileName = os.path.split(self.fullName)
        else:
            self.fileName = []
            for file in self.fullName:
                self.pathName, fileTemp = os.path.split(file)
                self.fileName.append(fileTemp)
        self.settings.set("Folder", self.pathName)

    def get_files(self):
        """Get filenames and pathname for file(s) to be processed

        Allows the user to select one *.mmt or one *_QRev.mat or one or more SonTek *.mat files for
        processing. The selected folder becomes the default folder for subsequent
        selectFile requests.
        """

        # Get the current folder setting.
        folder = self.default_folder()

        # Get the full names (path + file) of the selected files
        self.fullName = QtWidgets.QFileDialog.getOpenFileNames(
                    self, self.tr('Open File'), folder,
                    self.tr('All (*.mat *.mmt);;SonTek Matlab File (*.mat);;TRDI mmt File (*.mmt);;'
                            'QRev File (*_QRev.mat)'))[0]

        # Initialize parameters
        self.type = ''
        self.checked = False

        # Process fullName if selection was made
        if self.fullName:
            self.process_names()
        self.close()

    def process_names(self):
        """Parses fullnames into filenames and pathnames, sets default folder, determines the type of files selected,
        checks that the files selected are consistent with the type of files.
        """
        # Parse filenames and pathname from fullName
        if isinstance(self.fullName, str):
            self.pathName, self.fileName = os.path.split(self.fullName)
        else:
            self.fileName = []
            for file in self.fullName:
                self.pathName, fileTemp = os.path.split(file)
                self.fileName.append(fileTemp)

        # Update the folder setting
        self.settings.set('Folder', self.pathName)

        # Determine file type
        if len(self.fileName) == 1:
            file_name, file_extension = os.path.splitext(self.fileName[0])

            # TRDI file
            if file_extension == '.mmt':
                self.type = 'TRDI'
                checked_transect_dialog = QtWidgets.QMessageBox()
                checked_transect_dialog.setIcon(QtWidgets.QMessageBox.Question)
                checked_transect_dialog.setWindowTitle("Checked Transects?")
                checked_transect_dialog.setText(
                    "Do you want to load ONLY checked transects?")
                checked_transect_dialog.setStandardButtons(QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No)
                checked_transect_dialog.setDefaultButton(QtWidgets.QMessageBox.No)
                checked_transect_dialog = checked_transect_dialog.exec()

                if checked_transect_dialog == QtWidgets.QMessageBox.Yes:
                    self.checked = True

            # SonTek, Nortek, or QRev file
            else:
                mat_data = sio.loadmat(self.fullName[0], struct_as_record=False, squeeze_me=True)
                if 'version' in mat_data:
                    self.type = 'QRev'
                elif hasattr(mat_data['System'], 'InstrumentModel'):
                    self.type = 'Nortek'
                else:
                    self.type = 'SonTek'

        else:
            # If multiple files are selected they must all be SonTek or Nortek files
            for name in self.fileName:
                file_name, file_extension = os.path.splitext(name)
                if file_extension == '.mmt':
                    self.popup_message("Selected files contain an mmt file. An mmt file must be loaded separately")
                    break
                elif file_extension == '.mat':
                    mat_data = sio.loadmat(self.fullName[0], struct_as_record=False, squeeze_me=True)
                    if 'version' in mat_data:
                        self.popup_message("Selected files contain a QRev file. A QRev file must be opened separately")
                        break
                    elif hasattr(mat_data['System'], 'InstrumentModel'):
                        self.type = 'Nortek'
                        break
                    else:
                        self.type = 'SonTek'
                        break


    def default_folder(self):
        """Returns default folder.

        Returns the folder stored in settings or if no folder is stored, then the current
        working folder is returned.
        """
        try:
            folder = self.settings.get('Folder')
            if not folder:
                folder = os.getcwd()
        except KeyError:
            self.settings.new('Folder', os.getcwd())
            folder = self.settings.get('Folder')
        return folder

    @staticmethod
    def popup_message(text):
        """Display a message box with messages specified in text.

        Parameters
        ----------
        text: str
            Message to be displayed.
        """
        msg = QtWidgets.QMessageBox()
        msg.setIcon(QtWidgets.QMessageBox.Critical)
        msg.setText("Error")
        msg.setInformativeText(text)
        msg.setWindowTitle("Error")
        msg.exec_()
