# -*- coding: utf-8 -*-
"""
Created on Fri Feb 18 14:28:25 2022

@author: blais
"""
# ========================================
# External imports
# ========================================
import os
import glob
import tkinter as tk
import tkinter.filedialog
import numpy as np
import warnings
import sys

# ========================================
# Internal imports
# ========================================
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'qrevint_21_03'))

# from Classes.Measurement import Measurement
from Classes_vermeulen.Measurement_new import Measurement


# =============================================================================
# Functions
# =============================================================================
def select_file(path_window=None):
    if not path_window:
        path_window = os.getcwd()
    # Open a window to select a measurement
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    path_meas = tk.filedialog.askopenfilenames(parent=root, initialdir=path_window)
    if path_meas:
        # Normalize extensions and folder name in a robust, cross-platform way
        keys = [os.path.splitext(elem)[1].lstrip('.').lower() for elem in path_meas]
        name_folders = os.path.basename(os.path.dirname(path_meas[0])).split('.')[0]

        fileName = None
        type_meas = None

        if 'mmt' in keys:
            type_meas = 'TRDI'
            fileName = path_meas[0]
        elif 'mat' in keys:
            type_meas = 'SonTek'
            ind_meas = [i for i, s in enumerate(keys) if "mat" in s]
            fileName = [(path_meas[x]) for x in ind_meas]

        # Fallback: if we couldn't detect expected extensions, return first file and a generic type
        if fileName is None:
            fileName = path_meas[0]
            ext = keys[0] if keys else ''
            if ext == 'mmt':
                type_meas = 'TRDI'
            elif ext == 'mat':
                type_meas = 'SonTek'
            else:
                type_meas = 'Unknown'

        return fileName, type_meas, name_folders
    else:
        warnings.warn('No file selected - end')
        sys.exit()


def select_directory(path_window=None):
    if not path_window:
        path_window = os.getcwd()
    # Open a window to select a folder which contains measurements
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    path_folder = tk.filedialog.askdirectory(parent=root, initialdir=path_window,
                                             title='Select folder')
    if path_folder:
        if os.path.exists(path_folder + '/measurement_data.csv') and \
                os.path.exists(path_folder + '/transects_data.csv'):
            return path_folder
        else:
            path_folder = '\\'.join(path_folder.split('/'))
            path_folders = np.array(glob.glob(path_folder + "/*"))
            name_folders = np.array([os.path.basename((x)) for x in path_folders])
            excluded_folders = [s.find('.') == -1 for s in name_folders]
            path_folders = path_folders[excluded_folders]
            name_folders = name_folders[excluded_folders]

            type_meas = list()
            path_meas = list()
            name_meas = list()

            for id_meas in range(len(path_folders)):
                list_files = os.listdir(path_folders[id_meas])
                exte_files = list([i.split('.', 1)[-1] for i in list_files])
                if 'mmt' in exte_files or 'mat' in exte_files:
                    if 'mmt' in exte_files:
                        type_meas.append('TRDI')
                        loca_meas = exte_files.index("mmt")  # index of the measurement file
                        ind_meas = [i for i, x in enumerate(exte_files) if x == "PD0"]  # transects index
                        fileName = list_files[loca_meas]
                        path_meas.append(os.path.join(path_folders[id_meas], fileName))
                        name_meas.append(name_folders[id_meas])

                    elif 'mat' in exte_files:
                        type_meas.append('SonTek')
                        loca_meas = [i for i, s in enumerate(exte_files) if "mat" in s]  # transects index
                        fileNameRaw = [(list_files[x]) for x in loca_meas]
                        fileName = [s for i, s in enumerate(fileNameRaw)
                                    if "QRev.mat" not in s]
                        path_meas.append([os.path.join(path_folders[id_meas], fileName[x]) for x in range(len(fileName))])
                        name_meas.append(name_folders[id_meas])
            return path_meas, type_meas, name_meas
    else:
        warnings.warn('No folder selected - end')
        sys.exit()


def open_measurement(path_meas, type_meas, apply_settings=False, navigation_reference=None,
                     checked_transect=None, extrap_velocity=False, run_oursin=False, use_weighted=False,
                     use_measurement_thresholds=False):
    # Open measurement
    meas = Measurement(in_file=path_meas, source=type_meas, proc_type='QRev', run_oursin=run_oursin,
                       use_weighted=use_weighted)

    if apply_settings:
        if navigation_reference == 'GPS':
            if not meas.transects[meas.checked_transect_idx[0]].gps:
                print('No GPS available : switch to BT')
                navigation_reference = 'BT'
        meas, checked_transect, navigation_reference = new_settings(meas,
                                                                    navigation_reference,
                                                                    checked_transect,
                                                                    extrap_velocity)
    else:
        if meas.current_settings()['NavRef'] == 'bt_vel':
            navigation_reference = 'BT'
        elif meas.current_settings()['NavRef'] == 'gga_vel':
            navigation_reference = 'GPS'
        checked_transect = meas.checked_transect_idx

    return meas, checked_transect, navigation_reference


def new_settings(meas, navigation_reference_user=None, checked_transect_user=None, extrap_velocity=False):
    # Apply settings
    settings = meas.current_settings()

    settings_change = False

    # Default Navigation reference
    if navigation_reference_user == None:
        if meas.current_settings()['NavRef'] == 'bt_vel':
            navigation_reference = 'BT'
        elif meas.current_settings()['NavRef'] == 'gga_vel' or meas.current_settings()['NavRef'] == 'vtg_vel':
            navigation_reference = 'GPS'

    # Change Navigation reference
    else:
        navigation_reference = navigation_reference_user
        if navigation_reference_user == 'BT' and meas.current_settings()['NavRef'] != 'bt_vel':
            settings['NavRef'] = 'BT'
            settings_change = True
        elif navigation_reference_user == 'GPS' and meas.current_settings()['NavRef'] != 'gga_vel':
            settings['NavRef'] = 'GGA'
            settings_change = True

    # Change checked transects
    if not checked_transect_user or checked_transect_user == meas.checked_transect_idx:
        checked_transect_idx = meas.checked_transect_idx
    else:
        checked_transect_idx = checked_transect_user
        meas.checked_transect_idx = []
        for n in range(len(meas.transects)):
            if n in checked_transect_idx:
                meas.transects[n].checked = True
                meas.checked_transect_idx.append(n)
            else:
                meas.transects[n].checked = False
        meas.selected_transects_changed(checked_transect_user)
        # selected_transects_changed already contains apply_settings

    # Apply Extrapolation on velocities
    if extrap_velocity:
        meas.extrap_fit.change_data_type(meas.transects, 'v')
        settings_change = True

    if settings_change:
        meas.apply_settings(settings)

    return meas, checked_transect_idx, navigation_reference
