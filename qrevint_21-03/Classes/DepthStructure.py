import numpy as np
from Classes.DepthData import DepthData


class DepthStructure(object):
    """This class creates the data structure used store depths from different sources

    Attributes
    ----------
    selected: str
        Name of object DepthData that contains depth data.
    bt_depths: DepthData
        Object of DepthData for bottom track based depths.
    vb_depths: DepthData
        Object of DepthData for vertical beam based depths.
    ds_depths: DepthData
        Object of DepthData for depth sounder based depths.
    composite: str
        Indicates use of composite depths ("On" or "Off".
    """
    
    def __init__(self):
        """Creates object and initializes variables to None"""

        self.selected = None  # name of object DepthData that contains the depth data for q computation
        self.bt_depths = None  # object of DepthData for by depth data
        self.vb_depths = None  # object of DepthData for vertical beam depth data
        self.ds_depths = None  # object of DepthData for depth sounder depth data
        self.composite = "On"  # Turn composite depths "on" or "off"

    def add_depth_object(self, depth_in, source_in, freq_in, draft_in, cell_depth_in, cell_size_in):
        """Adds a DepthData object to the depth structure for the specified type of depths.

        Parameters
        ----------
        depth_in: np.array
            Depth data in meters.
        source_in: str
            Specifies source of depth data: bottom track (BT), vertical beam (VB), or depth sounder (DS)
        freq_in: np.array
            Acoustic frequency in kHz of beams used to determine depth.
        draft_in:
            Draft of transducer (in meters) used to measure depths.
        cell_depth_in
            Depth of each cell in the profile. If the referenced depth does not have depth cells the depth cell
            values from the bottom track (BT) depths should be used.
        cell_size_in
            Size of each depth cell. If the referenced depth does not have depth cells the cell size from
            the bottom track (BT) depths should be used.
        """

        if source_in == 'BT':
            self.bt_depths = DepthData()
            self.bt_depths.populate_data(depth_in, source_in, freq_in, draft_in, cell_depth_in, cell_size_in)
        elif source_in == 'VB':
            self.vb_depths = DepthData()
            self.vb_depths.populate_data(depth_in, source_in, freq_in, draft_in, cell_depth_in, cell_size_in)
        elif source_in == 'DS':
            self.ds_depths = DepthData()
            self.ds_depths.populate_data(depth_in, source_in, freq_in, draft_in, cell_depth_in, cell_size_in)

    def populate_from_qrev_mat(self, transect):
        """Populates the object using data from previously saved QRev Matlab file.

        Parameters
        ----------
        transect: mat_struct
           Matlab data structure obtained from sio.loadmat
        """
        if hasattr(transect, 'depths'):

            # try:
            self.bt_depths = DepthData()
            self.bt_depths.populate_from_qrev_mat(transect.depths.btDepths)
            # except AttributeError:
            #     self.bt_depths = None

            try:
                self.vb_depths = DepthData()
                self.vb_depths.populate_from_qrev_mat(transect.depths.vbDepths)
            except AttributeError:
                self.vb_depths = None

            try:
                self.ds_depths = DepthData()
                self.ds_depths.populate_from_qrev_mat(transect.depths.dsDepths)
            except AttributeError:
                self.ds_depths = None

            if transect.depths.selected == 'btDepths':
                self.selected = 'bt_depths'
            elif transect.depths.selected == 'vbDepths':
                self.selected = 'vb_depths'
            elif transect.depths.selected == 'dsDepths':
                self.selected = 'ds_depths'
            self.composite = transect.depths.composite
            if self.vb_depths is None and self.ds_depths is None:
                self.composite = 'Off'

    def composite_depths(self, transect, setting="Off"):
        """Depth composite is based on the following assumptions
        
        1. If a depth sounder is available the user must have assumed the ADCP beams
        (BT or vertical) might have problems and it will be the second alternative if 
        not selected as the preferred source
        
        2. For 4-beam BT depths, if 3 beams are valid the average is considered valid.
        It may be based on interpolation of the invalid beam.  However, if only 2 beams
        are valid even though the other two beams may be interpolated and included in the average the
        average will be replaced by an alternative if available.  If no alternative is 
        available the multi-beam average based on available beams and interpolation will
        be used.

        Parameters
        ----------
        transect: TransectData
            Transect object containing all data.
        setting: str
            Setting to use ("On") or not use ("Off") composite depths.
        """
        
        if setting is None:
            setting = self.composite
        else:
            self.composite = setting
            
        # The primary depth reference is the selected reference
        ref = self.selected
        comp_depth = np.array([])

        if setting == 'On':
            # Prepare vector of valid BT averages, which are defined as having at least 2 valid beams
            bt_valid = self.bt_depths.valid_data
            n_ensembles = bt_valid.shape[-1]
            bt_filtered = np.copy(self.bt_depths.depth_processed_m)
            bt_filtered[np.logical_not(bt_valid)] = np.nan
            
            # Prepare vertical beam data, using only data prior to interpolation
            if self.vb_depths is not None:
                vb_filtered = np.copy(self.vb_depths.depth_processed_m)
                vb_filtered[np.squeeze(np.equal(self.vb_depths.valid_data, False))] = np.nan
            else:
                vb_filtered = np.tile(np.nan, n_ensembles)
                  
            # Prepare depth sounder data, using only data prior to interpolation
            if self.ds_depths is not None:
                ds_filtered = np.copy(self.ds_depths.depth_processed_m)
                ds_filtered[np.squeeze(np.equal(self.ds_depths.valid_data, False))] = np.nan
            else:
                ds_filtered = np.tile(np.nan, n_ensembles)

            comp_source = np.tile(np.nan, bt_filtered.shape)

            # Apply composite depths
            if ref == 'bt_depths':
                comp_depth = np.copy(bt_filtered)
                comp_source[np.isnan(comp_depth) == False] = 1
                comp_depth[np.isnan(comp_depth)] = np.squeeze(ds_filtered[np.isnan(comp_depth)])
                comp_source[np.logical_and((np.isnan(comp_depth) == False), (np.isnan(comp_source) == True))] = 3
                comp_depth[np.isnan(comp_depth)] = vb_filtered[np.isnan(comp_depth)]
                comp_source[np.logical_and((np.isnan(comp_depth) == False), (np.isnan(comp_source) == True))] = 2
                comp_depth[np.isnan(comp_depth)] = np.squeeze(self.bt_depths.depth_processed_m[np.isnan(comp_depth)])
                comp_source[np.logical_and((np.isnan(comp_depth) == False), (np.isnan(comp_source) == True))] = 4
                
            elif ref == 'vb_depths':
                comp_depth = np.copy(vb_filtered)
                comp_source[np.isnan(comp_depth) == False] = 2
                comp_depth[np.isnan(comp_depth)] = np.squeeze(ds_filtered[np.isnan(comp_depth)])
                comp_source[np.logical_and((np.isnan(comp_depth) == False), (np.isnan(comp_source) == True))] = 3
                comp_depth[np.isnan(comp_depth)] = np.squeeze(bt_filtered[np.isnan(comp_depth)])
                comp_source[np.logical_and((np.isnan(comp_depth) == False), (np.isnan(comp_source) == True))] = 1
                comp_depth[np.isnan(comp_depth)] = np.squeeze(self.vb_depths.depth_processed_m[np.isnan(comp_depth)])
                comp_source[np.logical_and((np.isnan(comp_depth) == False), (np.isnan(comp_source) == True))] = 4
                
            elif ref == 'ds_depths':
                comp_depth = np.copy(ds_filtered)
                comp_source[np.isnan(comp_depth) == False] = 3
                comp_depth[np.isnan(comp_depth)] = np.squeeze(vb_filtered[np.isnan(comp_depth)])
                comp_source[np.logical_and((np.isnan(comp_depth) == False), (np.isnan(comp_source) == True))] = 2
                comp_depth[np.isnan(comp_depth)] = np.squeeze(bt_filtered[np.isnan(comp_depth)])
                comp_source[np.logical_and((np.isnan(comp_depth) == False), (np.isnan(comp_source) == True))] = 1
                comp_depth[np.isnan(comp_depth)] = np.squeeze(self.ds_depths.depth_processed_m[np.isnan(comp_depth)])
                comp_source[np.logical_and((np.isnan(comp_depth) == False), (np.isnan(comp_source) == True))] = 4

            # Save composite depth to depth_processed of selected primary reference
            selected_data = getattr(self, ref)
            selected_data.apply_composite(comp_depth, comp_source.astype(int))
                
        else:
            selected_data = getattr(self, ref)
            comp_source = np.zeros(selected_data.depth_processed_m.shape)
            
            if ref == 'bt_depths':
                selected_data.valid_data[np.isnan(selected_data.valid_data)] = False
                comp_source[np.squeeze(selected_data.valid_data)] = 1
            elif ref == 'vb_depths':
                comp_source[np.squeeze(selected_data.valid_data)] = 2
            elif ref == 'ds_depths':
                comp_source[np.squeeze(selected_data.valid_data)] = 3

            selected_data.apply_interpolation(transect)
            comp_depth = selected_data.depth_processed_m
            selected_data.apply_composite(comp_depth, comp_source.astype(int))

    def set_draft(self, target, draft):
        """This function will change the ref_depth draft.

        Parameters
        ----------
        target: str
            Source of depth data.
        draft: float
            New draft.
        """
        
        if target == 'ADCP':
            self.bt_depths.change_draft(draft)
            self.vb_depths.change_draft(draft)
        else:
            self.ds_depths.change_draft(draft)    
            
    def depth_filter(self, transect, filter_method):
        """Method to apply filter to all available depth sources, so that
        all sources have the same filter applied.

        Parameters
        ----------
        transect: TransectData
            Object of TransectData
        filter_method: str
            Method to use to filter data (Smooth, TRDI, None).
        """
        
        if self.bt_depths is not None:
            self.bt_depths.apply_filter(transect, filter_method)
        if self.vb_depths is not None:
            self.vb_depths.apply_filter(transect, filter_method)
        if self.ds_depths is not None:
            self.ds_depths.apply_filter(transect, filter_method)
            
    def depth_interpolation(self, transect, method=None):
        """Method to apply interpolation to all available depth sources, so
        that all sources have the same filter applied.

        Parameters
        ----------
        transect: TransectData
            Object of TransectData
        method: str
            Interpolation method (None, HoldLast, Smooth, Linear)
            """
        
        if self.bt_depths is not None:
            self.bt_depths.apply_interpolation(transect, method)
        if self.vb_depths is not None:
            self.vb_depths.apply_interpolation(transect, method)
        if self.ds_depths is not None:
            self.ds_depths.apply_interpolation(transect, method)
            
    def sos_correction(self, ratio):
        """Correct depths for change in speed of sound.

        Parameters
        ----------
        ratio: float
            Ratio of new to old speed of sound.
        """
        
        # Bottom Track Depths
        if self.bt_depths is not None:
            self.bt_depths.sos_correction(ratio)
            
        # Vertical beam depths
        if self.vb_depths is not None:
            self.vb_depths.sos_correction(ratio)
