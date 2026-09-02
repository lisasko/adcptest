import os
import numpy as np
from datetime import datetime
from datetime import timezone
from scipy import signal, fftpack
# from Classes.Pd0TRDI import Pd0TRDI
from Classes.Pd0TRDI_2 import Pd0TRDI
from Classes.DepthStructure import DepthStructure
from Classes.WaterData import WaterData
from Classes.BoatStructure import BoatStructure
from Classes.GPSData import GPSData
from Classes.Edges import Edges
from Classes.ExtrapData import ExtrapData
from Classes.Sensors import Sensors
from Classes.SensorData import SensorData
from Classes.HeadingData import HeadingData
from Classes.DateTime import DateTime
from Classes.InstrumentData import InstrumentData
from Classes.MultiThread import MultiThread
from Classes.CoordError import CoordError
from MiscLibs.common_functions import nandiff, cosd, arctand, tand, nans, cart2pol, rad2azdeg


class TransectData(object):
    """Class to hold Transect properties.

    Attributes
    ----------
    adcp: InstrumentData
        Object of InstrumentData
    file_name: str
        Filename of transect data file
    w_vel: WaterData
        Object of WaterData
    boat_vel: BoatStructure
        Object of BoatStructure containing objects of BoatData for BT, GGA, and VTG
    gps: GPSData
        Object of GPSData
    sensors: SensorData
        Object of SensorData
    depths: DepthStructure
        Object of DepthStructure containing objects of Depth data for bt_depths, vb_depths, ds_depths)
    edges: Edges
        Object of Edges (left and right object of clsEdgeData)
    extrap: ExtrapData
        Object of ExtrapData
    start_edge: str
        Starting edge of transect looking downstream (Left or Right)
    orig_start_edge: str
        Original starting edge of transect looking downstream (Left or Right)
    date_time: DateTime
        Object of DateTime
    checked: bool
        Setting for if transect was checked for use in mmt file assumed checked for SonTek
    in_transect_idx: np.array(int)
        Index of ensemble data associated with the moving-boat portion of the transect
    """

    def __init__(self):
        self.adcp = None  # object of clsInstrument
        self.file_name = None  # filename of transect data file
        self.w_vel = None  # object of clsWaterData
        self.boat_vel = None  # class for various boat velocity references (btVel, ggaVel, vtgVel)
        self.gps = None  # object of clsGPSData
        self.sensors = None  # object of clsSensorData
        self.depths = None  # object of clsDepthStructure for depth data including cell depths & ref depths
        self.edges = None  # object of clsEdges(left and right object of clsEdgeData)
        self.extrap = None  # object of clsExtrapData
        self.start_edge = None  # starting edge of transect looking downstream (Left or Right)
        self.orig_start_edge = None
        self.date_time = None  # object of DateTime
        self.checked = None  # transect was checked for use in mmt file assumed checked for SonTek
        self.in_transect_idx = None  # index of ensemble data associated with the moving-boat portion of the transect

    def trdi(self, mmt_transect, pd0_data, mmt):
        """Create object, lists, and instance variables for TRDI data.

        Parameters
        ----------
        mmt_transect: MMT_Transect
            Object of Transect (from mmt)
        pd0_data: Pd0TRDI
            Object of Pd0TRDI
        mmt: MMT_TRDI
            Object of MMT_TRDI
        """

        # Get file name of pd0 file which is first file in list of file associated with the transect
        self.file_name = mmt_transect.Files[0]

        # Get the active configuration data for the transect
        mmt_config = getattr(mmt_transect, 'active_config')

        # If the pd0 file has water track data process all of the data
        if pd0_data.Wt is not None:

            # Get and compute ensemble beam depths
            temp_depth_bt = np.array(pd0_data.Bt.depth_m)

            # Screen out invalid depths
            temp_depth_bt[temp_depth_bt < 0.01] = np.nan

            # Add draft
            temp_depth_bt += mmt_config['Offsets_Transducer_Depth']
            
            # Get instrument cell data
            cell_size_all_m, cell_depth_m, sl_cutoff_per, sl_lag_effect_m = \
                TransectData.compute_cell_data(pd0_data)
            
            # Adjust cell depth of draft
            cell_depth_m = np.add(mmt_config['Offsets_Transducer_Depth'], cell_depth_m)
            
            # Create depth data object for BT
            self.depths = DepthStructure()
            self.depths.add_depth_object(depth_in=temp_depth_bt,
                                         source_in='BT',
                                         freq_in=pd0_data.Inst.freq,
                                         draft_in=mmt_config['Offsets_Transducer_Depth'],
                                         cell_depth_in=cell_depth_m,
                                         cell_size_in=cell_size_all_m)
            
            # Compute cells above side lobe
            cells_above_sl, sl_cutoff_m = \
                TransectData.side_lobe_cutoff(depths=self.depths.bt_depths.depth_orig_m,
                                              draft=self.depths.bt_depths.draft_orig_m,
                                              cell_depth=self.depths.bt_depths.depth_cell_depth_m,
                                              sl_lag_effect=sl_lag_effect_m,
                                              slc_type='Percent',
                                              value=1-sl_cutoff_per / 100)
            
            # Check for the presence of vertical beam data
            if np.nanmax(np.nanmax(pd0_data.Sensor.vert_beam_status)) > 0:
                temp_depth_vb = np.tile(np.nan, (1, cell_depth_m.shape[1]))
                temp_depth_vb[0, :] = pd0_data.Sensor.vert_beam_range_m
                
                # Screen out invalid depths
                temp_depth_vb[temp_depth_vb < 0.01] = np.nan
                
                # Add draft
                temp_depth_vb = temp_depth_vb + mmt_config['Offsets_Transducer_Depth']
                
                # Create depth data object for vertical beam
                self.depths.add_depth_object(depth_in=temp_depth_vb,
                                             source_in='VB',
                                             freq_in=pd0_data.Inst.freq,
                                             draft_in=mmt_config['Offsets_Transducer_Depth'],
                                             cell_depth_in=cell_depth_m,
                                             cell_size_in=cell_size_all_m)
                                   
            # Check for the presence of depth sounder
            if np.nansum(np.nansum(pd0_data.Gps2.depth_m)) > 1e-5:
                temp_depth_ds = pd0_data.Gps2.depth_m
                
                # Screen out invalid data
                temp_depth_ds[temp_depth_ds < 0.01] = np.nan
                
                # Use the last valid depth for each ensemble
                last_depth_col_idx = np.sum(np.isnan(temp_depth_ds) == False, axis=1)-1
                last_depth_col_idx[last_depth_col_idx == -1] = 0               
                row_index = np.arange(len(temp_depth_ds))
                last_depth = nans(row_index.size)
                for row in row_index:
                    last_depth[row] = temp_depth_ds[row, last_depth_col_idx[row]]

                # Determine if mmt file has a scale factor and offset for the depth sounder
                if mmt_config['DS_Cor_Spd_Sound'] == 0:
                    scale_factor = mmt_config['DS_Scale_Factor']
                else:
                    scale_factor = pd0_data.Sensor.sos_mps / 1500.
                    
                # Apply scale factor, offset, and draft
                # Note: Only the ADCP draft is stored.  The transducer
                # draft or scaling for depth sounder data cannot be changed in QRev
                ds_depth = np.tile(np.nan, (1, cell_depth_m.shape[1]))
                ds_depth[0, :] = (last_depth * scale_factor) \
                    + mmt_config['DS_Transducer_Depth']\
                    + mmt_config['DS_Transducer_Offset']
                
                self.depths.add_depth_object(depth_in=ds_depth,
                                             source_in='DS',
                                             freq_in=pd0_data.Inst.freq,
                                             draft_in=mmt_config['Offsets_Transducer_Depth'],
                                             cell_depth_in=cell_depth_m,
                                             cell_size_in=cell_size_all_m)
                
            # Set depth reference to value from mmt file
            if 'Proc_River_Depth_Source' in mmt_config:
                if mmt_config['Proc_River_Depth_Source'] == 0:
                    self.depths.selected = 'bt_depths'
                    self.depths.composite_depths(transect=self, setting='Off')

                elif mmt_config['Proc_River_Depth_Source'] == 1:
                    if self.depths.ds_depths is not None:
                        self.depths.selected = 'ds_depths'
                    else:
                        self.depths.selected = 'bt_depths'
                    self.depths.composite_depths(transect=self, setting='Off')

                elif mmt_config['Proc_River_Depth_Source'] == 2:
                    if self.depths.vb_depths is not None:
                        self.depths.selected = 'vb_depths'
                    else:
                        self.depths.selected = 'bt_depths'
                    self.depths.composite_depths(transect=self, setting='Off')

                elif mmt_config['Proc_River_Depth_Source'] == 3:
                    if self.depths.vb_depths is None:
                        self.depths.selected = 'bt_depths'
                        self.depths.composite_depths(transect=self, setting='Off')
                    else:
                        self.depths.selected = 'vb_depths'
                        self.depths.composite_depths(transect=self, setting='On')

                elif mmt_config['Proc_River_Depth_Source'] == 4:
                    if self.depths.bt_depths is not None:
                        self.depths.selected = 'bt_depths'
                        if self.depths.vb_depths is not None or self.depths.ds_depths is not None:
                            self.depths.composite_depths(transect=self, setting='On')
                        else:
                            self.depths.composite_depths(transect=self, setting='Off')
                    elif self.depths.vb_depths is not None:
                        self.depths.selected = 'vb_depths'
                        self.depths.composite_depths(transect=self, setting='On')
                    elif self.depths.ds_depths is not None:
                        self.depths.selected = 'ds_depths'
                        self.depths.composite_depths(transect=self, setting='On')
                else:
                    self.depths.selected = 'bt_depths'
                    self.depths.composite_depths(transect=self, setting='Off')
            else:
                if mmt_config['DS_Use_Process'] > 0:
                    if self.depths.ds_depths is not None:
                        self.depths.selected = 'ds_depths'
                    else:
                        self.depths.selected = 'bt_depths'
                else:
                    self.depths.selected = 'bt_depths'
                self.depths.composite_depths(transect=self, setting='Off')
                
            # Create water_data object
            # ------------------------
            
            # Check for RiverRay and RiverPro data
            firmware = str(pd0_data.Inst.firm_ver[0])
            excluded_dist = 0
            if (firmware[:2] == '56') and (np.nanmax(pd0_data.Sensor.vert_beam_status) < 0.9):
                excluded_dist = 0.25
                
            if (firmware[:2] == '44') or (firmware[:2] == '56'):
                # Process water velocities for RiverRay and RiverPro
                self.w_vel = WaterData()
                self.w_vel.populate_data(vel_in=pd0_data.Wt.vel_mps,
                                         freq_in=pd0_data.Inst.freq.T,
                                         coord_sys_in=pd0_data.Cfg.coord_sys,
                                         nav_ref_in='None',
                                         rssi_in=pd0_data.Wt.rssi,
                                         rssi_units_in='Counts',
                                         excluded_dist_in=excluded_dist,
                                         cells_above_sl_in=cells_above_sl,
                                         sl_cutoff_per_in=sl_cutoff_per,
                                         sl_cutoff_num_in=0,
                                         sl_cutoff_type_in='Percent',
                                         sl_lag_effect_in=sl_lag_effect_m,
                                         sl_cutoff_m=sl_cutoff_m,
                                         wm_in=pd0_data.Cfg.wm[0],
                                         blank_in=pd0_data.Cfg.wf_cm[0] / 100,
                                         corr_in=pd0_data.Wt.corr,
                                         surface_vel_in=pd0_data.Surface.vel_mps,
                                         surface_rssi_in=pd0_data.Surface.rssi,
                                         surface_corr_in=pd0_data.Surface.corr,
                                         surface_num_cells_in=pd0_data.Surface.no_cells)
                
            else:
                # Process water velocities for non-RiverRay ADCPs
                self.w_vel = WaterData()
                self.w_vel.populate_data(vel_in=pd0_data.Wt.vel_mps,
                                         freq_in=pd0_data.Inst.freq.T,
                                         coord_sys_in=pd0_data.Cfg.coord_sys[0],
                                         nav_ref_in='None',
                                         rssi_in=pd0_data.Wt.rssi,
                                         rssi_units_in='Counts',
                                         excluded_dist_in=excluded_dist,
                                         cells_above_sl_in=cells_above_sl,
                                         sl_cutoff_per_in=sl_cutoff_per,
                                         sl_cutoff_num_in=0,
                                         sl_cutoff_type_in='Percent',
                                         sl_lag_effect_in=sl_lag_effect_m,
                                         sl_cutoff_m=sl_cutoff_m,
                                         wm_in=pd0_data.Cfg.wm[0],
                                         blank_in=pd0_data.Cfg.wf_cm[0] / 100,
                                         corr_in=pd0_data.Wt.corr)

            # Initialize boat vel
            self.boat_vel = BoatStructure()
            # Apply 3-beam setting from mmt file
            if mmt_config['Proc_Use_3_Beam_BT'] < 0.5:
                min_beams = 4
            else:
                min_beams = 3
            self.boat_vel.add_boat_object(source='TRDI',
                                          vel_in=pd0_data.Bt.vel_mps,
                                          freq_in=pd0_data.Inst.freq.T,
                                          coord_sys_in=pd0_data.Cfg.coord_sys[0],
                                          nav_ref_in='BT',
                                          min_beams=min_beams,
                                          bottom_mode=pd0_data.Cfg.bm[0])
            
            self.boat_vel.set_nav_reference('BT')
            
            # Compute velocities from GPS Data
            # ------------------------------------
            # Raw Data
            raw_gga_utc = pd0_data.Gps2.utc
            raw_gga_lat = pd0_data.Gps2.lat_deg
            raw_gga_lon = pd0_data.Gps2.lon_deg

            # Determine correct sign for latitude
            for n, lat_ref in enumerate(pd0_data.Gps2.lat_ref):
                idx = np.nonzero(np.array(lat_ref) == 'S')
                raw_gga_lat[n, idx] = raw_gga_lat[n, idx] * -1

            # Determine correct sign for longitude
            for n, lon_ref in enumerate(pd0_data.Gps2.lon_ref):
                idx = np.nonzero(np.array(lon_ref) == 'W')
                raw_gga_lon[n, idx] = raw_gga_lon[n, idx] * -1

            # Assign data to local variables
            raw_gga_alt = pd0_data.Gps2.alt
            raw_gga_diff = pd0_data.Gps2.corr_qual
            raw_gga_hdop = pd0_data.Gps2.hdop
            raw_gga_num_sats = pd0_data.Gps2.num_sats
            raw_vtg_course = pd0_data.Gps2.course_true
            raw_vtg_speed = pd0_data.Gps2.speed_kph * 0.2777778
            raw_vtg_delta_time = pd0_data.Gps2.vtg_delta_time
            raw_vtg_mode_indicator = pd0_data.Gps2.mode_indicator
            raw_gga_delta_time = pd0_data.Gps2.gga_delta_time
            
            # RSL provided ensemble values, not supported for TRDI data
            ext_gga_utc = []
            ext_gga_lat = []
            ext_gga_lon = []
            ext_gga_alt = []
            ext_gga_diff = []
            ext_gga_hdop = []
            ext_gga_num_sats = []
            ext_vtg_course = []
            ext_vtg_speed = []
             
            # QRev methods GPS processing methods
            gga_p_method = 'Mindt'
            gga_v_method = 'Mindt'
            vtg_method = 'Mindt'
            
            # If valid gps data exist, process the data
            if (np.nansum(np.nansum(np.abs(raw_gga_lat))) > 0) \
                    or (np.nansum(np.nansum(np.abs(raw_vtg_speed))) > 0):
                
                # Process raw GPS data
                self.gps = GPSData()
                self.gps.populate_data(raw_gga_utc=raw_gga_utc,
                                       raw_gga_lat=raw_gga_lat,
                                       raw_gga_lon=raw_gga_lon,
                                       raw_gga_alt=raw_gga_alt,
                                       raw_gga_diff=raw_gga_diff,
                                       raw_gga_hdop=raw_gga_hdop,
                                       raw_gga_num_sats=raw_gga_num_sats,
                                       raw_gga_delta_time=raw_gga_delta_time,
                                       raw_vtg_course=raw_vtg_course,
                                       raw_vtg_speed=raw_vtg_speed,
                                       raw_vtg_delta_time=raw_vtg_delta_time,
                                       raw_vtg_mode_indicator=raw_vtg_mode_indicator,
                                       ext_gga_utc=ext_gga_utc,
                                       ext_gga_lat=ext_gga_lat,
                                       ext_gga_lon=ext_gga_lon,
                                       ext_gga_alt=ext_gga_alt,
                                       ext_gga_diff=ext_gga_diff,
                                       ext_gga_hdop=ext_gga_hdop,
                                       ext_gga_num_sats=ext_gga_num_sats,
                                       ext_vtg_course=ext_vtg_course,
                                       ext_vtg_speed=ext_vtg_speed,
                                       gga_p_method=gga_p_method,
                                       gga_v_method=gga_v_method,
                                       vtg_method=vtg_method)
                
                # If valid gga data exists create gga boat velocity object
                if np.nansum(np.nansum(np.abs(raw_gga_lat))) > 0:
                    self.boat_vel.add_boat_object(source='TRDI',
                                                  vel_in=self.gps.gga_velocity_ens_mps,
                                                  coord_sys_in='Earth',
                                                  nav_ref_in='GGA')

                # If valid vtg data exist create vtg boat velocity object
                if np.nansum(np.nansum(np.abs(raw_vtg_speed))) > 0:
                    self.boat_vel.add_boat_object(source='TRDI',
                                                  vel_in=self.gps.vtg_velocity_ens_mps,
                                                  coord_sys_in='Earth',
                                                  nav_ref_in='VTG')

            # Create Edges Object
            self.edges = Edges()
            self.edges.populate_data(rec_edge_method='Fixed', vel_method='MeasMag')
                
            # Determine number of ensembles to average
            n_ens_left = mmt_config['Q_Shore_Pings_Avg']
            # TRDI uses same number on left and right edges
            n_ens_right = n_ens_left
            
            # Set indices for ensembles in the moving-boat portion of the transect
            self.in_transect_idx = np.arange(0, pd0_data.Bt.vel_mps.shape[1])
            
            # Determine left and right edge distances
            if mmt_config['Edge_Begin_Left_Bank']:
                dist_left = float(mmt_config['Edge_Begin_Shore_Distance'])
                dist_right = float(mmt_config['Edge_End_Shore_Distance'])
                if 'Edge_End_Manual_Discharge' in mmt_config:
                    user_discharge_left = float(mmt_config['Edge_Begin_Manual_Discharge'])
                    user_discharge_right = float(mmt_config['Edge_End_Manual_Discharge'])
                    edge_method_left = mmt_config['Edge_Begin_Method_Distance']
                    edge_method_right = mmt_config['Edge_End_Method_Distance']
                else:
                    user_discharge_left = None
                    user_discharge_right = None
                    edge_method_left = 'Yes'
                    edge_method_right = 'Yes'
                self.start_edge = 'Left'
                self.orig_start_edge = 'Left'
            else:
                dist_left = float(mmt_config['Edge_End_Shore_Distance'])
                dist_right = float(mmt_config['Edge_Begin_Shore_Distance'])
                if 'Edge_End_Manual_Discharge' in mmt_config:
                    user_discharge_left = float(mmt_config['Edge_End_Manual_Discharge'])
                    user_discharge_right = float(mmt_config['Edge_Begin_Manual_Discharge'])
                    edge_method_left = mmt_config['Edge_End_Method_Distance']
                    edge_method_right = mmt_config['Edge_Begin_Method_Distance']
                else:
                    user_discharge_left = None
                    user_discharge_right = None
                    edge_method_left = 'Yes'
                    edge_method_right = 'Yes'
                self.start_edge = 'Right'
                self.orig_start_edge = 'Right'
                
            # Create left edge
            if edge_method_left == 'NO':
                self.edges.left.populate_data(edge_type='User Q',
                                              distance=dist_left,
                                              number_ensembles=n_ens_left,
                                              user_discharge=user_discharge_left)

            elif mmt_config['Q_Left_Edge_Type'] == 0:
                self.edges.left.populate_data(edge_type='Triangular',
                                              distance=dist_left,
                                              number_ensembles=n_ens_left,
                                              user_discharge=user_discharge_left)

            elif mmt_config['Q_Left_Edge_Type'] == 1:
                self.edges.left.populate_data(edge_type='Rectangular',
                                              distance=dist_left,
                                              number_ensembles=n_ens_left,
                                              user_discharge=user_discharge_left)

            elif mmt_config['Q_Left_Edge_Type'] == 2:
                self.edges.left.populate_data(edge_type='Custom',
                                              distance=dist_left,
                                              number_ensembles=n_ens_left,
                                              coefficient=mmt_config['Q_Left_Edge_Coeff'],
                                              user_discharge=user_discharge_left)

                
            # Create right edge
            if edge_method_right == 'NO':
                self.edges.right.populate_data(edge_type='User Q',
                                               distance=dist_right,
                                               number_ensembles=n_ens_right,
                                               user_discharge=user_discharge_right)
            elif mmt_config['Q_Right_Edge_Type'] == 0:
                self.edges.right.populate_data(edge_type='Triangular',
                                               distance=dist_right,
                                               number_ensembles=n_ens_right,
                                               user_discharge=user_discharge_right)

            elif mmt_config['Q_Right_Edge_Type'] == 1:
                self.edges.right.populate_data(edge_type='Rectangular',
                                               distance=dist_right,
                                               number_ensembles=n_ens_right,
                                               user_discharge=user_discharge_right)

            elif mmt_config['Q_Right_Edge_Type'] == 2:
                self.edges.right.populate_data(edge_type='Custom',
                                               distance=dist_right,
                                               number_ensembles=n_ens_right,
                                               coefficient=mmt_config['Q_Right_Edge_Coeff'],
                                               user_discharge=user_discharge_right)
                
            # Create extrap object
            # --------------------
            # Determine top method
            top = 'Power'
            if mmt_config['Q_Top_Method'] == 1:
                top = 'Constant'
            elif mmt_config['Q_Top_Method'] == 2:
                top = '3-Point'
                
            # Determine bottom method
            bot = 'Power'
            if mmt_config['Q_Bottom_Method'] == 2:
                bot = 'No Slip'
                
            self.extrap = ExtrapData()
            self.extrap.populate_data(top=top, bot=bot, exp=mmt_config['Q_Power_Curve_Coeff'])
            
            # Sensor Data
            self.sensors = Sensors() 
            
            # Heading
            
            # Internal Heading
            self.sensors.heading_deg.internal = HeadingData()
            self.sensors.heading_deg.internal.populate_data(data_in=pd0_data.Sensor.heading_deg.T,
                                                            source_in='internal',
                                                            magvar=mmt_config['Offsets_Magnetic_Variation'],
                                                            align=mmt_config['Ext_Heading_Offset'])

            # External Heading
            ext_heading_check = np.where(np.isnan(pd0_data.Gps2.heading_deg) == False)
            if len(ext_heading_check[0]) <= 0:
                self.sensors.heading_deg.selected = 'internal'
            else:
                # Determine external heading for each ensemble
                # Using the minimum time difference
                d_time = np.abs(pd0_data.Gps2.hdt_delta_time)
                d_time_min = np.nanmin(d_time.T, 0).T
                use = np.tile([np.nan], d_time.shape)
                for nd_time in range(len(d_time_min)):
                    use[nd_time, :] = np.abs(d_time[nd_time, :]) == d_time_min[nd_time]
                    
                ext_heading_deg = np.tile([np.nan], (len(d_time_min)))
                for nh in range(len(d_time_min)):
                    idx = np.where(use[nh, :])[0]
                    if len(idx) > 0:
                        idx = idx[0]
                        ext_heading_deg[nh] = pd0_data.Gps2.heading_deg[nh, idx]
                        
                # Create external heading sensor
                self.sensors.heading_deg.external = HeadingData()
                self.sensors.heading_deg.external.populate_data(data_in=ext_heading_deg,
                                                                source_in='external',
                                                                magvar=mmt_config['Offsets_Magnetic_Variation'],
                                                                align=mmt_config['Ext_Heading_Offset'])

                # Determine heading source to use from mmt setting
                source_used = mmt_config['Ext_Heading_Use']
                if source_used:
                    self.sensors.heading_deg.selected = 'external'
                else:
                    self.sensors.heading_deg.selected = 'internal'

            # Pitch
            pitch = arctand(tand(pd0_data.Sensor.pitch_deg) * cosd(pd0_data.Sensor.roll_deg))
            pitch_src = pd0_data.Cfg.pitch_src[0]
            
            # Create pitch sensor
            self.sensors.pitch_deg.internal = SensorData()
            self.sensors.pitch_deg.internal.populate_data(data_in=pitch, source_in=pitch_src)
            self.sensors.pitch_deg.selected = 'internal'
            
            # Roll
            roll = pd0_data.Sensor.roll_deg.T
            roll_src = pd0_data.Cfg.roll_src[0]
            
            # Create Roll sensor
            self.sensors.roll_deg.internal = SensorData()
            self.sensors.roll_deg.internal.populate_data(data_in=roll, source_in=roll_src)
            self.sensors.roll_deg.selected = 'internal'
            
            # Temperature
            temperature = pd0_data.Sensor.temperature_deg_c.T
            temperature_src = pd0_data.Cfg.temp_src[0]
            
            # Create temperature sensor
            self.sensors.temperature_deg_c.internal = SensorData()
            self.sensors.temperature_deg_c.internal.populate_data(data_in=temperature, source_in=temperature_src)
            self.sensors.temperature_deg_c.selected = 'internal'
            
            # Salinity
            pd0_salinity = pd0_data.Sensor.salinity_ppt.T
            pd0_salinity_src = pd0_data.Cfg.sal_src[0]
            
            # Create salinity sensor from pd0 data
            self.sensors.salinity_ppt.internal = SensorData()
            self.sensors.salinity_ppt.internal.populate_data(data_in=pd0_salinity, source_in=pd0_salinity_src)

            # Create salinity sensor from mmt data
            mmt_salinity = mmt_config['Proc_Salinity']
            self.sensors.salinity_ppt.user = SensorData()
            self.sensors.salinity_ppt.user.populate_data(data_in=mmt_salinity, source_in='mmt')

            # Set selected salinity
            self.sensors.salinity_ppt.selected = 'internal'
            
            # Speed of Sound
            speed_of_sound = pd0_data.Sensor.sos_mps.T
            speed_of_sound_src = pd0_data.Cfg.sos_src[0]
            self.sensors.speed_of_sound_mps.internal = SensorData()
            self.sensors.speed_of_sound_mps.internal.populate_data(data_in=speed_of_sound, source_in=speed_of_sound_src)
            
            # The raw data are referenced to the internal SOS
            self.sensors.speed_of_sound_mps.selected = 'internal'
            
            # Ensemble times
            # Compute time for each ensemble in seconds
            ens_time_sec = pd0_data.Sensor.time[:, 0] * 3600 \
                + pd0_data.Sensor.time[:, 1] * 60 \
                + pd0_data.Sensor.time[:, 2] \
                + pd0_data.Sensor.time[:, 3] / 100
            
            # Compute the duration of each ensemble in seconds adjusting for lost data
            ens_delta_time = np.tile([np.nan], ens_time_sec.shape)
            idx_time = np.where(np.isnan(ens_time_sec) == False)[0]
            ens_delta_time[idx_time[1:]] = nandiff(ens_time_sec[idx_time])
            
            # Adjust for transects tha last past midnight
            idx_24hr = np.where(np.less(ens_delta_time, 0))[0]
            ens_delta_time[idx_24hr] = 24 * 3600 + ens_delta_time[idx_24hr]
            ens_delta_time = ens_delta_time.T
            
            # Start date and time
            idx = np.where(np.isnan(pd0_data.Sensor.time[:, 0]) == False)[0][0]
            start_year = int(pd0_data.Sensor.date[idx, 0])

            # StreamPro doesn't include y2k dates
            if start_year < 100:
                start_year = 2000 + int(pd0_data.Sensor.date_not_y2k[idx, 0])
                
            start_month = int(pd0_data.Sensor.date[idx, 1])
            start_day = int(pd0_data.Sensor.date[idx, 2])
            start_hour = int(pd0_data.Sensor.time[idx, 0])
            start_min = int(pd0_data.Sensor.time[idx, 1])
            start_sec = int(pd0_data.Sensor.time[idx, 2] + pd0_data.Sensor.time[idx, 3] / 100)
            start_micro = int(((pd0_data.Sensor.time[idx, 2] + pd0_data.Sensor.time[idx, 3] / 100) - start_sec) * 10**6)
            
            start_dt = datetime(start_year, start_month, start_day, start_hour, start_min, start_sec, start_micro,
                                tzinfo=timezone.utc)
            start_serial_time = start_dt.timestamp()
            start_date = datetime.strftime(datetime.utcfromtimestamp(start_serial_time), '%m/%d/%Y')
            
            # End data and time
            idx = np.where(np.isnan(pd0_data.Sensor.time[:, 0]) == False)[0][-1]
            end_year = int(pd0_data.Sensor.date[idx, 0])
            # StreamPro does not include Y@K dates
            if end_year < 100:
                end_year = 2000 + int(pd0_data.Sensor.date_not_y2k[idx, 0])
                
            end_month = int(pd0_data.Sensor.date[idx, 1])
            end_day = int(pd0_data.Sensor.date[idx, 2])
            end_hour = int(pd0_data.Sensor.time[idx, 0])
            end_min = int(pd0_data.Sensor.time[idx, 1])
            end_sec = int(pd0_data.Sensor.time[idx, 2] + pd0_data.Sensor.time[idx, 3] / 100)
            end_micro = int(((pd0_data.Sensor.time[idx, 2] + pd0_data.Sensor.time[idx, 3] / 100) - end_sec) * 10**6)
            
            end_dt = datetime(end_year, end_month, end_day, end_hour, end_min, end_sec, end_micro, tzinfo=timezone.utc)
            end_serial_time = end_dt.timestamp()
            
            # Create date/time object
            self.date_time = DateTime()
            self.date_time.populate_data(date_in=start_date,
                                         start_in=start_serial_time,
                                         end_in=end_serial_time,
                                         ens_dur_in=ens_delta_time)
            
            # Transect checked for use in discharge computation
            self.checked = mmt_transect.Checked

            # Create class for adcp information
            self.adcp = InstrumentData()
            self.adcp.populate_data(manufacturer='TRDI', raw_data=pd0_data, mmt_transect=mmt_transect, mmt=mmt)

    def sontek(self, rsdata, file_name):
        """Reads Matlab file produced by RiverSurveyor Live and populates the transect instance variables.

        Parameters
        ----------
        rsdata: MatSonTek
            Object of Matlab data from SonTek Matlab files
        file_name: str
            Name of SonTek Matlab file not including path.
        """

        self.file_name = os.path.basename(file_name)

        # ADCP instrument information
        # ---------------------------
        self.adcp = InstrumentData()
        if hasattr(rsdata.System, 'InstrumentModel'):
            self.adcp.populate_data(manufacturer='Nortek', raw_data=rsdata)
        else:
            self.adcp.populate_data(manufacturer='SonTek', raw_data=rsdata)

        # Depth
        # -----

        # Initialize depth data structure
        self.depths = DepthStructure()

        # Determine array rows and cols
        max_cells = rsdata.WaterTrack.Velocity.shape[0]
        num_ens = rsdata.WaterTrack.Velocity.shape[2]

        # Compute cell sizes and depths
        cell_size = rsdata.System.Cell_Size.reshape(1, num_ens)
        cell_size_all = np.tile(cell_size, (max_cells, 1))
        top_of_cells = rsdata.System.Cell_Start.reshape(1, num_ens)
        cell_depth = ((np.tile(np.arange(1, max_cells+1, 1).reshape(max_cells, 1), (1, num_ens)) - 0.5)
                      * cell_size_all) + np.tile(top_of_cells, (max_cells, 1))

        # Prepare bottom track depth variable
        depth = rsdata.BottomTrack.BT_Beam_Depth.T
        depth[depth == 0] = np.nan

        # Create depth object for bottom track beams
        self.depths.add_depth_object(depth_in=depth,
                                     source_in='BT',
                                     freq_in=rsdata.BottomTrack.BT_Frequency,
                                     draft_in=rsdata.Setup.sensorDepth,
                                     cell_depth_in=cell_depth,
                                     cell_size_in=cell_size_all)

        # Prepare vertical beam depth variable
        depth_vb = np.tile(np.nan, (1, cell_depth.shape[1]))
        depth_vb[0, :] = rsdata.BottomTrack.VB_Depth
        depth_vb[depth_vb == 0] = np.nan

        # Create depth object for vertical beam
        self.depths.add_depth_object(depth_in=depth_vb,
                                     source_in='VB',
                                     freq_in=np.array([rsdata.Transformation_Matrices.Frequency[1]] * depth.shape[-1]),
                                     draft_in=rsdata.Setup.sensorDepth,
                                     cell_depth_in=cell_depth,
                                     cell_size_in=cell_size_all)

        # Set depth reference
        if rsdata.Setup.depthReference < 0.5:
            self.depths.selected = 'vb_depths'
        else:
            self.depths.selected = 'bt_depths'

        # Water Velocity
        # --------------

        # Rearrange arrays for consistency with WaterData class
        vel = np.swapaxes(rsdata.WaterTrack.Velocity, 1, 0)
        snr = np.swapaxes(rsdata.System.SNR, 1, 0)
        if hasattr(rsdata.WaterTrack, 'Correlation'):
            corr = np.swapaxes(rsdata.WaterTrack.Correlation, 1, 0)
        else:
            corr = np.array([])

        # Correct SonTek difference velocity for error in earlier transformation matrices.
        if abs(rsdata.Transformation_Matrices.Matrix[3, 0, 0]) < 0.5:
            vel[3, :, :] = vel[3, :, :] * 2

        # Apply TRDI scaling to SonTek difference velocity to convert to a TRDI compatible error velocity
        vel[3, :, :] = vel[3, :, :] / ((2**0.5) * np.tan(np.deg2rad(25)))

        # Convert velocity reference from what was used in RiverSurveyor Live to None by adding the boat velocity
        # to the reported water velocity
        boat_vel = np.swapaxes(rsdata.Summary.Boat_Vel, 1, 0)
        vel[0, :, :] = vel[0, :, :] + boat_vel[0, :]
        vel[1, :, :] = vel[1, :, :] + boat_vel[1, :]

        ref_water = 'None'
        ref_coord = None

        # The initial coordinate system must be set to earth for early versions of RiverSurveyor firmware.
        # This implementation forces all versions to use the earth coordinate system.
        if rsdata.Setup.coordinateSystem == 0:
            # ref_coord = 'Beam'
            raise CoordError('Beam Coordinates are not supported for all RiverSuveyor firmware releases, ' +
                             'use Earth coordinates.')
        elif rsdata.Setup.coordinateSystem == 1:
            # ref_coord = 'Inst'
            raise CoordError('Instrument Coordinates are not supported for all RiverSuveyor firmware releases, ' +
                             'use Earth coordinates.')
        elif rsdata.Setup.coordinateSystem == 2:
            ref_coord = 'Earth'

        # Compute side lobe cutoff using Transmit Length information if availalbe, if not it is assumed to be equal
        # to 1/2 depth_cell_size_m. The percent method is use for the side lobe cutoff computation.
        sl_cutoff_percent = rsdata.Setup.extrapolation_dDiscardPercent
        sl_cutoff_number = rsdata.Setup.extrapolation_nDiscardCells
        if hasattr(rsdata.Summary, 'Transmit_Length'):
            sl_lag_effect_m = (rsdata.Summary.Transmit_Length
                               + self.depths.bt_depths.depth_cell_size_m[0, :]) / 2.0
        else:
            sl_lag_effect_m = np.copy(self.depths.bt_depths.depth_cell_size_m[0, :])
        sl_cutoff_type = 'Percent'
        cells_above_sl, sl_cutoff_m = TransectData.side_lobe_cutoff(depths=self.depths.bt_depths.depth_orig_m,
                                                                    draft=self.depths.bt_depths.draft_orig_m,
                                                                    cell_depth=self.depths.bt_depths.depth_cell_depth_m,
                                                                    sl_lag_effect=sl_lag_effect_m,
                                                                    slc_type=sl_cutoff_type,
                                                                    value=1 - sl_cutoff_percent / 100)
        # Determine water mode
        if len(corr) > 0:
            corr_nan = np.isnan(corr)
            number_of_nan = np.count_nonzero(corr_nan)
            if number_of_nan == 0:
                wm = 'HD'
            elif corr_nan.size == number_of_nan:
                wm = 'IC'
            else:
                wm = 'Variable'
        else:
            wm = 'Unknown'

        # Determine excluded distance (Similar to SonTek's screening distance)
        excluded_distance = rsdata.Setup.screeningDistance - rsdata.Setup.sensorDepth
        if excluded_distance < 0:
            excluded_distance = 0

        # Create water velocity object
        self.w_vel = WaterData()
        self.w_vel.populate_data(vel_in=vel,
                                 freq_in=rsdata.WaterTrack.WT_Frequency,
                                 coord_sys_in=ref_coord,
                                 nav_ref_in=ref_water,
                                 rssi_in=snr,
                                 rssi_units_in='SNR',
                                 excluded_dist_in=excluded_distance,
                                 cells_above_sl_in=cells_above_sl,
                                 sl_cutoff_per_in=sl_cutoff_percent,
                                 sl_cutoff_num_in=sl_cutoff_number,
                                 sl_cutoff_type_in=sl_cutoff_type,
                                 sl_lag_effect_in=sl_lag_effect_m,
                                 sl_cutoff_m=sl_cutoff_m,
                                 wm_in=wm,
                                 blank_in=excluded_distance,
                                 corr_in=corr)

        # Bottom Track
        # ------------
        self.boat_vel = BoatStructure()
        self.boat_vel.add_boat_object(source='SonTek',
                                      vel_in=np.swapaxes(rsdata.BottomTrack.BT_Vel, 1, 0),
                                      freq_in=rsdata.BottomTrack.BT_Frequency,
                                      coord_sys_in=ref_coord,
                                      nav_ref_in='BT')

        # GPS Data
        # --------
        self.gps = GPSData()
        if np.nansum(rsdata.GPS.GPS_Quality) > 0:

            if len(rsdata.RawGPSData.GgaLatitude.shape) > 1:

                self.gps.populate_data(raw_gga_utc=rsdata.RawGPSData.GgaUTC,
                                       raw_gga_lat=rsdata.RawGPSData.GgaLatitude,
                                       raw_gga_lon=rsdata.RawGPSData.GgaLongitude,
                                       raw_gga_alt=rsdata.RawGPSData.GgaAltitude,
                                       raw_gga_diff=rsdata.RawGPSData.GgaQuality,
                                       raw_gga_hdop=np.swapaxes(np.tile(rsdata.GPS.HDOP,
                                                                        (rsdata.RawGPSData.GgaLatitude.shape[1],
                                                                         1)), 1, 0),
                                       raw_gga_num_sats=np.swapaxes(np.tile(rsdata.GPS.Satellites,
                                                                            (rsdata.RawGPSData.GgaLatitude.shape[1],
                                                                             1)), 1, 0),
                                       raw_gga_delta_time=None,
                                       raw_vtg_course=rsdata.RawGPSData.VtgTmgTrue,
                                       raw_vtg_speed=rsdata.RawGPSData.VtgSogMPS,
                                       raw_vtg_delta_time=None,
                                       raw_vtg_mode_indicator=rsdata.RawGPSData.VtgMode,
                                       ext_gga_utc=rsdata.GPS.Utc,
                                       ext_gga_lat=rsdata.GPS.Latitude,
                                       ext_gga_lon=rsdata.GPS.Longitude,
                                       ext_gga_alt=rsdata.GPS.Altitude,
                                       ext_gga_diff=rsdata.GPS.GPS_Quality,
                                       ext_gga_hdop=rsdata.GPS.HDOP,
                                       ext_gga_num_sats=rsdata.GPS.Satellites,
                                       ext_vtg_course=np.tile(np.nan, rsdata.GPS.Latitude.shape),
                                       ext_vtg_speed=np.tile(np.nan, rsdata.GPS.Latitude.shape),
                                       gga_p_method='End',
                                       gga_v_method='End',
                                       vtg_method='Average')
            else:
                # Nortek data
                rows = rsdata.RawGPSData.GgaLatitude.shape[0]
                self.gps.populate_data(raw_gga_utc=rsdata.GPS.Utc.reshape(rows, 1),
                                       raw_gga_lat=rsdata.GPS.Latitude.reshape(rows, 1),
                                       raw_gga_lon=rsdata.GPS.Longitude.reshape(rows, 1),
                                       raw_gga_alt=rsdata.GPS.Altitude.reshape(rows, 1),
                                       raw_gga_diff=rsdata.GPS.GPS_Quality.reshape(rows, 1),
                                       raw_gga_hdop=rsdata.GPS.HDOP.reshape(rows, 1),
                                       raw_gga_num_sats=rsdata.GPS.Satellites.reshape(rows, 1),
                                       raw_gga_delta_time=None,
                                       raw_vtg_course=rsdata.RawGPSData.VtgTmgTrue.reshape(rows, 1),
                                       raw_vtg_speed=rsdata.RawGPSData.VtgSogMPS.reshape(rows, 1),
                                       raw_vtg_delta_time=None,
                                       raw_vtg_mode_indicator=rsdata.RawGPSData.VtgMode.reshape(rows, 1),
                                       ext_gga_utc=rsdata.GPS.Utc,
                                       ext_gga_lat=rsdata.GPS.Latitude,
                                       ext_gga_lon=rsdata.GPS.Longitude,
                                       ext_gga_alt=rsdata.GPS.Altitude,
                                       ext_gga_diff=rsdata.GPS.GPS_Quality,
                                       ext_gga_hdop=rsdata.GPS.HDOP,
                                       ext_gga_num_sats=rsdata.GPS.Satellites,
                                       ext_vtg_course=np.tile(np.nan, rsdata.GPS.Latitude.shape),
                                       ext_vtg_speed=np.tile(np.nan, rsdata.GPS.Latitude.shape),
                                       gga_p_method='End',
                                       gga_v_method='End',
                                       vtg_method='Average')

            self.boat_vel.add_boat_object(source='SonTek',
                                          vel_in=self.gps.gga_velocity_ens_mps,
                                          freq_in=None,
                                          coord_sys_in='Earth',
                                          nav_ref_in='GGA')

            self.boat_vel.add_boat_object(source='SonTek',
                                          vel_in=self.gps.vtg_velocity_ens_mps,
                                          freq_in=None,
                                          coord_sys_in='Earth',
                                          nav_ref_in='VTG')
        ref = 'BT'
        if rsdata.Setup.trackReference == 1:
            ref = 'BT'
        elif rsdata.Setup.trackReference == 2:
            ref = 'GGA'
        elif rsdata.Setup.trackReference == 3:
            ref = 'VTG'
        self.boat_vel.set_nav_reference(ref)

        # Edges
        # -----
        # Create edge object
        self.edges = Edges()
        self.edges.populate_data(rec_edge_method='Variable',
                                 vel_method='VectorProf')

        # Determine number of ensembles for each edge
        if rsdata.Setup.startEdge > 0.1:
            ensembles_right = np.nansum(rsdata.System.Step == 2)
            ensembles_left = np.nansum(rsdata.System.Step == 4)
            self.start_edge = 'Right'
            self.orig_start_edge = 'Right'
        else:
            ensembles_right = np.nansum(rsdata.System.Step == 4)
            ensembles_left = np.nansum(rsdata.System.Step == 2)
            self.start_edge = 'Left'
            self.orig_start_edge = 'Left'
        self.in_transect_idx = np.where(rsdata.System.Step == 3)[0]

        # Create left edge object
        edge_type = None
        if rsdata.Setup.Edges_0__Method == 2:
            edge_type = 'Triangular'
        elif rsdata.Setup.Edges_0__Method == 1:
            edge_type = 'Rectangular'
        elif rsdata.Setup.Edges_0__Method == 0:
            edge_type = 'User Q'
        if np.isnan(rsdata.Setup.Edges_0__EstimatedQ):
            user_discharge = None
        else:
            user_discharge = rsdata.Setup.Edges_0__EstimatedQ
        self.edges.left.populate_data(edge_type=edge_type,
                                      distance=rsdata.Setup.Edges_0__DistanceToBank,
                                      number_ensembles=ensembles_left,
                                      coefficient=None,
                                      user_discharge=user_discharge)

        # Create right edge object
        if rsdata.Setup.Edges_1__Method == 2:
            edge_type = 'Triangular'
        elif rsdata.Setup.Edges_1__Method == 1:
            edge_type = 'Rectangular'
        elif rsdata.Setup.Edges_1__Method == 0:
            edge_type = 'User Q'
        if np.isnan(rsdata.Setup.Edges_1__EstimatedQ):
            user_discharge = None
        else:
            user_discharge = rsdata.Setup.Edges_1__EstimatedQ
        self.edges.right.populate_data(edge_type=edge_type,
                                       distance=rsdata.Setup.Edges_1__DistanceToBank,
                                       number_ensembles=ensembles_right,
                                       coefficient=None,
                                       user_discharge=user_discharge)

        # Extrapolation
        # -------------
        top = ''
        bottom = ''

        # Top extrapolation
        if rsdata.Setup.extrapolation_Top_nFitType == 0:
            top = 'Constant'
        elif rsdata.Setup.extrapolation_Top_nFitType == 1:
            top = 'Power'
        elif rsdata.Setup.extrapolation_Top_nFitType == 2:
            top = '3-Point'

        # Bottom extrapolation
        if rsdata.Setup.extrapolation_Bottom_nFitType == 0:
            bottom = 'Constant'
        elif rsdata.Setup.extrapolation_Bottom_nFitType == 1:
            if rsdata.Setup.extrapolation_Bottom_nEntirePro > 1.1:
                bottom = 'No Slip'
            else:
                bottom = 'Power'

        # Create extrapolation object
        self.extrap = ExtrapData()
        self.extrap.populate_data(top=top,
                                  bot=bottom,
                                  exp=rsdata.Setup.extrapolation_Bottom_dExponent)

        # Sensor data
        # -----------
        self.sensors = Sensors()

        # Internal heading
        self.sensors.heading_deg.internal = HeadingData()

        # Check for firmware supporting G3 compass and associated data
        if hasattr(rsdata, 'Compass'):
            # TODO need to find older file that had 3 columns in Magnetic error to test and modify code
            mag_error = rsdata.Compass.Magnetic_error
            pitch_limit = np.array((rsdata.Compass.Maximum_Pitch, rsdata.Compass.Minimum_Pitch)).T
            roll_limit = np.array((rsdata.Compass.Maximum_Roll, rsdata.Compass.Minimum_Roll)).T
            if np.any(np.greater_equal(np.abs(pitch_limit), 90)) or np.any(np.greater_equal(np.abs(roll_limit), 90)):
                pitch_limit = None
                roll_limit = None
        else:
            mag_error = None
            pitch_limit = None
            roll_limit = None
        self.sensors.heading_deg.internal.populate_data(data_in=rsdata.System.Heading,
                                                        source_in='internal',
                                                        magvar=rsdata.Setup.magneticDeclination,
                                                        mag_error=mag_error,
                                                        pitch_limit=pitch_limit,
                                                        roll_limit=roll_limit)

        # External heading
        ext_heading = rsdata.System.GPS_Compass_Heading
        if np.nansum(np.abs(np.diff(ext_heading))) > 0:
            self.sensors.heading_deg.external = HeadingData()
            self.sensors.heading_deg.external.populate_data(data_in=ext_heading,
                                                            source_in='external',
                                                            magvar=rsdata.Setup.magneticDeclination,
                                                            align=rsdata.Setup.hdtHeadingCorrection)

        # Set selected reference
        if rsdata.Setup.headingSource > 1.1:
            self.sensors.heading_deg.selected = 'external'
        else:
            self.sensors.heading_deg.selected = 'internal'

        # Pitch and roll
        pitch = None
        roll = None
        if hasattr(rsdata, 'Compass'):
            pitch = rsdata.Compass.Pitch
            roll = rsdata.Compass.Roll
        elif hasattr(rsdata.System, 'Pitch'):
            pitch = rsdata.System.Pitch
            roll = rsdata.System.Roll
        self.sensors.pitch_deg.internal = SensorData()
        self.sensors.pitch_deg.internal.populate_data(data_in=pitch, source_in='internal')
        self.sensors.pitch_deg.selected = 'internal'
        self.sensors.roll_deg.internal = SensorData()
        self.sensors.roll_deg.internal.populate_data(data_in=roll, source_in='internal')
        self.sensors.roll_deg.selected = 'internal'

        # Temperature
        if rsdata.System.Units.Temperature.find('C') >= 0:
            temperature = rsdata.System.Temperature
        else:
            temperature = (5. / 9.) * (rsdata.System.Temperature - 32)
        self.sensors.temperature_deg_c.internal = SensorData()
        self.sensors.temperature_deg_c.internal.populate_data(data_in=temperature, source_in='internal')
        self.sensors.temperature_deg_c.selected = 'internal'

        # Salinity
        self.sensors.salinity_ppt.user = SensorData()
        self.sensors.salinity_ppt.user.populate_data(data_in=rsdata.Setup.userSalinity, source_in='Manual')
        self.sensors.salinity_ppt.selected = 'user'
        # Matlab notes indicated that an internal sensor needed to be created for compatibility with
        # future computations
        self.sensors.salinity_ppt.internal = SensorData()
        self.sensors.salinity_ppt.internal.populate_data(data_in=rsdata.Setup.userSalinity, source_in='Manual')

        # Speed of sound
        # Not provided in SonTek data but is computed from equation used in TRDI BBSS.
        speed_of_sound = Sensors.speed_of_sound(temperature=temperature, salinity=rsdata.Setup.userSalinity)
        self.sensors.speed_of_sound_mps.internal = SensorData()
        self.sensors.speed_of_sound_mps.internal.populate_data(data_in=speed_of_sound, source_in='QRev')
        # Set selected salinity
        self.sensors.speed_of_sound_mps.selected = 'internal'

        # Ensemble times
        ensemble_delta_time = np.append([0], np.diff(rsdata.System.Time))
        # TODO potentially add popup message when there are missing ensembles. Matlab did that.

        # idx_missing = np.where(ensemble_delta_time > 1.5)
        # if len(idx_missing[0]) > 0:
        #     number_missing = np.sum(ensemble_delta_time[idx_missing]) - len(idx_missing)
        #     error_str = self.file_name + ' is missing ' + str(number_missing) + ' samples'

        start_serial_time = rsdata.System.Time[0] + ((30 * 365) + 7) * 24 * 60 * 60
        end_serial_time = rsdata.System.Time[-1] + ((30 * 365) + 7) * 24 * 60 * 60
        meas_date = datetime.strftime(datetime.fromtimestamp(start_serial_time), '%m/%d/%Y')
        self.date_time = DateTime()
        self.date_time.populate_data(date_in=meas_date,
                                     start_in=start_serial_time,
                                     end_in=end_serial_time,
                                     ens_dur_in=ensemble_delta_time)

        # Transect checked for use in discharge computations
        self.checked = True

        # Set composite depths as this is the only option in RiverSurveyor Live
        self.depths.composite_depths(transect=self, setting="On")

    @staticmethod
    def qrev_mat_in(meas_struct):
        """Processes the Matlab data structure to obtain a list of TransectData objects containing transect
           data from the Matlab data structure.

       Parameters
       ----------
       meas_struct: mat_struct
           Matlab data structure obtained from sio.loadmat

       Returns
       -------
       transects: list
           List of TransectData objects
       """

        transects = []
        if hasattr(meas_struct, 'transects'):
            # If only one transect the data are not a list or array of transects
            try:
                if len(meas_struct.transects) > 0:
                    for transect in meas_struct.transects:
                        trans = TransectData()
                        trans.populate_from_qrev_mat(transect)
                        transects.append(trans)
            except TypeError:
                trans = TransectData()
                trans.populate_from_qrev_mat(meas_struct.transects)
                transects.append(trans)

        return transects

    def populate_from_qrev_mat(self, transect):
        """Populates the object using data from previously saved QRev Matlab file.

        Parameters
        ----------
        transect: mat_struct
           Matlab data structure obtained from sio.loadmat
        """

        self.adcp = InstrumentData()
        self.adcp.populate_from_qrev_mat(transect)
        self.file_name = os.path.basename(transect.fileName)
        self.w_vel = WaterData()
        self.w_vel.populate_from_qrev_mat(transect)
        self.boat_vel = BoatStructure()
        self.boat_vel.populate_from_qrev_mat(transect)
        self.gps = GPSData()
        self.gps.populate_from_qrev_mat(transect)
        self.sensors = Sensors()
        self.sensors.populate_from_qrev_mat(transect)
        self.depths = DepthStructure()
        self.depths.populate_from_qrev_mat(transect)
        self.edges = Edges()
        self.edges.populate_from_qrev_mat(transect)
        self.extrap = ExtrapData()
        self.extrap.populate_from_qrev_mat(transect)
        self.start_edge = transect.startEdge
        if hasattr(transect, 'orig_start_edge'):
            self.orig_start_edge = transect.orig_start_edge
        else:
            self.orig_start_edge = transect.startEdge
        self.date_time = DateTime()
        self.date_time.populate_from_qrev_mat(transect)
        self.checked = bool(transect.checked)
        if type(transect.inTransectIdx) is int:
            self.in_transect_idx = np.array([transect.inTransectIdx - 1])
        else:
            self.in_transect_idx = transect.inTransectIdx - 1

    @staticmethod
    def compute_cell_data(pd0):
        
        # Number of ensembles
        num_ens = np.array(pd0.Wt.vel_mps).shape[-1]

        # Retrieve and compute cell information
        reg_cell_size = pd0.Cfg.ws_cm / 100
        reg_cell_size[reg_cell_size == 0] = np.nan
        dist_cell_1_m = pd0.Cfg.dist_bin1_cm / 100
        num_reg_cells = pd0.Wt.vel_mps.shape[1]
        
        # Surf data are to accommodate RiverRay and RiverPro.  pd0_read sets these
        # values to nan when reading Rio Grande or StreamPro data
        no_surf_cells = pd0.Surface.no_cells
        no_surf_cells[np.isnan(no_surf_cells)] = 0
        max_surf_cells = np.nanmax(no_surf_cells)
        surf_cell_size = pd0.Surface.cell_size_cm / 100
        surf_cell_dist = pd0.Surface.dist_bin1_cm / 100
        
        # Compute maximum number of cells
        max_cells = int(max_surf_cells + num_reg_cells)
        
        # Combine cell size and cell range from transducer for both
        # surface and regular cells
        cell_depth = np.tile(np.nan, (max_cells, num_ens))
        cell_size_all = np.tile(np.nan, (max_cells, num_ens))
        for i in range(num_ens):
            # Determine number of cells to be treated as regular cells
            if np.nanmax(no_surf_cells) > 0:
                
                num_reg_cells = max_cells - no_surf_cells[i]
            else:
                num_reg_cells = max_cells

            # Compute cell depth
            if no_surf_cells[i] > 1e-5:
                cell_depth[:int(no_surf_cells[i]), i] = surf_cell_dist[i] + \
                                                        np.arange(0, (no_surf_cells[i]-1) * surf_cell_size[i]+0.001,
                                                                  surf_cell_size[i])
                cell_depth[int(no_surf_cells[i]):, i] = cell_depth[int(no_surf_cells[i]-1), i] \
                    + (.5 * surf_cell_size[i] + 0.5 * reg_cell_size[i]) \
                    + np.arange(0, (num_reg_cells-1) * reg_cell_size[i]+0.001, reg_cell_size[i])
                cell_size_all[0:int(no_surf_cells[i]), i] = np.repeat(surf_cell_size[i], int(no_surf_cells[i]))
                cell_size_all[int(no_surf_cells[i]):, i] = np.repeat(reg_cell_size[i], int(num_reg_cells))
            else:
                cell_depth[:int(num_reg_cells), i] = dist_cell_1_m[i] + \
                                                     np.linspace(0, int(num_reg_cells) - 1,
                                                                 int(num_reg_cells)) * reg_cell_size[i]
                cell_size_all[:, i] = np.repeat(reg_cell_size[i], num_reg_cells)

        # Firmware is used to ID RiverRay data with variable modes and lags
        firmware = str(pd0.Inst.firm_ver[0])
            
        # Compute sl_lag_effect
        lag = pd0.Cfg.lag_cm / 100
        if firmware[0:2] == '44' or firmware[0:2] == '56':
            lag_near_bottom = np.array(pd0.Cfg.lag_near_bottom)
            lag_near_bottom[lag_near_bottom == np.nan] = 0
            lag[lag_near_bottom != 0] = 0
            
        pulse_len = pd0.Cfg.xmit_pulse_cm / 100
        sl_lag_effect_m = (lag + pulse_len + reg_cell_size) / 2
        sl_cutoff_per = (1 - (cosd(pd0.Inst.beam_ang[0]))) * 100
            
        return cell_size_all, cell_depth, sl_cutoff_per, sl_lag_effect_m

    def change_q_ensembles(self, proc_method):
        """Sets in_transect_idx to all ensembles, except in the case of SonTek data
        where RSL processing is applied.
        
        Parameters
        ----------
        proc_method: str
            Processing method (WR2, RSL, QRev)
        """
        
        if proc_method == 'RSL':
            num_ens = self.boat_vel.bt_vel.u_processed_mps.shape[1]
            # Determine number of ensembles for each edge
            if self.start_edge == 'Right':
                self.in_transect_idx = np.arange(self.edges.right.num_ens_2_avg, num_ens-self.edges.left.num_ens_2_avg)
            else:
                self.in_transect_idx = np.arange(self.edges.left.num_ens_2_avg, num_ens-self.edges.right.num_ens_2_avg)
        else:
            self.in_transect_idx = np.arange(0, self.boat_vel.bt_vel.u_processed_mps.shape[0])
        
    def change_coord_sys(self, new_coord_sys):
        """Changes the coordinate system of the water and boat data.

        Current implementation only allows changes for original to higher order coordinate
        systems: Beam - Inst - Ship - Earth.
        
        Parameters
        ----------
        new_coord_sys: str
            Name of new coordinate system (Beam, Int, Ship, Earth)
        """
        self.w_vel.change_coord_sys(new_coord_sys, self.sensors, self.adcp)
        self.boat_vel.change_coord_sys(new_coord_sys, self.sensors, self.adcp)
        
    def change_nav_reference(self, update, new_nav_ref):
        """Method to set the navigation reference for the water data.
        
        Parameters
        ----------
        update: bool
            Setting to determine if water data should be updated.
        new_nav_ref: str
            New navigation reference (bt_vel, gga_vel, vtg_vel)
        """
        
        self.boat_vel.change_nav_reference(reference=new_nav_ref, transect=self)
        
        if update:
            self.update_water()
            
    def change_mag_var(self, magvar):
        """Change magnetic variation.
        
        Parameters
        ----------
        magvar: float
            Magnetic variation in degrees.
        """
        
        # Update object
        if self.sensors.heading_deg.external is not None:
            self.sensors.heading_deg.external.set_mag_var(magvar, 'external')
        
        if self.sensors.heading_deg.selected == 'internal':
            heading_selected = getattr(self.sensors.heading_deg, self.sensors.heading_deg.selected)
            old_magvar = heading_selected.mag_var_deg
            magvar_change = magvar - old_magvar
            heading_selected.set_mag_var(magvar, 'internal')
            self.boat_vel.bt_vel.change_heading(magvar_change)
            self.w_vel.change_heading(self.boat_vel, magvar_change)
        else:
            self.sensors.heading_deg.internal.set_mag_var(magvar, 'internal')
        
        # self.update_water()
        
    def change_offset(self, h_offset):
        """Change the heading offset (alignment correction). Only affects external heading.
        
        Parameters
        ----------
        h_offset: float
            Heading offset in degrees
        """
        self.sensors.heading_deg.internal.set_align_correction(h_offset, 'internal')
        
        if self.sensors.heading_deg.selected == 'external':
            old = getattr(self.sensors.heading_deg, self.sensors.heading_deg.selected)
            old_offset = old.align_correction_deg
            offset_change = h_offset - old_offset
            self.boat_vel.bt_vel.change_heading(offset_change)
            self.w_vel.change_heading(self.boat_vel, offset_change)

        if self.sensors.heading_deg.external is not None:
            self.sensors.heading_deg.external.set_align_correction(h_offset, 'external')
        
        self.update_water()
        
    def change_heading_source(self, h_source):
        """Changes heading source (internal or external).

        Parameters
        ----------
        h_source: str
            Heading source (internal or external)
        """

        new_heading_selection = getattr(self.sensors.heading_deg, h_source)
        if h_source is not None:
            old_heading_selection = getattr(self.sensors.heading_deg, self.sensors.heading_deg.selected)
            old_heading = old_heading_selection.data
            new_heading = new_heading_selection.data
            heading_change = new_heading - old_heading
            self.sensors.heading_deg.set_selected(h_source)
            self.boat_vel.bt_vel.change_heading(heading_change)
            self.w_vel.change_heading(self.boat_vel, heading_change)
            
        self.update_water()
            
    def update_water(self):
        """Method called from set_nav_reference, boat_interpolation and boat filters
        to ensure that changes in boatvel are reflected in the water data"""

        self.w_vel.set_nav_reference(self.boat_vel)
        
        # Reapply water filters and interpolations
        # Note wt_filters calls apply_filter which automatically calls
        # apply_interpolation so both filters and interpolations
        # are applied with this one call

        self.w_vel.apply_filter(transect=self)
        self.w_vel.apply_interpolation(transect=self)

    @staticmethod
    def side_lobe_cutoff(depths, draft, cell_depth, sl_lag_effect, slc_type='Percent', value=None):
        """Computes side lobe cutoff.

        The side lobe cutoff is based on the beam angle and is computed to
        ensure that the bin and any lag beyond the actual bin cutoff is
        above the side lobe cutoff.

        Parameters
        ----------
        depths: np.array
            Bottom track (all 4 beams) and vertical beam depths for each ensemble, in m.
        draft: float
            Draft of transducers, in m.
        cell_depth: np.array
            Depth to the centerline of each depth cell, in m.
        sl_lag_effect: np.array
            The extra depth below the last depth cell that must be above the side lobe cutoff, in m.
        slc_type: str
            Method used for side lobe cutoff computation.
        value: float
            Value used in specified method to use for side lobe cutoff computation.
        """

        # Compute minimum depths for each ensemble
        min_depths = np.nanmin(depths, 0)
        
        # Compute range from transducer
        range_from_xducer = min_depths - draft
        
        # Adjust for transducer angle
        coeff = None
        if slc_type == 'Percent':
            coeff = value
        elif slc_type == 'Angle':
            coeff = np.cos(np.deg2rad(value))
        
        # Compute sidelobe cutoff to centerline
        cutoff = np.array(range_from_xducer * coeff - sl_lag_effect + draft)
        
        # Compute boolean side lobe cutoff matrix
        cells_above_sl = np.less(cell_depth, cutoff)
        return cells_above_sl, cutoff
        
    def boat_interpolations(self, update, target, method=None):
        """Coordinates boat velocity interpolations.
        
        Parameters
        ----------
        update: bool
            Setting to control if water data are updated.
        target: str
            Boat velocity reference (BT or GPS)
        method: str
            Type of interpolation
        """

        # Interpolate bottom track data
        if target == 'BT':
            self.boat_vel.bt_vel.apply_interpolation(transect=self, interpolation_method=method)
            
        if target == 'GPS':
            # Interpolate GGA data
            vel = getattr(self.boat_vel, 'gga_vel')
            if vel is not None:
                self.boat_vel.gga_vel.apply_interpolation(transect=self, interpolation_method=method)
            # Interpolate VTG data
            vel = getattr(self.boat_vel, 'vtg_vel')
            if vel is not None:
                self.boat_vel.vtg_vel.apply_interpolation(transect=self, interpolation_method=method)

        # Apply composite tracks setting
        self.composite_tracks(update=False)
        
        # Update water to reflect changes in boat_vel
        if update:
            self.update_water()
            
    def composite_tracks(self, update, setting=None):
        """Coordinate application of composite tracks.
        
        Parameters
        ----------
        update: bool
            Setting to control if water data are updated.
        setting: str
            Sets composite tracks ("On" or "Off").
        """
        
        # Determine if setting is specified
        if setting is None:
            # Process transect using saved setting
            self.boat_vel.composite_tracks(transect=self)
        else:
            # Process transect usin new setting
            self.boat_vel.composite_tracks(transect=self, setting=setting)
            
        # Update water data to reflect changes in boatvel
        if update:
            self.update_water()
            
    def boat_filters(self, update, **kwargs):
        """Coordinates application of boat filters to bottom track data
        
        Parameters
        ----------
        update: bool
            Setting to control if water data are updated.
        **kwargs: dict
            beam: int
                Setting for beam filter (3, 4, -1)
            difference: str
                Setting for difference velocity filter (Auto, Manual, Off)
            difference_threshold: float
                Threshold for manual setting
            vertical: str
                Setting for vertical velocity filter (Auto, Manual, Off)
            vertical_threshold: float
                Threshold for manual setting
            other: bool
                Setting to other filter
        """
        
        # Apply filter to transect
        self.boat_vel.bt_vel.apply_filter(self, **kwargs)
        
        if self.boat_vel.selected == 'bt_vel' and update:
            self.update_water()
            
    def gps_filters(self, update, **kwargs):
        """Coordinate filters for GPS based boat velocities
        
        Parameters
        ----------
        update: bool
            Setting to control if water data are updated.
        **kwargs: dict
            differential: str
                Differential filter setting (1, 2, 4)
            altitude: str
                New setting for altitude filter (Off, Manual, Auto)
            altitude_threshold: float
                Threshold provide by user for manual altitude setting
            hdop: str
                Filter setting (On, off, Auto)
            hdop_max_threshold: float
                Maximum HDOP threshold
            hdop_change_threshold: float
                HDOP change threshold
            other: bool
                Other filter typically a smooth.
        """
        
        if self.boat_vel.gga_vel is not None:
            self.boat_vel.gga_vel.apply_gps_filter(self, **kwargs)
        if self.boat_vel.vtg_vel is not None:
            self.boat_vel.vtg_vel.apply_gps_filter(self, **kwargs)
            
        if (self.boat_vel.selected == 'VTG' or self.boat_vel.selected == 'GGA') and update == True:
            self.update_water()
            
    def set_depth_reference(self, update, setting):
        """Coordinates setting the depth reference.
        
        Parameters
        ----------
        update: bool
            Determines if associated data should be updated
        setting: str
            Depth reference (bt_depths, vb_depths, ds_depths)
        """
        
        self.depths.selected = setting

        if update:
            self.process_depths(update)
            self.w_vel.adjust_side_lobe(self)
            
    def apply_averaging_method(self, setting):
        """Method to apply the selected averaging method to the BT team depths to achieve a single
        average depth.  It is only applicable to the multiple beams used for BT, not VB or DS.
        
        Input:
        setting: averaging method (IDW, Simple)
        """
        
        self.depths.bt_depths.compute_avg_bt_depth(setting)
        
        self.process_depths(update=False)
            
    def process_depths(self, update=False, filter_method=None, interpolation_method=None, composite_setting=None,
                       avg_method=None, valid_method=None):
        """Method applies filter, composite, and interpolation settings to  depth objects
        so that all are updated using the same filter and interpolation settings.
        
        Parameters
        ----------
        update: bool
            Determines if water data should be updated.
        filter_method: str
            Filter method to be used (None, Smooth, TRDI).
        interpolation_method: str
            Interpolation method to be used (None, HoldLast, Smooth, Linear).
        composite_setting: str
            Specifies use of composite depths ("On" or "Off").
        avg_method: str
            Defines averaging method: "Simple", "IDW", only applicable to bottom track.
        valid_method:
            Defines method to determine if depth is valid (QRev or TRDI).
        """
        
        # Get current settings
        depth_data = getattr(self.depths, self.depths.selected)
        if filter_method is None:
            filter_method = depth_data.filter_type

        if interpolation_method is None:
            interpolation_method = depth_data.interp_type

        if composite_setting is None:
            composite_setting = self.depths.composite

        if avg_method is None:
            avg_method = self.depths.bt_depths.avg_method

        if valid_method is None:
            valid_method = self.depths.bt_depths.valid_data_method

        self.depths.bt_depths.valid_data_method = valid_method
        self.depths.bt_depths.avg_method = avg_method
        self.depths.depth_filter(transect=self, filter_method=filter_method)
        self.depths.depth_interpolation(transect=self, method=interpolation_method)
        self.depths.composite_depths(transect=self, setting=composite_setting)
        self.w_vel.adjust_side_lobe(transect=self)
        
        if update:
            self.update_water()

    def change_draft(self, draft_in):
        """Changes the draft for the specified transects and selected depth.

        Parameters
        ----------
        draft_in: float
            New draft value in m
        """
        
        if self.depths.vb_depths is not None:
            self.depths.vb_depths.change_draft(draft_in)
        if self.depths.bt_depths is not None:
            self.depths.bt_depths.change_draft(draft_in)

    def change_sos(self, parameter=None, salinity=None, temperature=None, selected=None, speed=None):
        """Coordinates changing the speed of sounc.

        Parameters
        ----------
        parameter: str
            Speed of sound parameter to be changed ('temperatureSrc', 'temperature', 'salinity', 'sosSrc')
        salinity: float
            Salinity in ppt
        temperature: float
            Temperature in deg C
        selected: str
            Selected speed of sound ('internal', 'computed', 'user') or temperature ('internal', 'user')
        speed: float
            Manually supplied speed of sound for 'user' source
        """

        if parameter == 'temperatureSrc':

            temperature_internal = getattr(self.sensors.temperature_deg_c, 'internal')
            if selected == 'user':
                if self.sensors.temperature_deg_c.user is None:
                    self.sensors.temperature_deg_c.user = SensorData()
                ens_temperature = np.tile(temperature, temperature_internal.data.shape)

                self.sensors.temperature_deg_c.user.change_data(data_in=ens_temperature)
                self.sensors.temperature_deg_c.user.set_source(source_in='Manual Input')

            # Set the temperature data to the selected source
            self.sensors.temperature_deg_c.set_selected(selected_name=selected)
            # Update the speed of sound
            self.update_sos()

        elif parameter == 'temperature':
            adcp_temp = self.sensors.temperature_deg_c.internal.populate_data
            new_user_temperature = np.tile(temperature, adcp_temp.shape)
            self.sensors.temperature_deg_c.user.change_data(data_in=new_user_temperature)
            self.sensors.temperature_deg_c.user.set_source(source_in='Manual Input')
            # Set the temperature data to the selected source
            self.sensors.temperature_deg_c.set_selected(selected_name='user')
            # Update the speed of sound
            self.update_sos()

        elif parameter == 'salinity':
            if salinity is not None:
                self.sensors.salinity_ppt.user.change_data(data_in=salinity)
                if type(self.sensors.salinity_ppt.internal.data) is float:
                    sos_internal = self.sensors.salinity_ppt.internal.data
                else:
                    sos_internal = self.sensors.salinity_ppt.internal.data[0]
                if self.sensors.salinity_ppt.user.data == sos_internal:
                    self.sensors.salinity_ppt.set_selected(selected_name='internal')
                else:
                    self.sensors.salinity_ppt.set_selected(selected_name='user')
                self.update_sos()

        elif parameter == 'sosSrc':
            if selected == 'internal':
                self.update_sos()
            elif selected == 'user':
                self.update_sos(speed=speed, selected='user', source='Manual Input')

    def update_sos(self, selected=None, source=None, speed=None):
        """Sets a new specified speed of sound.

        Parameters
        ----------
        self: obj
            Object of TransectData
        selected: str
             Selected speed of sound ('internal', 'computed', 'user')
        source: str
            Source of speed of sound (Computer, Calculated)
        speed: float
            Manually supplied speed of sound for 'user' source
        """

        # Get current speed of sound
        sos_selected = getattr(self.sensors.speed_of_sound_mps, self.sensors.speed_of_sound_mps.selected)
        old_sos = sos_selected.data
        new_sos = None

        # Manual input for speed of sound
        if selected == 'user' and source == 'Manual Input':
            self.sensors.speed_of_sound_mps.set_selected(selected_name=selected)
            self.sensors.speed_of_sound_mps.user = SensorData()
            self.sensors.speed_of_sound_mps.user.populate_data(speed, source)

        # If called with no input set source to internal and determine whether computed or calculated based on
        # availability of user supplied temperature or salinity
        elif selected is None and source is None:
            self.sensors.speed_of_sound_mps.set_selected('internal')
            # If temperature or salinity is set by the user the speed of sound is computed otherwise it is consider
            # calculated by the ADCP.
            if (self.sensors.temperature_deg_c.selected == 'user') or (self.sensors.salinity_ppt.selected == 'user'):
                self.sensors.speed_of_sound_mps.internal.set_source('Computed')
            else:
                self.sensors.speed_of_sound_mps.internal.set_source('Calculated')

        # Determine new speed of sound
        if self.sensors.speed_of_sound_mps.selected == 'internal':

            if self.sensors.speed_of_sound_mps.internal.source == 'Calculated':
                # Internal: Calculated
                new_sos = self.sensors.speed_of_sound_mps.internal.data_orig
                self.sensors.speed_of_sound_mps.internal.change_data(data_in=new_sos)
                # Change temperature and salinity selected to internal
                self.sensors.temperature_deg_c.set_selected(selected_name='internal')
                self.sensors.salinity_ppt.set_selected(selected_name='internal')
            else:
                # Internal: Computed
                temperature_selected = getattr(self.sensors.temperature_deg_c, self.sensors.temperature_deg_c.selected)
                temperature = temperature_selected.data
                salinity_selected = getattr(self.sensors.salinity_ppt, self.sensors.salinity_ppt.selected)
                salinity = salinity_selected.data
                new_sos = Sensors.speed_of_sound(temperature=temperature, salinity=salinity)
                self.sensors.speed_of_sound_mps.internal.change_data(data_in=new_sos)
        else:
            if speed is not None:
                new_sos = np.tile(speed, len(self.sensors.speed_of_sound_mps.internal.data_orig))
                self.sensors.speed_of_sound_mps.user.change_data(data_in=new_sos)

        self.apply_sos_change(old_sos=old_sos, new_sos=new_sos)

    def apply_sos_change(self, old_sos, new_sos):
        """Computes the ratio and calls methods in WaterData and BoatData to apply change.

        Parameters
        ----------
        old_sos: float
            Speed of sound on which the current data are based, in m/s
        new_sos: float
            Speed of sound on which the data need to be based, in m/s
        """

        ratio = new_sos / old_sos

        # RiverRay horizontal velocities are not affected by changes in speed of sound
        if self.adcp.model != 'RiverRay':
            # Apply speed of sound change to water and boat data
            self.w_vel.sos_correction(ratio=ratio)
            self.boat_vel.bt_vel.sos_correction(ratio=ratio)
        # Correct depths
        self.depths.sos_correction(ratio=ratio)

    @staticmethod
    def raw_valid_data(transect):
        """Determines ensembles and cells with no interpolated water or boat data.

        For valid water track cells both non-interpolated valid water data and
        boat velocity data must be available. Interpolated depths are allowed.

        For valid ensembles water, boat, and depth data must all be non-interpolated.

        Parameters
        ----------
        transect: TransectData
            Object of TransectData

        Returns
        -------
        raw_valid_ens: np.array(bool)
            Boolean array identifying raw valid ensembles.
        raw_valid_depth_cells: np.array(bool)
            Boolean array identifying raw valid depth cells.
        """

        in_transect_idx = transect.in_transect_idx

        # Determine valid water track ensembles based on water track and navigation data.
        boat_vel_select = getattr(transect.boat_vel, transect.boat_vel.selected)
        if boat_vel_select is not None and np.nansum(np.logical_not(np.isnan(boat_vel_select.u_processed_mps))) > 0:
            valid_nav = boat_vel_select.valid_data[0, in_transect_idx]
        else:
            valid_nav = np.tile(False, in_transect_idx.shape[0])

        valid_wt = np.copy(transect.w_vel.valid_data[0, :, in_transect_idx])
        valid_wt_ens = np.any(valid_wt, 1)

        # Determine valid depths
        depths_select = getattr(transect.depths, transect.depths.selected)
        if transect.depths.composite:
            valid_depth = np.tile(True, (depths_select.depth_source_ens.shape[0]))
            idx_na = np.where(depths_select.depth_source_ens[in_transect_idx] == 'NA')[0]
            if len(idx_na) > 0:
                valid_depth[idx_na] = False
            interpolated_depth_idx = np.where(depths_select.depth_source_ens[in_transect_idx] == 'IN')[0]
            if len(interpolated_depth_idx) > 0:
                valid_depth[interpolated_depth_idx] = False
        else:
            valid_depth = depths_select.valid_data[in_transect_idx]
            idx = np.where(np.isnan(depths_select.depth_processed_m[in_transect_idx]))[0]
            if len(idx) > 0:
                valid_depth[idx] = False

        # Determine valid ensembles based on all data
        valid_ens = np.all(np.vstack((valid_nav, valid_wt_ens, valid_depth)), 0)

        return valid_ens, valid_wt.T

    @staticmethod
    def compute_gps_lag(transect):
        """Computes the lag between bottom track and GGA and/or VTG using an autocorrelation method.

        Parameters
        ----------
        transect: TransectData
            Object of TransectData

        Returns
        -------
        lag_gga: float
            Lag in seconds betweeen bottom track and gga
        lag_vtg: float
            Lag in seconds between bottom track and vtg
        """

        # Intialize returns
        lag_gga = None
        lag_vtg = None

        bt_speed = np.sqrt(transect.boat_vel.bt_vel.u_processed_mps ** 2
                           + transect.boat_vel.bt_vel.v_processed_mps ** 2)

        avg_ens_dur = np.nanmean(transect.date_time.ens_duration_sec)

        # Compute lag for gga, if available
        if transect.boat_vel.gga_vel is not None:
            gga_speed = np.sqrt(transect.boat_vel.gga_vel.u_processed_mps**2
                             + transect.boat_vel.gga_vel.v_processed_mps**2)

            # Compute lag if both bottom track and gga have valid data
            valid_data = np.all(np.logical_not(np.isnan(np.vstack((bt_speed, gga_speed)))), axis=0)
            if np.sometrue(valid_data):
                # Compute lag
                lag_gga = (np.count_nonzero(valid_data)
                          - np.argmax(signal.correlate(bt_speed[valid_data], gga_speed[valid_data])) - 1) * avg_ens_dur
            else:
                lag_gga = None

        # Compute lag for vtg, if available
        if transect.boat_vel.vtg_vel is not None:
            vtg_speed = np.sqrt(transect.boat_vel.vtg_vel.u_processed_mps**2
                             + transect.boat_vel.vtg_vel.v_processed_mps**2)

            # Compute lag if both bottom track and gga have valid data
            valid_data = np.all(np.logical_not(np.isnan(np.vstack((bt_speed, vtg_speed)))), axis=0)
            if np.sometrue(valid_data):
                # Compute lag
                lag_vtg = (np.count_nonzero(valid_data)
                           - np.argmax(signal.correlate(bt_speed[valid_data], vtg_speed[valid_data])) - 1) * avg_ens_dur
            else:
                lag_vtg = None

        return lag_gga, lag_vtg

    @staticmethod
    def compute_gps_lag_fft(transect):
        """Computes the lag between bottom track and GGA and/or VTG using fft method.

        Parameters
        ----------
        transect: TransectData
            Object of TransectData

        Returns
        -------
        lag_gga: float
            Lag in seconds betweeen bottom track and gga
        lag_vtg: float
            Lag in seconds between bottom track and vtg
        """
        lag_gga = None
        lag_vtg = None

        bt_speed = np.sqrt(transect.boat_vel.bt_vel.u_processed_mps ** 2
                           + transect.boat_vel.bt_vel.v_processed_mps ** 2)

        avg_ens_dur = np.nanmean(transect.date_time.ens_duration_sec)
        if transect.boat_vel.gga_vel is not None:
            gga_speed = np.sqrt(transect.boat_vel.gga_vel.u_processed_mps**2
                             + transect.boat_vel.gga_vel.v_processed_mps**2)
            valid_data = np.all(np.logical_not(np.isnan(np.vstack((bt_speed, gga_speed)))), axis=0)
            b = fftpack.fft(bt_speed[valid_data])
            g = fftpack.fft(gga_speed[valid_data])
            br = -b.conjugat()
            lag_gga = np.argmax(np.abs(fftpack.ifft(br*g)))

        if transect.boat_vel.vtg_vel is not None:
            vtg_speed = np.sqrt(transect.boat_vel.vtg_vel.u_processed_mps**2
                             + transect.boat_vel.vtg_vel.v_processed_mps**2)
            valid_data = np.all(np.logical_not(np.isnan(np.vstack((bt_speed, vtg_speed)))), axis=0)
            b = fftpack.fft(bt_speed[valid_data])
            g = fftpack.fft(vtg_speed[valid_data])
            br = -b.conjugat()
            lag_vtg = np.argmax(np.abs(fftpack.ifft(br * g)))

        return lag_gga, lag_vtg

    @staticmethod
    def compute_gps_bt(transect, gps_ref='gga_vel'):
        """Computes properties describing the difference between bottom track and the specified GPS reference.

        Parameters
        ----------
        transect: TransectData
            Object of TransectData
        gps_ref: str
            GPS referenced to be used in computation (gga_vel or vtg_vel)

        Returns
        -------
        gps_bt: dict
            course: float
                Difference in course computed from gps and bt, in degrees
            ratio: float
                Ratio of final distance made good for bt and gps (bt dmg / gps dmg)
            dir: float
                Direction of vector from end of GPS track to end of bottom track
            mag: float
                Length of vector from end of GPS track to end of bottom track
        """

        gps_bt = dict()
        gps_vel = getattr(transect.boat_vel, gps_ref)
        if gps_vel is not None and \
                1 < np.sum(np.logical_not(np.isnan(gps_vel.u_processed_mps))) and \
                1 < np.sum(np.logical_not(np.isnan(transect.boat_vel.bt_vel.u_processed_mps))):
            # Data prep
            bt_track = BoatStructure.compute_boat_track(transect, ref='bt_vel')
            bt_course, _ = cart2pol(bt_track['track_x_m'][-1], bt_track['track_y_m'][-1])
            bt_course = rad2azdeg(bt_course)
            gps_track = BoatStructure.compute_boat_track(transect, ref=gps_ref)
            gps_course, _ = cart2pol(gps_track['track_x_m'][-1], gps_track['track_y_m'][-1])
            gps_course = rad2azdeg(gps_course)

            # Compute course
            gps_bt['course'] = gps_course - bt_course
            if gps_bt['course'] < 0:
                gps_bt['course'] = gps_bt['course'] + 360

            # Compute ratio
            gps_bt['ratio'] = bt_track['dmg_m'][-1] / gps_track['dmg_m'][-1]

            # Compute closure vector
            x_diff = bt_track['track_x_m'][-1] - gps_track['track_x_m'][-1]
            y_diff = bt_track['track_y_m'][-1] - gps_track['track_y_m'][-1]
            gps_bt['dir'], gps_bt['mag'] = cart2pol(x_diff, y_diff)
            gps_bt['dir'] = rad2azdeg(gps_bt['dir'])

        return gps_bt

# ========================================================================
# Begin multithread function included in module but not TransectData class
# Currently this is coded only for TRDI data
# ========================================================================


# DSM changed 1/23/2018 def allocate_transects(source, mmt, kargs)
# TODO This needs a complete rewrite from what Greg did. However it works with no multi-threading for now
def allocate_transects(mmt, transect_type='Q', checked=False):
    """Method to load transect data. Changed from Matlab approach by Greg to allow possibility
    of multi-thread approach.

    Parameters
    ----------
    mmt: MMT_TRDI
        Object of MMT_TRDI
    transect_type: str
        Type of transect (Q: discharge or MB: moving-bed test)
    checked: bool
        Determines if all files are loaded (False) or only checked files (True)
    """

    # DEBUG, set threaded to false to get manual serial commands
    multi_threaded = False

    file_names = []
    file_idx = []

    # Setup processing for discharge or moving-bed transects
    if transect_type == 'Q':
        # Identify discharge transect files to load
        if checked:
            for idx, transect in enumerate(mmt.transects):
                if transect.Checked == 1:
                    file_names.append(transect.Files[0])
                    file_idx.append(idx)
            # file_names = [transect.Files[0] for transect in mmt.transects if transect.Checked == 1]
        else:
            file_names = [transect.Files[0] for transect in mmt.transects]
            file_idx = list(range(0, len(file_names)))
    elif transect_type == 'MB':
        file_names = [transect.Files[0] for transect in mmt.mbt_transects]
        file_idx = list(range(0, len(file_names)))

    # Determine if any files are missing
    valid_files = []
    valid_indices = []
    for index, name in enumerate(file_names):
        fullname = os.path.join(mmt.path, name)
        if os.path.exists(fullname):
            valid_files.append(fullname)
            valid_indices.append(file_idx[index])

    # Multi-thread for Pd0 files
    # -------------------------
    # Seems like this section belongs in Pd0TRDI.py
    # Initialize thread variables
    pd0_data = []
    pd0_threads = []
    thread_id = 0

    # DSM 1/24/2018 could this be moved to Pd0TRDI.py as a method
    def add_pd0(file_name):
        pd0_data.append(Pd0TRDI(file_name))
        
    if multi_threaded:
        # TODO this belongs in the pd0 class
        for file in valid_files:
            pd0_thread = MultiThread(thread_id=thread_id, function=add_pd0, args={'file_name': file})
            thread_id += 1
            pd0_thread.start()
            pd0_threads.append(pd0_thread)
    else:
        for file in valid_files:
            pd0_data.append(Pd0TRDI(file))

    for thrd in pd0_threads:
        thrd.join()

    # Multi-thread for transect data

    # Initialize thread variables
    processed_transects = []
    transect_threads = []
    thread_id = 0

    # DSM 1/24/2018 couldn't this be added to the TransectData class
    def add_transect(transect_data, mmt_transect, mt_pd0_data, mt_mmt):
        transect_data.trdi(mmt=mt_mmt,
                           mmt_transect=mmt_transect,
                           pd0_data=mt_pd0_data)
        processed_transects.append(transect_data)

    # Process each transect
    for k in range(len(pd0_data)):
        transect = TransectData()
        if pd0_data[k].Wt is not None:
            if transect_type == 'MB':
                # Process moving-bed transect
                if multi_threaded:
                    t_thread = MultiThread(thread_id=thread_id,
                                           function=add_transect,
                                           args={'transect': transect,
                                                 'mmt_transect': mmt.mbt_transects[valid_indices[k]],
                                                 'mt_pd0_data': pd0_data[k],
                                                 'mt_mmt': mmt})
                    t_thread.start()
                    transect_threads.append(t_thread)

                else:
                    transect = TransectData()
                    add_transect(transect_data=transect,
                                 mmt_transect=mmt.mbt_transects[valid_indices[k]],
                                 mt_pd0_data=pd0_data[k],
                                 mt_mmt=mmt)

            else:
                # Process discharge transects
                if multi_threaded:
                    t_thread = MultiThread(thread_id=thread_id,
                                           function=add_transect,
                                           args={'transect': transect,
                                                 'mmt_transect': mmt.transects[valid_indices[k]],
                                                 'mt_pd0_data': pd0_data[k],
                                                 'mt_mmt': mmt})
                    t_thread.start()
                    transect_threads.append(t_thread)

                else:
                    add_transect(transect_data=transect,
                                 mmt_transect=mmt.transects[valid_indices[k]],
                                 mt_pd0_data=pd0_data[k],
                                 mt_mmt=mmt)

    if multi_threaded:
        for x in transect_threads:
            x.join()
                
    return processed_transects   


def adjusted_ensemble_duration(transect, trans_type=None):
    """Applies the TRDI method of expanding the ensemble time when data are invalid.

    Parameters
    ----------
    transect: TransectData
        Object of TransectData
    trans_type: str
        Transect type. If mbt then bottom track is used.

    Returns
    -------
    delta_t: np.array(float)
        Array of delta time in seconds for each ensemble.
    """
        
    if transect.adcp.manufacturer == 'TRDI':
        if trans_type is None:
            # Determine valid data from water track
            valid = np.isnan(transect.w_vel.u_processed_mps) == False
            valid_sum = np.sum(valid)
        else:
            # Determine valid data from bottom track
            valid_sum = np.isnan(transect.boat_vel.bt_vel.u_processed_mps) == False
            
        valid_ens = valid_sum > 0
        n_ens = len(valid_ens)
        ens_dur = transect.date_time.ens_duration_sec
        delta_t = np.tile([np.nan], n_ens)
        cum_dur = 0
        for j in range(n_ens):
            cum_dur = np.nansum(np.hstack([cum_dur, ens_dur[j]]))
            if valid_ens[j]:
                delta_t[j] = cum_dur
                cum_dur = 0
    else:
        delta_t = transect.date_time.ens_duration_sec
        
    return delta_t
