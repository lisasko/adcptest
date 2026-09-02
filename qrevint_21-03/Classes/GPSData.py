import utm
import numpy as np
from MiscLibs.common_functions import azdeg2rad, pol2cart, nans


class GPSData(object):
    """Class containing the raw GPS data and algorithms to convert 
    that raw data to boat velocity.

    Attributes
    ----------
    # Raw properties:
        raw_gga_lat_deg: np.array(float)
            Raw latitude in degress, [ensemble,n]
        raw_gga_lon_deg: np.array(float)
            Raw longitude in degrees, [ensemble,n]
        raw_gga_altitude_m: np.array(float)
            Raw altitude in meters, [ensemble,n]
        raw_gga_differential: np.array(float)
            Raw differential correction indicator, [ensemble,n]
        raw_gga_hdop: np.array(float)
            Raw horizontal dilution of precision, [ensemble,n]
        raw_gga_utc: np.array(float)
            Raw UTC time, hhmmss.ss, [ensemble,n]
        raw_gga_serial_time: np.array(float)
            Raw UTC time of gga data in seconds past midnight, [ensemble,n]
        raw_gga_num_sats: np.array(float)
            Raw number of satellites reported in gga sentence, [ensemble,n]
        raw_vtg_course_deg:np.array(float)
            Raw course in degress, [ensemble,n]
        raw_vtg_speed_mps: np.array(float)
            Raw speed in m/s, [ensemble,n]
        raw_vtg_delta_time: np.array(float)
            Raw vtg delta time (sec), [ensemble,n]
        raw_vtg_mode_indicator: np.array(float)
            Raw vtg mode indicator, [ensemble,n]
        raw_gga_delta_time: np.array(float)
            Raw gga delta time (sec), [ensemble,n]

    # Manufacturer assigned ensemble values:
        ext_gga_lat_deg: np.array(float)
            Latitude for each ensemble, in degrees [ensemble]
        ext_gga_lon_deg: np.array(float)
            Longitude for each ensemble, in degrees [ensemble]
        ext_gga_altitude_m: np.array(float)
            Altitude for each ensemble, in meters [ensemble]
        ext_gga_differential: np.array(float)
            Differential correction indicator for each ensemble [ensemble]
        ext_gga_hdop: np.array(float)
            Horizontal dilution of precision for each ensemble [ensemble]
        ext_gga_utc: np.array(float)
            UTC time, hhmmss.ss for each ensemble [ensemble]
        ext_gga_serial_time: np.array(float)
            UTC time of gga data in seconds past midnight for each ensemble [ensemble]
        ext_gga_num_sats: np.array(float)
            Number of satellites for each ensemble [ensemble]
        ext_vtg_course_deg: np.array(float)
            Course for each ensemble, in degrees [ensemble]
        ext_vtg_speed_mps: np.array(float)
            Speed for each ensemble, in m/s [ensemble]

    # User specifications:
        gga_position_method: str
            Method used to process gga data for position ('End', 'Average' 'External')
        gga_velocity_method: str
            Method used to process gga data for velocity ('End','Average' 'External')
        vtg_velocity_method: str
            Method used to process vtg data for velocity ('Average' 'External)

    # Computed properties:
        gga_lat_ens_deg: np.array(float)
            Processed latitude, in degrees [ensemble]
        gga_lon_ens_deg: np.array(float)
            Processed longitude, in degrees [ensemble]
        utm_ens_m: np.array(float)
            UTM position from processed gga data, in m [2,ensemble]
        gga_velocity_ens_mps: np.array(float)
            Boat velocity components computed from gga data, in m/s [2,ensemble]
        gga_serial_time_ens: np.array(float)
            UTC time of gga data, in seconds past midnight [ensemble]
        vtg_velocity_ens_mps: np.array(float)
            Boat velocity components computed from vtg data, in m/s [2,ensemble]
        per_good_ens: np.array(float)
            Percentage of available data used to compute ensemble value [ensemble]
        hdop_ens: np.array(float)
            HDOP for each ensemble using velocity method [ensemble]
        num_sats_ens: np.array(float)
            Number of satellites for each ensemble, using velocity method [ensemble]
        altitude_ens_m: np.array(float)
            Altitude for each ensemble, using velocity method [ensemble]
        diff_qual_ens: np.array(float)
            Differential quality for each ensemble, using velocity method [ensemble]
    """
    
    def __init__(self):
        """Initialize instance variables.
        """
        
        # Raw properties
        self.raw_gga_lat_deg = None         # self.raw_ latitude, in degress [ensemble,n]
        self.raw_gga_lon_deg = None         # self.raw_ longitude, in degrees [ensemble,n]
        self.raw_gga_altitude_m = None     # self.raw_ altitude in meters, [ensemble,n]
        self.raw_gga_differential = None    # Differential correction indicator [ensemble,n]
        self.raw_gga_hdop = None            # Horizontal dilution of precision [ensemble,n]
        self.raw_gga_utc = None             # UTC time, in hhmmss.ss [ensemble,n]
        self.raw_gga_serial_time = None     # UTC time of gga data, in seconds past midnight [ensemble,n]
        self.raw_gga_num_sats = None        # Number of satellites reported in gga sentence [ensemble,n]
        self.raw_vtg_course_deg = None      # Course, in degress [ensemble,n]
        self.raw_vtg_speed_mps = None       # Speed, in m/s [ensemble,n]
        self.raw_vtg_delta_time = None      # vtg delta time, in sec [ensemble,n]
        self.raw_vtg_mode_indicator = None  # vtg mode indicator [ensemble,n]
        self.raw_gga_delta_time = None      # gga delta time, in sec [ensemble,n]
        
        # Manufacturer assigned ensemble values
        self.ext_gga_lat_deg = None         # Raw latitude, in degrees [1,ensemble]
        self.ext_gga_lon_deg = None         # Raw longitude, in degrees [1,ensemble]
        self.ext_gga_altitude_m = None      # Raw altitude, in meters [1,ensemble]
        self.ext_gga_differential = None    # Differential correction indicator [1,ensemble]
        self.ext_gga_hdop = None            # Horizontal dilution of precision [1,ensemble]
        self.ext_gga_utc = None             # UTC time, in hhmmss.ss [1, ensemble]
        self.ext_gga_serial_time = None     # UTC time of gga data, in seconds past midnight [1,ensemble]
        self.ext_gga_num_sats = None        # Number of satellites reported by software [1,ensemble]
        self.ext_vtg_course_deg = None      # Course, in degress [1, ensemble]
        self.ext_vtg_speed_mps = None       # Speed, in m/s [1, ensemble]
       
        # User specification
        self.gga_position_method = None     # Method used to process gga data for position ('End', 'Average' 'External')
        self.gga_velocity_method = None     # Method used to process gga data for velocity ('End','Average' 'External')
        self.vtg_velocity_method = None     # Method used to process vtg data for velocity ('Average' 'External)
        
        # Computed properties for ensembles
        self.gga_lat_ens_deg = None         # Processed latitude in degrees, [ensemble]
        self.gga_lon_ens_deg = None         # Processed longitude in degrees, [ensemble]
        self.utm_ens_m = None               # UTM position from processed gga data, [2,ensemble]
        self.gga_velocity_ens_mps = None    # Boat velocity computed from gga data [2,ensemble]
        self.gga_serial_time_ens = None     # UTC time of gga data in seconds past midnight, [ensemble]
        self.vtg_velocity_ens_mps = None    # Boat velocity computed from vtg data [2,ensemble]
        self.per_good_ens = None            # Percentage of available data used to compute ensemble value [ensemble]
        self.hdop_ens = None                # HDOP for each ensemble using velocity method [ensemble]
        self.num_sats_ens = None            # Number of satellites for each ensemble, using velocity method [ensemble]
        self.altitude_ens_m = None          # Altitude for each ensemble, using velocity method [ensemble]
        self.diff_qual_ens = None           # Differential quality for each ensemble, using velocity method [ensemble]
        
    def populate_data(self, raw_gga_utc, raw_gga_lat, raw_gga_lon, raw_gga_alt, raw_gga_diff,
                      raw_gga_hdop, raw_gga_num_sats, raw_gga_delta_time, raw_vtg_course, raw_vtg_speed,
                      raw_vtg_delta_time, raw_vtg_mode_indicator, ext_gga_utc, ext_gga_lat, ext_gga_lon, ext_gga_alt,
                      ext_gga_diff, ext_gga_hdop, ext_gga_num_sats, ext_vtg_course, ext_vtg_speed,
                      gga_p_method, gga_v_method, vtg_method):
        """Store and process provided data in GPSData class.

        Parameters
        ----------
        raw_gga_utc: np.array(float)
            Raw UTC time, hhmmss.ss, [ensemble,n]
        raw_gga_lat: np.array(float)
            Raw latitude in degress, [ensemble,n]
        raw_gga_lon: np.array(float)
            Raw longitude in degrees, [ensemble,n]
        raw_gga_alt: np.array(float)
            Raw altitude in meters, [ensemble,n]
        raw_gga_diff: np.array(float)
            Raw differential correction indicator, [ensemble,n]
        raw_gga_hdop: np.array(float)
            Raw horizontal dilution of precision, [ensemble,n]
        raw_gga_num_sats: np.array(float)
            Raw number of satellites reported in gga sentence, [ensemble,n]
        raw_gga_delta_time: np.array(float)
            Raw gga delta time (sec), [ensemble,n]
        raw_vtg_course:np.array(float)
            Raw course in degress, [ensemble,n]
        raw_vtg_speed: np.array(float)
            Raw speed in m/s, [ensemble,n]
        raw_vtg_delta_time: np.array(float)
            Raw vtg delta time (sec), [ensemble,n]
        raw_vtg_mode_indicator: np.array(float)
            Raw vtg mode indicator, [ensemble,n]
        ext_gga_utc: np.array(float)
            UTC time, hhmmss.ss for each ensemble [ensemble]
        ext_gga_lat: np.array(float)
            Latitude for each ensemble, in degrees [ensemble]
        ext_gga_lon: np.array(float)
            Longitude for each ensemble, in degrees [ensemble]
        ext_gga_alt: np.array(float)
            Altitude for each ensemble, in meters [ensemble]
        ext_gga_diff: np.array(float)
            Differential correction indicator for each ensemble [ensemble]
        ext_gga_hdop: np.array(float)
            Horizontal dilution of precision for each ensemble [ensemble]
        ext_gga_num_sats: np.array(float)
            Number of satellites for each ensemble [ensemble]
        ext_vtg_course: np.array(float)
            Course for each ensemble, in degrees [ensemble]
        ext_vtg_speed: np.array(float)
            Speed for each ensemble, in m/s [ensemble]
        gga_p_method: str
            Method used to process gga data for position ('End', 'Average' 'External')
        gga_v_method: str
            Method used to process gga data for velocity ('End','Average' 'External')
        vtg_method: str
            Method used to process vtg data for velocity ('Average' 'External)
        """

        # Assign input to raw properties
        if raw_gga_utc is None:
            self.raw_gga_utc = np.tile([np.nan], raw_gga_lat.shape)
            self.raw_gga_serial_time = np.tile([np.nan], raw_gga_lat.shape)
        else:
            self.raw_gga_utc = raw_gga_utc
            self.raw_gga_serial_time = np.floor(raw_gga_utc / 10000) * 3600 \
                + np.floor(np.mod(raw_gga_utc, 10000) / 100) * 60 + np.mod(raw_gga_utc, 100)

        ## 03/08 : ajout pour méthode vermeulen 
        self.raw_gga_lat_deg_unfiltered = np.array(raw_gga_lat, dtype=float, copy=True)
        self.raw_gga_lon_deg_unfiltered = np.array(raw_gga_lon, dtype=float, copy=True)
        ##

        self.raw_gga_lat_deg = raw_gga_lat
        self.raw_gga_lon_deg = raw_gga_lon
        self.raw_gga_lat_deg[np.where(np.logical_and((self.raw_gga_lat_deg == 0),
                                                     (self.raw_gga_lon_deg == 0)))] = np.nan
        self.raw_gga_lat_deg[raw_gga_diff < 1] = np.nan
        self.raw_gga_lon_deg[np.isnan(self.raw_gga_lat_deg)] = np.nan
        self.raw_gga_altitude_m = raw_gga_alt
        self.raw_gga_altitude_m[np.isnan(self.raw_gga_lat_deg)] = np.nan
        self.raw_gga_differential = raw_gga_diff.astype('float')
        self.raw_gga_differential[np.isnan(self.raw_gga_lat_deg)] = np.nan
        self.raw_gga_hdop = raw_gga_hdop.astype('float')
        self.raw_gga_hdop[np.isnan(self.raw_gga_lat_deg)] = np.nan
        self.raw_gga_num_sats = raw_gga_num_sats.astype('float')
        self.raw_gga_num_sats[np.isnan(self.raw_gga_lat_deg)] = np.nan
        self.raw_gga_serial_time[np.isnan(self.raw_gga_lat_deg)] = np.nan

        # Delta time is a TRDI only variable
        if raw_gga_delta_time is None:
            self.raw_gga_delta_time = np.tile(np.nan, raw_gga_lat.shape)
        else:
            self.raw_gga_delta_time = raw_gga_delta_time

        self.raw_vtg_course_deg = raw_vtg_course
        self.raw_vtg_speed_mps = raw_vtg_speed
        self.raw_vtg_course_deg[np.where(np.logical_and((self.raw_vtg_course_deg == 0),
                                                        (self.raw_vtg_speed_mps == 0)))] = np.nan
        self.raw_vtg_speed_mps[np.isnan(self.raw_vtg_course_deg)] = np.nan

        # Delta time is a TRDI only variable
        if raw_vtg_delta_time is None:
            self.raw_vtg_delta_time = np.tile(np.nan, raw_gga_lat.shape)
        else:
            self.raw_vtg_delta_time = raw_vtg_delta_time

        self.raw_vtg_mode_indicator = np.array(raw_vtg_mode_indicator)
        
        # Assign input data to ensemble values computed by other software
        self.ext_gga_utc = ext_gga_utc
        self.ext_gga_lat_deg = ext_gga_lat
        self.ext_gga_lon_deg = ext_gga_lon
        self.ext_gga_altitude_m = ext_gga_alt
        self.ext_gga_differential = ext_gga_diff
        self.ext_gga_hdop = ext_gga_hdop
        self.ext_gga_num_sats = ext_gga_num_sats
        self.ext_gga_serial_time = np.floor(np.array(ext_gga_utc) / 10000) * 3600 + \
            np.floor(np.mod(ext_gga_utc, 10000) / 100) * 60 + np.mod(ext_gga_utc, 100)
        self.ext_vtg_course_deg = ext_vtg_course
        self.ext_vtg_speed_mps = ext_vtg_speed
        
        # Assign input data to method properties
        self.gga_position_method = gga_p_method
        self.gga_velocity_method = gga_v_method
        self.vtg_velocity_method = vtg_method
        
        # If gga data exist compute position and velocity
        if np.sum(np.sum(np.isnan(raw_gga_lat) == False)) > 0:
            self.process_gga()
        
        # If vtg data exist compute velocity
        if np.sum(np.sum(np.isnan(raw_vtg_speed) == False)) > 0:
            self.process_vtg()

    def populate_from_qrev_mat(self, transect):
        """Populates the object using data from previously saved QRev Matlab file.

        Parameters
        ----------
        transect: mat_struct
           Matlab data structure obtained from sio.loadmat
        """

        if hasattr(transect, 'gps'):
            if hasattr(transect.gps, 'diffQualEns'):

                # Raw properties
                self.raw_gga_lat_deg = transect.gps.rawGGALat_deg
                self.raw_gga_lon_deg = transect.gps.rawGGALon_deg
                self.raw_gga_altitude_m = transect.gps.rawGGAAltitude_m
                self.raw_gga_differential = transect.gps.rawGGADifferential
                self.raw_gga_hdop = transect.gps.rawGGAHDOP
                self.raw_gga_utc = transect.gps.rawGGAUTC
                self.raw_gga_serial_time = transect.gps.rawGGASerialTime
                self.raw_gga_num_sats = transect.gps.rawGGANumSats
                self.raw_vtg_course_deg = transect.gps.rawVTGCourse_deg
                self.raw_vtg_speed_mps = transect.gps.rawVTGSpeed_mps
                self.raw_vtg_delta_time = transect.gps.rawVTGDeltaTime

                # Older versions of QRev Matlab files represented the VTG mode differently.
                try:
                    if transect.gps.rawVTGModeIndicator.ndim == 2 and \
                            type(transect.gps.rawVTGModeIndicator[0][0]) is np.float64:
                        indicator = []
                        for row in transect.gps.rawVTGModeIndicator.astype(int):
                            row_indicator = []
                            for value in row:
                                if 127 > value > 0:
                                    row_indicator.append(chr(value))
                                else:
                                    row_indicator. append('')
                            indicator.append(row_indicator)
                        self.raw_vtg_mode_indicator = np.array(indicator)
                    else:
                        raw_vtg_mode_indicator = transect.gps.rawVTGModeIndicator.tolist()
                        new_list = []
                        for row in raw_vtg_mode_indicator:
                            new_list.append(list(row))
                        self.raw_vtg_mode_indicator = np.array(new_list)
                except AttributeError:
                    self.raw_vtg_mode_indicator = transect.gps.rawVTGModeIndicator

                self.raw_gga_delta_time = transect.gps.rawGGADeltaTime

                # Manufacturer assigned ensemble values
                self.ext_gga_lat_deg = transect.gps.extGGALat_deg
                self.ext_gga_lon_deg = transect.gps.extGGALon_deg
                self.ext_gga_altitude_m = transect.gps.extGGAAltitude_m
                self.ext_gga_differential = transect.gps.extGGADifferential
                self.ext_gga_hdop = transect.gps.extGGAHDOP
                self.ext_gga_utc = transect.gps.extGGAUTC
                self.ext_gga_serial_time = transect.gps.extGGASerialTime
                self.ext_gga_num_sats = transect.gps.extGGANumSats
                self.ext_vtg_course_deg = transect.gps.extVTGCourse_deg
                self.ext_vtg_speed_mps = transect.gps.extVTGSpeed_mps

                # User specification
                self.gga_position_method = transect.gps.ggaPositionMethod
                self.gga_velocity_method = transect.gps.ggaVelocityMethod
                self.vtg_velocity_method = transect.gps.vtgVelocityMethod

                # Computed properties for ensembles
                self.gga_lat_ens_deg = transect.gps.ggaLatEns_deg
                self.gga_lon_ens_deg = transect.gps.ggaLonEns_deg
                self.utm_ens_m = transect.gps.UTMEns_m
                self.gga_velocity_ens_mps = transect.gps.ggaVelocityEns_mps
                self.gga_serial_time_ens = transect.gps.ggaSerialTimeEns
                self.vtg_velocity_ens_mps = transect.gps.vtgVelocityEns_mps
                if len(transect.gps.perGoodEns) > 0:
                    self.per_good_ens = transect.gps.perGoodEns
                else:
                    self.per_good_ens = None
                self.hdop_ens = transect.gps.hdopEns
                self.num_sats_ens = transect.gps.numSatsEns
                self.altitude_ens_m = transect.gps.altitudeEns_m
                self.diff_qual_ens = transect.gps.diffQualEns

    def process_gga(self, p_setting=None, v_setting=None):
        """Computes boat velocity from gga data.

        Parameters
        ----------
        p_setting: str
            Specifies method to use for computing positions from gga data (External, End, First, Average, Mindt).
        v_setting: str
            Specifies method to use for computing velocity from gga data (External, End, First, Average, Mindt).
        """

        if p_setting is None:
            p_setting = self.gga_position_method

        if v_setting is None:
            v_setting = self.gga_velocity_method
            
        # Use only valid gga data
        valid = np.copy(self.raw_gga_num_sats)
        valid[np.isnan(valid)] = 0
        valid[valid > 0] = 1
        gga_lat_deg = np.copy(self.raw_gga_lat_deg)
        gga_lat_deg[valid == False] = np.nan
        gga_lon_deg = np.copy(self.raw_gga_lon_deg)
        gga_lon_deg[valid == False] = np.nan
        gga_serial_time = np.copy(self.raw_gga_serial_time)
        gga_serial_time[valid == False] = np.nan
        gga_delta_time = np.copy(self.raw_gga_delta_time)
        gga_delta_time[valid == False] = np.nan
        gga_hdop = np.copy(self.raw_gga_hdop)
        gga_hdop[valid == False] = np.nan
        gga_num_sats = np.copy(self.raw_gga_num_sats)
        gga_num_sats[valid == False] = np.nan
        gga_altitude_m = np.copy(self.raw_gga_altitude_m)
        gga_altitude_m[valid == False] = np.nan
        gga_differential = np.copy(self.raw_gga_differential)
        gga_differential[valid == False] = np.nan
        n_ensembles = gga_lat_deg.shape[0]

        # Apply method for computing position of ensemble

        # Use ensemble data from other software
        if p_setting == 'External':
            self.gga_lat_ens_deg = self.ext_gga_lat_deg
            self.gga_lon_ens_deg = self.ext_gga_lon_deg

        # Uses last valid data for each ensemble
        elif p_setting == 'End':
            self.gga_lat_ens_deg = np.tile(np.nan, gga_lat_deg.shape[0])
            self.gga_lon_ens_deg = np.tile(np.nan, gga_lon_deg.shape[0])
            for n in range(n_ensembles):
                idx = np.argwhere(~np.isnan(gga_lat_deg[n, :]))
                if idx.size < 1:
                    idx = 0
                else:
                    idx = idx[-1][0]
                self.gga_lat_ens_deg[n] = gga_lat_deg[n, idx]
                self.gga_lon_ens_deg[n] = gga_lon_deg[n, idx]

        # Use first valid data for each ensemble
        elif p_setting == 'First':
            self.gga_lat_ens_deg = np.tile(np.nan, gga_lat_deg.shape[0])
            self.gga_lon_ens_deg = np.tile(np.nan, gga_lon_deg.shape[0])
            for n in range(n_ensembles):
                idx = 0
                self.gga_lat_ens_deg[n] = gga_lat_deg[n, idx]
                self.gga_lon_ens_deg[n] = gga_lon_deg[n, idx]

        # Use minimum delta time
        elif p_setting == 'Mindt':
            self.gga_lat_ens_deg = np.tile(np.nan, gga_lat_deg.shape[0])
            self.gga_lon_ens_deg = np.tile(np.nan, gga_lon_deg.shape[0])
            d_time = np.abs(gga_delta_time)
            d_time_min = np.nanmin(d_time.T, 0).T
            
            use = []
            for n in range(len(d_time_min)):
                use.append(np.abs(d_time[n, :]) == d_time_min[n])
                
            use = np.array(use)
            self.gga_lat_ens_deg = np.tile([np.nan], (len(d_time_min))) 
            self.gga_lon_ens_deg = np.tile([np.nan], (len(d_time_min)))
            for n in range(len(d_time_min)):
                idx = np.where(use[n, :] == True)[0]
                if len(idx) > 0:
                    idx = idx[0]
                    self.gga_lat_ens_deg[n] = gga_lat_deg[n, idx]
                    self.gga_lon_ens_deg[n] = gga_lon_deg[n, idx]
                    
        y_utm, x_utm = self.compute_utm(self.gga_lat_ens_deg, self.gga_lon_ens_deg)
        self.utm_ens_m = (x_utm, y_utm)

        # Prepare variables for velocity computations
        lat = np.tile([np.nan], n_ensembles)
        lon = np.tile([np.nan], n_ensembles)
        self.gga_serial_time_ens = np.tile([np.nan], n_ensembles)
        self.altitude_ens_m = np.tile([np.nan], n_ensembles)
        self.diff_qual_ens = np.tile([np.nan], n_ensembles)
        self.hdop_ens = np.tile([np.nan], n_ensembles)
        self.num_sats_ens = np.tile([np.nan], n_ensembles)
        
        # Apply method for computing velocity of ensemble
        if v_setting == 'External':
            lat = self.ext_gga_lat_deg
            lon = self.ext_gga_lon_deg
            self.gga_serial_time_ens = self.ext_gga_serial_time
            self.hdop_ens = self.ext_gga_hdop
            self.num_sats_ens = self.ext_gga_num_sats
            self.altitude_ens_m = self.ext_gga_altitude_m
            self.diff_qual_ens = self.ext_gga_differential
            
        # Average all position during an ensemble
        elif v_setting == 'Average':
            lat = np.nanmean(gga_lat_deg, 1)
            lon = np.nanmean(gga_lon_deg, 1)
            self.gga_serial_time_ens = np.nanmean(gga_serial_time, 1)
            self.hdop_ens = np.nanmean(gga_hdop, 1)
            self.num_sats_ens = np.floor(np.nanmean(gga_num_sats, 1))
            self.altitude_ens_m = np.nanmean(self.raw_gga_altitude_m, 1)
            self.diff_qual_ens = np.floor(np.nanmean(self.raw_gga_differential, 1))
            
        # Use the last valid data in an ensemble
        elif v_setting == 'End':

            for n in range(n_ensembles):
                idx = np.where(np.isnan(gga_lat_deg[n, :]) == False)[0]
                if len(idx) > 0:
                    idx = idx[-1]
                    lat[n] = gga_lat_deg[n, idx]
                    lon[n] = gga_lon_deg[n, idx]
                    self.gga_serial_time_ens[n] = gga_serial_time[n, idx]
                    self.altitude_ens_m[n] = gga_altitude_m[n, idx]
                    self.diff_qual_ens[n] = gga_differential[n, idx]

                if idx <= len(self.raw_gga_hdop):
                    self.hdop_ens[n] = gga_hdop[n, idx]
                    
                if idx <= len(gga_num_sats[n]):
                    self.num_sats_ens[n] = gga_num_sats[n, idx]

        # Use the first valid data in an ensemble
        elif v_setting == 'First':
            for n in range(n_ensembles):
                idx = 0
                lat[n] = gga_lat_deg[n, idx]
                lon[n] = gga_lon_deg[n, idx]
                self.gga_serial_time_ens[n] = gga_serial_time[n, idx]
                self.altitude_ens_m[n] = gga_altitude_m[n, idx]
                self.diff_qual_ens[n] = gga_differential[n, idx]
                
                if idx <= len(self.raw_gga_hdop):
                    self.hdop_ens[n] = gga_hdop[n, idx]
                    
                if idx <= len(gga_num_sats[n]):
                    self.num_sats_ens[n] = gga_num_sats[n, idx]

        # Use the minimum delta time to assign data to an ensemble
        elif v_setting == 'Mindt':
            d_time = np.abs(gga_delta_time)
            d_time_min = np.nanmin(d_time, 1)
            use = []
            for n in range(len(d_time_min)):
                use.append(np.abs(d_time[n, :]) == d_time_min[n])
            use = np.array(use)  
            for n in range(len(d_time_min)):
                idx = np.where(use[n, :] == True)[0]
                if len(idx) > 0:
                    idx = idx[0]
                    lat[n] = gga_lat_deg[n, idx]
                    lon[n] = gga_lon_deg[n, idx]
                    self.gga_serial_time_ens[n] = gga_serial_time[n, idx]
                    self.altitude_ens_m[n] = gga_altitude_m[n, idx]
                    self.diff_qual_ens[n] = gga_differential[n, idx]
                    
                if idx <= len(gga_hdop[n]):
                    self.hdop_ens[n] = gga_hdop[n, idx]
                    
                if idx <= len(gga_num_sats[n]):
                    self.num_sats_ens[n] = gga_num_sats[n, idx]
                    
        # Identify valid values
        idx_values = np.where(np.isnan(x_utm) == False)[0]
        if len(idx_values) > 1:
            u, v = self.gga2_vel_trdi(lat, lon, self.gga_serial_time_ens, idx_values)
            self.gga_velocity_ens_mps = np.tile([np.nan], (2, len(lat)))
            self.gga_velocity_ens_mps[0, idx_values[1:]] = u[idx_values[1:]]
            self.gga_velocity_ens_mps[1, idx_values[1:]] = v[idx_values[1:]]
        else:
            self.gga_velocity_ens_mps = np.tile([np.nan], (2, len(lat)))

    def process_vtg(self, v_setting=None):
        """Processes raw vtg data to achieve a velocity for each ensemble containing data.

        Parameters
        ----------
        v_setting: str
            Method to used to compute ensemble velocity.
        """
        
        # Determine method used to compute ensemble velocity
        if v_setting is None:
            v_setting = self.vtg_velocity_method

        # Use only valid data
        vtg_speed_mps = np.copy(self.raw_vtg_speed_mps)
        vtg_course_deg = np.copy(self.raw_vtg_course_deg)
        vtg_delta_time = np.copy(self.raw_vtg_delta_time)

        # Use mode indicator to identify invalid original data
        idx = np.where(self.raw_vtg_mode_indicator == 'N')
        vtg_speed_mps[idx] = np.nan
        vtg_course_deg[idx] = np.nan
        vtg_delta_time[idx] = np.nan

        # Use average velocity for ensemble velocity
        if v_setting == 'Average':
            # Compute vtg velocity in x y coordinates from speed and course
            direction = azdeg2rad(vtg_course_deg)
            vx, vy = pol2cart(direction, vtg_speed_mps)
            vx[np.logical_and(vx == 0, vy == 0)] = np.nan
            vy[np.isnan(vx)] = np.nan
            vx_mean = np.nanmean(vx, 1)
            vy_mean = np.nanmean(vy, 1)
            self.vtg_velocity_ens_mps = np.vstack([vx_mean.T, vy_mean.T])

        # Use last velocity for ensemble velocity
        elif v_setting == 'End':
            n_ensembles = vtg_speed_mps.shape[0]
            vtg_vel = nans(n_ensembles)
            vtg_dir = nans(n_ensembles)
            
            for n in range(n_ensembles):
                idx = np.where(~np.isnan(vtg_speed_mps[n, :]))[0]
                if len(idx) > 0:
                    idx = idx[-1]
                else:
                    idx = 0
                vtg_vel[n] = vtg_speed_mps[n, idx]
                vtg_dir[n] = vtg_course_deg[n, idx]
                
            direction = azdeg2rad(vtg_dir)
            vx, vy = pol2cart(direction, vtg_vel)
            vx[np.logical_and(vx == 0, vy == 0)] = np.nan
            vy[np.isnan(vx)] = np.nan
            self.vtg_velocity_ens_mps = np.vstack([vx, vy])

        # Use first velocity for ensemble velocity
        elif v_setting == 'First':
            n_ensembles = vtg_speed_mps.shape[0]
            vtg_vel = nans(n_ensembles)
            vtg_dir = nans(n_ensembles)
            
            for n in range(n_ensembles):
                idx = 0
                vtg_vel[n] = vtg_speed_mps[n, idx]
                vtg_dir[n] = vtg_course_deg[n, idx]
            direction = azdeg2rad(vtg_dir)
            vx, vy = pol2cart(direction, vtg_vel)
            vx[np.logical_and(vx == 0, vy == 0)] = np.nan
            vy[np.isnan(vx)] = np.nan
            self.vtg_velocity_ens_mps = np.vstack([vx, vy])

        # Use the velocity with the minimum delta time for the ensemble velocity
        elif v_setting == 'Mindt':
            d_time = np.abs(vtg_delta_time)
            # d_time[d_time==0] = np.nan
            d_time_min = np.nanmin(d_time.T, 0).T
            
            use = []
            vtg_speed = []
            vtg_dir = []
            
            for n in range(len(d_time_min)):
                use.append(np.abs(d_time[n, :]) == d_time_min[n])
                
            use = np.array(use)
            for n in range(len(d_time_min)):
                idx = np.where(use[n, :] == True)[0]
                if len(idx) > 0:
                    idx = idx[0]
                    vtg_speed.append(vtg_speed_mps[n, idx])
                    vtg_dir.append(vtg_course_deg[n, idx])
                else:
                    vtg_speed.append(np.nan)
                    vtg_dir.append(np.nan)
                    
                direction = azdeg2rad(np.array(vtg_dir))
                vx, vy = pol2cart(direction, np.array(vtg_speed))
                self.vtg_velocity_ens_mps = np.vstack([vx, vy])

        # Use velocity selected by external algorithm for ensemble velocity
        elif v_setting == 'External':
            direction = azdeg2rad(self.ext_vtg_course_deg)
            vx, vy = pol2cart(direction, self.ext_vtg_speed_mps)
            self.vtg_velocity_ens_mps = np.vstack([vx.T, vy.T])

    @staticmethod
    def compute_utm(lat_in, lon_in):
        """Compute UTM coordinates from latitude and longitude.

        Parameters
        ----------
        lat_in: np.array(float)
            Latitude in degrees.
        lon_in: np.array(float)
            Longitude in degrees.
        """

        # Set invalid data to nan
        lat_in[lat_in == 0] = np.nan
        lon_in[lon_in == 0] = np.nan

        lat2 = np.deg2rad(lat_in)
        lon2 = np.deg2rad(lon_in)
        
        y = np.tile([np.nan], lat_in.shape)
        x = np.tile([np.nan], lon_in.shape)
        idx = np.where(np.logical_and((np.isnan(lat2) == False), (np.isnan(lon2) == False)))
        for ind in idx[0]:
            y[ind], x[ind], _, _ = utm.from_latlon(lat2[ind], lon2[ind])
        x_utm = x.reshape(lon_in.shape)
        y_utm = y.reshape(lat_in.shape)
        
        return y_utm, x_utm

    @staticmethod
    def gga2_vel_trdi(lat, lon, t, idx_values):
        """Computes velocity from gga data using approach from TRDI WinRiver II.

        Parameters
        ----------
        lat: np.array(float)
            Latitude for each ensemble used for velocity computations, in degrees.
        lon: np.array(float)
            Longitude for each ensemble used for velocity computations, in degrees.
        t: np.array(float)
            GGA time associated with the latitude and longitude selected for velocity computations.
        idx_values: np.array(bool)
            Index of valid lat-lon data.
        """
        
        u = np.zeros(lat.shape)
        v = np.zeros(lat.shape)
        
        for n in range(1, len(idx_values)):
            lat1 = lat[idx_values[n-1]]
            lat2 = lat[idx_values[n]]
            lon1 = lon[idx_values[n-1]]
            lon2 = lon[idx_values[n]]
            t1 = t[idx_values[n-1]]
            t2 = t[idx_values[n]]

            lat_avg_rad = ((lat1 + lat2) / 2) * np.pi / 180
            sin_lat_avg_rad = np.sin(lat_avg_rad)
            coefficient = 6378137 * np.pi / 180
            ellipticity = 1 / 298.257223563
            re = coefficient * (1 + ellipticity * sin_lat_avg_rad ** 2)
            rn = coefficient * (1 - 2 * ellipticity + 3 * ellipticity * sin_lat_avg_rad ** 2)
            delta_x = re * (lon2 - lon1) * np.cos(lat_avg_rad)
            delta_y = rn * (lat2 - lat1)
            delta_time = t2 - t1
            if delta_time > 0.0001:
                u[idx_values[n]] = delta_x / delta_time
                v[idx_values[n]] = delta_y / delta_time
            else:
                u[idx_values[n]] = np.nan
                v[idx_values[n]] = np.nan
            
        return u, v

