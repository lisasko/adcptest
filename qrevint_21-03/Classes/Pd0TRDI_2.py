import os
import re
import numpy as np
import struct
from MiscLibs.common_functions import pol2cart, valid_number, nans


class Pd0TRDI(object):
    """Class to read data from PD0 files

    Attributes
    ----------
    file_name: str
        Full name including path of pd0 file to be read
    Hdr: Hdr
        Object of Hdr for heading information
    Inst: Inst
        Object of Inst to hold instrument information
    Cfg: Cfg
        Object of Cfg to hold configuration information
    Sensor: Sensor
        Object of Sensor to hold sensor data
    Wt: Wt
        Object of Wt to hold water track data
    Bt: Bt
        Object of Bt to hold bottom track data
    Gps: Gps
        Object of Gps to hold GPS data from previous versions of WR
    Gps2: Gps2
        Object of Gps2 to hold GPS data from WR2
    Surface: Surface
        Object of Surface to hold surface cell data
    AutoMode: AutoMode
        Object of AutoMode to hold auto configuration settings
    Nmea: Nmea
        Object of Nmea to hold Nmea data
    """

    def __init__(self, file_name):
        """Constructor initializing instance variables.

        Parameters
        ----------
        file_name: str
            Full name including path of pd0 file to be read
        """

        self.file_name = file_name
        self.Hdr = None
        self.Inst = None
        self.Cfg = None
        self.Sensor = None
        self.Wt = None
        self.Bt = None
        self.Gps = None
        self.Gps2 = None
        self.Surface = None
        self.AutoMode = None
        self.Nmea = None

        self.data_decoders = {
            0x0000: ('fixed_leader', self.decode_fixed_leader),
            0x0080: ('variable_leader', self.decode_variable_leader),
            0x0100: ('velocity', self.decode_velocity),
            0x0200: ('correlation', self.decode_correlation),
            0x0300: ('echo_intensity', self.decode_echo_intensity),
            0x0400: ('percent_good', self.decode_percent_good),
            0x0500: ('status', self.decode_status),
            0x0600: ('bottom_track', self.decode_bottom_track),
            0x2022: ('nmea', self.decode_nmea),
            0x2100: ('dbt_sentence', self.decode_dbt_sentence),
            0x2101: ('gga_sentence', self.decode_gga_sentence),
            0x2102: ('vtg_sentence', self.decode_vtg_sentence),
            0x2103: ('gsa_sentence', self.decode_gsa_sentence),
            0x0010: ('surface_leader', self.decode_surface_leader),
            0x0110: ('surface_velocity', self.decode_surface_velocity),
            0x0210: ('surface_correlation', self.decode_surface_correlation),
            0x0310: ('surface_intensity', self.decode_surface_intensity),
            0x0410: ('surface_percent_good', self.decode_surface_percent_good),
            0x0510: ('surface_status', self.decode_surface_status),
            0x4401: ('auto_configuration', self.decode_auto_config),
            0x4100: ('vertical_beam', self.decode_vertical_beam),
            0x3200: ('transformation_matrix', self.decode_transformation_matrix)
        }

        self.nmea_decoders = {100: ('gga', self.decode_gga_100),
                              101: ('vtg', self.decode_vtg_101),
                              102: ('ds', self.decode_ds_102),
                              103: ('ext_heading', self.decode_ext_heading_103),
                              104: ('gga', self.decode_gga_104),
                              105: ('vtg', self.decode_vtg_105),
                              106: ('ds', self.decode_ds_106),
                              107: ('ext_heading', self.decode_ext_heading_107),
                              204: ('gga', self.decode_gga_204),
                              205: ('vtg', self.decode_vtg_205),
                              206: ('ds', self.decode_ds_206),
                              207: ('ext_heading', self.decode_ext_heading_207)}

        self.n_velocities = 4
        self.max_surface_bins = 5

        self.pd0_read(file_name)

    def create_objects(self, n_ensembles, n_types, n_bins, max_surface_bins, n_velocities, wr2=False):
        """Create objects for instance variables.

        Parameters
        ----------
        n_ensembles: int
            Number of ensembles
        n_types: int
            Number of data types
        n_bins: int
            Number of bins or depth cells
        max_surface_bins: int
            Maximum number of surface cells
        n_velocities: int
            Number of velocities
        wr2: bool
            Whether WR2 processing of GPS data should be applied
        """

        self.Hdr = Hdr(n_ensembles, n_types)
        self.Inst = Inst(n_ensembles)
        self.Cfg = Cfg(n_ensembles)
        self.Sensor = Sensor(n_ensembles)
        self.Wt = Wt(n_bins, n_ensembles, n_velocities)
        self.Bt = Bt(n_ensembles, n_velocities)
        self.Gps = Gps(n_ensembles)
        self.Gps2 = Gps2(n_ensembles, wr2)
        self.Surface = Surface(n_ensembles, n_velocities, max_surface_bins)
        self.AutoMode = AutoMode(n_ensembles)
        self.Nmea = Nmea(n_ensembles)

    def pd0_read(self, fullname, wr2=False):
        """Reads the binary pd0 file and assigns values to object instance variables.

        Parameters
        ----------
        fullname: str
            Full file name including path
        wr2: bool
            Determines if WR2 processing should be applied to GPS data
        """

        # Check to ensure file exists
        if os.path.exists(fullname):
            file_info = os.path.getsize(fullname)

            if file_info > 0:
                # Open file for processing
                with open(fullname, 'rb') as f:
                    pd0 = f.read()
                pd0_bytes = bytearray(pd0)

                # Intialize classes and arrays
                n_ensembles, max_types, max_beams, max_bins = self.number_of_ensembles(self, file_info, pd0_bytes)
                self.create_objects(n_ensembles, max_types, max_bins, self.max_surface_bins, self.n_velocities, wr2)
                self.decode_all(pd0_bytes, file_info)
                self.screen_and_convert(wr2)

    def screen_and_convert(self, wr2):

        # Screen for bad data, and do the unit conversions
        self.Wt.vel_mps[self.Wt.vel_mps == -32768] = np.nan
        self.Wt.vel_mps = self.Wt.vel_mps / 1000
        self.Wt.corr[self.Wt.corr == -32768] = np.nan
        self.Wt.rssi[self.Wt.rssi == -32768] = np.nan
        self.Wt.pergd[self.Wt.pergd == -32768] = np.nan

        # Remove bad data, convert units
        self.Bt.depth_m[self.Bt.depth_m == -32768] = np.nan
        self.Bt.depth_m = self.Bt.depth_m / 100
        self.Bt.vel_mps[self.Bt.vel_mps == -32768] = np.nan
        self.Bt.vel_mps = self.Bt.vel_mps / 1000
        self.Bt.corr[self.Bt.corr == -32768] = np.nan
        self.Bt.eval_amp[self.Bt.eval_amp == -32768] = np.nan
        self.Bt.pergd[self.Bt.pergd == -32768] = np.nan

        # Remove bad data from Surface structure (RR), convert where needed
        self.Surface.vel_mps[self.Surface.vel_mps == -32768] = np.nan
        self.Surface.vel_mps = self.Surface.vel_mps / 1000
        self.Surface.corr[self.Surface.corr == -32768] = np.nan
        self.Surface.rssi[self.Surface.rssi == -32768] = np.nan
        self.Surface.pergd[self.Surface.pergd == -32768] = np.nan

        # If requested compute WR2 compatible GPS-based boat velocities
        if wr2:

            # If vtg data are available compute north and east components
            if self.Gps2.vtg_header[0, 0] == '$':

                # Find minimum of absolute value of delta time from raw data
                vtg_delta_time = np.abs(self.Gps2.vtg_delta_time)
                vtg_min = np.nanmin(vtg_delta_time, 1)

                # Compute the velocity components in m/s
                for i in range(len(vtg_delta_time)):
                    idx = np.where(vtg_delta_time == vtg_min)[0][0]
                    self.Gps2.vtg_velE_mps[i], self.Gps2.vtg_velN_mps[i] = \
                        pol2cart((90 - self.Gps2.course_true[i, idx]) * np.pi / 180,
                                 self.Gps2.speed_kph[i, idx] * 0.2777778)

            if self.Gps2.gga_header[0, 0] == '$':

                # Initialize constants
                e_radius = 6378137
                coeff = e_radius * np.pi / 180
                ellip = 1 / 298.257223563

                # Find minimum of absolute value of delta time from raw data
                gga_delta_time = np.abs(self.Gps2.gga_delta_time)
                gga_min = np.nanmin(gga_delta_time, axis=1)

                # Process gga data
                for i in range(len(gga_delta_time)):
                    idx = np.where(gga_delta_time[i:] == gga_min)
                    if idx > 0:
                        lat_avg_rad = (self.Gps2.lat_deg[i, idx[i]]
                                       + self.Gps2.lat_deg[i - 1, idx[i - 1]]) / 2
                        sin_lat_avg_rad = np.sin(np.deg2rad(lat_avg_rad))
                        r_e = coeff * (1 + ellip * sin_lat_avg_rad * sin_lat_avg_rad)
                        rn = coeff * (1 - 2 * ellip + 3 * ellip * sin_lat_avg_rad * sin_lat_avg_rad)
                        dx = r_e * (self.Gps2.lon_deg[i, idx[i]] -
                                    self.Gps2.lon_deg(i - 1, idx[i - 1])) * np.cos(np.deg2rad(lat_avg_rad))
                        dy = rn * (self.Gps2.lat_deg[i, idx[i]] - self.Gps2.lat_deg[i - 1, idx[i - 1]])
                        dt = self.Gps2.utc[i, idx[i]] - self.Gps2.utc[i - 1, idx[i - 1]]
                        self.Gps2.gga_velE_mps[i] = dx / dt
                        self.Gps2.gga_velN_mps[i] = dy / dt
                    else:
                        self.Gps2.gga_velE_mps[i] = np.nan
                        self.Gps2.gga_velN_mps[i] = np.nan

    def decode_all(self, pd0_bytes, file_info):

        start_byte = 0
        n = 0
        ensemble_number = 0
        while start_byte < file_info:
            data = self.decode_pd0_bytearray(self.data_decoders, pd0_bytes[start_byte:])
            if data['checksum']:
                # Adjust index for lost ensembles
                if ensemble_number > 0:
                    n = n + data['variable_leader']['ensemble_number'] - ensemble_number

                self.Hdr.populate_data(n, data)
                self.Inst.populate_data(n, data)
                self.Cfg.populate_data(n, data)
                self.Sensor.populate_data(n, data)
                self.Wt.populate_data(n, data, self)
                self.Bt.populate_data(n, data)
                # self.Gps.populate_data(n, data)
                self.Gps2.populate_data(n, data)
                self.Surface.populate_data(n, data, self)
                self.AutoMode.populate_data(n, data)
                self.Nmea.populate_data(n, data)
                start_byte = start_byte + data['header']['number_of_bytes'] + 2
                ensemble_number = data['variable_leader']['ensemble_number']
            else:
                start_byte = Pd0TRDI.find_next(pd0_bytes, start_byte, file_info)


    @staticmethod
    def number_of_ensembles(self, file_info, pd0_bytes):
        """Determines the number of ensembles in the data file.

        Parameters
        ----------
        self: Pd0TRDI
            Current class
        file_info: int
            File size in bytes
        pd0_bytes: bytearray
            Contents of pd0 file

        Returns
        -------
        n_ensembles: int
            Number of ensembles
        max_data_types: int
            Maximum number of data types in file
        max_beams: int
            Maximum number of beamse
        max_bins: int
            Maximum number of regular bins
        """

        # Configure data decoders to be used
        data_decoders = {0x0000: ('fixed_leader', self.preload_fixed_leader),
                         0x0080: ('variable_leader', self.preload_variable_leader)}

        # Intitialize variables
        start_byte = 0
        n_beams = []
        n_bins = []
        n_data_types = []
        ens_num = []

        # Loop through entire file
        while start_byte < file_info:

            data = self.decode_pd0_bytearray(data_decoders, pd0_bytes[start_byte:])
            # start_byte = start_byte + data['header']['number_of_bytes'] + 2
            if data['checksum']:
                # if 'number_of_bytes' in data['header'] and data['header']['number_of_bytes'] > 0:
                if 'number_of_bytes' in data['header'] and 'fixed_leader' in data and 'variable_leader' in data:
                    n_data_types.append(data['header']['number_of_data_types'])
                    n_beams.append(data['fixed_leader']['number_of_beams'])
                    n_bins.append(data['fixed_leader']['number_of_cells'])
                    ens_num.append(data['variable_leader']['ensemble_number'])
                    start_byte = start_byte + data['header']['number_of_bytes'] + 2
                else:
                    start_byte = Pd0TRDI.find_next(pd0_bytes, start_byte, file_info)
            else:
                start_byte = Pd0TRDI.find_next(pd0_bytes, start_byte, file_info)


        # Compute maximums
        max_data_types = np.nanmax(n_data_types)
        max_beams = np.nanmax(n_beams)
        max_bins = np.nanmax(n_bins)
        n_ensembles = ens_num[-1] - ens_num[0] + 1

        return n_ensembles, max_data_types, max_beams, max_bins

    @staticmethod
    def find_next (pd0_bytes, start_byte, file_info):

        try:
            start_byte = start_byte + 1
            skip_forward = pd0_bytes[start_byte:].index(b'\x7f\x7f')
            # data['header'] = Pd0TRDI.decode_fixed_header(pd0_bytes[start_byte + skip_forward:])
            start_byte = start_byte + skip_forward
        except ValueError:
            start_byte = file_info

        return start_byte

    @staticmethod
    def preload_fixed_leader(pd0_bytes, offset, data):
        """Parses the fixed leader for number of beams and number of cells.

        Parameters
        ----------
        pd0_bytes: bytearray
            Bytearray of all pd0 data
        offset: int
            Pointer into pd0_bytes
        data: dict
            Not used, included for compatibilty with other decoders

        Returns
        -------
        number_of_beams: int
            Number of beams in ensemble
        number_of_cells: int
            Number of regular cells in ensemble
        """

        fixed_leader_format = (('number_of_beams', 'B', 8), ('number_of_cells', 'B', 9))

        return Pd0TRDI.unpack_bytes(pd0_bytes, fixed_leader_format, offset)

    @staticmethod
    def preload_variable_leader(pd0_bytes, offset, data):
        """Decodes variable leader ensemble number

        Parameters
        ----------
        pd0_bytes: bytearray
            Bytearray of all pd0 data
        offset: int
            Pointer into pd0_bytes
        data: dict
            Dictionary containing previously decoded data

        Returns
        -------
        :dict
            Dictionary of decoded data
        """

        # Define format
        variable_leader_format = (('ensemble_number', '<H', 2),)

        return Pd0TRDI.unpack_bytes(pd0_bytes, variable_leader_format, offset)

    @staticmethod
    def decode_pd0_bytearray(data_decoders, pd0_bytes):
        """Loops through data and calls appropriate parsing method for each header ID.

        Parameters
        ----------
        data_decoders: dict
            Dictionary associating a method with a leader ID
        pd0_bytes: bytearray
            Byte array of entire pd0 file

        Returns
        -------
        data: dict
            Dictionary of decoded data
        """

        data = {}

        # Read in header
        data['header'] = Pd0TRDI.decode_fixed_header(pd0_bytes)
        data['checksum'] = False
        if 'number_of_bytes' in data['header'] and data['header']['number_of_bytes'] > 0:
            if 'number_of_data_types' in data['header']:
                # If checksum is OK then decode address offsets to the data types
                if Pd0TRDI.validate_checksum(pd0_bytes, data['header']['number_of_bytes']):
                    data['checksum'] = True
                    data['header']['address_offsets'] = Pd0TRDI.decode_address_offsets(pd0_bytes,
                                                                                       data['header']['number_of_data_types'])
                    data['header']['invalid'] = []
                    # Loop to decode all data types for which a data decoder is provided
                    for offset in data['header']['address_offsets']:
                        if len(pd0_bytes) > offset + 2:
                            header_id = struct.unpack('<H', pd0_bytes[offset: offset + 2])[0]
                            if header_id in data_decoders:
                                key = data_decoders[header_id][0]
                                decoder = data_decoders[header_id][1]
                                data[key] = decoder(pd0_bytes, offset, data)
                            else:
                                data['header']['invalid'].append(header_id)

        return data

    @staticmethod
    def unpack_bytes(pd0_bytes, data_format_tuples, offset=0):
        """Unpackes the data based on the supplied data format tuples and offset.

        Parameters
        ----------
        pd0_bytes: bytearray
            Bytearray of all pd0 data
        data_format_tuples: tuple
            A tuple of tuples providing the data name, format, and byte location
        offset: int
            Pointer into pd0_bytes

        Returns
        -------
        data: dict
            Dictionary of decoded data
        """
        data = {}

        # Decode data for each format specified in the data format tuples and assign to the data dictionary
        for fmt in data_format_tuples:
            try:
                struct_offset = offset + fmt[2]
                size = struct.calcsize(fmt[1])
                data[fmt[0]] = struct.unpack(fmt[1], pd0_bytes[struct_offset: struct_offset + size])[0]
            except:
                print('Error parsing %s with the arguments ')

        return data

    @staticmethod
    def validate_checksum(pd0_bytes, offset):
        """Validates that the checksum is correct to ensure data integrity.

        Parameters
        ----------
        pd0_bytes: bytearray
            Bytearray of all pd0 data
        offset: int
            Pointer into pd0_bytes

        Returns
        -------
        :bool
            True if checksum is valid

        """
        if len(pd0_bytes) > offset + 1:
            calc_checksum = sum(pd0_bytes[:offset]) & 0xFFFF
            given_checksum = struct.unpack('<H', pd0_bytes[offset: offset + 2])[0]

            if calc_checksum == given_checksum:
                return True
            else:
                return False
        return False

    @staticmethod
    def bin2str(bin_in):

        try:
            str_out = bin_in.decode('utf-8')
        except:
            str_out = ''
        return str_out

    @staticmethod
    def decode_address_offsets(pd0_bytes, num_data_types, offset=6):
        """Decodes the address offsets for each data type.

        Parameters
        ----------
        pd0_bytes: bytearray
            Bytearray of all pd0 data
        num_data_types: int
            Number of data types for which to find offsets
        offset: int
            Pointer into pd0_bytes

        Returns
        -------
        address_data: list
            List of offsets to each data type
        """

        address_data = []

        # Loop through each data type
        for bytes_start in range(offset, offset + (num_data_types * 2), 2):
            data = struct.unpack_from('<H', pd0_bytes[bytes_start: bytes_start + 2])[0]
            address_data.append(data)

        return address_data

    @staticmethod
    def decode_fixed_header(pd0_bytes):
        """Decodes fixed header

        Parameters
        ----------
        pd0_bytes: bytearray
            Bytearray of all pd0 data

        Returns
        -------
        header: dict
            Dictionary of header data
        """

        header_data_format = (('id', 'B', 0),
                              ('data_source', 'B', 1),
                              ('number_of_bytes', '<H', 2),
                              ('spare', 'B', 4),
                              ('number_of_data_types', 'B', 5))

        header = Pd0TRDI.unpack_bytes(pd0_bytes, header_data_format)
        return header

    @staticmethod
    def decode_fixed_leader(pd0_bytes, offset, data):
        """Decodes fixed leader data

        Parameters
        ----------
        pd0_bytes: bytearray
            Bytearray of all pd0 data
        offset: int
            Pointer into pd0_bytes
        data: dict
            Dictionary containing previously decoded data

        Returns
        -------
        :dict
            Dictionary of decoded data
        """

        # Define format
        fixed_leader_format = (
            ('id', '<H', 0),
            ('cpu_firmware_version', 'B', 2),
            ('cpu_firmware_revision', 'B', 3),
            ('system_configuration_ls', 'B', 4),
            ('system_configuration_ms', 'B', 5),
            ('simulation_data_flag', 'B', 6),
            ('lag_length', 'B', 7),
            ('number_of_beams', 'B', 8),
            ('number_of_cells', 'B', 9),
            ('number_of_water_pings', '<H', 10),
            ('depth_cell_size', '<H', 12),
            ('blank_after_transmit', '<H', 14),
            ('water_mode', 'B', 16),
            ('low_correlation_threshold', 'B', 17),
            ('number_of_code_repetitions', 'B', 18),
            ('minimum_percentage_water_profile_pings', 'B', 19),
            ('error_velocity_threshold', '<H', 20),
            ('minutes', 'B', 22),
            ('seconds', 'B', 23),
            ('hundredths', 'B', 24),
            ('coordinate_transformation_process', 'B', 25),
            ('heading_alignment', '<H', 26),
            ('heading_bias', '<H', 28),
            ('sensor_source', 'B', 30),
            ('sensor_available', 'B', 31),
            ('bin_1_distance', '<H', 32),
            ('transmit_pulse_length', '<H', 34),
            ('starting_depth_cell', 'B', 36),
            ('ending_depth_cell', 'B', 37),
            ('false_target_threshold', 'B', 38),
            ('low_latency_trigger', 'B', 39),
            ('transmit_lag_distance', '<H', 40),
            ('cpu_board_serial_number', '<Q', 42),
            ('system_bandwidth', '<H', 50),
            ('system_power', 'B', 52),
            ('spare', 'B', 53),
            ('serial_number', '<I', 54),
            ('beam_angle', 'B', 58)
        )

        return Pd0TRDI.unpack_bytes(pd0_bytes, fixed_leader_format, offset)

    @staticmethod
    def decode_variable_leader(pd0_bytes, offset, data):
        """Decodes variabl leader data

        Parameters
        ----------
        pd0_bytes: bytearray
            Bytearray of all pd0 data
        offset: int
            Pointer into pd0_bytes
        data: dict
            Dictionary containing previously decoded data

        Returns
        -------
        :dict
            Dictionary of decoded data
        """

        # Define format
        variable_leader_format = (
            ('id', '<H', 0),
            ('ensemble_number', '<H', 2),
            ('rtc_year', 'B', 4),
            ('rtc_month', 'B', 5),
            ('rtc_day', 'B', 6),
            ('rtc_hour', 'B', 7),
            ('rtc_minutes', 'B', 8),
            ('rtc_seconds', 'B', 9),
            ('rtc_hundredths', 'B', 10),
            ('ensemble_number_msb', 'B', 11),
            ('bit_fault', 'B', 12),
            ('bit_count', 'B', 13),
            ('speed_of_sound', '<H', 14),
            ('depth_of_transducer', '<H', 16),
            ('heading', '<H', 18),
            ('pitch', '<h', 20),
            ('roll', '<h', 22),
            ('salinity', '<H', 24),
            ('temperature', '<h', 26),
            ('mpt_minutes', 'B', 28),
            ('mpt_seconds', 'B', 29),
            ('mpt_hundredths', 'B', 30),
            ('heading_standard_deviation', 'B', 31),
            ('pitch_standard_deviation', 'B', 32),
            ('roll_standard_deviation', 'B', 33),
            ('transmit_current', 'B', 34),
            ('transmit_voltage', 'B', 35),
            ('ambient_temperature', 'B', 36),
            ('pressure_positive', 'B', 37),
            ('pressure_negative', 'B', 38),
            ('attitude_temperature', 'B', 39),
            ('attitude', 'B', 40),
            ('contamination_sensor', 'B', 41),
            ('error_status_word', '<I', 42),
            ('reserved', '<H', 46),
            ('pressure', '<I', 48),
            ('pressure_variance', '<I', 52),
            ('spare', 'B', 56),
            ('rtc_y2k_century', 'B', 57),
            ('rtc_y2k_year', 'B', 58),
            ('rtc_y2k_month', 'B', 59),
            ('rtc_y2k_day', 'B', 60),
            ('rtc_y2k_hour', 'B', 61),
            ('rtc_y2k_minutes', 'B', 62),
            ('rtc_y2k_seconds', 'B', 63),
            ('rtc_y2k_hundredths', 'B', 64),
            ('lag_near_bottom', 'B', 65)
        )

        return Pd0TRDI.unpack_bytes(pd0_bytes, variable_leader_format, offset)

    def decode_per_cell_per_beam(pd0_bytes, offset, number_of_cells, number_of_beams, struct_format):
        """Parses fields that are stored in serial cells and beams structures.
        Returns an array of cell readings where each reading is an array containing the value at that beam.

        Parameters
        ----------
        pd0_bytes: bytearray
            Bytearray of all pd0 data
        offset: int
            Pointer into pd0_bytes
        number_of_cells: int
            Number of cells in data
        number of beams: int
            Number of beams in data
        struct_format: str
            A string identifying the type of data to decode

        Returns
        -------
        data: list
            A list of list containing cell data for each beam
        """

        data_size = struct.calcsize(struct_format)
        data = []
        # Loop through cells
        for cell in range(0, number_of_cells):
            cell_start = offset + cell * number_of_beams * data_size
            cell_data = []
            # Loop through beams in each cell
            for field in range(0, number_of_beams):
                field_start = cell_start + field * data_size
                data_bytes = pd0_bytes[field_start: field_start + data_size]
                field_data = struct.unpack(struct_format, data_bytes)[0]
                cell_data.append(field_data)
            data.append(cell_data)

        return data

    @staticmethod
    def decode_velocity(pd0_bytes, offset, data):
        """Decodes velocity data

        Parameters
        ----------
        pd0_bytes: bytearray
            Bytearray of all pd0 data
        offset: int
            Pointer into pd0_bytes
        data: dict
            Dictionary containing previously decoded data

        Returns
        -------
        velocity_data:dict
            Dictionary of decoded data
        """

        # Define format
        velocity_format = (('id', '<h', 0),)

        # Unpack data
        velocity_data = Pd0TRDI.unpack_bytes(pd0_bytes, velocity_format, offset)
        # Move past id field
        offset += 2
        # Arrange data in list of depth cells and beams or velocity components
        velocity_data['data'] = Pd0TRDI.decode_per_cell_per_beam(pd0_bytes,
                                                                 offset,
                                                                 data['fixed_leader']['number_of_cells'],
                                                                 data['fixed_leader']['number_of_beams'],
                                                                 '<h')

        return velocity_data

    @staticmethod
    def decode_correlation(pd0_bytes, offset, data):
        """Decodes correlation data

        Parameters
        ----------
        pd0_bytes: bytearray
            Bytearray of all pd0 data
        offset: int
            Pointer into pd0_bytes
        data: dict
            Dictionary containing previously decoded data

        Returns
        -------
        correlation_data:dict
            Dictionary of decoded data
        """

        correlation_format = (('id', '<H', 0),)
        # Unpack data
        correlation_data = Pd0TRDI.unpack_bytes(pd0_bytes, correlation_format, offset)
        # Move past id field
        offset += 2
        # Arrange data in list of depth cells and beams
        correlation_data['data'] = Pd0TRDI.decode_per_cell_per_beam(pd0_bytes,
                                                                    offset,
                                                                    data['fixed_leader']['number_of_cells'],
                                                                    data['fixed_leader']['number_of_beams'],
                                                                    'B')

        return correlation_data

    @staticmethod
    def decode_echo_intensity(pd0_bytes, offset, data):
        """Decodes echo intensity data

        Parameters
        ----------
        pd0_bytes: bytearray
            Bytearray of all pd0 data
        offset: int
            Pointer into pd0_bytes
        data: dict
            Dictionary containing previously decoded data

        Returns
        -------
        echo_intensity_data:dict
            Dictionary of decoded data
        """

        echo_intensity_format = (('id', '<H', 0),)
        # Unpack data
        echo_intensity_data = Pd0TRDI.unpack_bytes(pd0_bytes, echo_intensity_format, offset)
        # Move past id field
        offset += 2
        # Arrange data in list of depth cells and beams
        echo_intensity_data['data'] = Pd0TRDI.decode_per_cell_per_beam(pd0_bytes,
                                                                       offset,
                                                                       data['fixed_leader']['number_of_cells'],
                                                                       data['fixed_leader']['number_of_beams'],
                                                                       'B')

        return echo_intensity_data

    @staticmethod
    def decode_percent_good(pd0_bytes, offset, data):
        """Decodes percent good data

        Parameters
        ----------
        pd0_bytes: bytearray
            Bytearray of all pd0 data
        offset: int
            Pointer into pd0_bytes
        data: dict
            Dictionary containing previously decoded data

        Returns
        -------
        percent_good_data:dict
            Dictionary of decoded data
        """

        percent_good_format = (('id', '<H', 0),)
        # Unpack data
        percent_good_data = Pd0TRDI.unpack_bytes(pd0_bytes, percent_good_format, offset)
        # Move past id field
        offset += 2
        # Arrange data in list of depth cells and beams
        percent_good_data['data'] = Pd0TRDI.decode_per_cell_per_beam(pd0_bytes,
                                                                     offset,
                                                                     data['fixed_leader']['number_of_cells'],
                                                                     data['fixed_leader']['number_of_beams'],
                                                                     'B')

        return percent_good_data

    @staticmethod
    def decode_status(pd0_bytes, offset, data):
        """Decodes percent good data

        Parameters
        ----------
        pd0_bytes: bytearray
            Bytearray of all pd0 data
        offset: int
            Pointer into pd0_bytes
        data: dict
            Dictionary containing previously decoded data

        Returns
        -------
        status_data:dict
            Dictionary of decoded data
        """

        status_format = (('id', '<H', 0),)
        # Unpack data
        status_data = Pd0TRDI.unpack_bytes(pd0_bytes, status_format, offset)
        # Move past id field
        offset += 2
        # Arrange data in list of depth cells and beams
        status_data['data'] = Pd0TRDI.decode_per_cell_per_beam(pd0_bytes,
                                                               offset,
                                                               data['fixed_leader']['number_of_cells'],
                                                               data['fixed_leader']['number_of_beams'],
                                                               'B')

        return status_data

    @staticmethod
    def decode_bottom_track(pd0_bytes, offset, data):
        """Decodes bottom track data

        Parameters
        ----------
        pd0_bytes: bytearray
            Bytearray of all pd0 data
        offset: int
            Pointer into pd0_bytes
        data: dict
            Dictionary containing previously decoded data (not used)

        Returns
        -------
        bottom_track_data:dict
            Dictionary of decoded data
        """
        bottom_track_format = (('id', '<H', 0),
                               ('pings_per_ensemble_bp', '<H', 2),
                               ('delay_before_reaquire', '<H', 4),
                               ('correlation_magnitude_minimum_bc', 'B', 6),
                               ('evaluation_amplitude_minimum_ba', 'B', 7),
                               ('percent_good_minimum_bg', 'B', 8),
                               ('bottom_track_mode_bm', 'B', 9),
                               ('error_velocity_maximum_be', '<H', 10))

        bottom_track_data = Pd0TRDI.unpack_bytes(pd0_bytes, bottom_track_format, offset)
        bottom_track_data['range_lsb'] = Pd0TRDI.decode_per_cell_per_beam(pd0_bytes, offset + 16, 1, 4, '<H')
        bottom_track_data['velocity'] = Pd0TRDI.decode_per_cell_per_beam(pd0_bytes, offset + 24, 1, 4, '<h')
        bottom_track_data['correlation'] = Pd0TRDI.decode_per_cell_per_beam(pd0_bytes, offset + 32, 1, 4, 'B')
        bottom_track_data['amplitude'] = Pd0TRDI.decode_per_cell_per_beam(pd0_bytes, offset + 36, 1, 4, 'B')
        bottom_track_data['percent_good'] = Pd0TRDI.decode_per_cell_per_beam(pd0_bytes, offset + 40, 1, 4, 'B')
        bottom_track_data['rssi'] = Pd0TRDI.decode_per_cell_per_beam(pd0_bytes, offset + 72, 1, 4, 'B')
        bottom_track_data['range_msb'] = Pd0TRDI.decode_per_cell_per_beam(pd0_bytes, offset + 77, 1, 4, 'B')

        return bottom_track_data

    def decode_nmea(self, pd0_bytes, offset, data):
        """Decodes nmea data

        Parameters
        ----------
        pd0_bytes: bytearray
            Bytearray of all pd0 data
        offset: int
            Pointer into pd0_bytes
        data: dict
            Dictionary containing previously decoded data

        Returns
        -------
        nmea_data:dict
            Dictionary of decoded data
        """
        nmea_leader_format = (('id', '<H', 0),
                              ('msg_id', '<H', 2),
                              ('msg_size', '<H', 4),
                              ('delta_time', 'd', 6))

        nmea_data = Pd0TRDI.unpack_bytes(pd0_bytes, nmea_leader_format, offset)
        if nmea_data['msg_id'] in self.nmea_decoders:
            key = self.nmea_decoders[nmea_data['msg_id']][0]
            decoder = self.nmea_decoders[nmea_data['msg_id']][1]
            if key in data:
                data[key].append(decoder(pd0_bytes, offset + 14, nmea_data))
            else:
                data[key] = [decoder(pd0_bytes, offset + 14, nmea_data)]
        return nmea_data

    @staticmethod
    def decode_gga_100(pd0_bytes, offset, data):
        """Decodes gga data for WinRiver versions prior to 2.00

        Parameters
        ----------
        pd0_bytes: bytearray
            Bytearray of all pd0 data
        offset: int
            Pointer into pd0_bytes
        data: dict
            Dictionary containing previously decoded data

        Returns
        -------
        decoded_data:dict
            Dictionary of decoded data
        """

        # Define format
        format = (('header', '10s', 0),
                  ('utc', '10s', 10),
                  ('lat_deg', 'd', 20),
                  ('lat_ref', 'c', 28),
                  ('lon_deg', 'd', 29),
                  ('lon_ref', 'c', 37),
                  ('corr_qual', 'B', 38),
                  ('num_sats', 'B', 39),
                  ('hdop', 'f', 40),
                  ('alt', 'f', 44),
                  ('alt_unit', 'c', 48),
                  ('geoid', 'f', 49),
                  ('geoid_unit', 'c', 53),
                  ('d_gps_age', 'f', 54),
                  ('ref_stat_id', '<H', 58))

        # Decode data
        decoded_data = Pd0TRDI.unpack_bytes(pd0_bytes, format, offset)
        decoded_data['header'] = Pd0TRDI.bin2str(decoded_data['header']).rstrip('\x00')
        try:
            decoded_data['utc'] = float(re.findall(b'^\d+\.\d+|\d+', decoded_data['utc'])[0])
        except BaseException:
            decoded_data['utc'] = np.nan
        decoded_data['lat_ref'] = Pd0TRDI.bin2str(decoded_data['lat_ref'])
        decoded_data['lon_ref'] = Pd0TRDI.bin2str(decoded_data['lon_ref'])
        decoded_data['geoid_unit'] = Pd0TRDI.bin2str(decoded_data['geoid_unit'])
        decoded_data['alt_unit'] = Pd0TRDI.bin2str(decoded_data['alt_unit'])
        decoded_data['delta_time'] = data['delta_time']

        return decoded_data

    @staticmethod
    def decode_vtg_101(pd0_bytes, offset, data):
        """Decodes vtg data for WinRiver versions prior to 2.00

        Parameters
        ----------
        pd0_bytes: bytearray
            Bytearray of all pd0 data
        offset: int
            Pointer into pd0_bytes
        data: dict
            Dictionary containing previously decoded data

        Returns
        -------
        decoded_data:dict
            Dictionary of decoded data
        """

        # Define format
        format = (('header', '10s', 0),
                  ('course_true', 'f', 10),
                  ('true_indicator', 'c', 14),
                  ('course_mag', 'f', 15),
                  ('mag_indicator', 'c', 19),
                  ('speed_knots', 'f', 20),
                  ('knots_indicator', 'c', 24),
                  ('speed_kph', 'f', 25),
                  ('kph_indicator', 'c', 29),
                  ('mode_indicator', 'c', 30))

        # Decode data
        decoded_data = Pd0TRDI.unpack_bytes(pd0_bytes, format, offset)
        decoded_data['header'] = Pd0TRDI.bin2str(decoded_data['header']).rstrip('\x00')
        decoded_data['true_indicator'] = Pd0TRDI.bin2str(decoded_data['true_indicator'])
        decoded_data['mag_indicator'] = Pd0TRDI.bin2str(decoded_data['mag_indicator'])
        decoded_data['knots_indicator'] = Pd0TRDI.bin2str(decoded_data['knots_indicator'])
        decoded_data['kph_indicator'] = Pd0TRDI.bin2str(decoded_data['kph_indicator'])
        decoded_data['delta_time'] = data['delta_time']

        return decoded_data

    @staticmethod
    def decode_ds_102(pd0_bytes, offset, data):
        """Decodes depth sounder for WinRiver versions prior to 2.00

        Parameters
        ----------
        pd0_bytes: bytearray
            Bytearray of all pd0 data
        offset: int
            Pointer into pd0_bytes
        data: dict
            Dictionary containing previously decoded data

        Returns
        -------
        decoded_data:dict
            Dictionary of decoded data
        """

        # Define format
        format = (('header', '10s', 0),
                  ('depth_ft', 'f', 10),
                  ('ft_indicator', 'c', 14),
                  ('depth_m', 'f', 15),
                  ('m_indicator', 'c', 19),
                  ('depth_fath', 'f', 20),
                  ('fath_indicator', 'c', 24))

        # Decode data
        decoded_data = Pd0TRDI.unpack_bytes(pd0_bytes, format, offset)
        decoded_data['header'] = Pd0TRDI.bin2str(decoded_data['header']).rstrip('\x00')
        decoded_data['ft_indicator'] = Pd0TRDI.bin2str(decoded_data['ft_indicator'])
        decoded_data['m_indicator'] = Pd0TRDI.bin2str(decoded_data['m_indicator'])
        decoded_data['fath_indicator'] = Pd0TRDI.bin2str(decoded_data['fath_indicator'])
        decoded_data['delta_time'] = data['delta_time']

        return decoded_data

    @staticmethod
    def decode_ext_heading_103(pd0_bytes, offset, data):
        """Decodes external heading for WinRiver versions prior to 2.00

        Parameters
        ----------
        pd0_bytes: bytearray
            Bytearray of all pd0 data
        offset: int
            Pointer into pd0_bytes
        data: dict
            Dictionary containing previously decoded data

        Returns
        -------
        decoded_data:dict
            Dictionary of decoded data
        """

        # Define format
        format = (('header', '10s', 0),
                  ('heading_deg', 'd', 10),
                  ('h_true_indicator', 'c', 14))

        # Decode data
        decoded_data = Pd0TRDI.unpack_bytes(pd0_bytes, format, offset)
        decoded_data['header'] = Pd0TRDI.bin2str(decoded_data['header']).rstrip('\x00')
        decoded_data['h_true_indicator'] = Pd0TRDI.bin2str(decoded_data['h_true_indicator'])
        decoded_data['delta_time'] = data['delta_time']

        return decoded_data

    @staticmethod
    def decode_gga_104(pd0_bytes, offset, data):
        """Decodes gga data for WinRiver 2.00 and greater with ADCP's without integrated NMEA data

        Parameters
        ----------
        pd0_bytes: bytearray
            Bytearray of all pd0 data
        offset: int
            Pointer into pd0_bytes
        data: dict
            Dictionary containing previously decoded data

        Returns
        -------
        decoded_data:dict
            Dictionary of decoded data
        """

        # Define format
        format = (('header', '7s', 0),
                  ('utc', '10s', 7),
                  ('lat_deg', 'd', 17),
                  ('lat_ref', 'c', 25),
                  ('lon_deg', 'd', 26),
                  ('lon_ref', 'c', 34),
                  ('corr_qual', 'B', 35),
                  ('num_sats', 'B', 36),
                  ('hdop', 'f', 37),
                  ('alt', 'f', 41),
                  ('alt_unit', 'c', 45),
                  ('geoid', 'f', 46),
                  ('geoid_unit', 'c', 50),
                  ('d_gps_age', 'f', 51),
                  ('ref_stat_id', '<H', 55))

        # Decode data
        decoded_data = Pd0TRDI.unpack_bytes(pd0_bytes, format, offset)
        decoded_data['header'] = Pd0TRDI.bin2str(decoded_data['header']).rstrip('\x00')
        try:
            decoded_data['utc'] = float(re.findall(b'^\d+\.\d+|\d+', decoded_data['utc'])[0])
        except BaseException:
            decoded_data['utc'] = np.nan
        decoded_data['lat_ref'] = Pd0TRDI.bin2str(decoded_data['lat_ref'])
        decoded_data['lon_ref'] = Pd0TRDI.bin2str(decoded_data['lon_ref'])
        decoded_data['geoid_unit'] = Pd0TRDI.bin2str(decoded_data['geoid_unit'])
        decoded_data['alt_unit'] = Pd0TRDI.bin2str(decoded_data['alt_unit'])
        decoded_data['delta_time'] = data['delta_time']

        return decoded_data

    @staticmethod
    def decode_vtg_105(pd0_bytes, offset, data):
        """Decodes vtg data for WinRiver 2.00 and greater with ADCP's without integrated NMEA data

        Parameters
        ----------
        pd0_bytes: bytearray
            Bytearray of all pd0 data
        offset: int
            Pointer into pd0_bytes
        data: dict
            Dictionary containing previously decoded data

        Returns
        -------
        decoded_data:dict
            Dictionary of decoded data
        """

        # Define format
        format = (('header', '7s', 0),
                  ('course_true', 'f', 7),
                  ('true_indicator', 'c', 11),
                  ('course_mag', 'f', 12),
                  ('mag_indicator', 'c', 16),
                  ('speed_knots', 'f', 17),
                  ('knots_indicator', 'c', 21),
                  ('speed_kph', 'f', 22),
                  ('kph_indicator', 'c', 26),
                  ('mode_indicator', 'c', 27))

        # Decode data
        decoded_data = Pd0TRDI.unpack_bytes(pd0_bytes, format, offset)
        decoded_data['header'] = Pd0TRDI.bin2str(decoded_data['header']).rstrip('\x00')
        decoded_data['true_indicator'] = Pd0TRDI.bin2str(decoded_data['true_indicator'])
        decoded_data['mag_indicator'] = Pd0TRDI.bin2str(decoded_data['mag_indicator'])
        decoded_data['knots_indicator'] = Pd0TRDI.bin2str(decoded_data['knots_indicator'])
        decoded_data['kph_indicator'] = Pd0TRDI.bin2str(decoded_data['kph_indicator'])
        decoded_data['delta_time'] = data['delta_time']

        return decoded_data

    @staticmethod
    def decode_ds_106(pd0_bytes, offset, data):
        """Decodes depth sounder for WinRiver 2.00 and greater with ADCP's without integrated NMEA data

        Parameters
        ----------
        pd0_bytes: bytearray
            Bytearray of all pd0 data
        offset: int
            Pointer into pd0_bytes
        data: dict
            Dictionary containing previously decoded data

        Returns
        -------
        decoded_data:dict
            Dictionary of decoded data
        """

        # Define format
        format = (('header', '7s', 0),
                  ('depth_ft', 'f', 7),
                  ('ft_indicator', 'c', 11),
                  ('depth_m', 'f', 12),
                  ('m_indicator', 'c', 16),
                  ('depth_fath', 'f', 17),
                  ('fath_indicator', 'c', 21))

        # Decode data
        decoded_data = Pd0TRDI.unpack_bytes(pd0_bytes, format, offset)
        decoded_data['header'] = Pd0TRDI.bin2str(decoded_data['header']).rstrip('\x00')
        decoded_data['ft_indicator'] = Pd0TRDI.bin2str(decoded_data['ft_indicator'])
        decoded_data['m_indicator'] = Pd0TRDI.bin2str(decoded_data['m_indicator'])
        decoded_data['fath_indicator'] = Pd0TRDI.bin2str(decoded_data['fath_indicator'])
        decoded_data['delta_time'] = data['delta_time']

        return decoded_data

    @staticmethod
    def decode_ext_heading_107(pd0_bytes, offset, data):
        """Decodes external heading for WinRiver 2.00 and greater with ADCP's without integrated NMEA data

        Parameters
        ----------
        pd0_bytes: bytearray
            Bytearray of all pd0 data
        offset: int
            Pointer into pd0_bytes
        data: dict
            Dictionary containing previously decoded data

        Returns
        -------
        decoded_data:dict
            Dictionary of decoded data
        """

        # Define format
        format = (('header', '7s', 0),
                  ('heading_deg', 'd', 7),
                  ('h_true_indicator', 'c', 15))

        # Decode data
        decoded_data = Pd0TRDI.unpack_bytes(pd0_bytes, format, offset)
        decoded_data['header'] = Pd0TRDI.bin2str(decoded_data['header']).rstrip('\x00')
        if abs(decoded_data['heading_deg']) < 360:
            try:
                decoded_data['h_true_indicator'] = Pd0TRDI.bin2str(decoded_data['h_true_indicator'])
            except:
                decoded_data['h_true_indicator'] = ''
        else:
            decoded_data['heading_deg'] = np.nan
            decoded_data['h_true_indicator'] = ''
        decoded_data['delta_time'] = data['delta_time']

        return decoded_data

    @staticmethod
    def decode_gga_204(pd0_bytes, offset, data):
        """Decodes gga data for ADCP's with integrated NMEA data

        Parameters
        ----------
        pd0_bytes: bytearray
            Bytearray of all pd0 data
        offset: int
            Pointer into pd0_bytes
        data: dict
            Dictionary containing previously decoded data

        Returns
        -------
        decoded_data:dict
            Dictionary of decoded data
        """

        # Initialize dictionary
        decoded_data = {}
        decoded_data['header'] = ''
        decoded_data['utc'] = np.nan
        decoded_data['lat_deg'] = np.nan
        decoded_data['lat_ref'] = ''
        decoded_data['lon_deg'] = np.nan
        decoded_data['lon_ref'] = ''
        decoded_data['corr_qual'] = np.nan
        decoded_data['num_sats'] = np.nan
        decoded_data['hdop'] = np.nan
        decoded_data['alt'] = np.nan
        decoded_data['alt_unit'] = ''
        decoded_data['geoid'] = ''
        decoded_data['geoid_unit'] = ''
        decoded_data['d_gps_age'] = np.nan
        decoded_data['ref_stat_id'] = np.nan
        decoded_data['delta_time'] = np.nan

        # Decode NMEA sentence and split into an array
        format = str(data['msg_size']) + 'c'
        sentence = Pd0TRDI.bin2str(b''.join(list(struct.unpack(format, pd0_bytes[offset: offset + data['msg_size']]))))
        temp_array = np.array(sentence.split(','))
        temp_array[temp_array == '999.9'] = ''

        # Assign parts of array to dictionary
        try:
            decoded_data['delta_time'] = data['delta_time']
            decoded_data['header'] = temp_array[0]
            decoded_data['utc'] = valid_number(temp_array[1])
            lat_str = temp_array[2]
            lat_deg = valid_number(lat_str[0:2])
            decoded_data['lat_deg'] = lat_deg + valid_number(lat_str[2:]) / 60
            decoded_data['lat_ref'] = temp_array[3]
            lon_str = temp_array[4]
            lon_num = valid_number(lon_str)
            lon_deg = np.floor(lon_num / 100.)
            decoded_data['lon_deg'] = lon_deg + (((lon_num / 100.) - lon_deg) * 100.) / 60.
            decoded_data['lon_ref'] = temp_array[5]
            decoded_data['corr_qual'] = valid_number(temp_array[6])
            decoded_data['num_sats'] = valid_number(temp_array[7])
            decoded_data['hdop'] = valid_number(temp_array[8])
            decoded_data['alt'] = valid_number(temp_array[9])
            decoded_data['alt_unit'] = temp_array[10]
            decoded_data['geoid'] = temp_array[11]
            decoded_data['geoid_unit'] = temp_array[12]
            decoded_data['d_gps_age'] = valid_number(temp_array[13])
            idx_star = temp_array[14].find('*')
            decoded_data['ref_stat_id'] = valid_number(temp_array[15][:idx_star])

        except (ValueError, EOFError, IndexError):
            pass

        return decoded_data

    @staticmethod
    def decode_vtg_205(pd0_bytes, offset, data):
        """Decodes vtg data for ADCP's with integrated NMEA data

        Parameters
        ----------
        pd0_bytes: bytearray
            Bytearray of all pd0 data
        offset: int
            Pointer into pd0_bytes
        data: dict
            Dictionary containing previously decoded data

        Returns
        -------
        decoded_data:dict
            Dictionary of decoded data
        """

        # Initialize dictionary
        decoded_data = {}
        decoded_data['header'] = ''
        decoded_data['course_true'] = np.nan
        decoded_data['true_indicator'] = ''
        decoded_data['course_mag'] = np.nan
        decoded_data['mag_indicator'] = ''
        decoded_data['speed_knots'] = np.nan
        decoded_data['knots_indicator'] = ''
        decoded_data['speed_kph'] = np.nan
        decoded_data['kph_indicator'] = ''
        decoded_data['mode_indicator'] = ''
        decoded_data['delta_time'] = np.nan

        # Decode NMEA sentence and split into an array
        format = str(data['msg_size']) + 'c'
        sentence = Pd0TRDI.bin2str(b''.join(list(struct.unpack(format, pd0_bytes[offset: offset + data['msg_size']]))))
        temp_array = np.array(sentence.split(','))
        temp_array[temp_array == '999.9'] = ''

        # Assign parts of array to dictionary
        try:
            decoded_data['vtg_header'] = temp_array[0]
            decoded_data['course_true'] = valid_number(temp_array[1])
            decoded_data['true_indicator'] = temp_array[2]
            decoded_data['course_mag'] = valid_number(temp_array[3])
            decoded_data['mag_indicator'] = temp_array[4]
            decoded_data['speed_knots'] = valid_number(temp_array[5])
            decoded_data['knots_indicator'] = temp_array[6]
            decoded_data['speed_kph'] = valid_number(temp_array[7])
            decoded_data['kph_indicator'] = temp_array[8]
            idx_star = temp_array[9].find('*')
            decoded_data['mode_indicator'] = temp_array[9][:idx_star]
            decoded_data['delta_time'] = data['delta_time']

        except (ValueError, EOFError, IndexError):
            pass

        return decoded_data

    @staticmethod
    def decode_ds_206(pd0_bytes, offset, data):
        """Decodes depth sounder for ADCP's with integrated NMEA data

        Parameters
        ----------
        pd0_bytes: bytearray
            Bytearray of all pd0 data
        offset: int
            Pointer into pd0_bytes
        data: dict
            Dictionary containing previously decoded data

        Returns
        -------
        decoded_data:dict
            Dictionary of decoded data
        """

        # Initialize dictionary
        decoded_data = {}
        decoded_data['header'] = ''
        decoded_data['depth_ft'] = np.nan
        decoded_data['ft_indicator'] = ''
        decoded_data['depth_m'] = np.nan
        decoded_data['m_indicator'] = ''
        decoded_data['depth_fath'] = np.nan
        decoded_data['fath_indicator'] = ''
        decoded_data['delta_time'] = np.nan

        # Decode NMEA sentence and split into an array
        format = str(data['msg_size']) + 'c'
        sentence = Pd0TRDI.bin2str(b''.join(list(struct.unpack(format, pd0_bytes[offset: offset + data['msg_size']]))))
        temp_array = np.array(sentence.split(','))
        temp_array[temp_array == '999.9'] = ''

        # Assign parts of array to dictionary
        try:
            decoded_data['dbt_header'] = temp_array[0]
            decoded_data['depth_ft'] = valid_number(temp_array[1])
            decoded_data['ft_indicator'] = temp_array[2]
            decoded_data['depth_m'] = valid_number(temp_array[3])
            decoded_data['m_indicator'] = temp_array[4]
            decoded_data['depth_fath'] = valid_number(temp_array[5])
            idx_star = temp_array[6].find('*')
            decoded_data['fath_indicator'] = temp_array[6][:idx_star]
            decoded_data['delta_time'] = data['delta_time']

        except (ValueError, EOFError, IndexError):
            pass

        return decoded_data

    @staticmethod
    def decode_ext_heading_207(pd0_bytes, offset, data):
        """Decodes external heading for ADCP's with integrated NMEA data

        Parameters
        ----------
        pd0_bytes: bytearray
            Bytearray of all pd0 data
        offset: int
            Pointer into pd0_bytes
        data: dict
            Dictionary containing previously decoded data

        Returns
        -------
        decoded_data:dict
            Dictionary of decoded data
        """

        # Initialize dictionary
        decoded_data = {}
        decoded_data['header'] = ''
        decoded_data['heading_deg'] = np.nan
        decoded_data['h_true_indicator'] = ''
        decoded_data['delta_time'] = np.nan

        # Decode NMEA sentence and split into an array
        format = str(data['msg_size']) + 'c'
        sentence = Pd0TRDI.bin2str(b''.join(list(struct.unpack(format, pd0_bytes[offset: offset + data['msg_size']]))))
        temp_array = np.array(sentence.split(','))
        temp_array[temp_array == '999.9'] = ''

        # Assign parts of array to dictionary
        try:
            decoded_data['header'] = temp_array[0]
            decoded_data['heading_deg'] = valid_number(temp_array[1])
            idx_star = temp_array[2].find('*')
            decoded_data['h_true_indicator'] = temp_array[2][:idx_star]
            decoded_data['delta_time'] = data['delta_time']

        except (ValueError, EOFError, IndexError):
            pass

        return decoded_data

    @staticmethod
    def decode_dbt_sentence(pd0_bytes, offset, data):
        """Stores dbt sentence

        Parameters
        ----------
        pd0_bytes: bytearray
            Bytearray of all pd0 data
        offset: int
            Pointer into pd0_bytes
        data: dict
            Dictionary containing previously decoded data

        Returns
        -------
        decoded_data:dict
            Dictionary of decoded data
        """

        return Pd0TRDI.decode_nmea_sentence(pd0_bytes, offset, data, 'dbt_sentence')

    @staticmethod
    def decode_gga_sentence(pd0_bytes, offset, data):
        """Stores dbt sentence

        Parameters
        ----------
        pd0_bytes: bytearray
            Bytearray of all pd0 data
        offset: int
            Pointer into pd0_bytes
        data: dict
            Dictionary containing previously decoded data

        Returns
        -------
        decoded_data:dict
            Dictionary of decoded data
        """

        return Pd0TRDI.decode_nmea_sentence(pd0_bytes, offset, data, 'gga_sentence')

    @staticmethod
    def decode_vtg_sentence(pd0_bytes, offset, data):
        """Stores dbt sentence

        Parameters
        ----------
        pd0_bytes: bytearray
            Bytearray of all pd0 data
        offset: int
            Pointer into pd0_bytes
        data: dict
            Dictionary containing previously decoded data

        Returns
        -------
        decoded_data:dict
            Dictionary of decoded data
        """

        return Pd0TRDI.decode_nmea_sentence(pd0_bytes, offset, data, 'vtg_sentence')

    @staticmethod
    def decode_gsa_sentence(pd0_bytes, offset, data):
        """Stores dbt sentence

        Parameters
        ----------
        pd0_bytes: bytearray
            Bytearray of all pd0 data
        offset: int
            Pointer into pd0_bytes
        data: dict
            Dictionary containing previously decoded data

        Returns
        -------
        decoded_data:dict
            Dictionary of decoded data
        """

        return Pd0TRDI.decode_nmea_sentence(pd0_bytes, offset, data, 'gsa_sentence')

    @staticmethod
    def decode_nmea_sentence(pd0_bytes, offset, data, target):
        """Decodes nmea sentence

        Parameters
        ----------
        pd0_bytes: bytearray
            Bytearray of all pd0 data
        offset: int
            Pointer into pd0_bytes
        data: dict
            Dictionary containing previously decoded data
        target: str
            Dictionary key for decoded data in data

        Returns
        -------
        decoded_data:dict
            Dictionary of decoded data
        """

        # Compute number of characters in the sentence
        offset_idx = data['header']['address_offsets'].index(offset)

        if offset_idx + 1 == data['header']['number_of_data_types']:
            end_offset = data['header']['number_of_bytes']
        else:
            end_offset = data['header']['address_offsets'][offset_idx + 1]
        number_of_characters = end_offset - data['header']['address_offsets'][offset_idx]

        # Generate format string
        format_str = str(number_of_characters - 4) + 'c'
        format = (('sentence', format_str, 0))
        offset = data['header']['address_offsets'][offset_idx]
        # Decode data
        sentence = struct.unpack(format_str, pd0_bytes[offset + 4: offset + number_of_characters ])
        try:
            end_of_sentence = sentence.index(b'\n') + 1
            sentence = b''.join(sentence[0:end_of_sentence]).decode('utf-8')
        except ValueError:
            sentence = ''
        # Create or add to list of target sentences
        if target in data:
            decoded_data = data[target]
            decoded_data.append(sentence)
        else:
            decoded_data = sentence

        return decoded_data

    @staticmethod
    def decode_surface_leader(pd0_bytes, offset, data):
        """Decodes surface velocity leader

        Parameters
        ----------
        pd0_bytes: bytearray
            Bytearray of all pd0 data
        offset: int
            Pointer into pd0_bytes
        data: dict
            Dictionary containing previously decoded data

        Returns
        -------
        surface_leader_data:dict
            Dictionary of decoded data
        """
        surface_leader_format = (('id', '<H', 0),
                                 ('cell_count', 'B', 2),
                                 ('cell_size', '<H', 3),
                                 ('range_cell_1', '<H', 5))

        surface_leader_data = Pd0TRDI.unpack_bytes(pd0_bytes, surface_leader_format, offset)
        return surface_leader_data

    @staticmethod
    def decode_surface_velocity(pd0_bytes, offset, data):
        """Decodes surface velocity data

        Parameters
        ----------
        pd0_bytes: bytearray
            Bytearray of all pd0 data
        offset: int
            Pointer into pd0_bytes
        data: dict
            Dictionary containing previously decoded data

        Returns
        -------
        surface_velocity_data:dict
            Dictionary of decoded data
        """
        surface_velocity_format = (('id', '<H', 0),)

        surface_velocity_data = Pd0TRDI.unpack_bytes(pd0_bytes, surface_velocity_format, offset)
        surface_velocity_data['velocity'] = Pd0TRDI.decode_per_cell_per_beam(pd0_bytes, offset + 2,
                                                                             data['surface_leader']['cell_count'],
                                                                             4, '<h')
        return surface_velocity_data

    @staticmethod
    def decode_surface_correlation(pd0_bytes, offset, data):
        """Decodes surface correlation data

        Parameters
        ----------
        pd0_bytes: bytearray
            Bytearray of all pd0 data
        offset: int
            Pointer into pd0_bytes
        data: dict
            Dictionary containing previously decoded data

        Returns
        -------
        surface_velocity_data:dict
            Dictionary of decoded data
        """
        surface_correlation_format = (('id', '<H', 0),)

        surface_correlation_data = Pd0TRDI.unpack_bytes(pd0_bytes, surface_correlation_format, offset)
        surface_correlation_data['correlation'] = Pd0TRDI.decode_per_cell_per_beam(pd0_bytes, offset + 2,
                                                                                   data['surface_leader']['cell_count'],
                                                                                   4, 'B')
        return surface_correlation_data

    @staticmethod
    def decode_surface_intensity(pd0_bytes, offset, data):
        """Decodes surface intensity data

        Parameters
        ----------
        pd0_bytes: bytearray
            Bytearray of all pd0 data
        offset: int
            Pointer into pd0_bytes
        data: dict
            Dictionary containing previously decoded data

        Returns
        -------
        surface_rssi_data:dict
            Dictionary of decoded data
        """
        surface_rssi_format = (('id', '<H', 0),)

        surface_rssi_data = Pd0TRDI.unpack_bytes(pd0_bytes, surface_rssi_format, offset)
        surface_rssi_data['rssi'] = Pd0TRDI.decode_per_cell_per_beam(pd0_bytes, offset + 2,
                                                                     data['surface_leader']['cell_count'],
                                                                     4, 'B')
        return surface_rssi_data

    @staticmethod
    def decode_surface_percent_good(pd0_bytes, offset, data):
        """Decodes surface percent good data

        Parameters
        ----------
        pd0_bytes: bytearray
            Bytearray of all pd0 data
        offset: int
            Pointer into pd0_bytes
        data: dict
            Dictionary containing previously decoded data

        Returns
        -------
        surface_per_good_data:dict
            Dictionary of decoded data
        """
        surface_per_good_format = (('id', '<H', 0),)

        surface_per_good_data = Pd0TRDI.unpack_bytes(pd0_bytes, surface_per_good_format, offset)
        surface_per_good_data['percent_good'] = Pd0TRDI.decode_per_cell_per_beam(pd0_bytes, offset + 2,
                                                                                 data['surface_leader']['cell_count'],
                                                                                 4, 'B')
        return surface_per_good_data

    @staticmethod
    def decode_surface_status(pd0_bytes, offset, data):
        """Decodes surface percent good data

        Parameters
        ----------
        pd0_bytes: bytearray
            Bytearray of all pd0 data
        offset: int
            Pointer into pd0_bytes
        data: dict
            Dictionary containing previously decoded data

        Returns
        -------
        surface_statusdata:dict
            Dictionary of decoded data
        """
        surface_status_format = (('id', '<H', 0),)

        surface_status_data = Pd0TRDI.unpack_bytes(pd0_bytes, surface_status_format, offset)
        surface_status_data['percent_good'] = Pd0TRDI.decode_per_cell_per_beam(pd0_bytes, offset + 2,
                                                                               data['surface_leader']['cell_count'],
                                                                               4, 'B')
        return surface_status_data

    @staticmethod
    def decode_auto_config(pd0_bytes, offset, data):
        """Decodes auto configuration data

        Parameters
        ----------
        pd0_bytes: bytearray
            Bytearray of all pd0 data
        offset: int
            Pointer into pd0_bytes
        data: dict
            Dictionary of previously decoded data

        Returns
        -------
        auto_config_data:dict
            Dictionary of decoded data
        """
        auto_config_leader_format = (('id', '<H', 0), ('beam_count', 'B', 2))
        auto_config_beam_format = (('setup', 'B', 0),
                                   ('depth', '<H', 1),
                                   ('ping_count', 'B', 3),
                                   ('ping_type', 'B', 4),
                                   ('cell_count', '<H', 5),
                                   ('cell_size', '<H', 7),
                                   ('bin_1_mid', '<H', 9),
                                   ('code_reps', 'B', 11),
                                   ('transmit_length', '<H', 12),
                                   ('lag_length', '<H', 15),
                                   ('transmit_bandwidth', 'B', 16),
                                   ('receive_bandwidth', 'B', 17),
                                   ('min_ping_interval', '<H', 18))
        auto_config_data = {}
        auto_config_data['leader'] = Pd0TRDI.unpack_bytes(pd0_bytes, auto_config_leader_format, offset)

        for n in range(1, auto_config_data['leader']['beam_count'] + 1):
            label = 'beam_' + str(n)
            beam_offset = offset + 3 + (20 * (n - 1))
            auto_config_data[label] = Pd0TRDI.unpack_bytes(pd0_bytes, auto_config_beam_format, beam_offset)

        return auto_config_data

    @staticmethod
    def decode_vertical_beam(pd0_bytes, offset, data):
        """Decodes vertical beam data

        Parameters
        ----------
        pd0_bytes: bytearray
            Bytearray of all pd0 data
        offset: int
            Pointer into pd0_bytes
        data: dict
            Dictionary containing fixed leader data

        Returns
        -------
        vertical_beam_data:dict
            Dictionary of decoded data
        """
        vertical_beam_format = (('id', '<H', 0),
                                ('eval_amp', 'B', 2),
                                ('rssi', 'B', 3),
                                ('range', 'L', 4),
                                ('status', 'B', 8))

        vertical_beam_data = Pd0TRDI.unpack_bytes(pd0_bytes, vertical_beam_format, offset)
        return vertical_beam_data

    @staticmethod
    def decode_transformation_matrix(pd0_bytes, offset, data):
        """Decodes transformation matrix

        Parameters
        ----------
        pd0_bytes: bytearray
            Bytearray of all pd0 data
        offset: int
            Pointer into pd0_bytes
        data: dict
            Dictionary containing fixed leader data

        Returns
        -------
        matrix_data:dict
            Dictionary of decoded data
        """
        matrix_id_format = (('id', '<H', 0),)
        matrix_data_format = (('element', '<h', 0),)

        matrix_data = Pd0TRDI.unpack_bytes(pd0_bytes, matrix_id_format, offset)
        matrix = []
        for row in range(4):
            row_list = []
            for col in range(4):
                offset = offset + 2
                # row.append(struct.unpack('<H', pd0_bytes[offset: offset + 2])[0])
                row_list.append(Pd0TRDI.unpack_bytes(pd0_bytes, matrix_data_format, offset)['element'])
            matrix.append(row_list)
        matrix_data['matrix'] = matrix

        return matrix_data


class Hdr(object):
    """Class to hold header variables.

    Attributes
    ----------
    bytes_per_ens: int
        Number of bytes in ensemble
    data_offsets: int
        File offset to start of ensemble
    n_data_types: int
        Number of data types in ensemble
    data_ok: int

    invalid: str
        Leader ID that was not recognized
    """

    def __init__(self, n_ensembles, n_types):
        """Initialize instance variables to empty arrays.

        Parameters
        ----------
        n_ensembles: int
            Number of ensembles
        n_types: int
            Number of data types
        """
        self.bytes_per_ens = nans(n_ensembles)
        self.data_offsets = nans([n_ensembles, n_types])
        self.n_data_types = nans(n_ensembles)
        self.data_ok = nans(n_ensembles)
        self.invalid = [''] * n_ensembles

    def populate_data(self, n_ens, data):
        """Populates the class with data for an ensemble.

        Parameters
        ----------
        i_ens: int
            Ensemble index
        data: dict
            Dictionary of all data for this ensemble
        """

        if 'header' in data:
            self.bytes_per_ens[n_ens] = data['header']['number_of_bytes']
            self.data_offsets[n_ens, :len(data['header']['address_offsets'])] = \
                np.array(data['header']['address_offsets'])
            self.n_data_types[n_ens] = data['header']['number_of_data_types']
            self.invalid[n_ens] = data['header']['invalid']


class Inst(object):
    """Class to hold information about the instrument.

    Attributes
    ----------
    beam_ang: np.array(int)
        Angle of transducers in degrees
    beams: np.array(int)
        Number of beams used for velocity
    data_type: list
        Data type
    firm_ver: np.array(str)
        Firmware version
    freq: np.array(int)
        Frequency of ADCP in kHz
    pat = list
        Beam pattern
    res_RDI:
        Reserved for TRDI
    sensor_CFG: np.array(int)
        Sensor configuration
    xducer: list
        Indicates if transducer is attached
    t_matrix: np.array(float)
        Transformation matrix
    demod: np.array(int)
        Demodulation code
    serial_number: int
        serial number of ADCP
    """

    def __init__(self, n_ensembles):
        """Initialize instance variables.

        Parameters
        ----------
        n_ensembles: int
            Number of ensembles
        """

        #TODO change n_ensembles to (ensembles,)
        self.beam_ang = nans(n_ensembles)
        self.beams = nans(n_ensembles)
        self.data_type = [''] * n_ensembles
        self.firm_ver = nans(n_ensembles)
        self.freq = nans(n_ensembles)
        self.pat = [''] * n_ensembles
        self.res_RDI = 0
        self.sensor_CFG = nans(n_ensembles)
        self.xducer = [''] * n_ensembles
        self.t_matrix = np.tile([np.nan], [4, 4])
        self.demod = nans(n_ensembles)
        self.serial_number = np.nan

    def populate_data(self, i_ens, data):
        """Populates the class with data for an ensemble.

        Parameters
        ----------
        i_ens: int
            Ensemble index
        data: dict
            Dictionary of all data for this ensemble
        """


        if 'fixed_leader' in data:
            self.firm_ver[i_ens] = data['fixed_leader']['cpu_firmware_version'] + \
                            (data['fixed_leader']['cpu_firmware_revision']  / 100)

            # Convert system_configuration_ls to individual bits
            bitls = "{0:08b}".format(data['fixed_leader']['system_configuration_ls'])
            val = int(bitls[5:], 2)
            if val == 0:
                self.freq[i_ens] = 75
            elif val == 1:
                self.freq[i_ens] = 150
            elif val == 2:
                self.freq[i_ens] = 300
            elif val == 3:
                self.freq[i_ens] = 600
            elif val == 4:
                self.freq[i_ens] = 1200
            elif val == 5:
                self.freq[i_ens] = 2400
            else:
                self.freq[i_ens] = np.nan

            val = int(bitls[4], 2)
            if val == 0:
                self.pat[i_ens] = 'Concave'
            elif val == 1:
                self.pat[i_ens] = 'Convex'
            else:
                self.pat[i_ens] = 'n/a'

            self.sensor_CFG[i_ens] = int(bitls[2:3], 2) + 1

            val = int(bitls[1], 2)
            if val == 0:
                self.xducer[i_ens] = 'Not Attached'
            elif val == 1:
                self.xducer[i_ens] = 'Attached'
            else:
                self.xducer[i_ens] = 'n/a'

            # Convert system_configuration_ms to individual bits
            bitms = "{0:08b}".format(data['fixed_leader']['system_configuration_ms'])

            val = int(bitms[6:], 2)
            if val == 0:
                self.beam_ang[i_ens] = 15
            elif val == 1:
                self.beam_ang[i_ens] = 20
            elif val == 2:
                self.beam_ang[i_ens] = 30
            elif val == 3:
                self.beam_ang[i_ens] = np.nan
            else:
                self.beam_ang[i_ens] = np.nan

            val = int(bitms[:4], 2)
            if val == 4:
                self.beams[i_ens] = 4
            elif val == 5:
                self.beams[i_ens] = 5
                self.demod[i_ens] = 1
            elif val == 15:
                self.beams[i_ens] = 5
                self.demod[i_ens] = 2
            else:
                self.beams[i_ens] = np.nan
                self.demod[i_ens] = np.nan

            if data['fixed_leader']['simulation_data_flag'] == 0:
                self.data_type[i_ens] = 'Real'
            else:
                self.data_type[i_ens] = 'Simu'

            self.serial_number = data['fixed_leader']['serial_number']

        if 'transformation_matrix' in data:
            self.res_RDI = 0
            # Scale transformation matrix
            self.t_matrix = np.array(data['transformation_matrix']['matrix']) / 10000


class AutoMode(object):
    """Class to hold auto configuration mode settings for each beam.

    Attributes
    ----------
    beam_count: np.array(int)
        Number of beams
    Beam1: Beam
        Object of class Beam
    Beam2: Beam
        Object of class Beam
    Beam3: Beam
        Object of class Beam
    Beam4: Beam
        Object of class Beam
    Reserved: np.array
    """

    def __init__(self, n_ensembles):
        """Initialize instance variables.

        Parameters
        ----------
        n_ensembles: int
            Number of ensembles
        """
        self.beam_count = nans(n_ensembles)
        self.Beam1 = Beam(n_ensembles)
        self.Beam2 = Beam(n_ensembles)
        self.Beam3 = Beam(n_ensembles)
        self.Beam4 = Beam(n_ensembles)
        self.Reserved = nans(n_ensembles)

    def populate_data(self, i_ens, data):
        """Populates the class with data for an ensemble.

        Parameters
        ----------
        i_ens: int
            Ensemble index
        data: dict
            Dictionary of all data for this ensemble
        """

        if 'auto_configuration' in data:
            self.beam_count[i_ens] = data['auto_configuration']['leader']['beam_count']
            self.Beam1.populate_data(i_ens, data['auto_configuration']['beam_1'])
            self.Beam2.populate_data(i_ens, data['auto_configuration']['beam_2'])
            self.Beam3.populate_data(i_ens, data['auto_configuration']['beam_3'])
            self.Beam4.populate_data(i_ens, data['auto_configuration']['beam_4'])


class Beam(object):
    """Class to hold auto configuration settings for a beam.

    Attributes
    ----------
    mode: np.array(int)
        Water mode
    depth_cm: np.array(int)
        Depth in cm
    ping_count: np.array(int)
        Number of pings
    ping_type: np.array(int)
        Type of pings
    cell_count: np.array(int)
        Number of cells
    cell_size_cm: np.array(int)
        Cell size in cm
    cell_mid_cm: np.array(int)
        Distance to center of cell 1 in cm
    code_repeat: np.array(int)
        Number of code repeats
    trans_length_cm: np.array(int)
        Transmit length in cm
    lag_length_cm: np.array(int)
        Lag length in cm
    transmit_bw: np.array(int)
        Transmit bandwidth
    receive_bw: np.array(int)
        Receive bandwidth
    ping_interval_ms: np.array(int)
        Time between pings in ms
    """

    def __init__(self, n_ensembles):
        """Initialize instance variables.

        Parameters
        ----------
        n_ensembles: int
            Number of ensembles
        """

        self.mode = nans(n_ensembles)
        self.depth_cm = nans(n_ensembles)
        self.ping_count = nans(n_ensembles)
        self.ping_type = nans(n_ensembles)
        self.cell_count = nans(n_ensembles)
        self.cell_size_cm = nans(n_ensembles)
        self.cell_mid_cm = nans(n_ensembles)
        self.code_repeat = nans(n_ensembles)
        self.trans_length_cm = nans(n_ensembles)
        self.lag_length_cm = nans(n_ensembles)
        self.transmit_bw = nans(n_ensembles)
        self.receive_bw = nans(n_ensembles)
        self.ping_interval_ms = nans(n_ensembles)

    def populate_data(self, i_ens, beam_data):
        """Populates the class with data for an ensemble.

        Parameters
        ----------
        i_ens: int
            Ensemble index
        data: dict
            Dictionary of all data for this ensemble
        """

        self.mode = beam_data['setup']
        self.depth_cm = beam_data['depth']
        self.ping_count = beam_data['ping_count']
        self.ping_type = beam_data['ping_type']
        self.cell_count = beam_data['cell_count']
        self.cell_size_cm = beam_data['cell_size']
        self.cell_mid_cm = beam_data['bin_1_mid']
        self.code_repeat = beam_data['code_reps']
        self.trans_length_cm = beam_data['transmit_length']
        self.lag_length_cm = beam_data['lag_length']
        self.transmit_bw = beam_data['transmit_bandwidth']
        self.receive_bw = beam_data['receive_bandwidth']
        self.ping_interval_ms = beam_data['min_ping_interval']


class Bt(object):
    """Class to hold bottom track data.

    Attributes
    ----------
    corr: np.array(int)
        Correlation for each beam
    depth_m: np.array(float)
        Depth for each beam
    eval_amp: np.array(int)
        Return amplitude for each beam
    ext_depth_cm: np.array(int)
        External depth in cm
    pergd: np.array(int)
        Percent good
    rssi: np.array(int)
        Return signal strength indicator in counts for each beam
    vel_mps: np.array(float)
        Velocity in m/s, rows depend on coordinate system
    """

    def __init__(self, n_ensembles, n_velocities):
        """Initialize instance variables.

        Parameters
        ----------
        n_ensembles: int
            Number of ensembles
        n_velocities: int
            Number of velocity beams
        """

        self.corr = nans([n_velocities, n_ensembles])
        self.depth_m = nans([n_velocities, n_ensembles])
        self.eval_amp = nans([n_velocities, n_ensembles])
        self.ext_depth_cm = nans(n_ensembles)
        self.pergd = nans([n_velocities, n_ensembles])
        self.rssi = nans([n_velocities, n_ensembles])
        self.vel_mps = nans([n_velocities, n_ensembles])

    def populate_data(self, i_ens, data):
        """Populates the class with data for an ensemble.

        Parameters
        ----------
        i_ens: int
            Ensemble index
        data: dict
            Dictionary of all data for this ensemble
        """

        if 'bottom_track' in data:
            # Combine bytes to compute depth
            self.depth_m[0:4, i_ens] = np.squeeze(np.array(data['bottom_track']['range_lsb']).T) + \
                                       np.squeeze(np.array(data['bottom_track']['range_msb']).T) * 2e16 / 100
            self.vel_mps[0:4, i_ens] = np.squeeze(np.array(data['bottom_track']['velocity']).T)
            self.corr[0:4, i_ens] = np.squeeze(np.array(data['bottom_track']['correlation']).T)
            self.eval_amp[0:4, i_ens] = np.squeeze(np.array(data['bottom_track']['amplitude']).T)
            self.pergd[0:4, i_ens] = np.squeeze(np.array(data['bottom_track']['percent_good']).T)


class Cfg(object):
    """Class to hold configuration settings.

    Attributes
    ----------
    ba: np.array(int)
        Bottom track amplitude threshold
    bc: np.array(int)
        Bottom track correlation threshold
    be_mmps: np.array(int)
        Bottom track error velocity threshold
    bg: np.array(int)
        Bottom track percent good threshold
    bm: np.array(int)
        Bottom mode
    bp: np.array(int)
        Number of bottom pings
    bx_dm: np.array(int)
        Maximum tracking depth in decimeters
    code_reps: np.array(int)
        Number of code repetitions
    coord_sys: np.array(str)
        Coordinate system
    cpu_ser_no: np.array(int)
        CPU serial number
    cq: np.array(int)
        Transmit power
    cx: np.array(int)
        Low latency trigger
    dist_bin1_cm: np.array(int)
        Distance to center of bin 1 from transducer
    ea_deg: np.array(int)
        Heading alignment
    eb_deg: np.array(int)
        Heading bias
    sensor_avail: np.array(str)
        Sensor availability codes
    ex: np.array(str)
        Coordinate transformation codes
    ez: np.array(str)
        Sensor codes
    head_src: np.array(str)
        Heading source
    lag_cm: np.array(int)
        Lag
    map_bins: np.array(str)
        Bin mapping
    n_beams: np.array(int)
        Number of velocity beams
    pitch_src: np.array(str)
        Source of pitch data
    ref_lay_end_cell: np.array(int)
        Reference layer end
    ref_lay_str_cell: np.array(int)
        Reference layer start
    roll_src: np.array(str)
        Roll source
    sal_src: np.array(str)
        Salinity source
    wm: np.array(int)
        Water mode
    sos_src: np.array(str)
        Speed of sound source
    temp_src: np.array(str)
        Temperature source
    tp_sec: np.array(int)
        Time between pings
    use_3beam: np.array(str)
        Setting on whether to use 3-beam solutions or not
    use_pr =: np.array(str)
        Setting to use pitch and roll or not
    wa: np.array(int)
        Water track amplitude threshold
    wb: np.array(int)
        Water track bandwidth control
    wc: np.array(int)
        Water track correlation threshold
    we_mmps: np.array(int)
        Water track error velocity threshold
    wf_cm: np.array(int)
        Blank after transmit
    wg_per: np.array(int)
        Water track percent good threshold
    wj: np.array(int)
        Receiver gain setting
    wn: np.array(int)
        Number of depth cells (bins)
    wp: np.array(int)
        Number of water pings
    ws_cm: np.array(int)
        Bin size
    xdcr_dep_srs: np.array(str)
        Salinity source
    xmit_pulse_cm: np.array(int)
        Transmit pulse length
    lag_near_bottom: np.array(int)
        Lag near bottom setting
    """

    def __init__(self, n_ensembles):
        """Initialize instance variables.

        Parameters
        ----------
        n_ensembles: int
            Number of ensembles
        """

        self.ba = nans(n_ensembles)
        self.bc = nans(n_ensembles)
        self.be_mmps = nans(n_ensembles)
        self.bg = nans(n_ensembles)
        self.bm = nans(n_ensembles)
        self.bp = nans(n_ensembles)
        self.bx_dm = nans(n_ensembles)
        self.code_reps = nans(n_ensembles)
        self.coord_sys = [''] * n_ensembles
        self.cpu_ser_no = nans([n_ensembles, 8])
        self.cq = nans(n_ensembles)
        self.cx = nans(n_ensembles)
        self.dist_bin1_cm = nans(n_ensembles)
        self.ea_deg = nans(n_ensembles)
        self.eb_deg = nans(n_ensembles)
        self.sensor_avail = [''] * n_ensembles
        self.ex = [''] * n_ensembles
        self.ez = [''] * n_ensembles
        self.head_src = [''] * n_ensembles
        self.lag_cm = nans(n_ensembles)
        self.map_bins = [''] * n_ensembles
        self.n_beams = nans(n_ensembles)
        self.pitch_src = [''] * n_ensembles
        self.ref_lay_end_cell = nans(n_ensembles)
        self.ref_lay_str_cell = nans(n_ensembles)
        self.roll_src = [''] * n_ensembles
        self.sal_src = [''] * n_ensembles
        self.wm = nans(n_ensembles)
        self.sos_src = [''] * n_ensembles
        self.temp_src = [''] * n_ensembles
        self.tp_sec = nans(n_ensembles)
        self.use_3beam = [''] * n_ensembles
        self.use_pr = [''] * n_ensembles
        self.wa = nans(n_ensembles)
        self.wb = nans(n_ensembles)
        self.wc = nans(n_ensembles)
        self.we_mmps = nans(n_ensembles)
        self.wf_cm = nans(n_ensembles)
        self.wg_per = nans(n_ensembles)
        self.wj = nans(n_ensembles)
        self.wn = nans(n_ensembles)
        self.wp = nans(n_ensembles)
        self.ws_cm = nans(n_ensembles)
        self.xdcr_dep_srs = [''] * n_ensembles
        self.xmit_pulse_cm = nans(n_ensembles)
        self.lag_near_bottom = nans(n_ensembles)

    def populate_data (self, i_ens, data):
        """Populates the class with data for an ensemble.

        Parameters
        ----------
        i_ens: int
            Ensemble index
        data: dict
            Dictionary of all data for this ensemble
        """

        if 'fixed_leader' in data:
            self.n_beams[i_ens] = data['fixed_leader']['number_of_beams']
            self.wn[i_ens] = data['fixed_leader']['number_of_cells']
            self.wp[i_ens] = data['fixed_leader']['number_of_water_pings']
            self.ws_cm[i_ens] = data['fixed_leader']['depth_cell_size']
            self.wf_cm[i_ens] = data['fixed_leader']['blank_after_transmit']
            self.wm[i_ens] = data['fixed_leader']['water_mode']
            self.wc[i_ens] = data['fixed_leader']['low_correlation_threshold']
            self.code_reps[i_ens] = data['fixed_leader']['number_of_code_repetitions']
            self.wg_per[i_ens] = data['fixed_leader']['minimum_percentage_water_profile_pings']
            self.we_mmps[i_ens] = data['fixed_leader']['error_velocity_threshold']
            self.tp_sec[i_ens] = data['fixed_leader']['minutes'] * 60. + \
                                     data['fixed_leader']['seconds'] + \
                                     data['fixed_leader']['hundredths'] * 0.01

            # Convert coordinate_transformation_process to individual bits
            self.ex[i_ens] = "{0:08b}".format(data['fixed_leader']['coordinate_transformation_process'])

            val = int(self.ex[i_ens][3:5], 2)
            if val == 0:
                self.coord_sys[i_ens] = 'Beam'
            elif val == 1:
                self.coord_sys[i_ens] = 'Inst'
            elif val == 2:
                self.coord_sys[i_ens] = 'Ship'
            elif val == 3:
                self.coord_sys[i_ens] = 'Earth'
            else:
                self.coord_sys[i_ens] = "N/a"

            val = int(self.ex[i_ens][5], 2)
            if val == 0:
                self.use_pr = 'No'
            elif val == 1:
                self.use_pr = 'Yes'
            else:
                self.use_pr = 'N/a'

            val = int(self.ex[i_ens][6], 2)
            if val == 0:
                self.use_3beam = 'No'
            elif val == 1:
                self.use_3beam = 'Yes'
            else:
                self.use_3beam = 'N/a'

            val = int(self.ex[i_ens][7], 2)
            if val == 0:
                self.map_bins = 'No'
            elif val == 1:
                self.map_bins = 'Yes'
            else:
                self.map_bins = 'N/a'

            self.ea_deg[i_ens] = data['fixed_leader']['heading_alignment'] * 0.01
            self.eb_deg[i_ens] = data['fixed_leader']['heading_bias'] * 0.01

            # Convert sensour_source to individual bits
            self.ez[i_ens] = "{0:08b}".format(data['fixed_leader']['sensor_source'])

            val = int(self.ez[i_ens][:2], 2)
            if val == 0:
                self.sos_src[i_ens] = 'Manual EC'
            elif val == 1:
                self.sos_src[i_ens] = 'Calculated'
            elif val == 3:
                self.sos_src[i_ens] = 'SVSS Sensor'
            else:
                self.sos_src[i_ens] = 'N/a'

            val = int(self.ez[i_ens][2], 2)
            if val == 0:
                self.xdcr_dep_srs[i_ens] = 'Manual ED'
            if val == 1:
                self.xdcr_dep_srs[i_ens] = 'Sensor'
            else:
                self.xdcr_dep_srs[i_ens] = 'N/a'

            val = int(self.ez[i_ens][3], 2)
            if val == 0:
                self.head_src[i_ens] = 'Manual EH'
            if val == 1:
                self.head_src[i_ens] = 'Int. Sensor'
            else:
                self.head_src[i_ens] = 'N/a'

            val = int(self.ez[i_ens][4], 2)
            if val == 0:
                self.pitch_src[i_ens] = 'Manual EP'
            if val == 1:
                self.pitch_src[i_ens] = 'Int. Sensor'
            else:
                self.pitch_src[i_ens] = 'N/a'

            val = int(self.ez[i_ens][5], 2)
            if val == 0:
                self.roll_src[i_ens] = 'Manual ER'
            if val == 1:
                self.roll_src[i_ens] = 'Int. Sensor'
            else:
                self.roll_src[i_ens] = 'N/a'

            val = int(self.ez[i_ens][6], 2)
            if val == 0:
                self.xdcr_dep_srs[i_ens] = 'Manual ES'
            if val == 1:
                self.xdcr_dep_srs[i_ens] = 'Int. Sensor'
            else:
                self.xdcr_dep_srs[i_ens] = 'N/a'

            val = int(self.ez[i_ens][7], 2)
            if val == 0:
                self.temp_src[i_ens] = 'Manual ET'
            if val == 1:
                self.temp_src[i_ens] = 'Int. Sensor'
            else:
                self.temp_src[i_ens] = 'N/a'

            self.sensor_avail[i_ens] = "{0:08b}".format(data['fixed_leader']['sensor_available'])
            self.dist_bin1_cm[i_ens] = data['fixed_leader']['bin_1_distance']
            self.xmit_pulse_cm[i_ens] = data['fixed_leader']['transmit_pulse_length']
            self.ref_lay_str_cell[i_ens] = data['fixed_leader']['starting_depth_cell']
            self.ref_lay_end_cell[i_ens] = data['fixed_leader']['ending_depth_cell']
            self.wa[i_ens] = data['fixed_leader']['false_target_threshold']
            self.cx[i_ens] = data['fixed_leader']['low_latency_trigger']
            self.lag_cm[i_ens] = data['fixed_leader']['transmit_lag_distance']
            self.cpu_ser_no[i_ens] = data['fixed_leader']['cpu_board_serial_number']
            self.wb[i_ens] = data['fixed_leader']['system_bandwidth']
            self.cq[i_ens] = data['fixed_leader']['system_power']

        if 'variable_leader' in data:
            self.lag_near_bottom[i_ens] = data['variable_leader']['lag_near_bottom']

        if 'bottom_track' in data:
            self.bp[i_ens] = data['bottom_track']['pings_per_ensemble_bp']
            self.bc[i_ens] = data['bottom_track']['correlation_magnitude_minimum_bc']
            self.ba[i_ens] = data['bottom_track']['evaluation_amplitude_minimum_ba']
            self.bg[i_ens] = data['bottom_track']['percent_good_minimum_bg']
            self.bm[i_ens] = data['bottom_track']['bottom_track_mode_bm']
            self.be_mmps[i_ens] = data['bottom_track']['error_velocity_maximum_be']


class Gps(object):
    """Class to hold GPS data from WinRiver. CLASS NOT USED

    Attributes
    ----------
    alt_m: np.array(float)
        Altitude in meters
    gga_diff: np.array(int)
        Differential correction indicator
    gga_hdop: np.array(float)
        Horizontal dilution of precision
    gga_n_stats: np.array(int)
        Number of satellites
    gga_vel_e_mps: np.array(float)
        Velocity in east direction from GGA data
    gga_vel_n_mps: np.array(float)
        Velocity in north directio from GGA data
    gsa_p_dop: np.array(int)
        Position dilution of precision
    gsa_sat: np.array(int)
        Satellites
    gsa_v_dop: np.array(float)
        Vertical dilution of precision
    lat_deg: np.array(float)
        Latitude in degrees
    long_deg: np.array(float)
        Longitude in degrees
    vtg_vel_e_mps: np.array(float)
        Velocity in east direction from VTG data
    vtg_vel_n_mps: np.array(float)
        Velocity in north direction from VTG data
    """

    def __init__(self, n_ensembles):
        """Initialize instance variables.

        Parameters
        ----------
        n_ensembles: int
            Number of ensembles
        """

        self.alt_m = nans(n_ensembles)
        self.gga_diff = nans(n_ensembles)
        self.gga_hdop = nans(n_ensembles)
        self.gga_n_stats = nans(n_ensembles)
        self.gga_vel_e_mps = nans(n_ensembles)
        self.gga_vel_n_mps = nans(n_ensembles)
        self.gsa_p_dop = nans(n_ensembles)
        self.gsa_sat = nans([n_ensembles, 6])
        self.gsa_v_dop = nans(n_ensembles)
        self.lat_deg = nans(n_ensembles)
        self.long_deg = nans(n_ensembles)
        self.vtg_vel_e_mps = nans(n_ensembles)
        self.vtg_vel_n_mps = nans(n_ensembles)


class Gps2(object):
    """Class to hold GPS data for WinRiver II.

    Attributes
    ----------
    gga_delta_time: np.array(float)
        Time between ping and gga data
    gga_header: list
        GGA header
    gga_sentence: list
        GGA sentence
    utc: np.array(float)
        UTC time
    lat_deg: np.array(float)
        Latitude in degrees
    lat_ref: list
        Latitude reference
    lon_deg: np.array(float)
        Longitude in degrees
    lon_ref: list
        Longitude reference
    corr_qual: np.array(float)
        Differential quality indicator
    num_sats: np.array(int)
        Number of satellites
    hdop: np.array(float)
        Horizontal dilution of precision
    alt: np.array(float)
        Altitude
    alt_unit: list
        Units for altitude
    geoid: np.array(float)
        Geoid height
    geoid_unit: list
        Units for geoid height
    d_gps_age: np.array(float)
        Age of differential correction
    ref_stat_id: np.array(float)
        Reference station ID
    vtg_delta_time: np.array(float)
        Time between ping and VTG data
    vtg_header: list
        VTG header
    vtg_sentence: list
        VTG sentence
    course_true: np.array(float)
        Course relative to true north
    true_indicator: list
        True north indicator
    course_mag: np.array(float)
        Course relative to magnetic north
    mag_indicator: list
        Magnetic north indicator
    speed_knots: np.array(float)
        Speed in knots
    knots_indicator: list
        Knots indicator
    speed_kph: np.array(float)
        Speed in kilometers per hour
    kph_indicator: list
        Kilometers per hour indicator
    mode_indicator: list
        Mode indicator
    dbt_delta_time: np.array(float)
        Time between ping and echo sounder data
    dbt_header: list
        Echo sounder header
    depth_ft: np.array(float)
        Depth in ft from echo sounder
    ft_indicator: list
        Feet indicator
    depth_m: np.array(float)
        Depth in meters from echo sounder
    m_indicator: list
        Meters indicator
    depth_fath: np.array(float)
        Depth in fathoms from echo sounder
    fath_indicator: list
        Fathoms indicator
    hdt_delta_time: np.array(float)
        Time between ping and external heading data
    hdt_header: list
        External heading header
    heading_deg: np.array(float)
        Heading in degrees from external heading
    h_true_indicator: list
        Heading indicator to true north
    gga_velE_mps: np.array(float)
        Velocity in east direction in m/s from GGA for WR
    gga_velN_mps: np.array(float)
        Velocity in north direction in m/s from GGA for WR
    vtg_velE_mps: np.array(float)
        Velocity in east direction in m/s from VTG for WR
    vtg_velN_mps: np.array(float)
        Velocity in north direction in m/s from VTG for WR
    """

    def __init__(self, n_ensembles, wr2):
        """Initialize instance variables.

        Parameters
        ----------
        n_ensembles: int
            Number of ensembles
        wr2: bool
            Setting of whether data is from WR or WR2
        """

        self.gga_delta_time = np.full([n_ensembles, 20], np.nan)
        self.gga_header = np.full([n_ensembles, 20], '      ')
        self.gga_sentence = np.full([n_ensembles, 20], '')
        self.utc = np.full([n_ensembles, 20], np.nan)
        self.lat_deg = np.zeros([n_ensembles, 20])
        self.lat_ref = np.full([n_ensembles, 20], '')
        self.lon_deg = np.zeros([n_ensembles, 20])
        self.lon_ref = np.full([n_ensembles, 20], '')
        self.corr_qual = np.full([n_ensembles, 20], np.nan)
        self.num_sats = np.full([n_ensembles, 20], np.nan)
        self.hdop = np.full([n_ensembles, 20], np.nan)
        self.alt = np.full([n_ensembles, 20], np.nan)
        self.alt_unit = np.full([n_ensembles, 20], '')
        self.geoid = np.full([n_ensembles, 20], np.nan)
        self.geoid_unit = np.full([n_ensembles, 20], '')
        self.d_gps_age = np.full([n_ensembles, 20], np.nan)
        self.ref_stat_id = np.full([n_ensembles, 20], np.nan)
        self.vtg_delta_time = np.full([n_ensembles, 20], np.nan)
        self.vtg_header = np.full([n_ensembles, 20], '      ')
        self.vtg_sentence = np.full([n_ensembles, 20], '')
        self.course_true = np.full([n_ensembles, 20], np.nan)
        self.true_indicator = np.full([n_ensembles, 20], '')
        self.course_mag = np.full([n_ensembles, 20], np.nan)
        self.mag_indicator = np.full([n_ensembles, 20], '')
        self.speed_knots = np.full([n_ensembles, 20], np.nan)
        self.knots_indicator = np.full([n_ensembles, 20], '')
        self.speed_kph = np.zeros([n_ensembles, 20])
        self.kph_indicator = np.full([n_ensembles, 20], '')
        self.mode_indicator = np.full([n_ensembles, 20], '')
        self.dbt_delta_time = np.full([n_ensembles, 20], np.nan)
        self.dbt_header = np.full([n_ensembles, 20], '      ')
        self.depth_ft = np.full([n_ensembles, 20], np.nan)
        self.ft_indicator = np.full([n_ensembles, 20], '')
        self.depth_m = np.zeros([n_ensembles, 20])
        self.m_indicator = np.full([n_ensembles, 20], '')
        self.depth_fath = np.full([n_ensembles, 20], np.nan)
        self.fath_indicator = np.full([n_ensembles, 20], '')
        self.hdt_delta_time = np.full([n_ensembles, 20], np.nan)
        self.hdt_header = np.full([n_ensembles, 20], '      ')
        self.heading_deg = np.full([n_ensembles, 20], np.nan)
        self.h_true_indicator = np.full([n_ensembles, 20], '')

        # if wr2:
        self.gga_velE_mps = nans(n_ensembles)
        self.gga_velN_mps = nans(n_ensembles)
        self.vtg_velE_mps = nans(n_ensembles)
        self.vtg_velN_mps = nans(n_ensembles)

    def populate_data(self, i_ens, data):
        """Populates the class with data for an ensemble.

        Parameters
        ----------
        i_ens: int
            Ensemble index
        data: dict
            Dictionary of all data for this ensemble
        """

        if 'gga' in data:

            # Check size and expand if needed
            if len(data['gga']) > self.gga_delta_time.shape[1]:
                self.gga_expand(len(data['gga']))

            for n, gga_data in enumerate(data['gga']):
                # Try implemented because of occasional garbage in data stream.
                # This prevents a crash and data after garbage are not used, but any data before garbage is saved
                try:
                    self.gga_delta_time[i_ens, n] = gga_data['delta_time']
                    self.gga_header[i_ens, n] = gga_data['header']
                    self.utc[i_ens, n] = gga_data['utc']
                    self.lat_deg[i_ens, n] = gga_data['lat_deg']
                    self.lat_ref[i_ens, n] = gga_data['lat_ref']
                    self.lon_deg[i_ens, n] = gga_data['lon_deg']
                    self.lon_ref[i_ens, n] = gga_data['lon_ref']
                    self.corr_qual[i_ens, n] = gga_data['corr_qual']
                    self.num_sats[i_ens, n] = gga_data['num_sats']
                    self.hdop[i_ens, n] = gga_data['hdop']
                    self.alt[i_ens, n] = gga_data['alt']
                    self.alt_unit[i_ens, n] = gga_data['alt_unit']
                    self.geoid[i_ens, n] = gga_data['geoid']
                    self.geoid_unit[i_ens, n] = gga_data['geoid_unit']
                    self.d_gps_age[i_ens, n] = gga_data['d_gps_age']
                    self.ref_stat_id[i_ens, n] = gga_data['ref_stat_id']
                except:
                    pass

        if 'vtg' in data:

            # Check size and expand if needed
            if len(data['vtg']) > self.vtg_delta_time.shape[1]:
                self.vtg_expand(len(data['vtg']))

            for n, vtg_data in enumerate(data['vtg']):
                # Try implemented because of occasional garbage in data stream.
                # This prevents a crash and data after garbage are not used, but any data before garbage is saved
                try:
                    self.vtg_delta_time[i_ens, n] = vtg_data['delta_time']
                    self.vtg_header[i_ens, n] = vtg_data['header']
                    self.course_true[i_ens, n] = vtg_data['course_true']
                    self.true_indicator[i_ens, n] = vtg_data['true_indicator']
                    self.course_mag[i_ens, n] = vtg_data['course_mag']
                    self.mag_indicator[i_ens, n] = vtg_data['mag_indicator']
                    self.speed_knots[i_ens, n] = vtg_data['speed_knots']
                    self.knots_indicator[i_ens, n] = vtg_data['knots_indicator']
                    self.speed_kph[i_ens, n] = vtg_data['speed_kph']
                    self.kph_indicator[i_ens, n] = vtg_data['kph_indicator']
                    self.mode_indicator[i_ens, n] = vtg_data['mode_indicator']
                except:
                    pass

        if 'ds' in data:

            # Check size and expand if needed
            if len(data['ds']) > self.dbt_delta_time.shape[1]:
                self.dbt_expand(len(data['ds']))

            for n, dbt_data in enumerate(data['ds']):
                # Try implemented because of occasional garbage in data stream.
                # This prevents a crash and data after garbage are not used, but any data before garbage is saved
                try:
                    self.dbt_delta_time[i_ens, n] = dbt_data['delta_time']
                    self.dbt_header[i_ens, n] = dbt_data['header']
                    self.depth_ft[i_ens, n] = dbt_data['depth_ft']
                    self.ft_indicator[i_ens, n] = dbt_data['ft_indicator']
                    self.depth_m[i_ens, n] = dbt_data['depth_m']
                    self.m_indicator[i_ens, n] = dbt_data['m_indicator']
                    self.depth_fath[i_ens, n] = dbt_data['depth_fath']
                    self.fath_indicator[i_ens, n] = dbt_data['fath_indicator']
                except:
                    pass

        if 'ext_heading' in data:

            # Check size and expand if needed
            if len(data['ext_heading']) > self.hdt_delta_time.shape[1]:
                self.hdt_expand(len(data['ext_heading']))

            for n, hdt_data in enumerate(data['ext_heading']):
                # Try implemented because of occasional garbage in data stream.
                # This prevents a crash and data after garbage are not used, but any data before garbage is saved
                try:
                    self.hdt_delta_time[i_ens, n] = hdt_data['delta_time']
                    self.hdt_header[i_ens, n] = hdt_data['header']
                    self.heading_deg[i_ens, n] = hdt_data['heading_deg']
                    self.h_true_indicator[i_ens, n] = hdt_data['h_true_indicator']
                except:
                    pass

    def gga_expand(self, n_samples):
        """Expand arrays.

        Parameters
        ----------
        n_samples: int
            Desired size of array
        """

        # Determine amount of required expansion
        n_expansion = n_samples - self.gga_delta_time.shape[1]
        n_ensembles = self.gga_delta_time.shape[0]

        # Expand arrays
        self.gga_delta_time = np.concatenate(
            (self.gga_delta_time, np.tile(np.nan, (n_ensembles, n_expansion))), axis=1)
        self.utc = np.concatenate(
            (self.utc, np.tile(np.nan, (n_ensembles, n_expansion))), axis=1)
        self.lat_deg = np.concatenate(
            (self.lat_deg, np.tile(np.nan, (n_ensembles, n_expansion))), axis=1)
        self.lon_deg = np.concatenate(
            (self.lon_deg, np.tile(np.nan, (n_ensembles, n_expansion))), axis=1)
        self.corr_qual = np.concatenate(
            (self.corr_qual, np.tile(np.nan, (n_ensembles, n_expansion))), axis=1)
        self.num_sats = np.concatenate(
            (self.num_sats, np.tile(np.nan, (n_ensembles, n_expansion))), axis=1)
        self.hdop = np.concatenate(
            (self.hdop, np.tile(np.nan, (n_ensembles, n_expansion))), axis=1)
        self.alt = np.concatenate(
            (self.alt, np.tile(np.nan, (n_ensembles, n_expansion))), axis=1)
        self.geoid = np.concatenate(
            (self.geoid, np.tile(np.nan, (n_ensembles, n_expansion))), axis=1)
        self.d_gps_age = np.concatenate(
            (self.d_gps_age, np.tile(np.nan, (n_ensembles, n_expansion))), axis=1)
        self.ref_stat_id = np.concatenate(
            (self.ref_stat_id, np.tile(np.nan, (n_ensembles, n_expansion))), axis=1)

        self.gga_header = np.concatenate(
            (self.gga_header, np.tile('', (n_ensembles, n_expansion))), axis=1)
        self.geoid_unit = np.concatenate(
            (self.geoid_unit, np.tile('', (n_ensembles, n_expansion))), axis=1)
        self.alt_unit = np.concatenate(
            (self.alt_unit, np.tile('', (n_ensembles, n_expansion))), axis=1)
        self.lon_ref = np.concatenate(
            (self.lon_ref, np.tile('', (n_ensembles, n_expansion))), axis=1)
        self.lat_ref = np.concatenate(
            (self.lat_ref, np.tile('', (n_ensembles, n_expansion))), axis=1)

    def vtg_expand(self, n_samples):
        """Expand arrays.

        Parameters
        ----------
        n_samples: int
            Desired size of array
        """

        # Determine amount of required expansion
        n_expansion = n_samples - self.vtg_delta_time.shape[1]
        n_ensembles = self.vtg_delta_time.shape[0]

        # Expand arrays
        self.vtg_delta_time = np.concatenate(
            (self.vtg_delta_time, np.tile(np.nan, (n_ensembles, n_expansion))), axis=1)
        self.course_true = np.concatenate(
            (self.course_true, np.tile(np.nan, (n_ensembles, n_expansion))), axis=1)
        self.course_mag = np.concatenate(
            (self.course_mag, np.tile(np.nan, (n_ensembles, n_expansion))), axis=1)
        self.speed_knots = np.concatenate(
            (self.speed_knots, np.tile(np.nan, (n_ensembles, n_expansion))), axis=1)
        self.speed_kph = np.concatenate(
            (self.speed_kph, np.tile(np.nan, (n_ensembles, n_expansion))), axis=1)

        self.kph_indicator = np.concatenate(
            (self.kph_indicator, np.tile('', (n_ensembles, n_expansion))), axis=1)
        self.mode_indicator = np.concatenate(
            (self.mode_indicator, np.tile('', (n_ensembles, n_expansion))), axis=1)
        self.vtg_header = np.concatenate(
            (self.vtg_header, np.tile('', (n_ensembles, n_expansion))), axis=1)
        self.true_indicator = np.concatenate(
            (self.true_indicator, np.tile('', (n_ensembles, n_expansion))), axis=1)
        self.mag_indicator = np.concatenate(
            (self.mag_indicator, np.tile('', (n_ensembles, n_expansion))), axis=1)
        self.knots_indicator = np.concatenate(
            (self.knots_indicator, np.tile('', (n_ensembles, n_expansion))), axis=1)

    def dbt_expand(self, n_samples):
        """Expand arrays.

        Parameters
        ----------
        n_samples: int
            Desired size of array
        """

        # Determine amount of required expansion
        n_expansion = n_samples - self.dbt_delta_time.shape[1]
        n_ensembles = self.dbt_delta_time.shape[0]

        # Expand arrays
        self.dbt_delta_time = np.concatenate(
            (self.dbt_delta_time, np.tile(np.nan, (n_ensembles, n_expansion))), axis=1)
        self.depth_ft = np.concatenate(
            (self.depth_ft, np.tile(np.nan, (n_ensembles, n_expansion))), axis=1)
        self.depth_m = np.concatenate(
            (self.depth_m, np.tile(np.nan, (n_ensembles, n_expansion))), axis=1)
        self.depth_fath = np.concatenate(
            (self.depth_fath, np.tile(np.nan, (n_ensembles, n_expansion))), axis=1)

        self.fath_indicator = np.concatenate(
            (self.fath_indicator, np.tile(np.nan, (n_ensembles, n_expansion))), axis=1)
        self.dbt_header = np.concatenate(
            (self.dbt_header, np.tile(np.nan, (n_ensembles, n_expansion))), axis=1)
        self.ft_indicator = np.concatenate(
            (self.ft_indicator, np.tile(np.nan, (n_ensembles, n_expansion))), axis=1)
        self.m_indicator = np.concatenate(
            (self.m_indicator, np.tile(np.nan, (n_ensembles, n_expansion))), axis=1)

    def hdt_expand(self, n_samples):
        """Expand arrays.

        Parameters
        ----------
        n_samples: int
            Desired size of array
        """

        # Determine amount of required expansion
        n_expansion = n_samples - self.hdt_delta_time.shape[1]
        n_ensembles = self.hdt_delta_time.shape[0]

        # Expand the arrays
        self.hdt_delta_time = np.concatenate(
            (self.hdt_delta_time, np.tile(np.nan, (n_ensembles, n_expansion))), axis=1)
        self.heading_deg = np.concatenate(
            (self.heading_deg, np.tile(np.nan, (n_ensembles, n_expansion))), axis=1)
        self.h_true_indicator = np.concatenate(
            (self.h_true_indicator, np.tile(np.nan, (n_ensembles, n_expansion))), axis=1)
        self.hdt_header = np.concatenate(
            (self.hdt_header, np.tile(np.nan, (n_ensembles, n_expansion))), axis=1)


class Nmea(object):
    """Class to hold raw NMEA sentences.

    Attributes
    ----------
    gga: list
        List of GGA sentences
    gsa: list
        List of GSA sentences
    vtg: list
        List of VTG sentences
    dbt: list
        List of DBT sentences
    """

    def __init__(self, n_ensembles):
        """Initialize instance variables.

        Parameters
        ----------
        n_ensembles: int
            Number of ensembles
        """
        self.gga = [''] * n_ensembles
        self.gsa = [''] * n_ensembles
        self.vtg = [''] * n_ensembles
        # self.raw = ['']*n_ensembles DSM: not sure this was used
        self.dbt = [''] * n_ensembles

    def populate_data(self, i_ens, data):
        """Populates the class with data for an ensemble.

        Parameters
        ----------
        i_ens: int
            Ensemble index
        data: dict
            Dictionary of all data for this ensemble
        """

        if 'gga_sentence' in data:
            self.gga[i_ens] = data['gga_sentence']

        if 'vtg_sentence' in data:
            self.vtg[i_ens] = data['vtg_sentence']

        if 'gsa_sentence' in data:
            self.gsa[i_ens] = data['gsa_sentence']

        if 'dbt_sentence' in data:
            self.dbt[i_ens] = data['dbt_sentence']


class Sensor(object):
    """Class to hold sensor data.

    Attributes
    ----------
    ambient_temp: np.array(int)
        ADC ambient temperature
    attitude_temp: np.array(int)
        ADC attitude temperature
    attitude: np.array(int)
        ADC attitude
    bit_test: np.array(int)
        Bit test results
    bit_test_count: np.array(int)
        Number of fails for newer ADCPs, not used for Rio Grande
    contam_sensor: np.array(int)
        ADC contamination sensor
    date: np.array(int)
        Date
    date_y2k: np.array(int)
        Y2K compatible date
    date_not_y2k: np.array(int)
        Date not Y2K compatible
    error_status_word: np.array(int)
        Error status codes
    heading_deg: np.array(float)
        Heading to magnetic north in degrees
    heading_std_dev_deg: np.array(float)
        Standard deviation of headings for an ensemble
    mpt_msc: np.array(int)
        Minimum time prior to ping
    num: np.array(int)
        Ensemble number
    num_fact: np.array(int)
        Number fraction
    num_tot: np.array(int)
        Number total
    orient: list
        Orientation of ADCP
    pitch_std_dev_deg: np.array(float)
        Standard deviation of pitch for an ensemble
    pitch_deg: np.array(float)
        Pitch in degrees
    pressure_neg: np.array(int)
        ADC pressure negative
    pressure_pos: np.array(int)
        ADC pressure positive
    pressure_pascal: np.array(int)
        Pressure at transducer face in deca-pascals
    pressure_var_pascal: np.array(int)
        Pressure variance in deca-pascals
    roll_std_dev_deg: np.array(float)
        Standard deviation of roll for an ensemble
    roll_deg: np.array(float)
        Roll in degrees
    salinity_ppt: np.array(int)
        Salinit in parts per thousand
    sos_mps: np.array(int)
        Speed of sound in m/s
    temperature_deg_c: np.array(float)
        Water temperatuer in degrees C
    time: np.array(int)
        Time
    time_y2k: np.array(int)
        Y2K compatible time
    xdcr_depth_dm: np.array(int)
        Transducer depth in decimeters
    xmit_current: np.array(int)
        Transmit current
    self.xmit_voltage = nans(n_ensembles)
        Transmit voltage
    self.vert_beam_eval_amp: np.array(int)
        Vertical beam amplitude
    self.vert_beam_RSSI_amp: np.array(int)
        Vertical beam return signal stength indicator
    self.vert_beam_range_m: np.array(float)
        Vertical beam range in m
    self.vert_beam_gain: list
        Vertical beam gain setting
    self.vert_beam_status: np.array(int)
        Vertical beam status code
    """

    def __init__(self, n_ensembles):
        """Initialize instance variables.

        Parameters
        ----------
        n_ensembles: int
            Number of ensembles
        """

        self.ambient_temp = nans(n_ensembles)
        self.attitude_temp = nans(n_ensembles)
        self.attitude = nans(n_ensembles)
        self.bit_test = nans(n_ensembles)
        self.bit_test_count = nans(n_ensembles)
        self.contam_sensor = nans(n_ensembles)
        self.date = nans([n_ensembles, 3])
        self.date_y2k = nans([n_ensembles, 4])
        self.date_not_y2k = nans([n_ensembles, 3])
        self.error_status_word = [''] * n_ensembles
        self.heading_deg = nans(n_ensembles)
        self.heading_std_dev_deg = nans(n_ensembles)
        self.mpt_msc = nans([n_ensembles, 3])
        self.num = nans(n_ensembles)
        self.num_fact = nans(n_ensembles)
        self.num_tot = nans(n_ensembles)
        self.orient = [''] * n_ensembles
        self.pitch_std_dev_deg = nans(n_ensembles)
        self.pitch_deg = nans(n_ensembles)
        self.pressure_neg = nans(n_ensembles)
        self.pressure_pos = nans(n_ensembles)
        self.pressure_pascal = nans(n_ensembles)
        self.pressure_var_pascal = nans(n_ensembles)
        self.roll_std_dev_deg = nans(n_ensembles)
        self.roll_deg = nans(n_ensembles)
        self.salinity_ppt = nans(n_ensembles)
        self.sos_mps = nans(n_ensembles)
        self.temperature_deg_c = nans(n_ensembles)
        self.time = nans([n_ensembles, 4])
        self.time_y2k = nans([n_ensembles, 4])
        self.xdcr_depth_dm = nans(n_ensembles)
        self.xmit_current = nans(n_ensembles)
        self.xmit_voltage = nans(n_ensembles)
        self.vert_beam_eval_amp = nans(n_ensembles)
        self.vert_beam_RSSI_amp = nans(n_ensembles)
        self.vert_beam_range_m = nans(n_ensembles)
        self.vert_beam_gain = [''] * n_ensembles
        self.vert_beam_status = np.zeros(n_ensembles)

    def populate_data(self, i_ens, data):
        """Populates the class with data for an ensemble.

        Parameters
        ----------
        i_ens: int
            Ensemble index
        data: dict
            Dictionary of all data for this ensemble
        """

        if 'fixed_leader' in data and 'variable_leader' in data:
            # Convert system_configuration_ls to 1s and 0s
            bitls = "{0:08b}".format(data['fixed_leader']['system_configuration_ls'])

            # Convert first two bits to integer
            val = int(bitls[0], 2)
            if val == 0:
                self.orient[i_ens] = 'Down'
            elif val == 1:
                self.orient[i_ens] = 'Up'
            else:
                self.orient[i_ens] = 'n/a'

            self.num[i_ens] = data['variable_leader']['ensemble_number']

            # Store data and time as list
            self.date_not_y2k[i_ens, :] = [data['variable_leader']['rtc_year'],
                                                  data['variable_leader']['rtc_month'],
                                                  data['variable_leader']['rtc_day']]
            self.time[i_ens, :] = [data['variable_leader']['rtc_hour'],
                                          data['variable_leader']['rtc_minutes'],
                                          data['variable_leader']['rtc_seconds'],
                                          data['variable_leader']['rtc_hundredths']]

            self.num_fact[i_ens] = data['variable_leader']['ensemble_number_msb']
            self.num_tot[i_ens] = self.num[i_ens] + self.num_fact[i_ens] * 65535
            self.bit_test[i_ens] = data['variable_leader']['bit_fault']
            self.bit_test_count[i_ens] = data['variable_leader']['bit_count']
            self.sos_mps[i_ens] = data['variable_leader']['speed_of_sound']
            self.xdcr_depth_dm[i_ens] = data['variable_leader']['depth_of_transducer']
            self.heading_deg[i_ens] = data['variable_leader']['heading'] / 100.
            self.pitch_deg[i_ens] = data['variable_leader']['pitch'] / 100.
            self.roll_deg[i_ens] = data['variable_leader']['roll'] / 100.
            self.salinity_ppt[i_ens] = data['variable_leader']['salinity']
            self.temperature_deg_c[i_ens] = data['variable_leader']['temperature'] / 100.
            self.mpt_msc[i_ens, :] = [data['variable_leader']['mpt_minutes'],
                                             data['variable_leader']['mpt_seconds'],
                                             data['variable_leader']['mpt_hundredths']]
            self.heading_std_dev_deg[i_ens] = data['variable_leader']['heading_standard_deviation']
            self.pitch_std_dev_deg[i_ens] = data['variable_leader']['pitch_standard_deviation'] / 10.
            self.roll_std_dev_deg[i_ens] = data['variable_leader']['roll_standard_deviation'] / 10.
            self.xmit_current[i_ens] = data['variable_leader']['transmit_current']
            self.xmit_voltage[i_ens] = data['variable_leader']['transmit_voltage']
            self.ambient_temp[i_ens] = data['variable_leader']['ambient_temperature']
            self.pressure_pos[i_ens] = data['variable_leader']['pressure_positive']
            self.pressure_neg[i_ens] = data['variable_leader']['pressure_negative']
            self.attitude_temp[i_ens] = data['variable_leader']['attitude_temperature']
            self.attitude[i_ens] = data['variable_leader']['attitude']
            self.contam_sensor[i_ens] = data['variable_leader']['contamination_sensor']
            self.error_status_word[i_ens] = "{0:032b}".format(data['variable_leader']['error_status_word'])
            self.pressure_pascal[i_ens] = data['variable_leader']['pressure']
            self.pressure_var_pascal[i_ens] = data['variable_leader']['pressure_variance']

            # Store Y2K date and time as list
            self.date_y2k[i_ens, :] = [data['variable_leader']['rtc_y2k_century'],
                                              data['variable_leader']['rtc_y2k_year'],
                                              data['variable_leader']['rtc_y2k_month'],
                                              data['variable_leader']['rtc_y2k_day']]
            self.time_y2k[i_ens, :] = [data['variable_leader']['rtc_y2k_hour'],
                                              data['variable_leader']['rtc_y2k_minutes'],
                                              data['variable_leader']['rtc_y2k_seconds'],
                                              data['variable_leader']['rtc_y2k_hundredths']]
            self.date[i_ens, :] = self.date_not_y2k[i_ens, :]
            self.date[i_ens, 0] = self.date_y2k[i_ens, 0] * 100 + \
                                         self.date_y2k[i_ens, 1]

            if 'vertical_beam' in data:
                self.vert_beam_eval_amp[i_ens] = data['vertical_beam']['eval_amp']
                self.vert_beam_RSSI_amp[i_ens] = data['vertical_beam']['rssi']
                self.vert_beam_range_m[i_ens] = data['vertical_beam']['range'] / 1000

                # Use first 8 bits of status and the 6 the bit to determine the gain
                temp = "{0:08b}".format(data['vertical_beam']['status'])
                self.vert_beam_status[i_ens] = int(temp[6:], 2)
                if temp[5] == '0':
                    self.vert_beam_gain[i_ens] = 'L'
                else:
                    self.vert_beam_gain[i_ens] = 'H'


class Surface(object):
    """Class to hold surface cell data.

    Attributes
    ----------
    no_cells: np.array(int)
        Number of surface cells in the ensemble
    cell_size_cm: np.array(int)
        Cell size in cm
    dist_bin1_cm: np.array(int)
        Distance to center of cell 1 in cm
    vel_mps: np.array(float)
        3D array of velocity data in each cell and ensemble
    corr: np.array(int)
        3D array of correlation data for each beam, cell, and ensemble
    pergd: np.array(int)
        3D array of percent good for each beam, cell, and ensemble
    rssi: np.array(int)
        3D array of return signal strength indicator for each beam, cell, and ensemble
    """

    def __init__(self, n_ensembles, n_velocities, max_surface_bins):
        """Initialize instance variables.

        Parameters
        ----------
        n_ensembles: int
            Number of ensembles
        n_velocities: int
            Number of velocity beams
        max_surface_bins: int
            Maximum number of surface bins in an ensemble in the transect
        """

        self.no_cells = np.zeros(n_ensembles)
        self.cell_size_cm = nans(n_ensembles)
        self.dist_bin1_cm = nans(n_ensembles)
        self.vel_mps = np.tile([np.nan], [n_velocities, max_surface_bins, n_ensembles])
        self.corr = nans([n_velocities, max_surface_bins, n_ensembles])
        self.pergd = nans([n_velocities, max_surface_bins, n_ensembles])
        self.rssi = nans([n_velocities, max_surface_bins, n_ensembles])

    def populate_data(self, i_ens, data, main_data):
        """Populates the class with data for an ensemble.

        Parameters
        ----------
        i_ens: int
            Ensemble index
        data: dict
            Dictionary of all data for this ensemble
        main_data: Pd0TRDI
            Object of PD0TRDI
        """

        if 'surface_leader' in data:
            self.no_cells[i_ens] = data['surface_leader']['cell_count']
            self.cell_size_cm[i_ens] = data['surface_leader']['cell_size']
            self.dist_bin1_cm[i_ens] = data['surface_leader']['range_cell_1']

        if 'surface_velocity' in data:
            self.vel_mps[:main_data.n_velocities, :len(data['surface_velocity']['velocity']), i_ens] = \
                np.array(data['surface_velocity']['velocity']).T

        if 'surface_correlation' in data:
            self.corr[:main_data.n_velocities, :len(data['surface_correlation']['correlation']), i_ens] = \
                np.array(data['surface_correlation']['correlation']).T

        if 'surface_intensity' in data:
            self.rssi[:main_data.n_velocities, :len(data['surface_intensity']['rssi']), i_ens] = \
                np.array(data['surface_intensity']['rssi']).T

        if 'surface_percent_good' in data:
            self.pergd[:main_data.n_velocities, :len(data['surface_percent_good']['percent_good']), i_ens] = \
                np.array(data['surface_percent_good']['percent_good']).T


class Wt(object):
    """Class to hold water track data.

    Attributes
    ----------
    vel_mps: np.array(float)
        3D array of velocity data in each cell and ensemble
    corr: np.array(int)
        3D array of correlation data for each beam, cell, and ensemble
    pergd: np.array(int)
        3D array of percent good for each beam, cell, and ensemble
    rssi: np.array(int)
        3D array of return signal strength indicator for each beam, cell, and ensemble
    """

    def __init__(self, n_bins, n_ensembles, n_velocities):
        """Initialize instance variables.

        Parameters
        ----------
        n_ensembles: int
            Number of ensembles
        n_velocities: int
            Number of velocity beams
        n_bins: int
            Maximum number of bins in an ensemble in the transect
        """

        self.corr = nans([n_velocities, n_bins, n_ensembles])
        self.pergd = nans([n_velocities, n_bins, n_ensembles])
        self.rssi = nans([n_velocities, n_bins, n_ensembles])
        self.vel_mps = nans([n_velocities, n_bins, n_ensembles])

    def populate_data(self, i_ens, data, main_data):
        """Populates the class with data for an ensemble.

        Parameters
        ----------
        i_ens: int
            Ensemble index
        data: dict
            Dictionary of all data for this ensemble
        main_data: Pd0TRDI
            Object of PD0TRDI
        """


        if 'velocity' in data:
            # Check size in case array needs to be expanded
            if main_data.Cfg.wn[i_ens] > self.vel_mps.shape[1]:
                append = np.zeros([self.vel_mps.shape[0],
                                   int(main_data.Cfg.wn[i_ens] - self.vel_mps.shape[1]),
                                   self.vel_mps.shape[2]])
                self.vel_mps = np.hstack([self.vel_mps, append])
                self.corr = np.hstack([self.corr, append])
                self.rssi = np.hstack([self.rssi, append])
                self.pergd = np.hstack([self.pergd, append])

            # Reformat and assign data
            if 'velocity' in data:
                self.vel_mps[:main_data.n_velocities, :int(main_data.Cfg.wn[i_ens]), i_ens] = \
                    np.array(data['velocity']['data']).T
            if 'correlation' in data:
                self.corr[:main_data.n_velocities, :int(main_data.Cfg.wn[i_ens]), i_ens] = \
                    np.array(data['correlation']['data']).T
            if 'echo_intensity' in data:
                self.rssi[:main_data.n_velocities, :int(main_data.Cfg.wn[i_ens]), i_ens] = \
                    np.array(data['echo_intensity']['data']).T
            if 'percent_good' in data:
                self.pergd[:main_data.n_velocities, :int(main_data.Cfg.wn[i_ens]), i_ens] = \
                    np.array(data['percent_good']['data']).T
