import numpy as np
from Classes.TransformationMatrix import TransformationMatrix


class InstrumentData(object):
    """Container for characteristics of the ADCP used to make the measurement

    Attributes
    ----------
    serial_num: str
        Serial number of ADCP.
    manufacturer: str
        Name of manufacturer.
    model: str
        Model name of ADCP.
    firmware: str
        Firmware version in the ADCP.
    frequency_khz:
        Frequency or frequencies used by ADCP.
    beam_angle_deg:
        Angle of the beams from vertical in degrees.
    beam_pattern:
        Pattern of the beam angles, concave or convex.
    t_matrix: TransformationMatrix
        Object of TransformationMatrix.
    configuration_commands:
        Commands used to configure the instrument.
    """
     
    def __init__(self):
        """Constructor initializes the variables to None.
        """

        self.serial_num = None  # Serial number of ADCP
        self.manufacturer = None  # manufacturer of ADCP (SonTek, TRDI)
        self.model = None  # model of ADCP (Rio Grande, StreamPro, RiverRay, M9, S5)
        self.firmware = None  # firmware version
        self.frequency_khz = None  # frquency of ADCP (could be "Multi")
        self.beam_angle_deg = None  # angle of beam from vertical
        self.beam_pattern = None  # pattern of beams
        self.t_matrix = None  # object of TransformationMatrix
        self.configuration_commands = None  # configuration commands sent to ADCP
        
    def populate_data(self, manufacturer, raw_data, mmt_transect=None, mmt=None):
        """Manages method calls for different manufacturers.

        Parameters
        ----------
        manufacturer: str
            Name of manufacturer.
        raw_data: object
            Object of Pd0TRDI for TRDI or Object of MatSonTek for SonTek
        mmt_transect: MMT_Transect
            Object of Transect (mmt object)
        mmt: MMT_TRDI
            Object of MMT_TRDI
        """

        # Process based on manufacturer
        if manufacturer == 'TRDI':
            self.manufacturer = manufacturer
            self.trdi(pd0=raw_data, mmt_transect=mmt_transect, mmt=mmt)
        elif manufacturer == 'SonTek':
            self.manufacturer = manufacturer
            self.sontek(rs=raw_data)
        elif manufacturer == 'Nortek':
            self.manufacturer = manufacturer
            self.nortek(rs=raw_data)

    def trdi(self, pd0, mmt_transect, mmt):
        """Populates the variables with data from TRDI ADCPs.

        Parameters
        ----------
        pd0: Pd0TRDI
            Object of Pd0TRDI
        mmt_transect: MMT_Transect
            Object of MMT_Transect
        mmt: MMT_Transect
            Object of MMT_Transect
        """

        # Instrument frequency
        self.frequency_khz = pd0.Inst.freq[0]

        # Firmware
        self.firmware = pd0.Inst.firm_ver[0]

        # Instrument beam angle and pattern
        self.beam_angle_deg = pd0.Inst.beam_ang[0]
        self.beam_pattern = pd0.Inst.pat[0]

        # Instrument characteristics
        mmt_site = getattr(mmt, 'site_info')
        mmt_config = getattr(mmt_transect, 'active_config')

        self.serial_num = mmt_site['ADCPSerialNmb']

        # Determine TRDI model
        num = float(self.firmware)
        model_switch = np.floor(num)

        if model_switch == 10:
            self.model = 'Rio Grande'
            if 'Fixed_Commands' in mmt_config.keys():
                self.configuration_commands = np.array(['Fixed'], dtype=object)
                self.configuration_commands = np.append(self.configuration_commands, mmt_config['Fixed_Commands'])

        elif model_switch == 31:
            self.model = 'StreamPro'
            self.frequency_khz = 2000
            if 'Fixed_Commands_StreamPro' in mmt_config.keys():
                self.configuration_commands = np.array(['Fixed'], dtype=object)
                self.configuration_commands = np.append(self.configuration_commands,
                                                        mmt_config['Fixed_Commands_StreamPro'])

        elif model_switch == 44:
            self.model = 'RiverRay'
            if 'Fixed_Commands_RiverRay' in mmt_config.keys():
                self.configuration_commands = np.array(['Fixed'], dtype=object)
                self.configuration_commands = np.append(self.configuration_commands,
                                                        mmt_config['Fixed_Commands_RiverRay'])

        elif model_switch == 56:
            self.model = 'RiverPro'
            if pd0.Cfg.n_beams[0] < 5:
                if 'RG_Test' in mmt.qaqc.keys():
                    idx = mmt.qaqc['RG_Test'][0].find('RioPro')
                    if idx != -1:
                        self.model = 'RioPro'

            if 'Fixed_Commands_RiverPro' in mmt_config.keys():
                self.configuration_commands = np.array(['Fixed'], dtype=object)
                self.configuration_commands = np.append(self.configuration_commands,
                                                        mmt_config['Fixed_Commands_RiverPro'])
            else:
                self.configuration_commands = np.array(['Fixed'], dtype=object)
                self.configuration_commands = np.append(self.configuration_commands, ' ')

        else:
            self.model = 'Unknown'
            if 'Fixed_Commands' in mmt_config.keys():
                self.configuration_commands = np.array(['Fixed'], dtype=object)
                self.configuration_commands = np.append(self.configuration_commands, mmt_config['Fixed_Commands'])

        if 'Wizard_Commands' in mmt_config.keys():
            self.configuration_commands = np.append(self.configuration_commands, ['Wizard'])
            self.configuration_commands = np.append(self.configuration_commands,
                                                    mmt_config['Wizard_Commands'])

        if 'User_Commands' in mmt_config.keys():
            self.configuration_commands = np.append(self.configuration_commands, ['User'])
            self.configuration_commands = np.append(self.configuration_commands,
                                                    mmt_config['User_Commands'])

        # Obtain transformation matrix from one of the available sources
        if not np.isnan(pd0.Inst.t_matrix[0, 0]):
            self.t_matrix = TransformationMatrix()
            self.t_matrix.populate_data(manufacturer='TRDI', model='pd0', data_in=pd0)
        elif self.model == 'RiverRay':
            self.t_matrix = TransformationMatrix()
            self.t_matrix.populate_data(manufacturer='TRDI', model=self.model, data_in='Nominal')
        else:
            if isinstance(mmt.qaqc, dict) and len(mmt.qaqc) > 0:
                if 'RG_Test' in mmt.qaqc.keys():

                    self.t_matrix = TransformationMatrix()
                    self.t_matrix.populate_data(manufacturer='TRDI', model=self.model, data_in=mmt.qaqc['RG_Test'][0])

                elif 'Compass_Calibration' in mmt.qaqc.keys():

                    self.t_matrix = TransformationMatrix()
                    self.t_matrix.populate_data(manufacturer='TRDI',
                                                model=self.model,
                                                data_in=mmt.qaqc['Compass_Calibration'][0])

                elif 'Compass_Eval_Timestamp' in mmt.qaqc.keys():

                    self.t_matrix = TransformationMatrix()
                    self.t_matrix.populate_data(manufacturer='TRDI',
                                                model=self.model,
                                                data_in=mmt.qaqc['Compass_Evaluation'][0])

                else:
                    self.t_matrix = TransformationMatrix()
                    self.t_matrix.populate_data(manufacturer='TRDI',
                                                model=self.model,
                                                data_in='Nominal')
            else:
                self.t_matrix = TransformationMatrix()
                self.t_matrix.populate_data(manufacturer='TRDI',
                                            model=self.model,
                                            data_in='Nominal')

    def sontek(self, rs):
        """Populates the variables with data from SonTek ADCPs.

        Parameters
        ----------
        rs: MatSonTek
        """

        self.serial_num = rs.System.SerialNumber
        self.frequency_khz = rs.Transformation_Matrices.Frequency
        if self.frequency_khz[2] > 0:
            self.model = 'M9'
        else:
            self.model = 'S5'
        if hasattr(rs, 'SystemHW'):
            revision = str(rs.SystemHW.FirmwareRevision)
            if len(revision) < 2:
                revision = '0' + revision
            self.firmware = str(rs.SystemHW.FirmwareVersion) + '.' + revision
        else:
            self.firmware = ''
        self.beam_angle_deg = 25
        self.beam_pattern = 'Convex'
        self.t_matrix = TransformationMatrix()
        self.t_matrix.populate_data('SonTek', data_in=rs.Transformation_Matrices.Matrix)
        self.configuration_commands = None

    def nortek(self, rs):
        self.serial_num = rs.System.SerialNumber
        self.frequency_khz = rs.Transformation_Matrices.Frequency
        self.model = rs.System.InstrumentModel
        if hasattr(rs, 'SystemHW'):
            revision = str(rs.SystemHW.FirmwareRevision)
            if len(revision) < 2:
                revision = '0' + revision
            self.firmware = str(rs.SystemHW.FirmwareVersion) + '.' + revision
        else:
            self.firmware = ''
        self.beam_angle_deg = 25
        self.beam_pattern = 'Convex'
        self.t_matrix = TransformationMatrix()
        self.t_matrix.populate_data('SonTek', data_in=rs.Transformation_Matrices.Matrix)
        self.configuration_commands = None

    def populate_from_qrev_mat(self, transect):
        """Populates the object using data from previously saved QRev Matlab file.

        Parameters
       ----------
       transect: mat_struct
           Matlab data structure obtained from sio.loadmat
       """

        self.serial_num = str(transect.adcp.serialNum)
        self.manufacturer = transect.adcp.manufacturer
        self.model = transect.adcp.model
        self.firmware = transect.adcp.firmware
        self.frequency_khz = transect.adcp.frequency_hz
        self.beam_angle_deg = transect.adcp.beamAngle_deg
        self.beam_pattern = transect.adcp.beamPattern
        self.t_matrix = TransformationMatrix()
        self.t_matrix.populate_from_qrev_mat(transect.adcp.tMatrix)

        if len(transect.adcp.configurationCommands) > 0:
            self.configuration_commands = []
            for command in transect.adcp.configurationCommands:
                if type(command) == str:
                    self.configuration_commands.append(command)
            self.configuration_commands = np.array(self.configuration_commands)

        else:
            self.configuration_commands = None
