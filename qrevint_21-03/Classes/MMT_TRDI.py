import os
import re
import xmltodict
import numpy as np


class MMTtrdi(object):
    """Class to read and store data from a WinRiver 2 mmt file.

    Attributes
    ----------
    project: dict
        Dictionary of measurement information
    site_info: dict
        Dictionary of site information
    transects: list
        List of Transect objects containing information for each discharge transect
    summary: dict
        Dictionary of measurement summary for each available boat velocity reference
    qaqc: dict
        Dictionary of premeasurement tests, calibrations, and evaluations
    mbt_transects: list
        List of Transect objects containing information for each moving-bed test transect
    path: str
        Path for mmt file and associated files
    """

    def __init__(self, mmt_file):
        """Initialize instance variables and reads mmt file.

        Parameters
        ----------
        mmt_file: str
            Full filename including path of mmt file.
        """

        # Intialize instance variables
        self.project = {}
        self.site_info = {}
        self.transects = []
        self.summary = {}
        self.qaqc = {}
        self.mbt_transects = []
        self.path = None

        # Process mmt file
        self.process_mmt(mmt_file)

    def process_mmt(self, mmt_file):
        """Method to read and process the mmt file.

        Parameters
        ----------
        mmt_file: str
            Full filename including path of mmt file.
        """

        # Open the file and convert to an ordered dictionary tree
        with open(mmt_file, 'r', encoding='utf-8') as fd:
            xml_data = fd.read()
            clean_xml_data = ''
            remove_re = re.compile(u'[\x00-\x08\x0B-\x0C\x0E-\x1F\x7F%]')
            for line in xml_data:
                new_line, count = remove_re.subn('', line)
                clean_xml_data = clean_xml_data + new_line

            win_river = xmltodict.parse(clean_xml_data)
        # UnicodeDecodeError
        win_river = win_river['WinRiver']

        self.path = os.path.split(mmt_file)[0]

        # Process project settings
        self.project['Name'] = win_river['Project']['@Name']
        self.project['Version'] = win_river['Project']['@Version']
        if 'Locked' in win_river['Project'].keys():
            self.project['Locked'] = win_river['Project']['Locked']
        else:
            self.project['Locked'] = None

        # Process site information
        siteinfo_keys = win_river['Project']['Site_Information'].keys()

        # Iterate through all of the keys and values of site info
        for x in siteinfo_keys:
            site_data = win_river['Project']['Site_Information'][x]
            if site_data is not None:
                # Remove @ symbol from properties
                if '@' in x:
                    x = x[1:]
                if x == 'Water_Temperature':
                    self.site_info[x] = float(site_data)
                    # -32768 used to denote no data
                    if self.site_info[x] < -100:
                        self.site_info[x] = ''
                else:
                    self.site_info[x] = site_data
            else:
                self.site_info[x] = ''

        trans = win_river['Project']['Site_Discharge']['Transect']

        # Create a Transect class for each transect found under Site_Discharge
        if type(trans) == list:
            for i in range(len(trans)):
                if 'File' in trans[i]:
                    self.transects.append(MMTtransect(trans[i]))
        else:
            self.transects = [MMTtransect(trans)]

        # Discharge Summary
        if 'Discharge_Summary' in win_river['Project']['Site_Discharge'].keys():
            discharge_summary = win_river['Project']['Site_Discharge']['Discharge_Summary']

            self.summary['NONE'] = self.mmtqsum(discharge_summary['None'])
            self.summary['BT'] = self.mmtqsum(discharge_summary['BottomTrack'])
            self.summary['GGA'] = self.mmtqsum(discharge_summary['GGA'])
            self.summary['VTG'] = self.mmtqsum(discharge_summary['VTG'])

        # QA_QC
        if 'QA_QC' in win_river['Project'].keys():
            qaqc = win_river['Project']['QA_QC']
            for qaqc_type, data in qaqc.items():
                # Parse qaqc data from dictionary if the type is a test, cal, or eval
                if qaqc_type in ['RG_Test', 'Compass_Calibration', 'Compass_Evaluation']:
                    # There could be multiple tests of the same type so they are stored in a list
                    time_stamp = qaqc_type + '_TimeStamp'
                    if not isinstance(data['TestResult'], list):
                        self.qaqc[qaqc_type] = [data['TestResult']['Text']]
                        self.qaqc[time_stamp] = [data['TestResult']['TimeStamp']]
                    else:
                        self.qaqc[qaqc_type] = []
                        self.qaqc[time_stamp] = []
                        for result in data['TestResult']:
                            self.qaqc[qaqc_type].append(result['Text'])
                            self.qaqc[time_stamp].append(result['TimeStamp'])

                if qaqc_type == 'Moving_Bed_Test':
                    if 'Transect' in data.keys():
                        self.moving_bed_test(data)

    def moving_bed_test(self, mb_data):
        """Method to parse data from moving-bed test dictionary.

        Parameters
        ----------
        mb_data: dict
            Dictionary containing moving-bed test information
        """

        transects = mb_data['Transect']

        # If only one transect make it a list
        if not isinstance(transects, list):
            transects = [transects]

        # Process each transect dictionary
        for tsect in transects:
            transect = MMTtransect(tsect)

            # Determine type of moving-bed test
            if '@MBTType' in tsect:
                if tsect['@MBTType'] == '0':
                    transect.moving_bed_type = 'Loop'
                elif tsect['@MBTType'] == '1':
                    transect.moving_bed_type = 'Stationary'
            else:
                # Use the file name to determine the moving-bed test type
                file_name = transect.Files[0]
                fidx = file_name.rfind('.')
                if file_name[fidx-3:fidx] == 'SBT':
                    transect.moving_bed_type = 'Stationary'
                elif file_name[fidx-3:fidx] == 'LBT':
                    transect.moving_bed_type = 'Loop'
                else:
                    # If type can't be determined process as stationary
                    transect.moving_bed_type = 'Stationary'

            self.mbt_transects.append(transect)

    @staticmethod
    def mmtqsum(data):
        """Method to parse the MMT Q summary data.

        Parameters
        ----------
        data: dict
            A summary dictionary from mmt file.

        Returns
        -------
        sum_dict: dict
            Dictionary of summary with a couple of key names changed.
        """
        
        sum_dict = {
            'Use': [],
            'Begin_Left': [],
            'FileName': [],
            'LeftEdgeSlopeCoeff': [],
            'RightEdgeSlopeCoeff': []
            }

        # Iterate through each transect
        for transect in data.values():
            # Iterate through each key and val in the transect summary
            for key2, val2 in transect.items():
                # Append value from transect to appropriate key
                if key2 == 'UseInSummary':
                    sum_dict['Use'].append(float(val2))
                elif key2 == "BeginLeft":
                    sum_dict['Begin_Left'].append(float(val2))
                elif key2 == 'FileName':
                    sum_dict['FileName'].append(val2)
                elif key2 == 'LeftEdgeSlopeCoeff':
                    sum_dict['LeftEdgeSlopeCoeff'].append(float(val2))
                elif key2 == 'RightEdgeSlopeCoeff':
                    sum_dict['RightEdgeSlopeCoeff'].append(float(val2))
                else:
                    # If the key has not been specified use key from transect summary
                    if key2 not in sum_dict:
                        sum_dict[key2] = []
                    try:
                        sum_dict[key2].append(float(val2))
                    except ValueError:
                        sum_dict[key2].append(np.nan)
        return sum_dict


class MMTtransect(object):
    """Class to hold properties of MMT transect dictionary attributes.

    Attributes
    ----------
    Checked: int
    Files: list
    Notes: list
    """

    def __init__(self, trans):
        """Constructor immediately begins extraction of data"""

        self.Checked = int(trans['@Checked'])
        self.Files = []
        self.Notes = []
        self.field_config = None
        self.active_config = None
        self.moving_bed_type = None

        files = trans['File']

        # Create File classes for each file associated with transect
        if type(files) is list:
            for file in files:
                self.Files.append(file['#text'])
        else:
            self.Files.append(files['#text'])

        # Create Note classes for each file associated with transect
        if 'Note' in trans.keys():
            note = trans['Note']
            if type(note) is list:
                for n in note:
                    if type(trans['File']) is list:
                        self.Notes.append(self.note_dict(n, trans['File'][0]['@TransectNmb']))
                    else:
                        self.Notes.append(self.note_dict(n, trans['File']['@TransectNmb']))
            else:
                if type(trans['File']) is list:
                    self.Notes.append(self.note_dict(note, trans['File'][0]['@TransectNmb']))
                else:
                    self.Notes.append(self.note_dict(note, trans['File']['@TransectNmb']))

        # Create configuration dictionaries for each config attribute
        if type(trans['Configuration']) is list:
            for config in trans['Configuration']:
                if int(config['@Checked']) == 0:
                    self.field_config = self.parse_config(config)
                if int(config['@Checked']) == 1:
                    self.active_config = self.parse_config(config)
        else:
            if int(trans['Configuration']['@Checked']) == 0:
                self.field_config = self.parse_config(trans['Configuration'])
            if int(trans['Configuration']['@Checked']) == 1:
                self.active_config = self.parse_config(trans['Configuration'])

        # Assign active config to field config if there is no field config
        if self.field_config is None:
            self.field_config = self.active_config

    def set_moving_bed_type(self, mvb_type):
        """Setter for moving bed type in the case of MBT Transects

        Parameters
        ----------
        mvb_type: str
            Type of moving-bed test.
        """

        self.moving_bed_type = mvb_type

    @staticmethod
    def parse_config(config):
        """Method to parse configuration file from mmt xml.

        Parameters
        ----------
        config: dict
            Dictionary of configuration settings

        Returns
        -------
        config_dict: dict
            Processed dictionary of configuration settings
        """

        # Initialize dictionary for configuration
        config_dict = {}

        # Store all instrument commands
        command_groups = config['Commands']
        for group in command_groups.keys():
            config_dict[group] = []
            for key, command in command_groups[group].items():
                if key != '@Status':
                    config_dict[group].append(command)

        # Depth sounder configuration
        if 'Use_Depth_Sounder_In_Processing' in config['Depth_Sounder'].keys():
            if config['Depth_Sounder']['Use_Depth_Sounder_In_Processing']['#text'] == "YES":
                config_dict['DS_Use_Process'] = 1
            else:
                config_dict['DS_Use_Process'] = 0
        else:
            config_dict['DS_Use_Process'] = -1

        config_dict['DS_Transducer_Depth'] = float(config['Depth_Sounder']['Depth_Sounder_Transducer_Depth']['#text'])
        config_dict['DS_Transducer_Offset'] = float(config['Depth_Sounder']['Depth_Sounder_Transducer_Offset']['#text'])

        if config['Depth_Sounder']['Depth_Sounder_Correct_Speed_of_Sound']['#text'] == 'YES':
            config_dict['DS_Cor_Spd_Sound'] = 1
        else:
            config_dict['DS_Cor_Spd_Sound'] = 0

        config_dict['DS_Scale_Factor'] = float(config['Depth_Sounder']['Depth_Sounder_Scale_Factor']['#text'])

        # External heading configuration
        config_dict['Ext_Heading_Offset'] = float(config['Ext_Heading']['Offset']['#text'])

        if 'Use_Ext_Heading' in config['Ext_Heading'].keys():
            if config['Ext_Heading']['Use_Ext_Heading']['#text'] == 'NO':
                config_dict['Ext_Heading_Use'] = False
            else:
                config_dict['Ext_Heading_Use'] = True
        else:
            config_dict['Ext_Heading_Use'] = False

        # GPS configuration
        if 'GPS' in config.keys():
            config_dict['GPS_Time_Delay'] = config['GPS']['Time_Delay']['#text']

        # Discharge settings
        config_dict['Q_Top_Method'] = float(config['Discharge']['Top_Discharge_Estimate']['#text'])
        config_dict['Q_Bottom_Method'] = float(config['Discharge']['Bottom_Discharge_Estimate']['#text'])
        config_dict['Q_Power_Curve_Coeff'] = float(config['Discharge']['Power_Curve_Coef']['#text'])
        config_dict['Q_Cut_Top_Bins'] = float(config['Discharge']['Cut_Top_Bins']['#text'])
        config_dict['Q_Bins_Above_Sidelobe'] = float(config['Discharge']['Cut_Bins_Above_Sidelobe']['#text'])
        config_dict['Q_Left_Edge_Type'] = float(config['Discharge']['River_Left_Edge_Type']['#text'])
        config_dict['Q_Left_Edge_Coeff'] = float(config['Discharge']['Left_Edge_Slope_Coeff']['#text'])
        config_dict['Q_Right_Edge_Type'] = float(config['Discharge']['River_Right_Edge_Type']['#text'])
        config_dict['Q_Right_Edge_Coeff'] = float(config['Discharge']['Right_Edge_Slope_Coeff']['#text'])
        config_dict['Q_Shore_Pings_Avg'] = float(config['Discharge']['Shore_Pings_Avg']['#text'])

        # Edge estimate settings
        config_dict['Edge_Begin_Shore_Distance'] = config['Edge_Estimates']['Begin_Shore_Distance']['#text']
        config_dict['Edge_End_Shore_Distance'] = float(config['Edge_Estimates']['End_Shore_Distance']['#text'])
        if config['Edge_Estimates']['Begin_Left_Bank']['#text'] == 'YES':
            config_dict['Edge_Begin_Left_Bank'] = 1
        else:
            config_dict['Edge_Begin_Left_Bank'] = 0

        # Check for user discharge feature in mmt file
        if 'Begin_Manual_Discharge' in config['Edge_Estimates']:
            config_dict['Edge_Begin_Manual_Discharge'] = float(config['Edge_Estimates']['Begin_Manual_Discharge']['#text'])
            config_dict['Edge_Begin_Method_Distance'] = \
                config['Edge_Estimates']['Begin_Edge_Discharge_Method_Distance']['#text']
            config_dict['Edge_End_Manual_Discharge'] = float(config['Edge_Estimates']['End_Manual_Discharge']['#text'])
            config_dict['Edge_End_Method_Distance'] = \
                config['Edge_Estimates']['End_Edge_Discharge_Method_Distance']['#text']

        # Offsets
        for key in config['Offsets'].keys():
            if key == 'ADCP_Transducer_Depth':
                child = "Offsets_Transducer_Depth"
            else:
                child = "Offsets_" + key

            config_dict[child] = float(config['Offsets'][key]['#text'])

        # Processing settings
        for key in config['Processing'].keys():
            if key == 'Use_3_Beam_Solution_For_BT':
                child = 'Proc_Use_3_Beam_BT'
            elif key == 'Use_3_Beam_Solution_For_WT':
                child = 'Proc_Use_3_Beam_WT'
            elif key == 'BT_Error_Velocity_Threshold':
                child = 'Proc_BT_Error_Vel_Threshold'
            elif key == 'WT_Error_Velocity_Threshold':
                child = 'Proc_WT_Error_Velocity_Threshold'
            elif key == 'BT_Up_Velocity_Threshold':
                child = 'Proc_BT_Up_Vel_Threshold'
            elif key == 'WT_Up_Velocity_Threshold':
                child = 'Proc_WT_Up_Vel_Threshold'
            elif key == 'Fixed_Speed_Of_Sound':
                child = 'Proc_Fixed_Speed_Of_Sound'
            elif key == 'Mark_Below_Bottom_Bad':
                child = 'Proc_Mark_Below_Bottom_Bad'
            elif key == 'Use_Weighted_Mean':
                child = 'Proc_Use_Weighted_Mean'
            elif key == 'Absorption':
                child = 'Proc_Absorption'
            else:
                child = 'Proc_' + key

            # Try to cast to float otherwise assign 1 or 0 based on string value
            try:
                config_dict[child] = float(config['Processing'][key]['#text'])
            except ValueError:
                if config['Processing'][key]['#text'] == 'YES':
                    config_dict[child] = 1
                else:
                    config_dict[child] = 0

            # Recording
            config_dict['Rec_Filename_Prefix'] = config['Recording']['Filename_Prefix']['#text']
            config_dict['Rec_Output_Directory'] = config['Recording']['Output_Directory']['#text']

            if 'Root_Directory' in config['Recording'].keys():
                if '#text' in config['Recording']['Root_Directory']:
                    config_dict['Rec_Root_Directory'] = config['Recording']['Root_Directory']['#text']
                else:
                    config_dict['Rec_Root_Directory'] = None
            else:
                config_dict['Rec_Root_Directory'] = None

            if config['Recording']['MeasurmentNmb'] is None:
                config_dict['Rec_MeasNmb'] = config['Recording']['MeasurmentNmb']
            else:
                config_dict['Rec_MeasNmb'] = config['Recording']['MeasurmentNmb']
            config_dict['Rec_GPS'] = config['Recording']['GPS_Recording']['#text']
            config_dict['Rec_DS'] = config['Recording']['DS_Recording']['#text']
            config_dict['Rec_EH'] = config['Recording']['EH_Recording']['#text']
            config_dict['Rec_ASCII_Output'] = config['Recording']['ASCII_Output_Recording']['#text']
            config_dict['Rec_Max_File_Size'] = float(config['Recording']['Maximum_File_Size']['#text'])
            config_dict['Rec_Next_Transect_Number'] = float(config['Recording']['Next_Transect_Number']['#text'])
            config_dict['Rec_Add_Date_Time'] = float(config['Recording']['Add_Date_Time']['#text'])
            config_dict['Rec_Use_Delimiter'] = config['Recording']['Use_Delimiter']['#text']
            config_dict['Rec_Delimiter'] = config['Recording']['Custom_Delimiter']['#text']
            config_dict['Rec_Prefix'] = config['Recording']['Use_Prefix']['#text']
            config_dict['Rec_Use_MeasNmb'] = config['Recording']['Use_MeasurementNmb']['#text']
            config_dict['Rec_Use_TransectNmb'] = config['Recording']['Use_TransectNmb']['#text']
            config_dict['Rec_Use_SequenceNmb'] = config['Recording']['Use_SequenceNmb']['#text']

            # Wizard settings
            config_dict['Wiz_ADCP_Type'] = float(config['Wizard_Info']['ADCP_Type'])
            config_dict['Wiz_Firmware'] = float(config['Wizard_Info']['ADCP_FW_Version'])
            config_dict['Wiz_Use_Ext_Heading'] = config['Wizard_Info']['Use_Ext_Heading']
            config_dict['Wiz_Use_GPS'] = config['Wizard_Info']['Use_GPS']
            config_dict['Wiz_Use_DS'] = config['Wizard_Info']['Use_Depth_Sounder']
            config_dict['Wiz_Max_Water_Depth'] = float(config['Wizard_Info']['Max_Water_Depth'])
            config_dict['Wiz_Max_Water_Speed'] = float(config['Wizard_Info']['Max_Water_Speed'])
            config_dict['Wiz_Max_Boat_Space'] = float(config['Wizard_Info']['Max_Boat_Speed'])
            config_dict['Wiz_Material'] = float(config['Wizard_Info']['Material'])
            config_dict['Wiz_Water_Mode'] = float(config['Wizard_Info']['Water_Mode'])
            config_dict['Wiz_Bottom_Mode'] = float(config['Wizard_Info']['Bottom_Mode'])
            config_dict['Wiz_Beam_Angle'] = float(config['Wizard_Info']['Beam_Angle'])
            config_dict['Wiz_Pressure_Sensor'] = config['Wizard_Info']['Pressure_Sensor']
            config_dict['Wiz_Water_Mode_13'] = float(config['Wizard_Info']['Water_Mode_13_Avail'])
            config_dict['Wiz_StreamPro_Default'] = float(config['Wizard_Info']['Use_StreamPro_Def_Cfg'])
            config_dict['Wiz_StreamPro_Bin_Size'] = float(config['Wizard_Info']['StreamPro_Bin_Size'])
            config_dict['Wiz_StreamPro_Bin_Number'] = float(config['Wizard_Info']['StreamPro_Bin_Num'])

            if 'Use_GPS_Internal' in config['Wizard_Info'].keys():
                config_dict['Wiz_Use_GPS_Internal'] = config['Wizard_Info']['Use_GPS_Internal']
            if 'Internal_GPS_Baud_Rate_Index' in config['Wizard_Info'].keys():
                config_dict['Wiz_Internal_GPS_Baud_Rate_Index'] = float(config['Wizard_Info']
                                                                        ['Internal_GPS_Baud_Rate_Index'])

        return config_dict

    @staticmethod
    def file_dict(file):
        """Create dictionary for file information.

        Parameters
        ----------
        file: dict
            Dictionary for file from mmt

        Returns
        -------
        transect_file: dict
            Dictionary of transect file information
                Path: str
                    Full filename of transect including path
                File: str
                    Filename of transect
                Number: str
                    Transect number assigned in WinRiver 2
        """

        transect_file = {'Path': file['@PathName'], 'File': file['#text'], 'Number': file['@TransectNmb']}
        return transect_file

    @staticmethod
    def note_dict(note, number):
        """Create dictionary for notes.

        Parameters
        ----------
        note: dict
            Dictionary from mmt for notes
        number: str
            Transect number

        Returns
        -------
        note_dict_out: dict
            Dictionary for note information
                NoteFileNo: str
                    Transect number associated with the note
                NoteDate: str
                    Date note was entered
                NoteText: str
                    Text of note
        """

        note_dict_out = {'NoteFileNo': number, 'NoteDate': note['@TimeStamp'], 'NoteText': note['@Text']}
        return note_dict_out
