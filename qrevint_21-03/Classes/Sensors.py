import numpy as np
from Classes.SensorStructure import SensorStructure


class Sensors(object):
    """Class to store data from ADCP sensors.

    Attributes
    ----------
    heading_deg: HeadingData
        Object of HeadingData.
    pitch_deg: SensorStructure
        Pitch data, object of SensorStructure
    roll_deg: SensorStructure
        Roll data, object of SensorStructure
    temperature_deg_c: SensorStructure
        Temperature data, object of SensorStructure
    salinity_ppt: SensorStructure
        Salinity data, object of SensorStructure
    speed_of_sound_mps: SensorStructure
        Speed of sound, object of SensorStructure
    """

    def __init__(self):
        """Initialize class and create variable objects"""

        self.heading_deg = SensorStructure()  # Object of HeadingData
        self.pitch_deg = SensorStructure()  # Pitch data, object of SensorStructure
        self.roll_deg = SensorStructure()  # Roll data, object of SensorStructure
        self.temperature_deg_c = SensorStructure()  # Temperature data, object of SensorStructure
        self.salinity_ppt = SensorStructure()  # Salinity data, object of SensorStructure
        self.speed_of_sound_mps = SensorStructure()  # Speed of sound, object of SensorStructure

    def populate_from_qrev_mat(self, transect):
        """Populates the object using data from previously saved QRev Matlab file.

        Parameters
        ----------
        transect: mat_struct
           Matlab data structure obtained from sio.loadmat
        """
        if hasattr(transect, 'sensors'):
            if hasattr(transect.sensors, 'heading_deg'):
                self.heading_deg.populate_from_qrev_mat(transect.sensors.heading_deg, heading=True)
            if hasattr(transect.sensors, 'pitch_deg'):
                self.pitch_deg.populate_from_qrev_mat(transect.sensors.pitch_deg)
            if hasattr(transect.sensors, 'roll_deg'):
                self.roll_deg.populate_from_qrev_mat(transect.sensors.roll_deg)
            if hasattr(transect.sensors, 'salinity_ppt'):
                self.salinity_ppt.populate_from_qrev_mat(transect.sensors.salinity_ppt)
            if hasattr(transect.sensors, 'speedOfSound_mps'):
                self.speed_of_sound_mps.populate_from_qrev_mat(transect.sensors.speedOfSound_mps)
            if hasattr(transect.sensors, 'temperature_degC'):
                self.temperature_deg_c.populate_from_qrev_mat(transect.sensors.temperature_degC)

    @staticmethod
    def speed_of_sound(temperature, salinity):
        """Computes speed of sound from temperature and salinity.

        Parameters
        ----------
        temperature: float or np.array(float)
            Water temperature at transducer face, in degrees C.
        salinity: float or np.array(float)
            Water salinity at transducer face, in ppt.
        """

        # Not provided in RS Matlab file computed from equation used in TRDI BBSS
        sos = 1449.2 + 4.6 * temperature - 0.055 * temperature**2 + 0.00029 * temperature**3 \
            + (1.34 - 0.01 * temperature) * (salinity - 35.0)
        return sos

    @staticmethod
    def avg_temperature(transects):
        """Compute mean temperature from temperature data from all transects.

        Parameters
        ----------
        transects: list
            List of TransectData objects
        """

        temps = np.array([])
        for transect in transects:
            if transect.checked:
                temps = np.append(temps, transect.sensors.temperature_deg_c.internal.data)
        return np.nanmean(temps)
