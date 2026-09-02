import numpy as np

class DateTime(object):
    """This stores the date and time data in Python compatible format.

    Attributes
    ----------
    date: str
        Measurement date as mm/dd/yyyy
    start_serial_time: float
        Python serial time for start of transect (seconds since 1/1/1970), timestamp
    end_serial_time: float
        Python serial time for end of transect (seconds since 1/1/1970), timestamp
    transect_duration_sec: float
        Duration of transect, in seconds.
    ens_duration_sec: np.array(float)
        Duration of each ensemble, in seconds.
    """
    
    def __init__(self):
        """Initialize class and instance variables."""

        self.date = None  # Measurement date mm/dd/yyyy
        self.start_serial_time = None  # Python serial time for start of transect, timestamp
        self.end_serial_time = None  # Python serial time for end of transect, timestamp
        self.transect_duration_sec = None  # Duration of transect in seconds
        self.ens_duration_sec = None  # Duration of each ensemble in seconds
        
    def populate_data(self, date_in, start_in, end_in, ens_dur_in):
        """Populate data in object.

        Parameters
        ----------
        date_in: str
            Measurement date as mm/dd/yyyy
        start_in: float
            Python serial time for start of transect.
        end_in: float
            Python serial time for end of transect.
        ens_dur_in: np.array(float)
            Duration of each ensemble, in seconds.
        """
        
        self.date = date_in
        self.start_serial_time = start_in
        self.end_serial_time = end_in
        self.transect_duration_sec = float(end_in - start_in)
        self.ens_duration_sec = ens_dur_in.astype(float)

    def populate_from_qrev_mat(self, transect):
        """Populates the object using data from previously saved QRev Matlab file.

        Parameters
        ----------
        transect: mat_struct
           Matlab data structure obtained from sio.loadmat
        """

        if hasattr(transect, 'dateTime'):
            seconds_day = 86400
            time_correction = 719529.0000000003

            self.date = transect.dateTime.date
            self.start_serial_time = (transect.dateTime.startSerialTime - time_correction) * seconds_day
            self.end_serial_time = (transect.dateTime.endSerialTime - time_correction) * seconds_day
            self.transect_duration_sec = float(transect.dateTime.transectDuration_sec)
            try:
                self.ens_duration_sec = transect.dateTime.ensDuration_sec.astype(float)
            except AttributeError:
                self.ens_duration_sec = np.array([np.nan])

            #
            # self.date = transect.dateTime.date
            # self.start_serial_time = transect.dateTime.startSerialTime
            # self.end_serial_time = transect.dateTime.endSerialTime
            # self.transect_duration_sec = float(transect.dateTime.transectDuration_sec)
            # self.ens_duration_sec = transect.dateTime.ensDuration_sec.astype(float)
