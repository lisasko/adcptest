import re


class CompassCal(object):
    """Class stores compass calibration or evaluation data and parses the compass error from the raw data.

    Attributes
    ----------
    time_stamp: str
        Time of calibration or evaluation (mm/dd/yyyy).
    data: str
        All calibration or evaluation data provided by the manufacturer.
    error: float
        Remaining compass error after calibration or from evaluation, in degrees.
    """

    def __init__(self):
        """Initialize class and instance variables."""

        self.time_stamp = None
        self.data = None
        self.error = None

    def populate_data(self, time_stamp, data_in):
        """Store data and parse compass error from compass data.

        Parameters
        ----------
        time_stamp: str
            Time of calibration or evaluation (mm/dd/yyyy).
        data_in: str
            All calibration or evaluation data provided by the manufacturer.
        """
        self.time_stamp = time_stamp
        self.data = data_in

        splits = re.split('(Total error:|Double Cycle Errors:|Error from calibration:)', data_in)
        if len(splits) > 1:
            self.error = re.search('\d+\.*\d*', splits[2])[0]
        else:
            self.error = 'N/A'
