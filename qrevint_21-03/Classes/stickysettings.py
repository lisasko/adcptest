import os
import json


class StickySettings(object):
    """Provides methods to quickly store and retrieve settings to and from disk.

    This class is intended to be used to store simple settings that need to be retained between session of the subject
    application, such as, last folder opened, or units setting. Any setting that the application needs to know
    when it is run again can be stored using the methods in this class.

    Data are stored a dictionary which is then written to a json file having the filename provided by the user and
    stored in the folder defined by the APPDATA environment variable.

    Note
    ----
    The os and json packages are required for this class.

    Attributes
    ----------
    settings_file : str
        Filename of json file used to store data between sessions.
    settings: dict
        Dictionary used to store user defined settings.

    """

    def __init__(self, arg):
        """Constructor method which establishes the json file.

        If the filename (arg) provided by the user cannont be found a new file is created. If the filename (arg)
        provided by the user is found the file is opened and all keys and values are read and stored in settings for
        quick modification by the calling application.

        Parameters
        ----------
        arg : str
            User supplied filename excluding the suffix. Example 'myFile' but not 'myFile.json'

        """
        # Construct filename from user input.
        self.settings_file = os.path.join(os.getenv('APPDATA'), arg + '.json')
        if os.path.isfile(self.settings_file):
            # Read json into dictionary
            with open(self.settings_file, 'r') as f:
                self.settings = json.load(f)
        else:
            # Create json file with default dictionary
            self.settings = {}
            with open(self.settings_file, 'w') as f:
                json.dump(self.settings, f)

    def new(self, key, value):
        """Create new key value pair in settings.

        Method checks to see if key exists. If it exists an error is raised. If the key does not exist it is created.

        Paramenters
        -----------
        key : str
            Label for value in dictionary.
        value : any value that can be stored in a dictionary

        Raises
        ------
        KeyError
            If the key already exists in settings.

        """

        if key in self.settings:
            raise KeyError('Key already exists in settings')
        else:
            self.settings[key] = value
            with open(self.settings_file, 'w') as f:
                json.dump(self.settings, f)

    def set(self, key, value):
        """Set value of existing key.

        Method checks to ensure the key exists and sets the value of the key to value. If the key does not exist an
        error is raised.

        Parameters
        ----------
        key : str
            Label for value in dictionary.
        value : any value that can be stored in a dictionary

        Raises
        ------
        KeyError
            If the key does not exist in settings.

        """
        if key in self.settings:
            self.settings[key] = value
            with open(self.settings_file, 'w') as f:
                json.dump(self.settings, f)
        else:
            raise KeyError('Key does not exist in settings')

    def get(self, item):
        """Get value of item for settings.

        Returns the value of item from settings dictionary.

        Parameters
        ----------
        item : str
            Key for settings dictionary.

        Returns
        -------
        value
            Data type stored in key 'item'.

        """

        with open(self.settings_file, 'r') as f:
            self.settings = json.load(f)

        return self.settings[item]
