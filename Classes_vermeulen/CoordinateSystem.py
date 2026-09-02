from enum import Enum, auto, IntEnum

"""
    Enumeration defining the four coordinate systems used by ADCPs.

    Attributes:
        Beam: Coordinates point along the acoustic beam, towards the transducer.
              This is the raw coordinate system in which an ADCP measures velocity.
              The number of components matches the number of beams.

        Instrument: Cartesian coordinate system defined with respect to the ADCP.
                   Typically has three components (forward, left, up). In redundant systems,
                   an error velocity may be added.

        Ship: Cartesian coordinate system defined with respect to the Ship.
              Includes corrections for the tilts of the boat (pitch and roll) and any misalignment of the ADCP and the Ship.

        Earth: Geographical coordinate system, typically defined with east, north, and upward coordinates.
               Includes corrections for the heading of the ADCP.
    """

class CoordinateSystem(Enum):

    Beam = 0
    Instrument = 1
    Ship = 2
    Earth = 3

