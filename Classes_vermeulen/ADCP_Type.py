from enum import IntEnum

"""
Enumeration defining different ADCP models

"""

class ADCPType(IntEnum):
    Unknown = 0
    ChannelMaster = 1
    ExplorerPhasedArray = 2
    ExplorerPiston = 3
    SentinelV = 4
    MonitorV = 5
    RioGrande = 6
    RiverRay = 7
    StreamPro = 8
    RiverPro = 9

    Sentinel = 10
    Mariner = 11
    Monitor = 12
    QuarterMaster1500 = 13
    QuarterMaster3000 = 14
    QuarterMaster6000 = 15
    QuarterMaster1500ModBeams = 16
    LongRanger75 = 17
    LongRanger1500 = 18
    LongRanger3000 = 19