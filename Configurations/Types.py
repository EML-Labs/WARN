from enum import Enum,IntEnum

class FileTypes(Enum):
    HDF5 = '.h5'
    CSV = '.csv'
    TXT = '.txt'
    DAT = '.dat'

class DataTypes(Enum):
    ECG = 'ECG'
    PPG = 'PPG'

class ClassLabels(IntEnum):
    PRE_AF = 0
    AF = 1
    SR = 2

class TestTypes(Enum):
    MIMIC_PERFORM_ECG = 'MIMIC_ECG'
    MIMIC_PERFORM_PPG = 'MIMIC_PPG'
    AFPDB_ECG = 'AFPDB_ECG'
