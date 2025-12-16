from enum import Enum,IntEnum

class FileTypes(Enum):
    HDF5 = '.h5'
    CSV = '.csv'
    TXT = '.txt'

class DataTypes(Enum):
    ECG = 'ECG'
    PPG = 'PPG'

class ClassLabels(IntEnum):
    AF = 0
    PRE_AF = 1
    SR = 2

class TestTypes(Enum):
    MIMIC_PERFORM_ECG = 'MIMIC_ECG'
    MIMIC_PERFORM_PPG = 'MIMIC_PPG'
