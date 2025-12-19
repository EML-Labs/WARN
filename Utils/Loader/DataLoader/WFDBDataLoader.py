import os
import pandas as pd
import numpy as np
import wfdb
from typing import List
from Utils.Loader.Base import Loader
from Utils.logger import get_logger
from Configurations.Types import FileTypes,DataTypes


logger = get_logger()

class WFDBDataLoader(Loader):
    def __init__(self, file_path:os.PathLike,data_types:List[DataTypes]):
        super().__init__(file_path)
        self.data_types = [data_type.value for data_type in data_types]
        logger.info(f"Initialized WFDBDataLoader")

    def get_files(self)->List[os.PathLike]:
        all_files = os.listdir(self.file_path)
        files = []
        for file in all_files:
            if file.endswith(FileTypes.DAT.value):
                files.append(os.path.join(self.file_path, os.path.splitext(file)[0]))
        
        return files
                    

    def read_file(self, file_path)->np.ndarray:
        record = wfdb.rdrecord(file_path)
        data = np.array(record.p_signal[:,0])
        return data
        

    def load_data(self)->List[np.ndarray]:
        data = []
        files = self.get_files()
        for file in files:
            signal = self.read_file(file)
            data.append(signal)
        logger.info(f"Loaded {len(data)} files from {self.file_path}")
        return data


