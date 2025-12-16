import numpy as np

class BasePreProcessor:
    sampling_rate: int
    quality_threshold: float

    def __init__(self, sampling_rate: int,quality_threshold:float=0.9):
        self.sampling_rate = sampling_rate
        self.quality_threshold = quality_threshold
    
    def preprocess(self, data:np.ndarray)->np.ndarray:
        raise NotImplementedError("This method should be overridden by subclasses.")
    

class BaseRPGenerator:
    dim : int
    delay : int

    def __init__(self, dim:int=3, delay:int=2):
        self.dim = dim
        self.delay = delay

    def generate(self, rri_peaks:np.ndarray)->np.ndarray:
        raise NotImplementedError("This method should be overridden by subclasses.")