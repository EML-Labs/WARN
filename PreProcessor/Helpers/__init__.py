import numpy as np
from Utils.logger import get_logger

logger = get_logger()

class Segmenter:
    sampling_rate: int
    segment_size: int
    overlap: int

    def __init__(self,sampling_rate:int, segment_size:int, overlap:int):
        self.sampling_rate = sampling_rate
        self.segment_size = segment_size*sampling_rate
        self.overlap = overlap*sampling_rate
        self.step = self.segment_size - self.overlap
        logger.info(f"Initialized Segmenter with segment_size={segment_size}, overlap={overlap}")

    def segment(self, ppg_data:np.ndarray)->list[np.ndarray]:
        segments = []
        for start in range(0, len(ppg_data) - self.segment_size + 1, self.step):
            end = start + self.segment_size
            segments.append(ppg_data[start:end])
        return segments