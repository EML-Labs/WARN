import neurokit2 as nk
import numpy as np
import warnings
from Utils.logger import get_logger
from PreProcessor.Base import BasePreProcessor

warnings.simplefilter("error", category=nk.misc.NeuroKitWarning)
logger = get_logger()

class ECGPreProcessor(BasePreProcessor):
    total_segments: int = 0
    total_skipped: int = 0

    def __init__(self, sampling_rate, quality_threshold:float=0.9):
        super().__init__(sampling_rate, quality_threshold)
        logger.info(f"Initialized ECGPreProcessor with sampling_rate={sampling_rate}, quality_threshold={quality_threshold}")

    def preprocess(self, data:np.ndarray)->np.ndarray|None:
        try:
            cleaned_signal = nk.ecg_clean(data, sampling_rate=self.sampling_rate)
            quality = nk.ecg_quality(cleaned_signal, sampling_rate=self.sampling_rate)
            average_quality = np.mean(quality)
            if average_quality >= self.quality_threshold:
                info = nk.ecg_findpeaks(cleaned_signal, sampling_rate=self.sampling_rate)
                peak_distances = np.diff(info['ECG_R_Peaks'])
                rri_peaks = peak_distances / self.sampling_rate
                self.total_segments += 1
                return rri_peaks
            else:
                self.total_skipped += 1
                return None
        except nk.misc.NeuroKitWarning:
            logger.debug("Skipping segment due to NeuroKitWarning during quality assessment.")
            return None
        
    def process_dataset(self, dataset:list[np.ndarray])->list[np.ndarray]:
        processed_data = []
        for i,data in enumerate(dataset):
            rri_peaks = self.preprocess(data)
            if rri_peaks is not None:
                processed_data.append(rri_peaks)
            else:
                logger.debug(f"Segment {i} skipped due to low quality.")
        return processed_data
        




