import numpy as np
from PIL import Image
from Utils.logger import get_logger
from PreProcessor.Base import BaseRPGenerator

logger = get_logger()

class RPGenerator(BaseRPGenerator):

    def __init__(self, dim:int=3, delay:int=2):
        super().__init__(dim, delay)
        logger.info(f"Initialized RPGenerator with dim={dim}, delay={delay}")

    def generate_rm(self, rri_peaks:np.ndarray)->np.ndarray|None:
        n = len(rri_peaks)
        nrp = n - (self.dim - 1) * self.delay
        if nrp <= 0:
            logger.warning("Not enough data points to generate recurrence matrix with the given parameters.")
            return None
        
        embedding_vector = np.empty((nrp, self.dim), dtype=np.float32)
        for dim in range(self.dim):
            start = dim * self.delay
            end = start + nrp
            embedding_vector[:, dim] = rri_peaks[start:end]

        diff = embedding_vector[:, None, :] - embedding_vector[None, :, :]
        D2 = np.sum(diff**2, axis=2)
        rp_matrix = np.sqrt(D2)
        return rp_matrix
        


    def generate(self, rri_peaks:np.ndarray)->np.ndarray|None:
        rm = self.generate_rm(rri_peaks)
        if rm is None:
            return None
        rp_min = np.min(rm)
        rp_max = np.max(rm)
        rp_normalized = (rm - rp_min) / (rp_max - rp_min) 
        rp_im = (rp_normalized*255.0).round().astype(np.uint8)
        im = Image.fromarray(rp_im)
        im = im.resize((224, 224), resample=Image.BICUBIC)
        im = im.convert('L')  
        im = np.expand_dims(np.array(im, dtype=np.uint8), axis=0)
        return im
    

