import os
import requests
import zipfile
from tqdm import tqdm
from Utils.logger import get_logger
from Utils.Loader.DownLoader.Base import BaseDownLoader

logger = get_logger()

class MimicDatasetDownLoader(BaseDownLoader):
    MIMIC_PERFORM_AF_DATASET_URL:str = 'https://zenodo.org/record/6807403/files/mimic_perform_af_csv.zip'
    MIMIC_PERFORM_NON_AF_DATASET_URL:str = 'https://zenodo.org/record/6807403/files/mimic_perform_non_af_csv.zip'

    def __init__(self, file_path:os.PathLike,chunk_size:int=1024):
        super().__init__(file_path, name='MIMIC_PERFORM')
        self.chunk_size = chunk_size
        logger.info(f"Initialized MimicDatasetDownLoader")

    def load_data(self)->os.PathLike|None:
        logger.info("Downloading MIMIC-PERFORM dataset...")
        try:
            os.makedirs(self.dataset_path, exist_ok=True)
            os.chdir(self.file_path)

            def download_and_extract(url, extract_to='.'):
                local_filename = url.split('/')[-1]
                try:
                    with requests.get(url, stream=True) as r:
                        r.raise_for_status()
                        total_size = int(r.headers.get('content-length', 0))

                        with open(local_filename, 'wb') as f, tqdm(
                            total=total_size, unit='B', unit_scale=True, desc=f"Downloading {local_filename}"
                        ) as pbar:
                            for chunk in r.iter_content(chunk_size=self.chunk_size):
                                if chunk:
                                    f.write(chunk)
                                    pbar.update(len(chunk))

                    with zipfile.ZipFile(local_filename, 'r') as zip_ref:
                        zip_ref.extractall(extract_to)
                    os.remove(local_filename)

                except requests.exceptions.RequestException as e:
                    logger.error(f"Error downloading {url}: {e}")
                    return
                
                except zipfile.BadZipFile as e:
                    logger.error(f"Error extracting {local_filename}: {e}")
                    return
                
                except Exception as e:
                    logger.error(f"An unexpected error occurred: {e}")
                    return


            if not os.listdir(self.dataset_path):
                logger.info("MIMIC-PERFORM dataset not found locally. Downloading now...")
                download_and_extract(self.MIMIC_PERFORM_AF_DATASET_URL, extract_to=self.dataset_path)
                download_and_extract(self.MIMIC_PERFORM_NON_AF_DATASET_URL, extract_to=self.dataset_path)
                logger.info("MIMIC-PERFORM dataset downloaded and extracted successfully.")
            else:
                logger.info("MIMIC-PERFORM dataset already exists locally. Skipping download.")

            os.chdir('..')
            return self.dataset_path

        except Exception as e:
            logger.error(f"An error occurred while setting up the dataset directory: {e}")

