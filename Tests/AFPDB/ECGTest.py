import os
import numpy as np
from sklearn.utils import shuffle
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt

from Tests.Base import BaseTest

from Configurations.Types import DataTypes,ClassLabels

from PreProcessor.ECG import ECGPreProcessor
from PreProcessor.RP import RPGenerator
from PreProcessor.Helpers import Segmenter

from Utils.Loader.ModelLoader import ModelLoader
from Utils.Loader.DataLoader.WFDBDataLoader import WFDBDataLoader
from Utils.Loader.DownLoader.AFPDBDataset import AFPDBDatasetDownLoader
from Utils.logger import get_logger

logger = get_logger()

class AFPDBECGTest(BaseTest):
    name = 'AFPDB'
    SAMPLING_RATE = 128
    DIM = 3
    DELAY = 2

    accuracy:float = 0.0
    precision:float = 0.0
    recall:float = 0.0
    f1_score:float = 0.0
    confusion_matrix:np.ndarray = np.array([])

    def setup_method(self):
        self.artifact_path = os.path.join(os.getcwd(), 'artifacts')
        self.base_path = os.path.join(os.getcwd(), 'datasets')
        self.dataset_path = os.path.join(self.base_path, 'AFPDBDataset')
        os.makedirs(self.base_path, exist_ok=True)
        self.af_data_path = os.path.join(self.dataset_path, 'afpdb_pre_af_wfdb')
        self.non_af_data_path = os.path.join(self.dataset_path, 'afpdb_non_af_wfdb')
        self.pre_af_data_path = os.path.join(self.dataset_path, 'afpdb_pre_af_wfdb')

    def __init__(self,segment_size:int=30,overlap:int=5,quality_threshold:float=0.8,shuffle_data:bool=True,batch_size:int=32):
        logger.info(f"Initializing AFPDBECGTest")
        super().__init__(shuffle_data=shuffle_data, batch_size=batch_size)
        self.setup_method()
        self.dataloader = AFPDBDatasetDownLoader(file_path=self.base_path)
        self.af_dataloader = WFDBDataLoader(file_path=self.af_data_path, data_types=[DataTypes.ECG])
        self.non_af_dataloader = WFDBDataLoader(file_path=self.non_af_data_path, data_types=[DataTypes.ECG])
        self.pre_af_dataloader = WFDBDataLoader(file_path=self.pre_af_data_path, data_types=[DataTypes.ECG])
        self.segmentor = Segmenter(sampling_rate=self.SAMPLING_RATE, segment_size=segment_size, overlap=overlap)
        self.preprocessor = ECGPreProcessor(sampling_rate=self.SAMPLING_RATE, quality_threshold=quality_threshold)
        self.rp_generator = RPGenerator(dim=self.DIM, delay=self.DELAY)
        self.model_loader = ModelLoader()
        logger.info(f"Initialized AFPDBECGTest with batch_size={batch_size}")

        self.ytrue = []
        self.ypred = []


    def load_data(self):
        try:
            self.dataloader.load_data()
            self.af_dataset = self.af_dataloader.load_data()
            self.non_af_dataset = self.non_af_dataloader.load_data()
            self.pre_af_dataset = self.pre_af_dataloader.load_data()

        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise e
        

    def load_model(self):
        self.model = self.model_loader.load_model()

    def preprocess_data(self):
        X, y = [], []

        with tqdm(total=len(self.af_dataset) + len(self.non_af_dataset) + len(self.pre_af_dataset), desc="Preprocessing data") as pbar:
            for data in self.af_dataset:
                pbar.update(1)
                segments = self.segmentor.segment(data)
                rri_peaks_list = self.preprocessor.process_dataset(segments)
                if not rri_peaks_list:
                    continue

                for seg in rri_peaks_list:
                    rp_plot = self.rp_generator.generate(seg)
                    X.append(rp_plot)
                    y.append(ClassLabels.AF.value)


            for data in self.non_af_dataset:
                pbar.update(1)
                segments = self.segmentor.segment(data)
                rri_peaks_list = self.preprocessor.process_dataset(segments)
                if not rri_peaks_list:
                    continue

                for seg in rri_peaks_list:
                    rp_plot = self.rp_generator.generate(seg)
                    X.append(rp_plot)
                    y.append(ClassLabels.SR.value)
                
            for data in self.pre_af_dataset:
                pbar.update(1)
                segments = self.segmentor.segment(data)
                rri_peaks_list = self.preprocessor.process_dataset(segments)
                if not rri_peaks_list:
                    continue

                for seg in rri_peaks_list:
                    rp_plot = self.rp_generator.generate(seg)
                    X.append(rp_plot)
                    y.append(ClassLabels.PRE_AF.value)

        logger.info(f"Preprocessing completed. Total samples: {self.preprocessor.total_segments}, Skipped samples: {self.preprocessor.total_skipped}")


        X = np.stack(X,axis=0)
        y = np.array(y)

        if self.shuffle_data:
            X, y = shuffle(X, y, random_state=42)

        return X, y

    
    def update_metrics(self, y_true:np.ndarray, y_pred:np.ndarray)->None:
        self.ytrue.extend(y_true)
        self.ypred.extend(y_pred)

    def compute_final_metrics(self, y_true:np.ndarray, y_pred:np.ndarray)->None:
        self.accuracy = accuracy_score(y_true, y_pred)
        self.precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        self.recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        self.f1_score = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        self.confusion_matrix = confusion_matrix(y_true, y_pred)

        logger.info(f"Final Evaluation Metrics:")
        logger.info(f"Accuracy: {self.accuracy:.4f}")
        logger.info(f"Precision: {self.precision:.4f}")
        logger.info(f"Recall: {self.recall:.4f}")
        logger.info(f"F1 Score: {self.f1_score:.4f}")
        logger.info(f"Confusion Matrix:\n{self.confusion_matrix}")

    def plot_confusion_matrix(self):
        plt.figure(figsize=(8, 6))
        plt.imshow(self.confusion_matrix, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Confusion Matrix')
        plt.colorbar()
        labels = [ClassLabels.PRE_AF.name, ClassLabels.AF.name, ClassLabels.SR.name]
        tick_marks = np.arange(len(labels))
        plt.xticks(tick_marks, labels, rotation=45)
        plt.yticks(tick_marks, labels)

        thresh = self.confusion_matrix.max() / 2.
        for i, j in np.ndindex(self.confusion_matrix.shape):
            plt.text(j, i, format(self.confusion_matrix[i, j], 'd'),
                     horizontalalignment="center",
                     color="white" if self.confusion_matrix[i, j] > thresh else "black")

        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.tight_layout()
        plt.savefig(os.path.join(self.artifact_path, 'confusion_matrix.png'))
        plt.close()

    def evaluate_model(self):
        X, y = self.preprocess_data()
        num_batches = int(np.ceil(X.shape[0] / self.batch_size))
        batch_range = tqdm(range(num_batches), desc="Predicting batches")

        for batch_idx in batch_range:
            start_idx = batch_idx * self.batch_size
            end_idx = min((batch_idx + 1) * self.batch_size, X.shape[0])
            X_batch = X[start_idx:end_idx]
            y_batch = y[start_idx:end_idx]

            predictions = self.model.predict(X_batch,verbose=0)
            predicted_labels = np.argmax(predictions, axis=1)

            self.update_metrics(y_batch, predicted_labels)


        self.compute_final_metrics(np.array(self.ytrue), np.array(self.ypred))

        self.plot_confusion_matrix()

    def teardown_method(self):
        del self.dataloader
        del self.af_dataloader
        del self.non_af_dataloader
        del self.segmentor
        del self.preprocessor
        del self.rp_generator
        del self.model_loader
        del self.model
        del self.ytrue
        del self.ypred

    def run_test(self):
        self.load_data()
        self.load_model()
        self.evaluate_model()
        self.teardown_method()
             
        