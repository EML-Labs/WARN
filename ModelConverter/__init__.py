import os
import coremltools as ct
import numpy as np
from Utils.Loader.ModelLoader import ModelLoader
from Configurations import ModelConfig,ModelMetaData
from Utils.logger import get_logger

model:ct.models.MLModel.input_description

logger = get_logger()

OUTPUT_DIR = os.path.join(os.getcwd(),'ConvertedWeights')

class ModelConverter:

    def __init__(self):
        self.model_loader = ModelLoader()
        self.model = self.model_loader.load_model()
        os.makedirs(OUTPUT_DIR, exist_ok=True)

    def convert(self):
        mlmodel = ct.convert(
            self.model, 
            source="tensorflow",
            convert_to="mlprogram",
            inputs=[ct.TensorType(shape=ModelConfig.INPUT_SHAPE,dtype=np.float32)],
            classifier_config=ct.ClassifierConfig(class_labels=ModelConfig.CLASSES_LIST),
            minimum_deployment_target=ct.target.watchOS10,
            )
        output_path = os.path.join(OUTPUT_DIR, "Model.mlpackage")
        mlmodel = ModelMetaData(mlmodel).model
        mlmodel.save(output_path)
        logger.info(f"Model converted and saved to {output_path}")
        return mlmodel