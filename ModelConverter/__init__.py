import os
import coremltools as ct
from Utils.Loader.ModelLoader import ModelLoader
from Configurations import ModelConfig

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
            inputs=[ct.ImageType(shape=ModelConfig.INPUT_SHAPE)],
            classifier_config=ct.ClassifierConfig(class_labels=ModelConfig.CLASSES_LIST)
            )
        output_path = os.path.join(OUTPUT_DIR, "Model.mlmodel")
        mlmodel.save(output_path)
        print(f"Model converted and saved to {output_path}")