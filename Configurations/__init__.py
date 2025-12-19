import coremltools as ct
class ModelConfig:
    INPUT_SHAPE = (1,224,224,1)
    CLASSES_LIST = ['Normal', 'AFib', 'Pre-AFib']


class ModelMetaData:
    model:ct.models.MLModel
    def __init__(self, model:ct.models.MLModel):
        self.model = model
        self.model.input_description['input_1'] = "Input Reccurence Plot to be classified"
        self.model.output_description["classLabel"] = "AFib Classification Result"
        self.model.author = """Original Paper: Gavidia, M., Zhu, H., Montanari, A. N., Fuentes, J., Cheng, C., Dubner, S., ... & Goncalves, J.
        Model Converted by : Yasantha Niroshan (EML-Labs, University of Moratuwa)"""
        self.model.license = "Please see https://github.com/EML-Labs/WARN/blob/main/LICENSE for license information."
        self.model.short_description = "Early Warning of Atrial Fibrillation Using Deep Learning"
        self.model.version = "1.0"
