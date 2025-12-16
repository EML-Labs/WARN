import os
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from typing import Tuple
from Utils.Loader.Base import Loader
from Utils.logger import get_logger

logger = get_logger()


PATH:os.PathLike = os.path.join(os.getcwd(), 'NN_weights', 'WEIGHTS.hdf5')

class ModelLoader(Loader):
    input_shape:Tuple[int, int, int]=(224, 224, 1)
    lr:float=1e-5

    def __init__(self, file_path:os.PathLike=PATH):
        super().__init__(file_path)

        base_model = tf.keras.applications.EfficientNetV2S(
        include_top=False,
        weights=None,
        input_shape=self.input_shape)
        x = base_model.output
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Dropout(0.2)(x)
        predictions = tf.keras.layers.Dense(3, activation="softmax")(x)
        self.model = tf.keras.Model(inputs=base_model.input, outputs=predictions)
        self.model.compile(optimizer=Adam(learning_rate=self.lr),
                    loss='categorical_crossentropy',
                    metrics=['accuracy', 'AUC'])
        
        logger.info(f"Initialized ModelLoader")
    
    def load_model(self)->tf.keras.Model:
        self.model.load_weights(self.file_path)
        logger.info(f"Loaded model weights from {self.file_path}")
        return self.model