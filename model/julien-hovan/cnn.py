import tensorflow as tf
from tensorflow.keras import layers
from config import ModelConfig

def create_cnn_model(input_shape):
    """
    Create a CNN model for feature extraction from spectrograms.
    """
    layers_list = []
    
    # Create conv blocks dynamically based on config
    for i, filters in enumerate(ModelConfig.CNN_FILTERS):
        # First layer needs input_shape
        if i == 0:
            layers_list.extend([
                layers.Conv2D(filters, ModelConfig.KERNEL_SIZE, padding='same', input_shape=input_shape),
                layers.BatchNormalization(),
                layers.ReLU(),
                layers.MaxPooling2D(ModelConfig.POOL_SIZE),
            ])
        else:
            layers_list.extend([
                layers.Conv2D(filters, ModelConfig.KERNEL_SIZE, padding='same'),
                layers.BatchNormalization(),
                layers.ReLU(),
                layers.MaxPooling2D(ModelConfig.POOL_SIZE),
            ])
    
    cnn_model = tf.keras.Sequential(layers_list)
    return cnn_model 