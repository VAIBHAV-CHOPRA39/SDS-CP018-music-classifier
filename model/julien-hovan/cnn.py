import tensorflow as tf
from tensorflow.keras import layers
from config import ModelConfig

def create_cnn_model(input_shape):
    """
    Original CNN model implementation
    """
    inputs = layers.Input(shape=input_shape)
    x = inputs

    # Stack CNN layers based on filter configuration
    for filters in ModelConfig.CNN_FILTERS:
        x = layers.Conv2D(filters, ModelConfig.KERNEL_SIZE, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.MaxPooling2D(ModelConfig.POOL_SIZE)(x)

    return tf.keras.Model(inputs, x)

def residual_block(x, filters, kernel_size=(3,3)):
    """
    Creates a residual block with two convolution layers and skip connection
    """
    shortcut = x
    
    # First conv layer
    x = layers.Conv2D(filters, kernel_size, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    
    # Second conv layer
    x = layers.Conv2D(filters, kernel_size, padding='same')(x)
    x = layers.BatchNormalization()(x)
    
    # If number of filters changed, adjust shortcut dimensions
    if shortcut.shape[-1] != filters:
        shortcut = layers.Conv2D(filters, (1,1), padding='same')(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)
    
    # Add skip connection and final activation
    x = layers.Add()([x, shortcut])
    x = layers.ReLU()(x)
    
    return x

def create_resnet_model(input_shape):
    """
    ResNet-style CNN model for feature extraction from spectrograms
    """
    inputs = layers.Input(shape=input_shape)
    x = inputs
    
    # Initial convolution
    x = layers.Conv2D(ModelConfig.CNN_FILTERS[0], (5,5), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D(ModelConfig.POOL_SIZE)(x)
    
    # Create residual blocks based on config
    for filters in ModelConfig.CNN_FILTERS[1:]:
        # Two residual blocks for each filter size
        x = residual_block(x, filters)
        x = residual_block(x, filters)
        x = layers.MaxPooling2D(ModelConfig.POOL_SIZE)(x)
    
    return tf.keras.Model(inputs, x) 