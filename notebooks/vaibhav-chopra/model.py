import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications import MobileNetV2

def load_dataset(dataset_dir="kaggle_dataset/genres_original", target_size=(128, 128), batch_size=32):
    """
    Load the dataset from the specified directory for training and validation.

    Parameters:
    dataset_dir (str): Path to the dataset directory.
    target_size (tuple): Target size for the images (default: (128, 128)).
    batch_size (int): Batch size for training (default: 32).

    Returns:
    tuple: Training and validation data generators.
    """
    datagen = ImageDataGenerator(rescale=1.0 / 255, validation_split=0.2)

    train_data = datagen.flow_from_directory(
        dataset_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='training'
    )

    val_data = datagen.flow_from_directory(
        dataset_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation'
    )

    return train_data, val_data


def create_model(input_shape=(128, 128, 3), num_classes=10):
    """
    Create and compile the CNN model.

    Parameters:
    input_shape (tuple): Shape of the input data (default: (128, 128, 3)).
    num_classes (int): Number of output classes (default: 10).

    Returns:
    tf.keras.Model: Compiled CNN model.
    """
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def create_pretrained_model(input_shape=(128, 128, 3), num_classes=10):
    """
    Create and compile a model based on MobileNetV2 with transfer learning.

    Parameters:
    input_shape (tuple): Shape of the input data (default: (128, 128, 3)).
    num_classes (int): Number of output classes (default: 10).

    Returns:
    tf.keras.Model: Compiled pre-trained model.
    """
    # Load pre-trained MobileNetV2 model without the top layers
    base_model = MobileNetV2(input_shape=input_shape, include_top=False, weights='imagenet')

    # Freeze the base model layers to avoid retraining them
    base_model.trainable = False

    # Add custom top layers for genre classification
    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def train_model(model, train_data, val_data, epochs=10):
    """
    Train the CNN model with the given data.

    Parameters:
    model (tf.keras.Model): The CNN model to train.
    train_data: Training data generator.
    val_data: Validation data generator.
    epochs (int): Number of training epochs (default: 10).

    Returns:
    tf.keras.callbacks.History: Training history.
    """
    return model.fit(train_data, validation_data=val_data, epochs=epochs)


def save_model(model, model_path="saved_model"):
    """
    Save the trained model to the specified path using SavedModel format.

    Parameters:
    model (tf.keras.Model): Trained CNN model.
    model_path (str): Path to save the model directory (default: "saved_model").
    """
    model.save(model_path)  # Saves model in SavedModel format (directory)


def load_model(model_path="saved_model"):
    """
    Load the trained CNN model from the specified path.

    Parameters:
    model_path (str): Path to the saved model directory.

    Returns:
    tf.keras.Model: Loaded CNN model.
    """
    model_path = os.path.join(os.getcwd(), model_path)  # Ensure correct path handling

    if os.path.exists(model_path):
        print(f"Loading model from {model_path}")
        return tf.keras.models.load_model(model_path)
    else:
        raise FileNotFoundError(f"Model directory not found at {model_path}")


def predict_genre(model, spectrogram_path):
    """
    Predicts the genre of a given spectrogram image.

    Parameters:
    model (tf.keras.Model): Trained CNN model.
    spectrogram_path (str): Path to the spectrogram image.

    Returns:
    str: Predicted genre.
    """
    img = load_img(spectrogram_path, target_size=(128, 128))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)
    genres = ['Blues', 'Classical', 'Country', 'Disco', 'Hiphop', 'Jazz', 'Metal', 'Pop', 'Reggae', 'Rock']
    return genres[np.argmax(predictions)]
