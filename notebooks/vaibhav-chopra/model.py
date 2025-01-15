import os
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import load_img, img_to_array

def load_dataset(dataset_dir="kaggle_dataset/genres_original", target_size=(128, 128), batch_size=32):
    """
    Load the dataset from the specified directory for training and validation.

    Parameters:
    dataset_dir (str): Path to the dataset directory.
    target_size (tuple): Target size for the images (default: (128, 128)).
    batch_size (int): Batch size for training (default: 32).

    Returns:
    tuple: Training and validation data.
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


def prepare_data(train_data, val_data):
    """
    Prepare the data for XGBoost by flattening the images and encoding labels.

    Parameters:
    train_data: Training data generator.
    val_data: Validation data generator.

    Returns:
    tuple: Training and validation data and labels.
    """
    # Flatten the images and extract the labels
    X_train = []
    y_train = []
    for batch_images, batch_labels in train_data:
        X_train.append(batch_images)
        y_train.append(batch_labels)
        if len(X_train) * len(batch_images) >= train_data.samples:
            break

    X_train = np.concatenate(X_train, axis=0)
    y_train = np.concatenate(y_train, axis=0)

    X_val = []
    y_val = []
    for batch_images, batch_labels in val_data:
        X_val.append(batch_images)
        y_val.append(batch_labels)
        if len(X_val) * len(batch_images) >= val_data.samples:
            break

    X_val = np.concatenate(X_val, axis=0)
    y_val = np.concatenate(y_val, axis=0)

    # Flatten the images
    X_train_flattened = X_train.reshape(X_train.shape[0], -1)
    X_val_flattened = X_val.reshape(X_val.shape[0], -1)

    # Convert labels to integers
    label_encoder = LabelEncoder()
    y_train_int = label_encoder.fit_transform(np.argmax(y_train, axis=1))
    y_val_int = label_encoder.transform(np.argmax(y_val, axis=1))

    return X_train_flattened, y_train_int, X_val_flattened, y_val_int


def train_xgboost(X_train, y_train):
    """
    Train an XGBoost classifier on the provided data.

    Parameters:
    X_train (array): Training feature data.
    y_train (array): Training labels.

    Returns:
    xgb.Booster: Trained XGBoost model.
    """
    dtrain = xgb.DMatrix(X_train, label=y_train)
    param = {'objective': 'multi:softmax', 'num_class': 10, 'max_depth': 6, 'eta': 0.3}
    num_round = 10  # Number of boosting iterations
    model = xgb.train(param, dtrain, num_round)
    return model


def evaluate_model(model, X_val, y_val):
    """
    Evaluate the XGBoost model on validation data.

    Parameters:
    model (xgb.Booster): Trained XGBoost model.
    X_val (array): Validation feature data.
    y_val (array): Validation labels.

    Returns:
    float: Accuracy score.
    """
    dval = xgb.DMatrix(X_val)
    y_pred = model.predict(dval)
    return accuracy_score(y_val, y_pred)


def save_model(model, model_path="xgboost_model.json"):
    """
    Save the trained XGBoost model to the specified path.

    Parameters:
    model (xgb.Booster): Trained XGBoost model.
    model_path (str): Path to save the model file (default: "xgboost_model.json").
    """
    model.save_model(model_path)


def load_model(model_path="xgboost_model.json"):
    """
    Load the trained XGBoost model from the specified path.

    Parameters:
    model_path (str): Path to the saved model file.

    Returns:
    xgb.Booster: Loaded XGBoost model.
    """
    model = xgb.Booster()
    model.load_model(model_path)
    return model


def predict_genre(model, spectrogram_path):
    """
    Predicts the genre of a given spectrogram image.

    Parameters:
    model (xgb.Booster): Trained XGBoost model.
    spectrogram_path (str): Path to the spectrogram image.

    Returns:
    str: Predicted genre.
    """
    img = load_img(spectrogram_path, target_size=(128, 128))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Flatten the image
    img_flattened = img_array.reshape(1, -1)
    dtest = xgb.DMatrix(img_flattened)
    prediction = model.predict(dtest)
    genres = ['Blues', 'Classical', 'Country', 'Disco', 'Hiphop', 'Jazz', 'Metal', 'Pop', 'Reggae', 'Rock']
    return genres[int(prediction[0])]
