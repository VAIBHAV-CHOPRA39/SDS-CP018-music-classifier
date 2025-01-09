import tensorflow as tf
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import os
from attention import MultiHeadSelfAttention
from data_generator import TimeSegmentedSpectrogramGenerator

# Constants
BATCH_SIZE = 32
DATA_DIR = "Data/mel_spectrograms_images"  # Updated data directory
NUM_SEGMENTS = 7  # Number of segments expected by the model
IMG_HEIGHT = 128
IMG_WIDTH = 128

def load_data_from_npy(data_dir, img_size, num_segments):
    """
    Loads data from .npy files, creates labels, segments the spectrograms, and prepares the dataset.

    Args:
        data_dir: Path to the directory containing the class folders with .npy files.
        img_size: Size to resize the images to.
        num_segments: Number of segments to divide each spectrogram into.

    Returns:
        X: Array of segmented image data.
        y: Array of corresponding labels.
    """
    X = []
    y = []
    class_names = sorted(os.listdir(data_dir))

    for class_index, class_name in enumerate(class_names):
        class_dir = os.path.join(data_dir, class_name)
        for filename in os.listdir(class_dir):
            if filename.endswith(".npy"):
                filepath = os.path.join(class_dir, filename)
                try:
                    spectrogram = np.load(filepath)

                    # Segment the spectrogram
                    total_frames = spectrogram.shape[0]
                    frames_per_segment = total_frames // num_segments
                    segments = []
                    for i in range(num_segments):
                        start = i * frames_per_segment
                        end = (i + 1) * frames_per_segment
                        segment = spectrogram[start:end]
                        # Resize and remove extra dimension
                        segment = tf.image.resize(segment, (img_size, img_size)).numpy()
                        segments.append(segment)

                    X.append(segments)
                    y.append(class_index)
                except Exception as e:
                    print(f"Error processing {filepath}: {e}")

    return np.array(X), np.array(y)

def load_and_evaluate_model(model_path, data_dir, img_height, img_width, batch_size):
    """
    Loads a Keras model, evaluates it on a test dataset, and generates a confusion matrix and classification report.

    Args:
        model_path: Path to the saved Keras model.
        data_dir: Path to the directory containing the test dataset.
        img_size: Size of the images (assuming square images).
        batch_size: Batch size for evaluation.
    """

    # Load the model with custom_objects
    model = tf.keras.models.load_model(
        model_path,
        custom_objects={"MultiHeadSelfAttention": MultiHeadSelfAttention}
    )

    # Load data using the generator
    test_paths = []
    test_labels = {}
    class_names = sorted(os.listdir(data_dir))
    for class_index, class_name in enumerate(class_names):
        class_dir = os.path.join(data_dir, class_name)
        for filename in os.listdir(class_dir):
            if filename.endswith(".npy"):
                filepath = os.path.join(class_dir, filename)
                test_paths.append(filepath)
                test_labels[filepath] = class_index

    test_generator = TimeSegmentedSpectrogramGenerator(
        test_paths,
        test_labels,
        batch_size=batch_size,
        dim=(img_height, img_width),
        n_channels=1,
        n_classes=len(class_names),
        shuffle=False,
        augment=False  # Typically, no augmentation during evaluation
    )

    # Add debugging information for data loading
    print(f"Found {len(test_paths)} test files")
    print(f"Class distribution:", {name: sum(1 for path, label in test_labels.items() if label == i) 
                                 for i, name in enumerate(class_names)})

    # Modify the evaluation and prediction logic
    try:
        # Evaluate the model
        test_loss, test_accuracy = model.evaluate(test_generator)
        print(f"Test accuracy: {test_accuracy:.4f}")

        # Generate predictions
        y_pred = []
        y_true = []
        
        # Get the total number of batches
        num_batches = len(test_generator)
        print(f"Total number of batches: {num_batches}")
        
        for i, (x, y) in enumerate(test_generator):
            if x.size == 0:
                print(f"Warning: Empty batch encountered at index {i}")
                continue
                
            pred = model.predict(x, verbose=0)  # Reduced verbosity
            y_pred.extend(np.argmax(pred, axis=1))
            y_true.extend(np.argmax(y, axis=1))
            
            if i >= num_batches - 1:  # Stop after processing all batches
                break

        if not y_pred or not y_true:
            raise ValueError("No predictions were generated. Check if the test data is being loaded correctly.")

        # Create confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        class_names = sorted(os.listdir(data_dir))  # Get class names for labels
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=class_names,
                    yticklabels=class_names)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.show()

        # Generate classification report
        print(classification_report(y_true, y_pred, target_names=class_names))

    except Exception as e:
        print(f"Error during evaluation: {str(e)}")
        print("Debug information:")
        print(f"Number of test paths: {len(test_paths)}")
        print(f"Number of test labels: {len(test_labels)}")
        raise  # Re-raise the exception for full traceback

if __name__ == "__main__":
    MODEL_PATH = 'model/julien-hovan/saved_models/final_model.keras'
    load_and_evaluate_model(MODEL_PATH, DATA_DIR, IMG_HEIGHT, IMG_WIDTH, BATCH_SIZE)
