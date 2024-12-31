import tensorflow as tf
from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split
from classifier import create_music_genre_classifier, create_minimal_cnn_classifier, create_time_aware_classifier
from data_generator import TimeSegmentedSpectrogramGenerator
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from collections import Counter
import logging

def prepare_data(spectrograms_dir):
    """
    Prepare data paths and labels from the spectrograms directory
    """
    genres = ['blues', 'classical', 'country', 'disco', 'hiphop', 
              'jazz', 'metal', 'pop', 'reggae', 'rock']
    
    paths = []
    labels = {}
    
    # Add counter for each genre
    genre_counts = {}
    
    for idx, genre in enumerate(genres):
        genre_path = Path(spectrograms_dir) / genre
        genre_files = list(genre_path.glob('*.npy'))
        genre_counts[genre] = len(genre_files)
        
        for spec_path in genre_files:
            paths.append(str(spec_path))
            labels[str(spec_path)] = idx
    
    # Print summary of files found
    print("\nSpectrogram counts per genre:")
    print("-" * 30)
    for genre, count in genre_counts.items():
        print(f"{genre:10} : {count:4d} files")
    print("-" * 30)
    print(f"Total files: {len(paths)}\n")
    
    return paths, labels, genres

def plot_training_history(history):
    """
    Plot training history
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Accuracy plot
    ax1.plot(history.history['accuracy'], label='Training Accuracy')
    ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax1.set_title('Model Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    
    # Loss plot
    ax2.plot(history.history['loss'], label='Training Loss')
    ax2.plot(history.history['val_loss'], label='Validation Loss')
    ax2.set_title('Model Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('model/julien-hovan/visualizations/training_history.png')
    plt.close()

def plot_confusion_matrix(y_true, y_pred, classes):
    """
    Plot confusion matrix
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(12, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('model/julien-hovan/visualizations/confusion_matrix.png')
    plt.close()

def calculate_normalization_stats(train_paths, normalization_type="zscore"):
    """
    Calculate normalization statistics (min/max or mean/std) over the entire training set.
    
    Args:
        train_paths: List of paths to training data files.
        normalization_type: Either "minmax" or "zscore".
    
    Returns:
        A dictionary containing the normalization statistics.
    """
    all_data = []
    for path in train_paths:
        data = np.load(path)
        all_data.append(data)  # Append data to the list
    all_data = np.concatenate(all_data, axis=0)  # Concatenate along the first axis (time segments)

    if normalization_type == "minmax":
        min_val = np.min(all_data)
        max_val = np.max(all_data)
        return {"min": min_val, "max": max_val}
    elif normalization_type == "zscore":
        mean_val = np.mean(all_data)
        std_val = np.std(all_data)
        return {"mean": mean_val, "std": std_val}
    else:
        raise ValueError("Invalid normalization_type. Choose 'minmax' or 'zscore'.")


def main():
    # Set random seeds for reproducibility
    tf.random.set_seed(42)
    np.random.seed(42)
    
    # Parameters
    SPECTROGRAMS_DIR = '/Users/julienh/Desktop/SDS/SDS-CP018-music-classifier/Data/mel_spectrograms_images'
    BATCH_SIZE = 16  # Might need to reduce batch size due to larger data
    EPOCHS = 100
    IMG_HEIGHT = 128
    IMG_WIDTH = 128
    CHANNELS = 1
    NUM_SEGMENTS = 7 # For 30-second audio split into 4-second segments
    
    # Prepare data
    print("Preparing data...")
    spectrogram_paths, labels, genres = prepare_data(SPECTROGRAMS_DIR)
    num_classes = len(genres)
    
    # Debug: Check the first file
    first_file = spectrogram_paths[0]
    print(f"\nChecking first file: {first_file}")
    data = np.load(first_file)
    print(f"Data shape: {data.shape}")
    print(f"Data type: {data.dtype}")
    print(f"Data range: [{data.min()}, {data.max()}]")
    
    # Split data into train, validation, and test sets
    train_paths, test_paths, train_labels, test_labels = train_test_split(
        spectrogram_paths,
        list(labels.values()),  # Pass the actual list of labels
        test_size=0.2,
        random_state=42,
        stratify=list(labels.values())  # Stratify by genre labels
    )
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        train_paths,
        train_labels,  # Use the corresponding labels for the second split
        test_size=0.2,
        random_state=42,
        stratify=train_labels  # Stratify again for validation set
    )
    
    print(f"Number of training samples: {len(train_paths)}")
    print(f"Number of validation samples: {len(val_paths)}")
    print(f"Number of test samples: {len(test_paths)}")
    
    # After first split
    train_val_label_dist = Counter(train_labels)
    test_label_dist = Counter(test_labels)
    
    print("\nLabel distribution after train/test split:")
    print("-" * 50)
    print("Train + Val set:")
    for label, count in sorted(train_val_label_dist.items()):
        print(f"Genre {genres[label]:10}: {count:4d} files ({count/len(train_labels)*100:.1f}%)")
    print("\nTest set:")
    for label, count in sorted(test_label_dist.items()):
        print(f"Genre {genres[label]:10}: {count:4d} files ({count/len(test_labels)*100:.1f}%)")
    print("-" * 50)
    
    # After second split
    train_label_dist = Counter(train_labels)
    val_label_dist = Counter(val_labels)
    
    print("\nFinal distribution after validation split:")
    print("-" * 50)
    print("Training set:")
    for label, count in sorted(train_label_dist.items()):
        print(f"Genre {genres[label]:10}: {count:4d} files ({count/len(train_labels)*100:.1f}%)")
    print("\nValidation set:")
    for label, count in sorted(val_label_dist.items()):
        print(f"Genre {genres[label]:10}: {count:4d} files ({count/len(val_labels)*100:.1f}%)")
    print("-" * 50)
    
    # Calculate normalization statistics
    normalization_stats = calculate_normalization_stats(train_paths, normalization_type="zscore")
    print(f"Normalization stats: {normalization_stats}")

    # Create data generators
    train_generator = TimeSegmentedSpectrogramGenerator(
        train_paths,
        labels,
        batch_size=BATCH_SIZE,
        dim=(IMG_HEIGHT, IMG_WIDTH),
        n_channels=CHANNELS,
        n_classes=num_classes,
        augment=True,
        normalization_stats=normalization_stats
    )
    
    val_generator = TimeSegmentedSpectrogramGenerator(
        val_paths,
        labels,
        batch_size=BATCH_SIZE,
        dim=(IMG_HEIGHT, IMG_WIDTH),
        n_channels=CHANNELS,
        n_classes=num_classes,
        normalization_stats=normalization_stats
    )

    test_generator = TimeSegmentedSpectrogramGenerator(
        test_paths,
        labels,
        batch_size=BATCH_SIZE,
        dim=(IMG_HEIGHT, IMG_WIDTH),
        n_channels=CHANNELS,
        n_classes=num_classes,
        shuffle=False,
        normalization_stats=normalization_stats
    )
    
    # Create model
    model = create_time_aware_classifier(
        input_shape=(IMG_HEIGHT, IMG_WIDTH, CHANNELS),
        num_classes=num_classes,
        num_segments=NUM_SEGMENTS
    )
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),  # Slightly lower learning rate
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Callbacks
    callbacks = [
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=0.00001
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ModelCheckpoint(
            'saved_models/best_cnn_model.keras',
            monitor='val_accuracy',
            save_best_only=True,
            mode='max'
        )
    ]
    
    # Train model
    print("Training model...")
    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=EPOCHS,
        callbacks=callbacks
    )
    
    # Plot training history
    plot_training_history(history)
    
    # Evaluate on test set
    print("\nEvaluating model on test set...")
    test_loss, test_accuracy = model.evaluate(test_generator)
    print(f"Test accuracy: {test_accuracy:.4f}")
    
    # Generate predictions for confusion matrix
    y_pred = []
    y_true = []
    
    for i in range(len(test_generator)):
        x, y = test_generator[i]
        pred = model.predict(x)
        y_pred.extend(np.argmax(pred, axis=1))
        y_true.extend(np.argmax(y, axis=1))
    
    # Plot confusion matrix
    plot_confusion_matrix(y_true, y_pred, genres)
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=genres))
    
    # Save model
    model.save('saved_models/final_model.keras')
    print("\nModel saved as 'final_model.keras'")

if __name__ == "__main__":
    main() 