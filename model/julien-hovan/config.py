class ModelConfig:
    # CNN Configuration
    CNN_FILTERS = [32, 64, 128, 256]  # Number of filters in each conv block
    KERNEL_SIZE = (3, 3)
    POOL_SIZE = (2, 2)
    
    # Classifier Configuration
    EMBED_DIM = 512
    NUM_HEADS = 16
    NUM_TRANSFORMER_BLOCKS = 2
    DENSE_UNITS = [512, 256]  # Units in dense layers
    DROPOUT_RATE = 0.5
    
    # Time-aware Classifier Configuration
    NUM_SEGMENTS = 7  # Default number of time segments
    
    # Data Generator Configuration
    BATCH_SIZE = 16
    SPECTROGRAM_DIM = (128, 172)  # (height, width)
    N_CHANNELS = 1
    AUGMENTATION_PARAMS = {
        'freq_shift_range': 0.05,  # 5% of frequency range
        'noise_factor': 0.02,
        'time_stretch_range': (0.8, 1.2),
        'pitch_shift_range': (-2, 2),  # semitones
        'freq_mask_width': 0.1,  # 10% of frequency bins
        'time_mask_width': 0.1,  # 10% of time steps
    }
    
    # Training Configuration
    EPOCHS = 100
    LEARNING_RATE = 0.0001
    EARLY_STOPPING_PATIENCE = 15
    REDUCE_LR_PATIENCE = 5
    REDUCE_LR_FACTOR = 0.5
    MIN_LR = 0.00001
    
    # Paths Configuration
    SPECTROGRAMS_DIR = '/Users/julienh/Desktop/SDS/SDS-CP018-music-classifier/Data/mel_spectrograms_images'
    MODEL_CHECKPOINT_PATH = 'saved_models/best_cnn_model.keras'
    FINAL_MODEL_PATH = 'saved_models/final_model.keras'
    VISUALIZATION_DIR = 'model/julien-hovan/visualizations'