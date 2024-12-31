import numpy as np
import tensorflow as tf

class TimeSegmentedSpectrogramGenerator(tf.keras.utils.Sequence):
    def __init__(self, paths, labels, batch_size=32, dim=(128, 128), 
                 n_channels=1, n_classes=10, shuffle=True, augment=False,
                 normalization_stats=None):
        super().__init__()
        self.paths = paths
        self.labels = labels
        self.batch_size = batch_size
        self.dim = dim
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.augment = augment
        self.indexes = np.arange(len(self.paths))
        if self.shuffle:
            np.random.shuffle(self.indexes)
        self.normalization_stats = normalization_stats
    
    def __len__(self):
        return int(np.floor(len(self.paths) / self.batch_size))
    
    def __getitem__(self, index):
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        
        # Generate data
        X, y = self.__data_generation(indexes)
        
        return X, y
    
    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)
    
    def __data_generation(self, indexes):
        # Load data
        X = []
        y = []
        
        for idx in indexes:
            try:
                # Load saved numpy array (contains multiple time segments)
                segments = np.load(self.paths[idx])
                
                # Debug print
                #print(f"Original segments shape: {segments.shape}")
                
                # Resize segments to match expected dimensions
                resized_segments = []
                for segment in segments:
                    # Add channel dimension if needed for resize operation
                    if len(segment.shape) == 2:
                        segment = np.expand_dims(segment, axis=-1)
                    
                    # Debug print
                    #print(f"Segment shape before resize: {segment.shape}")
                    
                    # Resize each segment to match the expected dim
                    resized_segment = tf.image.resize(segment, self.dim)
                    resized_segments.append(resized_segment)
                
                segments = np.stack(resized_segments)
                
                # Debug print
                #print(f"Final segments shape: {segments.shape}")
                
                if self.augment:
                    segments = self._augment_segments(segments)
                
                X.append(segments)
                y.append(self.labels[self.paths[idx]])
                
            except Exception as e:
                print(f"Error processing file: {self.paths[idx]}")
                print(f"Error message: {str(e)}")
                raise
        
        # Stack all segments
        X = np.stack(X, axis=0)
        
        # Add channel dimension if needed
        if self.n_channels == 1:
            X = np.expand_dims(X, axis=-1)
        
        # Final debug print
        #print(f"Batch X shape: {X.shape}")
        
        return X, tf.keras.utils.to_categorical(y, num_classes=self.n_classes)
    
    def _augment_segments(self, segments):
        # Add time-sequence aware augmentations here
        # For example, small frequency shifts that are consistent across segments
        if self.augment:
            if np.random.random() < 0.5:
                segments = self._random_frequency_shift(segments)
            if np.random.random() < 0.5:
                segments = self._add_noise(segments)
            if np.random.random() < 0.5:
                segments = self._time_stretch(segments)
            if np.random.random() < 0.5:
                segments = self._pitch_shift(segments)
        return segments
    
    def _random_frequency_shift(self, segments):
        # Shift the frequency of the segments by a random amount
        max_shift = segments.shape[1] // 20  # Shift up to 5% of the frequency range
        shift = np.random.randint(-max_shift, max_shift)
        
        # Create a copy to avoid modifying the original array
        shifted_segments = np.zeros_like(segments)
        
        if shift > 0:
            shifted_segments[:, shift:, :] = segments[:, :-shift, :]
        elif shift < 0:
            shifted_segments[:, :shift, :] = segments[:, -shift:, :]
        else:
            shifted_segments = segments
        
        return shifted_segments
    
    def _add_noise(self, segments, noise_factor=0.02):
        # Add random noise to the segments
        noise = np.random.normal(0, noise_factor, segments.shape)
        noisy_segments = segments + noise
        return noisy_segments
    
    def _time_stretch(self, segments, rate=None):
        # Time stretch the segments by a small random amount
        if rate is None:
            rate = np.random.uniform(0.8, 1.2)  # Stretch between 80% and 120%
        
        stretched_segments = []
        for segment in segments:
            # Resize along the time axis (axis=1) to the target dimension
            stretched_segment = tf.image.resize(
                segment,
                (segment.shape[0], self.dim[1]),  # Resize to target time dimension
                method=tf.image.ResizeMethod.BILINEAR,
                preserve_aspect_ratio=False  # Force exact dimensions
            )
            stretched_segments.append(stretched_segment)
        
        return tf.stack(stretched_segments)
    
    def _pitch_shift(self, segments, semitones=None):
        # Pitch shift the segments by a small random amount
        if semitones is None:
            semitones = np.random.uniform(-2, 2)  # Shift up to 2 semitones
        
        # Convert semitones to a frequency ratio
        ratio = 2 ** (semitones / 12.0)
        
        pitch_shifted_segments = []
        for segment in segments:
            # Resize along the frequency axis (axis=0) to the target dimension
            pitch_shifted_segment = tf.image.resize(
                segment,
                (self.dim[0], segment.shape[1]),  # Resize to target frequency dimension
                method=tf.image.ResizeMethod.BILINEAR,
                preserve_aspect_ratio=False  # Force exact dimensions
            )
            pitch_shifted_segments.append(pitch_shifted_segment)
        
        return tf.stack(pitch_shifted_segments) 
    
    def _normalize(self, segments):
        if self.normalization_stats is not None:
            if "mean" in self.normalization_stats:
                # Z-score normalization
                mean_val = self.normalization_stats["mean"]
                std_val = self.normalization_stats["std"]
                if std_val != 0:
                    segments = (segments - mean_val) / std_val
                else:
                    segments = np.zeros_like(segments)
            elif "min" in self.normalization_stats:
                # Min-max normalization
                min_val = self.normalization_stats["min"]
                max_val = self.normalization_stats["max"]
                if max_val - min_val != 0:
                    segments = (segments - min_val) / (max_val - min_val)
                else:
                    segments = np.zeros_like(segments)
        return segments 