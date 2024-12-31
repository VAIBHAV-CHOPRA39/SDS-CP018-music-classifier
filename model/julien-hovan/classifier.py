import tensorflow as tf
from tensorflow.keras import layers, Model
from cnn import create_cnn_model
from attention import MultiHeadSelfAttention

def create_music_genre_classifier(
    input_shape,
    num_classes,
    embed_dim,
    num_heads,
    num_transformer_blocks
):
    """
    Create a music genre classifier with CNN, Multi-head Attention, and Classification layers
    """
    
    # CNN Feature Extractor
    cnn_model = create_cnn_model(input_shape)
    
    # Input Layer
    inputs = layers.Input(shape=input_shape)
    
    # Apply CNN
    x = cnn_model(inputs)
    
    # Reshape for attention mechanism
    x = layers.Reshape((-1, x.shape[-1]))(x)
    
    # Position Embedding
    x = layers.Dense(embed_dim)(x)
    
    # Multi-head Attention Blocks
    for _ in range(num_transformer_blocks):
        x = MultiHeadSelfAttention(embed_dim, num_heads)(x)
    
    # Global Average Pooling
    x = layers.GlobalAveragePooling1D()(x)
    
    # Classification Head
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    # Create model
    model = Model(inputs=inputs, outputs=outputs)
    
    return model 

def create_minimal_cnn_classifier(input_shape, num_classes):
    """
    Creates a simple CNN classifier for music genre classification
    """
    model = tf.keras.Sequential([
        # Input layer
        tf.keras.layers.Input(shape=input_shape),
        
        # First Conv Block
        tf.keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Dropout(0.25),
        
        # Second Conv Block
        tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Dropout(0.25),
        
        # Third Conv Block
        tf.keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Dropout(0.25),
        
        # Dense Layers
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    
    return model 

def create_time_aware_classifier(input_shape, num_classes, num_segments):
    """
    Creates a time-aware classifier that processes multiple segments of a spectrogram
    input_shape: (height, width, channels) for each segment
    num_segments: number of time segments per sample
    """
    # CNN to process each segment
    segment_input = tf.keras.layers.Input(shape=input_shape)
    
    # Use the CNN model from cnn.py
    cnn = create_cnn_model(input_shape)
    
    x = cnn(segment_input)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    
    # Create the time-distributed model
    segment_model = tf.keras.Model(inputs=segment_input, outputs=x)
    
    # Main model that processes all segments
    main_input = tf.keras.layers.Input(shape=(num_segments, *input_shape))
    
    # Apply the segment model to each time segment
    processed_segments = tf.keras.layers.TimeDistributed(segment_model)(main_input)
    
    # Add attention mechanism using MultiHeadSelfAttention from attention.py
    attention_layer = MultiHeadSelfAttention(embed_dim=256, num_heads=8)
    attention_output = attention_layer(processed_segments)
    
    # Global temporal pooling
    x = tf.keras.layers.GlobalAveragePooling1D()(attention_output)
    
    # Final classification
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
    
    model = tf.keras.Model(inputs=main_input, outputs=outputs)
    return model 