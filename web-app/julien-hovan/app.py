import streamlit as st
import numpy as np
from PIL import Image

# Page configuration
st.set_page_config(
    page_title="The Music Translator",
    page_icon="🎵",
    layout="wide"
)

# Main title and description
st.title("🎵 The Music Translator")
st.markdown("""
    Transform your music into visual patterns, display its spectrogram, and discover its genre! 
    Upload an audio file and let AI analyze its musical characteristics.
""")

# Create two columns for layout
col1, col2 = st.columns([3, 2])

with col1:
    # File upload section
    st.subheader("Upload Your Audio")
    uploaded_file = st.file_uploader(
        "Choose an MP3 or WAV file",
        type=["mp3", "wav"],
        help="Maximum file size: 200MB"
    )
    
    # Example audio section
    with st.expander("Or try an example"):
        if st.button("Use Example Audio"):
            st.info("Loading example audio file...")
            # Placeholder for example functionality
    
    # Display upload status and spectrogram
    if uploaded_file is not None:
        st.success("File successfully uploaded!")
        
        with st.spinner("Generating spectrogram..."):
            # Placeholder for spectrogram display
            st.image(
                "https://via.placeholder.com/600x300?text=Spectrogram+Visualization",
                caption="Audio Spectrogram"
            )

with col2:
    # Results section
    st.subheader("Analysis Results")
    
    if uploaded_file is not None:
        with st.spinner("Analyzing audio..."):
            # Placeholder for genre prediction
            st.markdown("### Predicted Genre")
            st.info("Rock", icon="🎸")
            
            # Confidence scores
            st.markdown("### Confidence Scores")
            genres = ["Rock", "Jazz", "Classical", "Hip Hop", "Electronic"]
            scores = np.random.uniform(0, 1, len(genres))
            
            for genre, score in zip(genres, scores):
                st.progress(score, text=f"{genre}: {score:.1%}")

# Additional information
with st.expander("About The Music Translator"):
    st.markdown("""
        This application uses deep learning to analyze audio files and predict their musical genre.
        The model has been trained on the GTZAN dataset, featuring 1000 audio tracks across 10 genres.
        
        **Supported Genres:**
        - Blues 🎸
        - Classical 🎻
        - Country 🤠
        - Disco 🕺
        - Hip Hop 🎤
        - Jazz 🎷
        - Metal 🤘
        - Pop 🎵
        - Reggae 🎼
        - Rock 🎸
        
        ---
        
        ### Model Architecture
        
        This project employs a sophisticated deep learning model that combines Convolutional Neural Networks (CNNs) 
        with multi-head attention mechanisms, achieving 85% accuracy on the test dataset.
        
        **The model processes audio in three main stages:**
        
        1. **Spatial Feature Extraction (CNN)**
           - Converts audio into spectrogram images
           - Processes 4-second segments through convolutional layers
           - Uses TimeDistributed layers to maintain sequential structure
        
        2. **Temporal Feature Extraction (Multi-Head Attention)**
           - Analyzes relationships between different time segments
           - Identifies key moments in the audio
           - Weighs the importance of different parts of the song
        
        3. **Genre Classification**
           - Processes the attention-weighted features
           - Makes final genre prediction through fully connected layers
        
        **Technical Details:**
        - Input: Spectrogram segments (4 seconds each)
        - Architecture: CNN + Multi-Head Attention
        - Performance: 85% accuracy on test set
        - Dataset: GTZAN (1000 tracks, 10 genres)
    """)

# Footer
st.markdown("---")
st.markdown("Created with ❤️ by Julien Hovan")