import os
import streamlit as st
from preprocessing import preprocess_audio
from model import predict_genre

def run_streamlit_app(model):
    """
    Runs the Streamlit app for audio genre classification.

    Parameters:
    model: The trained CNN model for genre prediction.
    """
    st.title("Music Genre Classifier")
    st.markdown("Upload an audio file to get its genre prediction.")

    audio_file = st.file_uploader("Choose an audio file", type=["mp3", "wav"])

    if audio_file is not None:
        try:
            with st.spinner("Processing the audio file..."):
                audio_path = f"uploaded_{audio_file.name}"
                with open(audio_path, "wb") as f:
                    f.write(audio_file.read())

                # Preprocess the audio file to generate spectrogram
                spectrogram_path = preprocess_audio(audio_path)

            # Display the generated spectrogram
            st.image(spectrogram_path, caption="Generated Spectrogram", use_column_width=True)

            with st.spinner("Predicting genre..."):
                genre = predict_genre(model, spectrogram_path)

            st.success(f"Predicted Genre: {genre}")

        except Exception as e:
            st.error(f"An error occurred: {e}")

        finally:
            # Clean up the temporary files
            if os.path.exists(audio_path):
                os.remove(audio_path)
            if os.path.exists(spectrogram_path):
                os.remove(spectrogram_path)
