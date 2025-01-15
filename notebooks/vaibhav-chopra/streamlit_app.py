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

    # Upload the audio file
    audio_file = st.file_uploader("Choose an audio file", type=["mp3", "wav"])

    if audio_file is not None:
        # Save the uploaded file locally
        try:
            with st.spinner("Processing the audio file..."):
                audio_path = f"uploaded_{audio_file.name}"
                with open(audio_path, "wb") as f:
                    f.write(audio_file.read())
                st.write(f"File saved as {audio_path}")

                # Preprocess the audio file to generate spectrogram
                spectrogram_path = preprocess_audio(audio_path)
                st.write(f"Spectrogram generated at {spectrogram_path}")

                if os.path.exists(spectrogram_path):
                    st.image(spectrogram_path, caption="Generated Spectrogram", use_column_width=True)
                else:
                    st.error("Failed to generate spectrogram.")
                    return  # Exit the function early if spectrogram generation fails

                # Predict genre using the model
                with st.spinner("Predicting genre..."):
                    genre = predict_genre(model, spectrogram_path)

                if genre:
                    st.success(f"Predicted Genre: {genre}")
                else:
                    st.error("Genre prediction failed.")

        except Exception as e:
            st.error(f"An error occurred: {e}")
            st.write(f"Exception details: {e}")  # Debug message

        finally:
            # Clean up temporary files
            if os.path.exists(audio_path):
                os.remove(audio_path)
            if os.path.exists(spectrogram_path):
                os.remove(spectrogram_path)
    else:
        st.info("Please upload an audio file.")
