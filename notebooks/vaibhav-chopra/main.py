import os
from preprocessing import preprocess_audio
from model import load_model, predict_genre
from streamlit_app import run_streamlit_app

# Main execution function
def main():
    # Path to the saved model directory
    model_path = "saved_model"
    
    # Check if model directory exists
    if os.path.exists(model_path):
        # Load the pre-trained model
        model = load_model(model_path)
    else:
        raise FileNotFoundError(f"Model directory not found at {model_path}. Please train and save the model first.")
    
    # Run the Streamlit app
    run_streamlit_app(model)

if __name__ == "__main__":
    main()
