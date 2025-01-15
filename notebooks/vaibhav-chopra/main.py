from preprocessing import preprocess_audio
from model import load_model, predict_genre
from streamlit_app import run_streamlit_app

# Main execution function
def main():
    # Load the pre-trained model
    model = load_model()

    # Run the Streamlit app
    run_streamlit_app(model)

if __name__ == "__main__":
    main()
