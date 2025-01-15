
import librosa
import numpy as np
import matplotlib.pyplot as plt
import os

def preprocess_audio(audio_path, output_dir="spectrograms"):
    """
    Converts an audio file into a spectrogram image.

    Parameters:
    audio_path (str): Path to the audio file.
    output_dir (str): Directory to save the spectrogram image.

    Returns:
    str: Path to the saved spectrogram image.
    """
    y, sr = librosa.load(audio_path, sr=None)
    spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    log_spectrogram = librosa.power_to_db(spectrogram, ref=np.max)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_file = os.path.join(output_dir, f"{os.path.basename(audio_path).split('.')[0]}.png")

    plt.figure(figsize=(10, 4))
    librosa.display.specshow(log_spectrogram, sr=sr, x_axis='time', y_axis='mel')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel-Spectrogram')
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.close()

    return output_file