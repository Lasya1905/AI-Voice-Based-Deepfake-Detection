import numpy as np
import librosa
import os
from tensorflow.keras.models import load_model

# SETTINGS (same as before)
SAMPLE_RATE = 22050
N_MFCC = 40
MAX_LEN = 100

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 🔹 Load model
model_path = os.path.join(BASE_DIR, "models/cnn_model.h5")  # or .keras
model = load_model(model_path)


# 🔹 Extract MFCC (same logic)
def extract_mfcc(file_path):
    audio, sr = librosa.load(file_path, sr=SAMPLE_RATE)

    if len(audio) == 0:
        return None

    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=N_MFCC)

    if mfcc.shape[1] < MAX_LEN:
        pad_width = MAX_LEN - mfcc.shape[1]
        mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)))
    else:
        mfcc = mfcc[:, :MAX_LEN]

    return mfcc


# 🔹 Predict function
def predict(file_path):
    mfcc = extract_mfcc(file_path)

    if mfcc is None:
        print("Invalid audio file")
        return

    mfcc = mfcc[np.newaxis, ..., np.newaxis]

    pred = model.predict(mfcc)[0][0]

    if pred > 0.5:
        print(f"🔴 Fake Voice (confidence: {pred:.2f})")
    else:
        print(f"🟢 Real Voice (confidence: {1 - pred:.2f})")


# 🔹 Run
if __name__ == "__main__":
    file_path = input("Enter audio file path: ")
    predict(file_path)