import os
import numpy as np
import librosa

# SETTINGS
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_PATH = os.path.join(BASE_DIR, "data", "raw")
REAL_PATH = os.path.join(DATA_PATH, "real")
FAKE_PATH = os.path.join(DATA_PATH, "fake")

SAMPLE_RATE = 22050
N_MFCC = 40
MAX_LEN = 100   # time frames


# 🔹 Extract MFCC from one file
def extract_mfcc(file_path):
    try:
        audio, sr = librosa.load(file_path, sr=SAMPLE_RATE)

        # skip empty audio
        if len(audio) == 0:
            return None

        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=N_MFCC)

        if mfcc.shape[1] < MAX_LEN:
            pad_width = MAX_LEN - mfcc.shape[1]
            mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)))
        else:
            mfcc = mfcc[:, :MAX_LEN]

        return mfcc

    except Exception:
        return None

# 🔹 Load dataset
def load_data():
    X = []
    y = []

    # REAL → label 0
    for file in os.listdir(REAL_PATH)[:15000]:
        file_path = os.path.join(REAL_PATH, file)
        if os.path.isfile(file_path):
            mfcc = extract_mfcc(file_path)
            if mfcc is not None:
                X.append(mfcc)
                y.append(0)

    # FAKE → label 1
    for file in os.listdir(FAKE_PATH)[:15000]:
        file_path = os.path.join(FAKE_PATH, file)
        if os.path.isfile(file_path):
            mfcc = extract_mfcc(file_path)
            if mfcc is not None:
                X.append(mfcc)
                y.append(1)

    return np.array(X), np.array(y)


# 🔹 Save processed data
def save_data():
    print("Extracting features...")

    X, y = load_data()

    # Add channel dimension for CNN
    X = X[..., np.newaxis]

    os.makedirs("data/processed", exist_ok=True)

    np.save("data/processed/X.npy", X)
    np.save("data/processed/y.npy", y)

    print("Saved:")
    print("X shape:", X.shape)
    print("y shape:", y.shape)


# 🔹 Run this file directly
if __name__ == "__main__":
    save_data()