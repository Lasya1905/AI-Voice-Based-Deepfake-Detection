import streamlit as st
import numpy as np
import librosa
import os
from tensorflow.keras.models import load_model

# SETTINGS
SAMPLE_RATE = 22050
N_MFCC = 40
MAX_LEN = 100

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Load model
model = load_model(os.path.join(BASE_DIR, "models/cnn_model.h5"))


# MFCC extraction
def extract_mfcc(audio):
    mfcc = librosa.feature.mfcc(y=audio, sr=SAMPLE_RATE, n_mfcc=N_MFCC)

    if mfcc.shape[1] < MAX_LEN:
        pad_width = MAX_LEN - mfcc.shape[1]
        mfcc = np.pad(mfcc, ((0, 0), (0, pad_width)))
    else:
        mfcc = mfcc[:, :MAX_LEN]

    return mfcc


# UI
st.title("🎙️ Voice Deepfake Detection")

uploaded_file = st.file_uploader("Upload audio file (.wav)", type=["wav"])

if uploaded_file is not None:
    st.audio(uploaded_file)

    # Load audio
    audio, sr = librosa.load(uploaded_file, sr=SAMPLE_RATE)

    mfcc = extract_mfcc(audio)
    mfcc = mfcc[np.newaxis, ..., np.newaxis]

    pred = model.predict(mfcc)[0][0]

    if pred > 0.5:
        st.error(f"🔴 Fake Voice (confidence: {pred:.2f})")
    else:
        st.success(f"🟢 Real Voice (confidence: {1 - pred:.2f})")