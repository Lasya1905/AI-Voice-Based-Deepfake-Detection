# AI Voice Deepfake Detection using MFCC & CNN

## Overview
This project detects whether an audio clip is **real (human voice)** or **AI-generated (deepfake)** using **MFCC (Mel-Frequency Cepstral Coefficients)** and a **Convolutional Neural Network (CNN)**.

The system processes raw audio, extracts meaningful features, and classifies the input with high accuracy.

---

## Features
- Audio-based deepfake detection  
- MFCC feature extraction  
- CNN-based classification  
- High accuracy (~96%)  
- Modular pipeline  
- Streamlit UI for easy testing  

---

## Methodology
Audio → MFCC Extraction → CNN → Classification

---

## 📁 Project Structure

voice-deepfake-detection/
│
├── app/
│ └── app.py
│
├── data/
│ ├── raw/
│ ├── processed/
│ └── splits/
│
├── models/
│
├── src/
│ ├── feature_extraction.py
│ ├── dataset.py
│ ├── model.py
│ ├── train.py
│ ├── evaluate.py
│ └── predict.py
│
├── .gitignore
└── README.md
