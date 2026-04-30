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

voice-deepfake-detection/ <br>
│<br>
├── app/<br>
│ └── app.py<br>
│<br>
├── data/<br>
│ ├── raw/<br>
│ ├── processed/<br>
│ └── splits/<br>
│<br>
├── models/<br>
│<br>
├── src/<br>
│ ├── feature_extraction.py<br>
│ ├── dataset.py<br>
│ ├── model.py<br>
│ ├── train.py<br>
│ ├── evaluate.py<br>
│ └── predict.py<br>
│<br>
├── .gitignore<br>
└── README.md<br>
