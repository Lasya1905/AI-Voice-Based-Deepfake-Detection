import numpy as np
import os
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

print("🔹 Loading test data...")

X_test = np.load(os.path.join(BASE_DIR, "data/splits/X_test.npy"))
y_test = np.load(os.path.join(BASE_DIR, "data/splits/y_test.npy"))

print(f"Test shape: {X_test.shape}")

print("🔹 Loading model...")

model_path = os.path.join(BASE_DIR, "models/cnn_model.h5")  # or .keras
model = load_model(model_path)

print("🔹 Evaluating...")

loss, acc = model.evaluate(X_test, y_test)
print(f"\n✅ Test Accuracy: {acc:.4f}")
print(f"Test Loss: {loss:.4f}")

# 🔹 Predictions
y_pred = model.predict(X_test)
y_pred = (y_pred > 0.5).astype(int)

# 🔹 Metrics
print("\n🔹 Classification Report:")
print(classification_report(y_test, y_pred))

print("\n🔹 Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))