import numpy as np
import os
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from model import build_model

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

print("🔹 Loading data...")

X_train = np.load(os.path.join(BASE_DIR, "data/splits/X_train.npy"))
y_train = np.load(os.path.join(BASE_DIR, "data/splits/y_train.npy"))

X_val = np.load(os.path.join(BASE_DIR, "data/splits/X_val.npy"))
y_val = np.load(os.path.join(BASE_DIR, "data/splits/y_val.npy"))

print(f"Train shape: {X_train.shape}")
print(f"Val shape: {X_val.shape}")

print("🔹 Building model...")
model = build_model()

# 🔹 Callbacks
checkpoint_path = os.path.join(BASE_DIR, "models/cnn_model.h5")

callbacks = [
    EarlyStopping(patience=3, restore_best_weights=True),
    ModelCheckpoint(checkpoint_path, save_best_only=True)
]

print("🔹 Starting training...")

history = model.fit(
    X_train, y_train,
    epochs=15,
    batch_size=32,
    validation_data=(X_val, y_val),
    callbacks=callbacks
)

print("🔹 Training finished")

# 🔹 Save model
model.save(checkpoint_path)

print(f"✅ Model saved at: {checkpoint_path}")