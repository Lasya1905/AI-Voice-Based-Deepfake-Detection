import numpy as np
from sklearn.model_selection import train_test_split
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Load data
X = np.load(os.path.join(BASE_DIR, "data/processed/X.npy"))
y = np.load(os.path.join(BASE_DIR, "data/processed/y.npy"))

# 🔹 Split: train (70), temp (30)
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# 🔹 Split temp → val (15), test (15)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
)

# 🔹 Save splits
os.makedirs(os.path.join(BASE_DIR, "data/splits"), exist_ok=True)

np.save(os.path.join(BASE_DIR, "data/splits/X_train.npy"), X_train)
np.save(os.path.join(BASE_DIR, "data/splits/y_train.npy"), y_train)

np.save(os.path.join(BASE_DIR, "data/splits/X_val.npy"), X_val)
np.save(os.path.join(BASE_DIR, "data/splits/y_val.npy"), y_val)

np.save(os.path.join(BASE_DIR, "data/splits/X_test.npy"), X_test)
np.save(os.path.join(BASE_DIR, "data/splits/y_test.npy"), y_test)

print("Done splitting!")
print("Train:", X_train.shape)
print("Val:", X_val.shape)
print("Test:", X_test.shape)