import pandas as pd
import joblib
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ==================================================
# CONFIG
# ==================================================

DATA_PATH = "datasets/dataset_7M.csv"
MODEL_PATH = "models/behaviour_model.pkl"
ENCODER_PATH = "models/behaviour_encoder.pkl"

RANDOM_STATE = 42
SAMPLE_SIZE = 100000   # mets None si tu veux tout utiliser

FEATURES = [
    "AccX",
    "AccY",
    "AccZ",
    "GyroX",
    "GyroY",
    "GyroZ"
]

TARGET = "Class"

# ==================================================
# LOAD DATASET
# ==================================================

print("Loading dataset...")
df = pd.read_csv(DATA_PATH)
print("Dataset loaded.")

print("\nDataset shape:", df.shape)

# ==================================================
# BASIC CLEANING
# ==================================================

df = df.dropna(subset=FEATURES + [TARGET])

# Si tu veux tester plus vite au début
if SAMPLE_SIZE is not None and SAMPLE_SIZE < len(df):
    df = df.sample(n=SAMPLE_SIZE, random_state=RANDOM_STATE)

print("Shape after cleaning/sampling:", df.shape)

# ==================================================
# FEATURES / TARGET
# ==================================================

X = df[FEATURES].copy()
y = df[TARGET].copy()

# ==================================================
# ENCODE TARGET
# ==================================================

encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

print("\nClasses found:")
for idx, cls in enumerate(encoder.classes_):
    print(f"{idx}: {cls}")

# Save encoder
joblib.dump(encoder, ENCODER_PATH)
print(f"\nLabel encoder saved to {ENCODER_PATH}")

# ==================================================
# TRAIN / TEST SPLIT
# ==================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.2,
    random_state=RANDOM_STATE,
    stratify=y_encoded
)

print("\nTrain size:", X_train.shape)
print("Test size:", X_test.shape)

# ==================================================
# MODEL
# ==================================================

print("\nTraining model...")

model = RandomForestClassifier(
    n_estimators=300,
    max_depth=20,
    min_samples_split=5,
    min_samples_leaf=2,
    class_weight="balanced",
    random_state=RANDOM_STATE,
    n_jobs=-1
)

model.fit(X_train, y_train)

# ==================================================
# PREDICTION
# ==================================================

predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print("\nBehaviour Model Accuracy:")
print(f"{accuracy:.4f}")

# ==================================================
# DETAILED EVALUATION
# ==================================================

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        predictions,
        target_names=encoder.classes_
    )
)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, predictions))

# ==================================================
# SAVE MODEL
# ==================================================

joblib.dump(model, MODEL_PATH)
print(f"\nBehaviour model saved to {MODEL_PATH}")