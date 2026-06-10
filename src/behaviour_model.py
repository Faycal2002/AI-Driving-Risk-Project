import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

print("Loading dataset...")

df = pd.read_csv("datasets/dataset_7M.csv")

print("Dataset loaded.")

# Pour aller vite au début
df = df.sample(
    n=100000,
    random_state=42
)

X = df[
    [
        "AccX",
        "AccY",
        "AccZ",
        "GyroX",
        "GyroY",
        "GyroZ"
    ]
]

y = df["Class"]

encoder = LabelEncoder()

y = encoder.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("Training model...")

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

predictions = model.predict(X_test)

accuracy = accuracy_score(
    y_test,
    predictions
)

print("\nBehaviour Model Accuracy:")
print(accuracy)


joblib.dump(
    model,
    "models/behaviour_model.pkl"
)

print("Behaviour model saved!")