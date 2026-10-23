

import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_PATH = os.path.join(
    BASE_DIR,
    "dataset",
    "heart.csv"
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "backend",
    "model"
)

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "heart_model.pkl"
)

SCALER_PATH = os.path.join(
    MODEL_DIR,
    "scaler.pkl"
)


# Create model directory
os.makedirs(MODEL_DIR, exist_ok=True)


# Load dataset
df = pd.read_csv(DATA_PATH)

# Remove unnamed columns
df = df.loc[:, ~df.columns.str.contains("^Unnamed")]

print("Dataset shape:", df.shape)
print("Columns:", list(df.columns))


# Target column
TARGET = "target"

if TARGET not in df.columns:
    raise ValueError(
        f"Target column '{TARGET}' not found. "
        f"Available columns: {list(df.columns)}"
    )


# Separate features and target
X = df.drop(TARGET, axis=1)
y = df[TARGET]


# Convert categorical columns into numerical columns
X = pd.get_dummies(X, drop_first=False)


# Save feature columns
feature_columns = X.columns.tolist()

feature_columns_path = os.path.join(
    MODEL_DIR,
    "feature_columns.pkl"
)

joblib.dump(feature_columns, feature_columns_path)


# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# Scaling
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# Model
model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

model.fit(X_train_scaled, y_train)


# Prediction
y_pred = model.predict(X_test_scaled)


# Evaluation
accuracy = accuracy_score(y_test, y_pred)

print("\n==============================")
print("MODEL PERFORMANCE")
print("==============================")

print("Accuracy:", round(accuracy * 100, 2), "%")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))


# Save model
joblib.dump(model, MODEL_PATH)

# Save scaler
joblib.dump(scaler, SCALER_PATH)


print("\nModel saved:", MODEL_PATH)
print("Scaler saved:", SCALER_PATH)
print("Feature columns saved:", feature_columns_path)