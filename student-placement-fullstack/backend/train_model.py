"""
train_model.py
----------------
Trains a Logistic Regression model that predicts student placement
and saves it (with its scaler) as placement_model.pkl for the Flask
backend (app.py) to load and use.

Run this file first, before starting the Flask server:
    python train_model.py
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib


# ---------- 1. Load the dataset ----------
print("Loading dataset...")
data = pd.read_csv("student_placement.csv")
print("Shape:", data.shape)
print(data.head())

# ---------- 2. Clean the data ----------
print("\nMissing values per column:")
print(data.isnull().sum())
data = data.dropna()

# ---------- 3. Separate features (X) and target (y) ----------
feature_columns = [
    "CGPA",
    "10th_Percentage",
    "12th_Percentage",
    "Technical_Score",
    "Aptitude_Score",
    "Communication_Score",
    "Internship",
]
X = data[feature_columns]
y = data["Placement"]

# ---------- 4. Split into training and testing data ----------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"\nTraining samples: {len(X_train)} | Testing samples: {len(X_test)}")

# ---------- 5. Scale the features ----------
# Logistic Regression performs better when features share a similar scale.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ---------- 6. Train the Logistic Regression model ----------
print("\nTraining Logistic Regression model...")
model = LogisticRegression()
model.fit(X_train_scaled, y_train)

# ---------- 7. Evaluate the model ----------
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy: {accuracy * 100:.2f}%")
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# ---------- 8. Save the model + scaler together ----------
# app.py needs both the model and the exact scaler used during training,
# so we bundle them into a single dictionary and save it with joblib.
model_bundle = {
    "model": model,
    "scaler": scaler,
    "feature_columns": feature_columns,
}
joblib.dump(model_bundle, "placement_model.pkl")
print("\nSaved trained model to 'placement_model.pkl'")
