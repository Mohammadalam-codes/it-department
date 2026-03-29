import os
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    RocCurveDisplay
)

# ------------------------------
# 1. Load dataset
# ------------------------------
DATA_PATH = "data_file.csv"  # your uploaded dataset
df = pd.read_csv(DATA_PATH)

print("Initial shape:", df.shape)
print("Columns:", df.columns.tolist())

# ------------------------------
# 2. Preprocess data
# ------------------------------
# Drop rows with missing target
df = df.dropna(subset=['Benign'])

# Convert target to numeric
df['Benign'] = df['Benign'].apply(lambda x: 1 if str(x).strip().lower() == 'benign' or str(x).strip() == '1' else 0)

# Keep numeric features only
X = df.select_dtypes(include=[np.number]).drop(columns=['Benign'], errors='ignore')
y = df['Benign']

# Handle missing values
X = X.fillna(X.mean())

print(f"Using {X.shape[1]} numeric features for training")

# ------------------------------
# 3. Split data
# ------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# ------------------------------
# 4. Scale and Train
# ------------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = RandomForestClassifier(
    n_estimators=300, random_state=42, n_jobs=-1
)
model.fit(X_train_scaled, y_train)

# ------------------------------
# 5. Evaluate
# ------------------------------
y_pred = model.predict(X_test_scaled)
y_proba = model.predict_proba(X_test_scaled)[:, 1]

acc = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_proba)
cm = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred, output_dict=True)

print("\n📊 Model Performance:")
print(f"Accuracy: {acc:.4f}")
print(f"ROC AUC: {roc_auc:.4f}")
print("\nConfusion Matrix:")
print(cm)

# ------------------------------
# 6. Save model and scaler
# ------------------------------
os.makedirs("model_artifacts", exist_ok=True)
joblib.dump(model, "model_artifacts/model.pkl")
joblib.dump(scaler, "model_artifacts/scaler.pkl")

# ------------------------------
# 7. Save metrics and plots
# ------------------------------
metrics = {
    "accuracy": acc,
    "roc_auc": roc_auc,
    "confusion_matrix": cm.tolist(),
    "classification_report": report
}
import json
with open("model_artifacts/metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)

# Confusion matrix plot
plt.figure(figsize=(4, 4))
plt.imshow(cm, cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.colorbar()
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, cm[i, j], ha="center", va="center", color="red")
plt.tight_layout()
plt.savefig("model_artifacts/confusion_matrix.png")
plt.close()

# ROC curve
plt.figure(figsize=(5, 4))
RocCurveDisplay.from_estimator(model, X_test_scaled, y_test)
plt.title("ROC Curve")
plt.tight_layout()
plt.savefig("model_artifacts/roc_curve.png")
plt.close()

print("\n✅ Training complete! Files saved in ./model_artifacts")
