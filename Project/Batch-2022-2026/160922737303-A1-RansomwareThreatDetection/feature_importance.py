import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# === 1. Load dataset ===
df = pd.read_csv("data_file.csv")

if "Benign" not in df.columns:
    raise ValueError("❌ Target column 'Benign' not found in dataset")

# Convert target to numeric if needed
if df["Benign"].dtype == "object":
    df["Benign"] = df["Benign"].apply(
        lambda x: 1 if str(x).strip().lower() in ["benign", "1", "true"] else 0
    )

# === 2. Keep numeric features only ===
numeric_df = df.select_dtypes(include=["number"])
X = numeric_df.drop(columns=["Benign"], errors="ignore")
y = df["Benign"]

print(f"✅ Using {X.shape[1]} numeric features out of {df.shape[1]} total columns.")

# === 3. Handle missing values ===
X = X.fillna(X.mean())

# === 4. Train Random Forest Model ===
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
model.fit(X_scaled, y)

# === 5. Compute Feature Importances ===
importances = model.feature_importances_
feature_names = X.columns
importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importances
}).sort_values(by="Importance", ascending=False)

print("\n=== Top Features ===\n")
print(importance_df.head(20))

# === 6. Save to CSV ===
importance_df.to_csv("feature_importance.csv", index=False)
print("\n✅ Saved feature importances to feature_importance.csv")

# === 7. Plot ===
plt.figure(figsize=(10, 6))
plt.barh(importance_df["Feature"][:20], importance_df["Importance"][:20], color='teal')
plt.gca().invert_yaxis()
plt.title("Top 20 Feature Importances (Random Forest)", fontsize=14)
plt.xlabel("Importance Score")
plt.tight_layout()
plt.savefig('static/f_imp.jpg')
