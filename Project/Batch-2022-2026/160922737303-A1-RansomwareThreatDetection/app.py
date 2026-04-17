import os
import joblib
import pandas as pd
from flask import Flask, render_template, request, redirect, url_for, flash

app = Flask(__name__)
app.secret_key = "ransomware_secret_key"

UPLOAD_FOLDER = "static/uploads"
EDA_FOLDER = "static/eda"
PERFORMANCE_FOLDER = "static/performance"
MODEL_FOLDER = "static/final_model"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load trained model & scaler
model = joblib.load("model_artifacts/model.pkl")
scaler = joblib.load("model_artifacts/scaler.pkl")

TOP_FEATURES = [
    "DllCharacteristics",
    "DebugSize",
    "DebugRVA",
    "MajorLinkerVersion",
    "MajorOSVersion",
    "ResourceSize"
]

# ==========================
# ROUTES
# ==========================

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/upload_dataset", methods=["GET", "POST"])
def upload_dataset():
    if request.method == "POST":
        file = request.files["dataset"]
        if file:
            filepath = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(filepath)
            flash("Dataset uploaded successfully!", "success")
            return redirect(url_for("upload_dataset"))
    return render_template("upload_dataset.html")

@app.route("/eda")
def eda():
    images = os.listdir(EDA_FOLDER)
    return render_template("eda.html", images=images)

@app.route("/performance")
def performance():
    files = os.listdir(PERFORMANCE_FOLDER)

    models_data = {}

    for file in files:
        # Extract model name (before first underscore)
        model_name = file.split("_")[0]

        if model_name not in models_data:
            models_data[model_name] = {
                "confusion_matrix": None,
                "classification_report": None,
                "other_files": []
            }

        if "confusion_matrix" in file:
            models_data[model_name]["confusion_matrix"] = file

        elif "classification_report" in file:
            # Read classification report content
            with open(os.path.join(PERFORMANCE_FOLDER, file), "r") as f:
                report_text = f.read()
            models_data[model_name]["classification_report"] = report_text

        else:
            models_data[model_name]["other_files"].append(file)

    return render_template("performance.html", models=models_data)

@app.route("/model_details")
def model_details():
    files = os.listdir(MODEL_FOLDER)

    artifacts = {
        "images": [],
        "reports": [],
        "others": []
    }

    for file in files:
        if file.endswith((".png", ".jpg", ".jpeg")):
            artifacts["images"].append(file)

        elif file.endswith((".txt", ".csv")):
            with open(os.path.join(MODEL_FOLDER, file), "r") as f:
                content = f.read()
            artifacts["reports"].append({
                "name": file,
                "content": content
            })

        else:
            artifacts["others"].append(file)

    return render_template("model_details.html", artifacts=artifacts)

@app.route("/predict", methods=["GET", "POST"])
def predict():
    prediction = None
    confidence = None

    if request.method == "POST":
        values = [float(request.form[f]) for f in TOP_FEATURES]
        scaled = scaler.transform([values])

        proba = model.predict_proba(scaled)[0]
        result = model.predict(scaled)[0]

        confidence = round(max(proba) * 100, 2)
        prediction = "Benign File" if result == 1 else "Potential Ransomware"

    return render_template("predict.html",
                           features=TOP_FEATURES,
                           prediction=prediction,
                           confidence=confidence)

if __name__ == "__main__":
    app.run(debug=True)