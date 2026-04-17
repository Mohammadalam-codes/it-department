from flask import Flask, render_template, request
import joblib, json, numpy as np
from validate import preprocess


app = Flask(__name__)

# Load model + scaler + metrics
model = joblib.load("model_artifacts/model.pkl")
scaler = joblib.load("model_artifacts/scaler.pkl")

with open("model_artifacts/metrics.json", "r") as f:
    metrics = json.load(f)

FEATURES = metrics["features_used"]

@app.route("/")
def home():
    if preprocess()=="valid":
        return render_template("index.html", title="Home")
    else:
        return render_template("base.html")

@app.route("/predict", methods=["GET", "POST"])
def predict():
    if preprocess()=="valid":
        if request.method == "POST":
            try:
                values = [float(request.form[feat]) for feat in FEATURES]
                scaled = scaler.transform([values])
                pred = model.predict(scaled)[0]
                result_text = "✅ Benign File" if pred == 1 else "⚠️ Potential Ransomware Detected"
                return render_template(
                    "predict.html",
                    title="Prediction",
                    features=FEATURES,
                    prediction=True,
                    result_text=result_text
                )
            except Exception as e:
                return f"Error: {e}"
        return render_template("predict.html", title="Prediction", features=FEATURES)
    else:
        return render_template("base.html")

@app.route("/metrics")
def metrics_page():
    if preprocess()=="valid":
        return render_template("metrics.html", title="Model Metrics", metrics=metrics)
    else:
        return render_template("base.html")

if __name__ == "__main__":
    app.run(debug=True)
