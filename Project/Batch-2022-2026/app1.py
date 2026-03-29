import os
import numpy as np
import joblib
from flask import Flask, render_template, request, redirect, url_for, flash
from werkzeug.utils import secure_filename
import pandas as pd
import pefile

app = Flask(__name__)
app.secret_key = "ransomware_detection_secret"
app.config['UPLOAD_FOLDER'] = 'uploads'

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load pre-trained model and scaler
MODEL_PATH = "model.pkl"
SCALER_PATH = "scaler.pkl"

if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
else:
    model = None
    scaler = None


def extract_features(filepath):
    with open(filepath, "rb") as f:
        byte_arr = np.frombuffer(f.read(), dtype=np.uint8)
    size = len(byte_arr)
    entropy = -np.sum((np.bincount(byte_arr, minlength=256) / size) *
                      np.log2((np.bincount(byte_arr, minlength=256) / size) + 1e-10))
    mean_val = np.mean(byte_arr)
    std_val = np.std(byte_arr)
    zeros = np.count_nonzero(byte_arr == 0)
    nonzeros = size - zeros
    zero_ratio = zeros / (size + 1e-10)
    return [size, entropy, mean_val, std_val, zero_ratio, nonzeros]

def extract_pe_features(filepath):

    try:
        pe = pefile.PE(filepath)
        return {
            "FileName": os.path.basename(filepath),
            "DllCharacteristics": getattr(pe.OPTIONAL_HEADER, "DllCharacteristics", 0),
            "DebugSize": getattr(pe.OPTIONAL_HEADER.DATA_DIRECTORY[6], "Size", 0) if len(pe.OPTIONAL_HEADER.DATA_DIRECTORY) > 6 else 0,
            "DebugRVA": getattr(pe.OPTIONAL_HEADER.DATA_DIRECTORY[6], "VirtualAddress", 0) if len(pe.OPTIONAL_HEADER.DATA_DIRECTORY) > 6 else 0,
            "MajorLinkerVersion": getattr(pe.OPTIONAL_HEADER, "MajorLinkerVersion", 0),
            "MajorOSVersion": getattr(pe.OPTIONAL_HEADER, "MajorOperatingSystemVersion", 0),
            "ResourceSize": getattr(pe.OPTIONAL_HEADER.DATA_DIRECTORY[2], "Size", 0) if len(pe.OPTIONAL_HEADER.DATA_DIRECTORY) > 2 else 0
        }
    except Exception as e:
        print(f"⚠️ Error processing {filepath}: {e}")
        return None


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/metrics')
def metrics():
    return render_template('metrics.html')


@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        flash('No file uploaded!')
        return redirect(url_for('index'))

    file = request.files['file']
    if file.filename == '':
        flash('No selected file!')
        return redirect(url_for('index'))

    filepath = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(file.filename))
    file.save(filepath)

    #features = np.array(extract_features(filepath)).reshape(1, -1)
    features =np.array(extract_pe_features(filepath)).reshape(1,-1)
    features_scaled = scaler.transform(features)
    pred = model.predict(features_scaled)[0]
    label = "Ransomware" if pred == 1 else "Benign"

    return render_template('result.html', filename=file.filename, result=label)


if __name__ == '__main__':
    app.run(debug=True)
