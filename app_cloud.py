# app_cloud.py

from flask import Flask, request, jsonify
import torch
import numpy as np
import os
import time
from flask_cors import CORS

from cloud_config import WINDOW_SAMPLES, RISK_THRESHOLD
from cloud_transformer import ECGTransformer
from cloud_mitbih_loader import load_record
from cloud_windowing import generate_windows
from cloud_features import extract_edge_features

torch.set_num_threads(1)

# ---------------- App ---------------- #

app = Flask(__name__)
CORS(app)

# ---------------- Constants ---------------- #

MAX_WINDOWS = 10
CLOUD_MODEL_ACCURACY = 0.92  # offline evaluated accuracy (reported)

# ---------------- Model loading ---------------- #

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "cloud_transformer_mitbih.pth")

print("Loading cloud Transformer model...")
model = ECGTransformer(seq_len=WINDOW_SAMPLES)
model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
model.eval()
print("Model loaded")

# ---------------- Preload data (for demo + UI) ---------------- #

print("Loading ECG record 100...")
ecg, ann_samples, ann_symbols = load_record("100")

print("Generating windows...")
WINDOWS, _ = generate_windows(ecg, ann_samples, ann_symbols)
WINDOWS = WINDOWS[:MAX_WINDOWS]

EDGE_FEATURES_SAMPLE = extract_edge_features(WINDOWS[0])

print("Cloud server ready")

# ---------------- Routes ---------------- #

@app.route("/", methods=["GET"])
def home():
    return "Cloud MIT-BIH Transformer Server is running."

@app.route("/health", methods=["GET"])
def health():
    return {"status": "ok"}, 200

@app.route("/analyze", methods=["POST"])
def analyze():
    """
    Cached demo-style endpoint:
    - Measures REAL inference time
    - Returns aggregate decision
    """

    api_start = time.perf_counter()
    infer_start = time.perf_counter()

    probs = []
    for w in WINDOWS:
        t = torch.tensor(w, dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
        with torch.no_grad():
            probs.append(model(t).item())

    inference_time_ms = (time.perf_counter() - infer_start) * 1000

    avg_prob = float(np.mean(probs))
    risk = "high" if avg_prob >= RISK_THRESHOLD else "low"

    api_latency_ms = (time.perf_counter() - api_start) * 1000

    response = {
        "status": "success",
        "risk_level": risk,
        "confidence": round(avg_prob, 4),
        "windows_used": len(WINDOWS),
        "edge_features": EDGE_FEATURES_SAMPLE,

        # timing
        "inference_time_ms": round(inference_time_ms, 2),
        "api_latency_ms": round(api_latency_ms - inference_time_ms, 2),
        "total_time_ms": round(api_latency_ms, 2),

        # reported accuracy
        "model_accuracy": CLOUD_MODEL_ACCURACY
    }

    return jsonify(response), 200

# ---------------- Render entrypoint ---------------- #

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
