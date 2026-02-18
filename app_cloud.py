# app_cloud.py

from flask import Flask, request, jsonify
import torch
torch.set_num_threads(1)

import numpy as np
import os
import time
from flask_cors import CORS

from cloud_config import WINDOW_SAMPLES, RISK_THRESHOLD
from cloud_transformer import ECGTransformer
from cloud_mitbih_loader import load_record
from cloud_windowing import generate_windows
from cloud_features import extract_edge_features

# ---------------- App ---------------- #

app = Flask(__name__)
CORS(app)

# ---------------- Constants ---------------- #

MAX_WINDOWS = 10
CLOUD_MODEL_ACCURACY = 0.92

# ---------------- Model loading ---------------- #

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "cloud_transformer_mitbih.pth")

print("Loading cloud Transformer model...")
model = ECGTransformer(seq_len=WINDOW_SAMPLES)
model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
model.eval()
print("Model loaded")

# ---------------- Preload data ---------------- #

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
    try:
        api_start = time.perf_counter()
        infer_start = time.perf_counter()

        # -------- Safe Batch Creation -------- #
        batch_np = np.array(WINDOWS, dtype=np.float32)
        batch = torch.from_numpy(batch_np).unsqueeze(-1)

        # -------- Model Forward -------- #
        with torch.no_grad():
            outputs = model(batch)

        probs = outputs.squeeze().tolist()

        if isinstance(probs, float):
            probs = [probs]

        inference_time_ms = (time.perf_counter() - infer_start) * 1000

        avg_prob = float(np.mean(probs))
        risk = "high" if avg_prob >= RISK_THRESHOLD else "low"

        total_time_ms = (time.perf_counter() - api_start) * 1000

        response = {
            "status": "success",
            "risk_level": risk,
            "confidence": round(avg_prob, 4),
            "windows_used": len(WINDOWS),
            "edge_features": EDGE_FEATURES_SAMPLE,
            "inference_time_ms": round(inference_time_ms, 2),
            "total_time_ms": round(total_time_ms, 2),
            "model_accuracy": CLOUD_MODEL_ACCURACY
        }

        return jsonify(response), 200

    except Exception as e:
        print("ERROR INSIDE /analyze:", str(e))
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500


# ---------------- Render Entry ---------------- #

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
