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

# ---------------- App ---------------- #

app = Flask(__name__)
CORS(app)

# ---------------- Constants ---------------- #

CLOUD_MODEL_ACCURACY = 0.92

# ---------------- Model loading ---------------- #

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "cloud_transformer_mitbih.pth")

print("Loading cloud Transformer model...")
model = ECGTransformer(seq_len=WINDOW_SAMPLES)
model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu", weights_only=True))
model.eval()
print("Model loaded")

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

        data = request.get_json()
        ecg_window = data.get("ecg_window")

        if ecg_window is None:
            return jsonify({
                "status": "error",
                "message": "ecg_window is required"
            }), 400

        # -------- Prepare Single Window -------- #
        window_np = np.array(ecg_window, dtype=np.float32)
        window_tensor = torch.from_numpy(window_np).unsqueeze(0).unsqueeze(-1)

        # -------- Model Forward -------- #
        infer_start = time.perf_counter()

        with torch.no_grad():
            output = model(window_tensor)
            output = torch.sigmoid(output)

        inference_time_ms = (time.perf_counter() - infer_start) * 1000

        prob = float(output.item())
        risk = "high" if prob >= RISK_THRESHOLD else "low"

        total_time_ms = (time.perf_counter() - api_start) * 1000

        return jsonify({
            "status": "success",
            "risk_level": risk,
            "confidence": round(prob, 4),
            "inference_time_ms": round(inference_time_ms, 2),
            "total_time_ms": round(total_time_ms, 2),
            "model_accuracy": CLOUD_MODEL_ACCURACY
        }), 200

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