# evaluate_cloud_real.py
import time
import numpy as np
import torch

from cloud_transformer import ECGTransformer
from cloud_mitbih_loader import load_record
from cloud_windowing import generate_windows
from cloud_config import WINDOW_SAMPLES, RISK_THRESHOLD

from sklearn.metrics import classification_report, confusion_matrix

# ---------------- Config ---------------- #

RECORDS = ["100", "101", "102", "103", "104"]
MODEL_PATH = "cloud_transformer_mitbih.pth"

# ---------------- Load model ---------------- #

print("\nLoading CLOUD Transformer model...")
model = ECGTransformer(seq_len=WINDOW_SAMPLES)
model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
model.eval()
print("Model loaded")

# ---------------- Evaluation ---------------- #

y_true = []
y_pred = []

low_samples = []
mid_samples = []
high_samples = []

latencies = []

print("\nEvaluating CLOUD Transformer on REAL MIT-BIH data\n")

for rid in RECORDS:
    print(f"Processing record {rid}")

    ecg, ann_samples, ann_symbols = load_record(rid)
    windows, labels = generate_windows(ecg, ann_samples, ann_symbols)

    for w, label in zip(windows, labels):

        x = torch.tensor(w, dtype=torch.float32).unsqueeze(0).unsqueeze(-1)

        start = time.time()
        with torch.no_grad():
            logit = model(x).item()
            prob = torch.sigmoid(torch.tensor(logit)).item()

        latency_ms = (time.time() - start) * 1000
        latencies.append(latency_ms)

        pred = 1 if prob >= RISK_THRESHOLD else 0

        y_true.append(label)
        y_pred.append(pred)

        # Representative samples
        if label == 0 and len(low_samples) < 1:
            low_samples.append((label, prob))
        elif label == 1 and prob < RISK_THRESHOLD and len(mid_samples) < 1:
            mid_samples.append((label, prob))
        elif label == 1 and prob >= RISK_THRESHOLD and len(high_samples) < 1:
            high_samples.append((label, prob))

# ---------------- Metrics ---------------- #

print("\n===== CLOUD TRANSFORMER PERFORMANCE (REAL DATA) =====\n")
print(classification_report(y_true, y_pred, digits=4))
print("Confusion Matrix:")
print(confusion_matrix(y_true, y_pred))

print(f"\nAverage inference latency: {np.mean(latencies):.2f} ms")
print(f"Median inference latency : {np.median(latencies):.2f} ms")

# ---------------- Representative cases ---------------- #

def show_case(title, case):
    label, prob = case
    print(f"\n{title}")
    print(f"Ground Truth : {'abnormal' if label else 'normal'}")
    print(f"Predicted    : {'abnormal' if prob >= RISK_THRESHOLD else 'normal'}")
    print(f"Confidence   : {prob:.4f}")

print("\n===== REPRESENTATIVE CASES =====")

if low_samples:
    show_case("LOW RISK CASE", low_samples[0])

if mid_samples:
    show_case("MID RISK CASE (borderline)", mid_samples[0])

if high_samples:
    show_case("HIGH RISK CASE", high_samples[0])
