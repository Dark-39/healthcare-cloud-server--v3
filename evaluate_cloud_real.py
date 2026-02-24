# evaluate_cloud_real.py

import time
import numpy as np
import torch

from cloud_transformer import ECGTransformer
from cloud_mitbih_loader import load_record
from cloud_windowing import generate_windows
from cloud_config import WINDOW_SAMPLES, RISK_THRESHOLD

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    roc_curve,
    auc
)

# ---------------- Config ---------------- #

RECORDS = ["106"]   # Evaluate unseen patient
MODEL_PATH = "cloud_transformer_best.pth"

# ---------------- Load model ---------------- #

print("\nLoading CLOUD Transformer model...")
model = ECGTransformer(seq_len=WINDOW_SAMPLES)
model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
model.eval()
print("Model loaded")

# ---------------- Evaluation ---------------- #

y_true = []
y_pred = []
y_probs = []

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
        y_probs.append(prob)

# ---------------- Standard Metrics ---------------- #

print("\n===== CLOUD TRANSFORMER PERFORMANCE =====\n")
print(classification_report(y_true, y_pred, digits=4))
print("Confusion Matrix:")
print(confusion_matrix(y_true, y_pred))

print(f"\nAverage inference latency: {np.mean(latencies):.2f} ms")
print(f"Median inference latency : {np.median(latencies):.2f} ms")

# ---------------- ROC ---------------- #

fpr, tpr, thresholds = roc_curve(y_true, y_probs)
roc_auc = auc(fpr, tpr)

print(f"\nAUC Score: {roc_auc:.4f}")

# ---------------- Threshold Sweep ---------------- #

print("\n===== THRESHOLD SWEEP =====\n")

threshold_range = np.arange(0.05, 0.45, 0.02)

best_f1 = 0
best_threshold = 0

for t in threshold_range:
    preds = [1 if p >= t else 0 for p in y_probs]

    precision = precision_score(y_true, preds)
    recall = recall_score(y_true, preds)
    f1 = f1_score(y_true, preds)

    print(
        f"Threshold {t:.2f} | "
        f"Precision: {precision:.3f} | "
        f"Recall: {recall:.3f} | "
        f"F1: {f1:.3f}"
    )

    if f1 > best_f1:
        best_f1 = f1
        best_threshold = t

print("\nBest Threshold based on F1:", round(best_threshold, 3))
print("Best F1:", round(best_f1, 4))