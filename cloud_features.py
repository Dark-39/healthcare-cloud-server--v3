# cloud_features.py

import numpy as np

def extract_edge_features(ecg_window):
    return {
        "mean": float(np.mean(ecg_window)),
        "std": float(np.std(ecg_window)),
        "energy": float(np.sum(ecg_window ** 2)),
        "min": float(np.min(ecg_window)),
        "max": float(np.max(ecg_window)),
        "zero_crossings": int(np.sum(np.diff(np.sign(ecg_window)) != 0))
    }
