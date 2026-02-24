# cloud_windowing.py

import numpy as np
from cloud_config import WINDOW_SAMPLES, STEP_SAMPLES

NORMAL_BEATS = {"N"}

def generate_windows(ecg, ann_samples, ann_symbols):
    windows, labels = [], []

    for start in range(0, len(ecg) - WINDOW_SAMPLES, STEP_SAMPLES):
        end = start + WINDOW_SAMPLES
        window = ecg[start:end]

        beats = [
            sym for s, sym in zip(ann_samples, ann_symbols)
            if start <= s < end
        ]

        label = 0
        if any(b not in NORMAL_BEATS for b in beats):
            label = 1

        # ❌ REMOVED per-window normalization
        # ❌ REMOVED multi-window concatenation

        windows.append(window)
        labels.append(label)

    return windows, labels