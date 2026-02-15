# cloud_mitbih_loader.py

import wfdb

def load_record(record_id):
    record = wfdb.rdrecord(record_id, pn_dir="mitdb")
    annotation = wfdb.rdann(record_id, "atr", pn_dir="mitdb")


    ecg = record.p_signal[:, 0].astype("float32")  # MLII
    ann_samples = annotation.sample
    ann_symbols = annotation.symbol

    return ecg, ann_samples, ann_symbols
