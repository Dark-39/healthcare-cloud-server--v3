# train_cloud_transformer_gpu.py

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

from cloud_mitbih_loader import load_record
from cloud_windowing import generate_windows
from cloud_transformer import ECGTransformer
from cloud_config import WINDOW_SAMPLES

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EPOCHS = 50
BATCH_SIZE = 32
LR = 6e-5
WEIGHT_DECAY = 2e-4
EARLY_STOPPING_PATIENCE = 10

# ---------------- Dataset ---------------- #

class ECGWindowDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        w, y = self.samples[idx]
        w = torch.from_numpy(w).float().unsqueeze(-1)
        y = torch.tensor(y, dtype=torch.float32)
        return w, y

# ---------------- Load data ---------------- #

def load_all_data(record_ids):
    samples = []
    for rid in record_ids:
        print(f"Loading record {rid}")
        ecg, ann_s, ann_sym = load_record(rid)
        windows, labels = generate_windows(ecg, ann_s, ann_sym)

        for w, l in zip(windows, labels):
            samples.append((w, l))

    print(f"Total windows collected: {len(samples)}")
    return samples

# ---------------- Training ---------------- #

def train():

    TRAIN_RECORDS = ["100","101","102","103","104","105"]
    VAL_RECORDS   = ["106"]

    train_samples = load_all_data(TRAIN_RECORDS)
    val_samples   = load_all_data(VAL_RECORDS)

    # ---- Compute class weights from training set ----
    labels_array = np.array([y for _, y in train_samples])
    num_pos = labels_array.sum()
    num_neg = len(labels_array) - num_pos
    pos_weight = num_neg / num_pos
    print("Cloud pos_weight:", round(pos_weight, 4))

    train_ds = ECGWindowDataset(train_samples)
    val_ds   = ECGWindowDataset(val_samples)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    model = ECGTransformer(
        seq_len=WINDOW_SAMPLES,
        d_model=128,
        nhead=8,
        layers=3,
        dropout=0.2
    ).to(DEVICE)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=4
    )

    loss_fn = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pos_weight]).to(DEVICE)
    )

    best_val_loss = float("inf")
    early_stop_counter = 0

    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    for epoch in range(EPOCHS):

        # -------- TRAIN -------- #
        model.train()
        total_loss, correct, total = 0, 0, 0

        for X, y in train_loader:
            X, y = X.to(DEVICE), y.to(DEVICE)

            optimizer.zero_grad()
            logits = model(X)
            loss = loss_fn(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            total_loss += loss.item()

            probs = torch.sigmoid(logits)
            preds = (probs >= 0.5).float()
            correct += (preds == y).sum().item()
            total += y.size(0)

        train_loss = total_loss / len(train_loader)
        train_acc = correct / total

        # -------- VALIDATION -------- #
        model.eval()
        val_loss, val_correct, val_total = 0, 0, 0
        y_true, y_probs = [], []

        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(DEVICE), y.to(DEVICE)

                logits = model(X)
                loss = loss_fn(logits, y)

                val_loss += loss.item()

                probs = torch.sigmoid(logits)
                preds = (probs >= 0.5).float()

                val_correct += (preds == y).sum().item()
                val_total += y.size(0)

                y_true.extend(y.cpu().numpy())
                y_probs.extend(probs.cpu().numpy())

        val_loss /= len(val_loader)
        val_acc = val_correct / val_total

        scheduler.step(val_loss)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        print(
            f"Epoch {epoch+1}/{EPOCHS} | "
            f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}"
        )

        # ---- Early Stopping ----
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "cloud_transformer_best.pth")
            early_stop_counter = 0
        else:
            early_stop_counter += 1
            if early_stop_counter >= EARLY_STOPPING_PATIENCE:
                print("Early stopping triggered.")
                break

    print("✅ Training Complete")

    # -------- Plot Loss -------- #
    plt.figure()
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Cloud Transformer Loss Curve")
    plt.savefig("cloud_loss_curve.png")
    plt.close()

    # -------- Plot Accuracy -------- #
    plt.figure()
    plt.plot(train_accs, label="Train Accuracy")
    plt.plot(val_accs, label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.title("Cloud Transformer Accuracy Curve")
    plt.savefig("cloud_accuracy_curve.png")
    plt.close()

    print("📊 Loss & Accuracy graphs saved.")

    # -------- ROC -------- #
    fpr, tpr, thresholds = roc_curve(y_true, y_probs)
    roc_auc = auc(fpr, tpr)

    print(f"\nCloud AUC (Validation Patient): {roc_auc:.4f}")

    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    print(f"Optimal threshold (Validation Patient): {optimal_threshold:.4f}")

    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
    plt.plot([0,1],[0,1],'--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.title("Cloud Transformer ROC Curve")
    plt.savefig("cloud_roc_curve.png")
    plt.close()

    print("📊 ROC curve saved.")

if __name__ == "__main__":
    train()