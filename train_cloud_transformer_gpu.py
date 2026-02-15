import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from cloud_mitbih_loader import load_record
from cloud_windowing import generate_windows
from cloud_transformer import ECGTransformer
from cloud_config import WINDOW_SAMPLES

# ---------------- Dataset ---------------- #

class ECGWindowDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        w, y = self.samples[idx]
        w = torch.from_numpy(w).float().unsqueeze(-1)  # (3600,1)
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
    RECORDS = ["100", "101", "102", "103", "104"]

    all_samples = load_all_data(RECORDS)

    train_samples, val_samples = train_test_split(
        all_samples, test_size=0.2, random_state=42, stratify=[y for _, y in all_samples]
    )

    train_ds = ECGWindowDataset(train_samples)
    val_ds   = ECGWindowDataset(val_samples)

    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, num_workers=2)
    val_loader   = DataLoader(val_ds, batch_size=16, shuffle=False, num_workers=2)

    device = torch.device("cuda")
    print("Using device:", device)

    model = ECGTransformer(seq_len=WINDOW_SAMPLES).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.BCELoss()

    EPOCHS = 10

    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    for epoch in range(EPOCHS):
        # -------- TRAIN -------- #
        model.train()
        total_loss, correct, total = 0, 0, 0

        for X, y in train_loader:
            X, y = X.to(device), y.to(device)

            optimizer.zero_grad()
            preds = model(X).squeeze()
            loss = criterion(preds, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            predicted = (preds >= 0.5).int()
            correct += (predicted == y.int()).sum().item()
            total += y.size(0)

        train_loss = total_loss / len(train_loader)
        train_acc = correct / total

        # -------- VALIDATION -------- #
        model.eval()
        val_loss, val_correct, val_total = 0, 0, 0

        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)
                preds = model(X).squeeze()
                loss = criterion(preds, y)

                val_loss += loss.item()
                predicted = (preds >= 0.5).int()
                val_correct += (predicted == y.int()).sum().item()
                val_total += y.size(0)

        val_loss /= len(val_loader)
        val_acc = val_correct / val_total

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        print(
            f"Epoch {epoch+1}/{EPOCHS} | "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
        )

    # ---------------- Save model ---------------- #
    torch.save(model.state_dict(), "cloud_transformer_mitbih.pth")
    print("✅ Model saved")

    # ---------------- Plot graphs ---------------- #
    epochs = range(1, EPOCHS + 1)

    plt.figure()
    plt.plot(epochs, train_losses, label="Train Loss")
    plt.plot(epochs, val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Cloud Transformer Loss")
    plt.savefig("loss_curve.png")
    plt.close()

    plt.figure()
    plt.plot(epochs, train_accs, label="Train Accuracy")
    plt.plot(epochs, val_accs, label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.title("Cloud Transformer Accuracy")
    plt.savefig("accuracy_curve.png")
    plt.close()

    print("📊 Saved loss_curve.png and accuracy_curve.png")


if __name__ == "__main__":
    train()
