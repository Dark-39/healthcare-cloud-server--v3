import torch
import torch.nn as nn


class ECGTransformer(nn.Module):
    def __init__(self, seq_len, d_model=128, nhead=8, layers=3, dropout=0.2):
        super().__init__()

        # -------- CNN Downsampling -------- #
        self.conv = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(32),
            nn.GELU(),

            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(64),
            nn.GELU(),

            nn.Conv1d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(128),
            nn.GELU(),

            nn.Conv1d(128, d_model, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(d_model),
            nn.GELU()
        )

        # -------- Dynamic Positional Encoding -------- #
        # After 3 stride-2 layers:
        # 3600  → 450
        # 10800 → 1350
        # So we allocate safely above max expected length

        self.max_seq_len = 2000  # Safe upper bound
        self.pos_encoding = nn.Parameter(
            torch.randn(1, self.max_seq_len, d_model)
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            batch_first=True,
            activation="gelu"
        )

        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=layers)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.norm = nn.LayerNorm(d_model)
        self.classifier = nn.Linear(d_model, 1)

    def forward(self, x):
        # x: (batch, seq_len, 1)

        x = x.transpose(1, 2)   # (batch, 1, seq_len)
        x = self.conv(x)        # (batch, d_model, new_seq)
        x = x.transpose(1, 2)   # (batch, new_seq, d_model)

        b = x.size(0)

        # Add CLS token
        cls = self.cls_token.expand(b, 1, x.size(-1))
        x = torch.cat((cls, x), dim=1)

        # Apply positional encoding dynamically
        seq_len = x.size(1)

        if seq_len > self.max_seq_len:
            raise ValueError(
                f"Sequence length {seq_len} exceeds max_seq_len {self.max_seq_len}"
            )

        x = x + self.pos_encoding[:, :seq_len]

        x = self.encoder(x)

        cls_out = x[:, 0, :]
        cls_out = self.norm(cls_out)

        logits = self.classifier(cls_out)

        return logits.view(-1)