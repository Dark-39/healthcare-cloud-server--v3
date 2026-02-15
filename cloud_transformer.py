# cloud_transformer.py

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # (1, max_len, d_model)

        self.register_buffer("pe", pe)

    def forward(self, x):
        # x shape: (batch, seq, d_model)
        x = x + self.pe[:, : x.size(1)]
        return x


class ECGTransformer(nn.Module):
    def __init__(self, seq_len, d_model=64, nhead=4, layers=2, dropout=0.1):
        super().__init__()

        self.seq_len = seq_len
        self.d_model = d_model

        # linear embedding for 1-D ECG samples
        self.embed = nn.Linear(1, d_model)

        # positional encoding
        self.pos_encoding = PositionalEncoding(d_model=d_model, max_len=seq_len)

        # transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            batch_first=True,
            activation="gelu"
        )

        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=layers)

        # CLS token for classification
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))

        self.norm = nn.LayerNorm(d_model)

        # output head (raw logits, NO sigmoid here)
        self.classifier = nn.Linear(d_model, 1)

    def forward(self, x):
        # x : (batch, seq_len, 1)

        b = x.size(0)

        # linear projection
        x = self.embed(x)  # (b, seq, d_model)

        # positional encoding
        x = self.pos_encoding(x)

        # prepend CLS token
        cls = self.cls_token.expand(b, 1, self.d_model)
        x = torch.cat((cls, x), dim=1)  # (b, seq+1, d_model)

        # transformer encoder
        x = self.encoder(x)

        # take CLS output
        cls_out = x[:, 0, :]  # (b, d_model)

        cls_out = self.norm(cls_out)

        logits = self.classifier(cls_out)  # (b,1)

        return logits.view(-1)  # return raw logits
