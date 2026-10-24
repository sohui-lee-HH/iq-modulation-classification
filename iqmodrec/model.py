"""1D CNN for modulation classification on raw IQ samples."""

import torch.nn as nn


class IQCNN(nn.Module):
    """Small 1D CNN: 4 conv blocks + global average pooling.

    Input  (B, 2, L)  -- I and Q as two channels
    Output (B, n_classes) logits
    Global average pooling makes the model independent of the input length L.
    """

    def __init__(self, n_classes: int, width: int = 64, dropout: float = 0.3):
        super().__init__()

        def block(c_in, c_out, k):
            return nn.Sequential(
                nn.Conv1d(c_in, c_out, k, padding=k // 2),
                nn.BatchNorm1d(c_out),
                nn.ReLU(),
                nn.Conv1d(c_out, c_out, k, padding=k // 2),
                nn.BatchNorm1d(c_out),
                nn.ReLU(),
                nn.MaxPool1d(2),
            )

        self.features = nn.Sequential(
            block(2, width, 7),
            block(width, width, 5),
            block(width, 2 * width, 3),
            block(2 * width, 2 * width, 3),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(2 * width, n_classes),
        )

    def forward(self, x):
        return self.head(self.features(x))
