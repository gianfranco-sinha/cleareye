"""PyTorch model architectures for ClearEye.

Three model types: classification (regime), regression (residual), autoencoder (anomaly).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class RegimeClassifierNet(nn.Module):
    """MLP classifier for turbidity regime prediction.

    Input: [turbidity_adc, tds, water_temperature]
    Output: 3-class softmax (solution, colloid, suspension)
    """

    def __init__(self, input_dim: int = 3, hidden_dim: int = 32) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ResidualCorrectionNet(nn.Module):
    """MLP for learning the residual between datasheet NTU and reference NTU.

    One model per regime. Predicts the additive correction to apply
    to the datasheet transfer function output.

    Input: [voltage, water_temperature, tds, d_adc_dt, hour_sin, hour_cos]
    Output: scalar residual (NTU correction)
    """

    def __init__(self, input_dim: int = 6, hidden_dim: int = 32) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class AnomalyAutoencoder(nn.Module):
    """Autoencoder for anomaly/drift detection.

    Trained on "normal" readings. High reconstruction error at inference
    signals sensor drift or biofouling. Used by BiofoulingMonitor (milestone 3).

    Input/Output: same dimensionality (reconstruction)
    Anomaly signal: MSE between input and reconstruction
    """

    def __init__(
        self, input_dim: int = 3, hidden_dim: int = 16, encoding_dim: int = 4
    ) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, encoding_dim),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(x)
        return self.decoder(encoded)

    def reconstruction_error(self, x: torch.Tensor) -> torch.Tensor:
        """Per-sample MSE reconstruction error."""
        reconstructed = self.forward(x)
        return ((x - reconstructed) ** 2).mean(dim=-1)
