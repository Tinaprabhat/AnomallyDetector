"""
Autoencoder detector.

Trained ONLY on licit transactions (learns the normal manifold).
Returns continuous reconstruction error as anomaly score, normalized to [0, 1].

Different mathematical foundation from IF:
  IF asks "are you isolated?", AE asks "do you live on the normal manifold?"

Falls back gracefully if torch isn't installed.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False

from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class AEConfig:
    hidden_dim_1: int = 128
    hidden_dim_2: int = 32
    epochs: int = 50
    batch_size: int = 256
    learning_rate: float = 1e-3
    random_state: int = 42


if _HAS_TORCH:
    class _AEModule(nn.Module):
        def __init__(self, input_dim: int, h1: int, h2: int):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, h1), nn.ReLU(), nn.Linear(h1, h2),
            )
            self.decoder = nn.Sequential(
                nn.Linear(h2, h1), nn.ReLU(), nn.Linear(h1, input_dim),
            )

        def forward(self, x):
            return self.decoder(self.encoder(x))


class AutoencoderDetector:
    """Reconstruction-error based anomaly detector."""

    def __init__(self, config: Optional[AEConfig] = None):
        if not _HAS_TORCH:
            raise ImportError("AutoencoderDetector requires torch. pip install torch")
        self.config = config or AEConfig()
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._err_min: Optional[float] = None
        self._err_max: Optional[float] = None
        self._is_fit = False

    def fit(
        self,
        X_train: np.ndarray,
        y_train: Optional[np.ndarray] = None,
        X_val: Optional[np.ndarray] = None,
    ) -> "AutoencoderDetector":
        torch.manual_seed(self.config.random_state)
        np.random.seed(self.config.random_state)

        if y_train is not None:
            licit_mask = (y_train == 0)
            X_fit = X_train[licit_mask]
            logger.info("autoencoder_training_on_licit_only", n_licit=int(licit_mask.sum()))
        else:
            X_fit = X_train

        self.model = _AEModule(
            input_dim=X_fit.shape[1],
            h1=self.config.hidden_dim_1,
            h2=self.config.hidden_dim_2,
        ).to(self.device)

        opt = optim.Adam(self.model.parameters(), lr=self.config.learning_rate)
        crit = nn.MSELoss()
        X_tensor = torch.tensor(X_fit, dtype=torch.float32).to(self.device)

        self.model.train()
        for epoch in range(self.config.epochs):
            perm = torch.randperm(X_tensor.size(0))
            epoch_loss = 0.0; n_batches = 0
            for i in range(0, X_tensor.size(0), self.config.batch_size):
                idx = perm[i: i + self.config.batch_size]
                batch = X_tensor[idx]
                opt.zero_grad()
                recon = self.model(batch)
                loss = crit(recon, batch)
                loss.backward()
                opt.step()
                epoch_loss += float(loss.item()); n_batches += 1
            if (epoch + 1) % 10 == 0:
                logger.info("autoencoder_epoch", epoch=epoch + 1,
                            loss=epoch_loss / max(n_batches, 1))

        norm_X = X_val if X_val is not None else X_train
        raw = self._compute_errors(norm_X)
        self._err_min = float(raw.min())
        self._err_max = float(raw.max())
        self._is_fit = True
        logger.info("autoencoder_fit_complete",
                    err_min=self._err_min, err_max=self._err_max)
        return self

    def _compute_errors(self, X: np.ndarray) -> np.ndarray:
        self.model.eval()
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            recon = self.model(X_tensor)
            errs = torch.mean((X_tensor - recon) ** 2, dim=1)
        return errs.cpu().numpy()

    def score(self, X: np.ndarray) -> np.ndarray:
        if not self._is_fit or self.model is None:
            raise RuntimeError("AutoencoderDetector not fit. Call .fit() first.")
        raw = self._compute_errors(X)
        rng = max(self._err_max - self._err_min, 1e-9)
        return np.clip((raw - self._err_min) / rng, 0.0, 1.0)

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.score(X) >= threshold).astype(int)
