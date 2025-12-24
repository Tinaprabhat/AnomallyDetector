import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.decomposition import PCA
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler


# ==============================================
# --- Utility: Dimensionality Reduction (PCA) ---
# ==============================================
def apply_pca(X, n_components=30):
    print(f"\n[INFO] Applying PCA: reducing from {X.shape[1]} to {n_components} features")
    pca = PCA(n_components=n_components, random_state=42)
    X_reduced = pca.fit_transform(X)
    print("[INFO] PCA complete.")
    return X_reduced


# =====================================================
# --- Utility: Sampling Large Datasets for Speed ---
# =====================================================
def sample_data(X, sample_size=30000):
    if len(X) > sample_size:
        print(f"[INFO] Sampling {sample_size}/{len(X)} samples for heavy models.")
        idx = np.random.choice(len(X), size=sample_size, replace=False)
        return X[idx]
    return X


# =====================================================
# --- Baseline 1: Isolation Forest ---
# =====================================================
def run_isolation_forest(X):
    print("\n[MODEL] Running Isolation Forest...")
    model = IsolationForest(contamination=0.05, n_jobs=-1, random_state=42)
    preds = model.fit_predict(X)
    return preds


# =====================================================
# --- Baseline 2: Local Outlier Factor ---
# =====================================================
def run_lof(X):
    print("\n[MODEL] Running Local Outlier Factor...")
    model = LocalOutlierFactor(n_neighbors=20, contamination=0.05, n_jobs=-1)
    preds = model.fit_predict(X)
    return preds


# =====================================================
# --- Baseline 3: One-Class SVM ---
# =====================================================
def run_oneclass_svm(X):
    print("\n[MODEL] Running One-Class SVM (on sample)...")
    X_small = sample_data(X, 10000)
    model = OneClassSVM(kernel="rbf", gamma="scale", nu=0.05)
    preds = model.fit_predict(X_small)
    return preds


# =====================================================
# --- Baseline 4: Autoencoder (GPU accelerated) ---
# =====================================================
class AutoEncoder(nn.Module):
    def __init__(self, input_dim):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 32)
        )
        self.decoder = nn.Sequential(
            nn.Linear(32, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


def run_autoencoder(X, epochs=10, batch_size=256):
    print("\n[MODEL] Running Autoencoder (PyTorch, GPU supported)...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[INFO] Using device:", device)

    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
    model = AutoEncoder(X.shape[1]).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    model.train()
    for epoch in range(epochs):
        perm = torch.randperm(X_tensor.size(0))
        epoch_loss = 0
        for i in range(0, X_tensor.size(0), batch_size):
            idx = perm[i:i + batch_size]
            batch = X_tensor[idx]

            optimizer.zero_grad()
            output = model(batch)
            loss = criterion(output, batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss/len(X_tensor):.6f}")

    # Reconstruction Error
    model.eval()
    with torch.no_grad():
        reconstructed = model(X_tensor)
        loss = torch.mean((X_tensor - reconstructed) ** 2, dim=1)
    loss = loss.cpu().numpy()

    # Threshold (95th percentile)
    threshold = np.percentile(loss, 95)
    preds = np.where(loss > threshold, -1, 1)

    print("[INFO] Autoencoder anomaly detection complete.")
    return preds


# =====================================================
# --- Master Function: Run All Baselines ---
# =====================================================
def run_baselines(X_scaled, use_autoencoder=True):
    results = {}

    # Step 1: PCA for all models
    X_reduced = apply_pca(X_scaled, n_components=30)

    # Step 2: Run Isolation Forest
    results["IsolationForest"] = run_isolation_forest(X_reduced)

    # Step 3: Run LOF
    X_small_lof = sample_data(X_reduced, 20000)
    results["LOF"] = run_lof(X_small_lof)

    # Step 4: Run One-Class SVM
    results["OneClassSVM"] = run_oneclass_svm(X_reduced)

    # Step 5: Run Autoencoder (Optional, GPU)
    if use_autoencoder:
        results["AutoEncoder"] = run_autoencoder(X_reduced)

    print("\n[INFO] All baseline models completed successfully.")
    return results
