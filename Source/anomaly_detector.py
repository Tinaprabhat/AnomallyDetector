# anomaly_detector.py

"""
Anomaly Detector for Elliptic Bitcoin Dataset
---------------------------------------------
Phase 3: Detect anomalies using GNN embeddings.
- Uses Isolation Forest
- Maps labels from dataset: 1 -> 0 (licit), 2 -> 1 (illicit)
- Handles alignment of embeddings and labels
- Avoids errors for single-class labels
"""

import os
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, roc_auc_score

def run_anomaly_detection(embeddings, labels_df=None, node_ids=None, save_path="outputs", contamination=0.05):
    """
    Run anomaly detection on GNN embeddings for Elliptic dataset.

    Args:
        embeddings (np.ndarray): Node embeddings from GNN (shape: num_nodes x embedding_dim)
        labels_df (pd.DataFrame, optional): Dataset's classes DataFrame with columns ["txId", "label"]
        node_ids (list or pd.Index, optional): Node IDs corresponding to embeddings (Phase 2 alignment)
        save_path (str): Directory to save predictions
        contamination (float): Estimated proportion of anomalies for Isolation Forest

    Returns:
        np.ndarray: Binary anomaly predictions (0=normal/licit, 1=anomaly/illicit)
    """
    os.makedirs(save_path, exist_ok=True)

    # -------------------------
    # Align labels to embeddings
    # -------------------------
    labels_aligned = None
    if labels_df is not None and node_ids is not None:
        # Set txId as index
        labels_series = labels_df.set_index("txId")["label"]

        # Map dataset-specific labels: 1 -> 0 (licit), 2 -> 1 (illicit)
        label_map = {"1": 0, "2": 1, 1:0, 2:1}
        labels_series = labels_series.map(label_map)

        # Keep only nodes present in embeddings
        labels_aligned = labels_series.reindex(node_ids)  # align safely, keep NaN if missing
        missing_count = labels_series.reindex(node_ids).isna().sum()
        if missing_count > 0:
            print(f"[Warning] {missing_count} node_ids had missing labels — filled as -1.")
        labels_aligned = labels_aligned.fillna(-1).astype(int).values  # mark missing labels as -1

    # -------------------------
    # Fit Isolation Forest
    # -------------------------
    iso = IsolationForest(n_estimators=100, contamination=contamination, random_state=42)
    iso.fit(embeddings)
    iso_preds = iso.predict(embeddings)  # {-1=anomaly, 1=normal}

    # Convert to 0=normal/licit, 1=anomaly/illicit
    iso_preds = np.where(iso_preds == 1, 0, 1)

    # -------------------------
    # Evaluation (if labels available)
    # -------------------------
    if labels_aligned is not None:
        if len(labels_aligned) != len(iso_preds):
            min_len = min(len(labels_aligned), len(iso_preds))
            labels_aligned = labels_aligned[:min_len]
            iso_preds = iso_preds[:min_len]

        unique_labels = np.unique(labels_aligned)
        if len(unique_labels) < 2:
            print("[WARNING] Only one class present. Some metrics (ROC AUC) will be skipped.")
            accuracy = np.mean(labels_aligned == iso_preds)
            print(f"[INFO] Accuracy: {accuracy:.4f}")
        else:
            print("[INFO] Classification Report:")
            print(classification_report(labels_aligned, iso_preds, digits=4))
            try:
                auc = roc_auc_score(labels_aligned, iso_preds)
                print(f"[INFO] ROC AUC: {auc:.4f}")
            except ValueError as e:
                print(f"[WARNING] ROC AUC could not be computed: {e}")

    # -------------------------
    # Save predictions
    # -------------------------
    save_file = os.path.join(save_path, "anomaly_preds.npy")
    np.save(save_file, iso_preds)
    print(f"[INFO] Anomaly predictions saved to {save_file}")

    return iso_preds
