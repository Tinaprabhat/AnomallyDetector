"""
trainer_gnn.py

Robust GNN trainer for Phase 2: Graph Neural Network anomaly detection.
- Dynamically detects label column
- Aligns features, labels, and graph nodes safely
- Handles unknown labels
- Trains GraphSAGE GNN
- Extracts embeddings and evaluates metrics
"""

import os
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

from Source.gnn_model import build_pyg_data, train_gnn, get_node_embeddings, evaluate
from Source.dataset import EllipticDataset


def run_gnn_phase2(dataset=None, output_dir="Outputs"):
    """
    Runs the full GNN pipeline and returns embeddings, model, and data object.
    """

    try:
        import os
        import numpy as np
        import torch
        import pandas as pd
        from sklearn.preprocessing import StandardScaler
        from Source.gnn_model import build_pyg_data, train_gnn, get_node_embeddings, evaluate
        from Source.dataset import EllipticDataset

        # -------------------------
        # Configuration
        # -------------------------
        DATA_PATH = "Data/elliptic_bitcoin_dataset"
        OUTPUT_DIR = output_dir
        MODEL_PATH = os.path.join(OUTPUT_DIR, "graphsage.pt")
        EMBEDDINGS_PATH = os.path.join(OUTPUT_DIR, "gnn_embeddings.npy")

        HIDDEN_CHANNELS = 128
        EPOCHS = 30
        LR = 1e-3
        VAL_SPLIT_RATIO = 0.2
        DROPOUT = 0.5

        # -------------------------
        # Step 1: Load dataset
        # -------------------------
        print("[INFO] Loading Elliptic dataset...")
        ds = dataset if dataset is not None else EllipticDataset(DATA_PATH)
        G = ds.load()
        features_df = ds.features.copy()
        classes_df = ds.classes.copy()

        # -------------------------
        # Step 2: Detect label column
        # -------------------------
        possible_label_cols = ['class_label', 'class', 'label']
        label_col = next((c for c in possible_label_cols if c in classes_df.columns), None)
        if label_col is None:
            raise ValueError(f"No label column found in classes CSV. Columns: {classes_df.columns}")

        print(f"[INFO] Using '{label_col}' as label column.")
        labels_series = classes_df.set_index("txId")[label_col]

        # Map string labels to integers (licit=0, illicit=1)
        label_map = {"unknown": -1, "licit": 0, "illicit": 1, "1": 0, "2": 1}
        labels_series = labels_series.astype(str).replace(label_map)
        labels_series = labels_series.apply(lambda x: -1 if not str(x).isdigit() else int(x))

        # -------------------------
        # Step 3: Align features and labels safely
        # -------------------------
        # If features has a txId column, use it
        if 0 in features_df.columns:  # assuming first column is txId
            features_df.columns = ['txId'] + [f'feat{i}' for i in range(1, features_df.shape[1])]
        features_df = features_df.set_index('txId')
        # After filtering valid nodes in Phase 2
        valid_nodes = labels_series.index  # These are nodes used in GNN

        # Ensure labels_series is int
        labels_series.index = labels_series.index.astype(str)
        features_df.index = features_df.index.astype(str)

        # Filter unknown labels
        valid_mask = labels_series != -1
        labels_series = labels_series[valid_mask]

        # Take intersection of features, labels, and graph nodes
        common_nodes = labels_series.index.intersection(features_df.index).intersection(G.nodes)
        if len(common_nodes) == 0:
            raise ValueError("No overlapping nodes between features, labels, and graph!")

        features_df = features_df.loc[common_nodes]
        labels_series = labels_series.loc[common_nodes]
        G = G.subgraph(common_nodes).copy()

        print(f"[INFO] Dataset after alignment: {features_df.shape[0]} nodes, {features_df.shape[1]} features")

        # -------------------------
        # Step 4: Feature scaling
        # -------------------------
        print("[INFO] Scaling features...")
        scaler = StandardScaler()
        features_scaled = pd.DataFrame(
            scaler.fit_transform(features_df),
            index=labels_series.index,
            columns=features_df.columns
        )

        # -------------------------
        # Step 5: Build PyG Data
        # -------------------------
        print("[INFO] Converting NetworkX graph to PyTorch Geometric Data...")
        data = build_pyg_data(G, features_scaled, labels_series)
        # After building PyG data
        data.node_ids = labels_series.index  # store aligned node IDs inside data


        # -------------------------
        # Step 6: Train GNN
        # -------------------------
        print("[INFO] Training GraphSAGE model...")
        model, history = train_gnn(
            data,
            epochs=EPOCHS,
            lr=LR,
            hidden_channels=HIDDEN_CHANNELS,
            out_channels=2,
            val_split_ratio=VAL_SPLIT_RATIO,
            save_path=MODEL_PATH
        )

        # -------------------------
        # Step 7: Extract embeddings
        # -------------------------
        print("[INFO] Extracting node embeddings...")
        embeddings = get_node_embeddings(model, data)
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        np.save(EMBEDDINGS_PATH, embeddings)
        print(f"[INFO] Node embeddings saved to {EMBEDDINGS_PATH}")

        # -------------------------
        # Step 8: Evaluation
        # -------------------------
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[INFO] Using device: {device}")

        model = model.to(device)
        data = data.to(device)

        n_nodes = data.x.shape[0]
        val_count = int(n_nodes * VAL_SPLIT_RATIO)
        perm = torch.randperm(n_nodes)
        val_mask = torch.zeros(n_nodes, dtype=torch.bool)
        val_mask[perm[:val_count]] = True

        metrics = evaluate(model, data, mask=val_mask.to(device), device=device)

        print("\n[INFO] Validation Metrics:")
        for k, v in metrics.items():
            print(f"{k}: {v:.4f}")

        print("\n[INFO] Phase 2 (GNN) completed successfully.")

        # ✅ Return results for main.py
        return embeddings, model, data, valid_nodes

    except Exception as e:
        print(f"[ERROR] run_gnn_phase2 failed: {e}")
        raise e

