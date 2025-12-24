from pathlib import Path
import pandas as pd
import networkx as nx


class EllipticDataset:
    def __init__(self, root: str = "../Data/elliptic_bitcoin_dataset"):
        self.root = Path(root)
        self.features = None
        self.classes = None
        self.edges = None
        self.labels = None  # ✅ ensure available externally

    def load(self):
        # --- Load CSVs safely ---
        self.features = pd.read_csv(self.root / "elliptic_txs_features.csv", header=None)
        self.classes = pd.read_csv(self.root / "elliptic_txs_classes.csv", header=None)
        self.edges = pd.read_csv(self.root / "elliptic_txs_edgelist.csv", header=None)

        # --- Remove accidental header rows (if any) ---
        if not str(self.classes.iloc[0, 0]).isdigit():
            print("[WARN] Detected header-like first row in classes file — skipping it.")
            self.classes = self.classes.iloc[1:].reset_index(drop=True)

        # --- Rename columns ---
        self.classes.columns = ["txId", "label"]
        self.edges.columns = ["txId1", "txId2"]

        # --- Build Graph ---
        G = nx.from_pandas_edgelist(self.edges, source="txId1", target="txId2")

        # --- Add node attributes ---
        label_dict = self.classes.set_index("txId")["label"].to_dict()
        nx.set_node_attributes(G, label_dict, "label")

        # --- Normalize labels ---
        label_map = {
            "unknown": -1, "licit": 0, "illicit": 1,
            "1": 0, "2": 1
        }

        # Convert everything to string, map, and safely to int
        self.classes["label"] = (
            self.classes["label"]
            .astype(str)
            .replace(label_map)
            .apply(lambda x: -1 if not str(x).isdigit() else int(x))
        )

        # ✅ Store labels as numpy array
        self.labels = self.classes["label"].astype(int).values

        print(f"[INFO] Labels loaded: {len(self.labels)} samples.")
        return G
