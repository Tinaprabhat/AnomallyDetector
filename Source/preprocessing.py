import networkx as nx
import pandas as pd
from sklearn.preprocessing import StandardScaler

def preprocess_features(features_df):
    X = features_df.iloc[:, 2:].values  # skip ID, time-step columns
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled


def build_node_features(G):
    deg = dict(G.degree())
    clustering = nx.clustering(G)
    features = pd.DataFrame({
        "node": list(deg.keys()),
        "degree": list(deg.values()),
        "clustering": [clustering[n] for n in deg.keys()]
    })
    return features
