# visualize_results.py
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

def visualize_anomaly_results(anomaly_results, labels_df, embeddings, save_path="visualizations"):
    print("[INFO] Starting Phase 5: Visualization...")

    # Create output directory
    os.makedirs(save_path, exist_ok=True)

    # Merge all info into one dataframe
    df = anomaly_results.merge(labels_df, on="txId", how="inner")
    df["label"] = df["label"].astype(int)

    # Ensure embeddings match shape
    if embeddings.shape[0] != len(df):
        print("[WARN] Aligning embeddings with available nodes...")
        embeddings = embeddings[:len(df)]

    # ----------------------------
    # 2D Projection using PCA
    # ----------------------------
    pca = PCA(n_components=2)
    reduced_pca = pca.fit_transform(embeddings)

    df["PCA1"] = reduced_pca[:, 0]
    df["PCA2"] = reduced_pca[:, 1]

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(
        df["PCA1"],
        df["PCA2"],
        c=df["anomaly_score"],
        cmap="coolwarm",
        alpha=0.6
    )
    plt.colorbar(scatter, label="Anomaly Score")
    plt.title("PCA Visualization of Anomalies")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.savefig(os.path.join(save_path, "pca_anomaly_plot.png"), dpi=300)
    plt.close()

    # ----------------------------
    # t-SNE for richer visualization
    # ----------------------------
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    reduced_tsne = tsne.fit_transform(embeddings)

    df["TSNE1"] = reduced_tsne[:, 0]
    df["TSNE2"] = reduced_tsne[:, 1]

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(
        df["TSNE1"],
        df["TSNE2"],
        c=df["anomaly_score"],
        cmap="viridis",
        alpha=0.7
    )
    plt.colorbar(scatter, label="Anomaly Score")
    plt.title("t-SNE Visualization of Anomalies")
    plt.xlabel("t-SNE Component 1")
    plt.ylabel("t-SNE Component 2")
    plt.savefig(os.path.join(save_path, "tsne_anomaly_plot.png"), dpi=300)
    plt.close()

    # ----------------------------
    # Anomaly Score Histogram
    # ----------------------------
    plt.figure(figsize=(7, 5))
    plt.hist(df["anomaly_score"], bins=50, color='steelblue', alpha=0.7)
    plt.title("Anomaly Score Distribution")
    plt.xlabel("Anomaly Score")
    plt.ylabel("Frequency")
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(save_path, "anomaly_score_histogram.png"), dpi=300)
    plt.close()

    print("[INFO] Visualization completed. Plots saved in:", save_path)
