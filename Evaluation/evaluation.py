import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score,
    roc_curve, precision_recall_curve, confusion_matrix, classification_report
)
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

def evaluate_anomaly_detection(anomaly_results, labels_df, embeddings, output_dir="Evaluation/outputs"):
    os.makedirs(f"{output_dir}/plots", exist_ok=True)
    os.makedirs(f"{output_dir}/reports", exist_ok=True)

    # Ensure txId alignment
    merged_df = anomaly_results.merge(labels_df, on="txId", how="inner")

    y_true = merged_df["label"].astype(int)
    # Use threshold to convert anomaly_score → anomaly_label
    y_pred = (merged_df["anomaly_score"] > merged_df["anomaly_score"].mean()).astype(int)

    y_scores = merged_df["anomaly_score"]

    print("[DEBUG] Evaluation summary:")
    print(f"y_true unique: {np.unique(y_true)}")
    print(f"y_pred unique: {np.unique(y_pred)}")

    # Example metrics
    from sklearn.metrics import classification_report, roc_auc_score
    print(classification_report(y_true, y_pred))
    print(f"AUC: {roc_auc_score(y_true, merged_df['anomaly_score']):.4f}")

    # Metrics
    metrics = {
        "Precision": precision_score(y_true, y_pred),
        "Recall": recall_score(y_true, y_pred),
        "F1 Score": f1_score(y_true, y_pred),
        "ROC AUC": roc_auc_score(y_true, y_scores)
    }

    # Save report
    report_path = f"{output_dir}/reports/metrics.json"
    with open(report_path, "w") as f:
        json.dump(metrics, f, indent=4)

    print("\nEvaluation Metrics:")
    print(json.dumps(metrics, indent=4))

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    print("\nConfusion Matrix:\n", cm)
    print("\nDetailed Report:\n", classification_report(y_true, y_pred))

    # Plot ROC Curve
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {metrics['ROC AUC']:.2f}")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.savefig(f"{output_dir}/plots/roc_curve.png")
    plt.close()

    # Plot Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    plt.figure()
    plt.plot(recall, precision)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.savefig(f"{output_dir}/plots/precision_recall_curve.png")
    plt.close()

    # Embedding Visualization (PCA + t-SNE)
    try:
        pca_2d = PCA(n_components=2).fit_transform(embeddings)
        tsne_2d = TSNE(n_components=2, perplexity=30, random_state=42).fit_transform(embeddings)

        plt.figure(figsize=(8, 6))
        plt.scatter(pca_2d[:, 0], pca_2d[:, 1], c=y_pred, cmap="coolwarm", s=10)
        plt.title("PCA - Anomaly Visualization")
        plt.savefig(f"{output_dir}/plots/pca_anomalies.png")
        plt.close()

        plt.figure(figsize=(8, 6))
        plt.scatter(tsne_2d[:, 0], tsne_2d[:, 1], c=y_pred, cmap="coolwarm", s=10)
        plt.title("t-SNE - Anomaly Visualization")
        plt.savefig(f"{output_dir}/plots/tsne_anomalies.png")
        plt.close()

    except Exception as e:
        print(f"Embedding visualization skipped: {e}")

    return metrics
