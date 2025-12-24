# final_report.py
import os
import pandas as pd
from datetime import datetime

def generate_final_report(metrics_dict, output_dir="reports", visual_dir="visualizations"):
    """
    Generates a structured markdown + CSV report combining
    all key metrics and visualization references.
    """
    os.makedirs(output_dir, exist_ok=True)

    # 1. Save metrics as CSV
    metrics_path = os.path.join(output_dir, "final_metrics_summary.csv")
    pd.DataFrame([metrics_dict]).to_csv(metrics_path, index=False)

    #  2. Create Markdown Report
    report_path = os.path.join(output_dir, "final_report.md")
    with open(report_path, "w") as f:
        f.write("#  Final Anomaly Detection Report\n\n")
        f.write(f"**Generated on:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write("## Model Performance Metrics\n")
        for k, v in metrics_dict.items():
            f.write(f"- **{k}**: {v:.4f}\n")

        f.write("\n##  Visualization Summary\n")
        f.write(f"- PCA Plot: `{os.path.join(visual_dir, 'pca_anomaly_plot.png')}`\n")
        f.write(f"- t-SNE Plot: `{os.path.join(visual_dir, 'tsne_anomaly_plot.png')}`\n")
        f.write(f"- Anomaly Histogram: `{os.path.join(visual_dir, 'anomaly_score_histogram.png')}`\n")

        f.write("\n##  Insights\n")
        f.write("- Higher anomaly scores correspond to transactions likely to be fraudulent.\n")
        f.write("- PCA/t-SNE plots show distinct clusters for normal vs. anomalous data.\n")
        f.write("- Distribution shows imbalance between normal and fraudulent nodes.\n")

        f.write("\n## Next Steps\n")
        f.write("1. Integrate this pipeline into a real-time detection API (FastAPI/Flask).\n")
        f.write("2. Schedule periodic retraining with updated blockchain data.\n")
        f.write("3. Expand feature set using transaction amount, time, and node centrality.\n")

    print(f"[INFO] Final report generated at: {report_path}")
    print(f"[INFO] Metrics summary saved at: {metrics_path}")
