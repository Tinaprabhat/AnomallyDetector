#  Final Anomaly Detection Report

**Generated on:** 2025-10-14 12:36:25

## Model Performance Metrics
- **AUC**: 0.8231
- **Accuracy**: 0.9456
- **Precision**: 0.9123
- **Recall**: 0.8674
- **F1-Score**: 0.8895

##  Visualization Summary
- PCA Plot: `visualizations\pca_anomaly_plot.png`
- t-SNE Plot: `visualizations\tsne_anomaly_plot.png`
- Anomaly Histogram: `visualizations\anomaly_score_histogram.png`

##  Insights
- Higher anomaly scores correspond to transactions likely to be fraudulent.
- PCA/t-SNE plots show distinct clusters for normal vs. anomalous data.
- Distribution shows imbalance between normal and fraudulent nodes.

## Next Steps
1. Integrate this pipeline into a real-time detection API (FastAPI/Flask).
2. Schedule periodic retraining with updated blockchain data.
3. Expand feature set using transaction amount, time, and node centrality.
