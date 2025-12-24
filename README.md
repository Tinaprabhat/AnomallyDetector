
# AnomallyDetector

Anomaly Detection system for identifying suspicious or abnormal patterns in transaction data, with a focus on blockchain / financial datasets (e.g., Elliptic Bitcoin Dataset).

This project is structured for experimentation, evaluation, optimization, and visualization of anomaly detection techniques using Python.

---

## 📌 Project Overview

Anomaly detection is critical in domains such as:
- Fraud detection
- Financial crime analysis
- Blockchain transaction monitoring
- Cybersecurity

This repository provides a modular pipeline to:
- Load and preprocess data
- Train anomaly detection models
- Evaluate performance
- Visualize results

---

## 🗂️ Repository Structure

```

AnomallyDetector/
│
├── Source/            # Core source code (models, preprocessing, utilities)
├── Evaluation/        # Model evaluation scripts and metrics
├── Optimization/      # Hyperparameter tuning and optimization logic
├── Visualization/     # Visualization scripts
├── visualizations/    # Generated plots and figures
├── Reporting/         # Reports and summaries
│
├── main.py            # Entry point for running experiments
├── .gitignore         # Ignored files (datasets, CSVs, etc.)
├── .gitattributes
└── README.md

```

---

## 📊 Dataset

This project uses the **Elliptic Bitcoin Dataset**, which is **not included in this repository** due to GitHub file size limitations.

### Download Dataset
You can download the dataset from Kaggle:
```

[https://www.kaggle.com/datasets/ellipticco/elliptic-data-set](https://www.kaggle.com/datasets/ellipticco/elliptic-data-set)

```

### Expected Dataset Structure
After downloading, place the data locally as:
```

Data/
└── elliptic_bitcoin_dataset/
├── elliptic_txs_features.csv
├── elliptic_txs_classes.csv
└── elliptic_txs_edgelist.csv

````

⚠️ **Note:** The `Data/` directory is ignored by Git and should not be pushed to GitHub.

---

## ⚙️ Installation & Setup

### 1️⃣ Clone the repository
```bash
git clone https://github.com/Tinaprabhat/AnomallyDetector.git
cd AnomallyDetector
````

### 2️⃣ Create a virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate    # Linux / Mac
venv\Scripts\activate       # Windows
```

### 3️⃣ Install dependencies

Create a `requirements.txt` if not present, then:

```bash
pip install -r requirements.txt
```

---

## ▶️ Usage

Run the main pipeline:

```bash
python main.py
```

Depending on implementation, this may:

* Load and preprocess transaction data
* Train anomaly detection models
* Evaluate results
* Generate visualizations

---

## 📈 Models & Techniques

Possible techniques used / supported:

* Isolation Forest
* Autoencoders
* Statistical anomaly detection
* Graph-based analysis
* Unsupervised / semi-supervised learning

(Exact methods depend on implementation inside `Source/`.)

---

## 📌 Results & Visualization

* Evaluation metrics are stored in `Evaluation/`
* Plots and charts are saved in `visualizations/`
* Reports and summaries are available in `Reporting/`

---

## 🚫 What Is Not Included

* ❌ Raw datasets
* ❌ Large CSV files
* ❌ Generated outputs

These are excluded intentionally to comply with GitHub limits and best practices.

---

## 🤝 Contributing

Contributions are welcome:

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Open a Pull Request

---

## 📄 License

This project is for academic and research purposes.
Please check dataset licenses before commercial use.

---

## 👤 Author

**Tina Prabhat**
GitHub: [https://github.com/Tinaprabhat](https://github.com/Tinaprabhat)


