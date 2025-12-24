import optuna
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split

def optimize_hyperparameters(embeddings, labels, n_trials=30):
    """
    Runs Optuna-based hyperparameter tuning for IsolationForest anomaly detection.
    Automatically handles both string ('licit', 'illicit') and numeric (0/1) labels.
    Handles unmapped string values by setting them to 0.
    """

    # ----------------------------
    # ✅ Label Preprocessing
    # ----------------------------
    labels = np.array(labels)  # Ensure labels is a NumPy array

    # If labels are strings or objects, process them
    if labels.dtype.type is np.str_ or labels.dtype == object:
        # Convert to string array first, then strip
        labels = np.char.strip(labels.astype(str))
        # Map strings to 0/1, and set unmapped values to 0
        labels = np.where(labels == "illicit", 1,  # If "illicit", set to 1
                          np.where(labels == "licit", 0, 0))  # If "licit", set to 0; otherwise, set to 0

    # Ensure final dtype is int
    labels = labels.astype(int)  # Now safe, as labels should be numeric

    # ----------------------------
    # ✅ Split Dataset
    # ----------------------------
    X_train, X_val, y_train, y_val = train_test_split(
        embeddings, labels, test_size=0.2, random_state=42
    )

    # ----------------------------
    # ✅ Objective Function for Optuna
    # ----------------------------
    def objective(trial):
        try:
            n_estimators = trial.suggest_int("n_estimators", 50, 300)
            max_samples = trial.suggest_float("max_samples", 0.6, 1.0)
            contamination = trial.suggest_float("contamination", 0.01, 0.2)
            max_features = trial.suggest_float("max_features", 0.5, 1.0)
            bootstrap = trial.suggest_categorical("bootstrap", [True, False])

            model = IsolationForest(
                n_estimators=n_estimators,
                max_samples=max_samples,
                contamination=contamination,
                max_features=max_features,
                bootstrap=bootstrap,
                random_state=42,
            )

            model.fit(X_train)
            preds = model.predict(X_val)
            y_pred = np.where(preds == -1, 1, 0)  # Anomalies (-1) to 1, normal (1) to 0

            # Compute F1-score safely
            if len(np.unique(y_pred)) == 1:  # All predictions are the same
                return 0.0  # F1-score is 0 if no anomalies are predicted

            f1 = f1_score(y_val, y_pred)
            return f1

        except Exception as e:
            print(f"[WARN] Trial failed: {e}")
            return 0.0  # Return 0 for failed trials

    # ----------------------------
    # ✅ Run Optuna Optimization
    # ----------------------------
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)

    best_params = study.best_params

    # ----------------------------
    # ✅ Retrain with Best Parameters
    # ----------------------------
    best_model = IsolationForest(
        **best_params, random_state=42
    ).fit(embeddings)

    print("\n[INFO]  Best Parameters Found:", best_params)
    print("[INFO]  Best F1 Score:", round(study.best_value, 4))

    return best_model, best_params, study
