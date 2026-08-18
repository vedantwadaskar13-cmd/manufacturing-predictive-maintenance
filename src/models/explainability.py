"""
Precision Manufacturing Plant - Explainability Engine
Generates Global SHAP Summary artifacts and per-machine SHAP diagnostics.
"""

import os
import sys
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.dirname(CURRENT_DIR)
ROOT_DIR = os.path.dirname(SRC_DIR)
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

try:
    from models.trainer import load_feature_splits
except ModuleNotFoundError:
    from trainer import load_feature_splits

MODELS_DIR = os.path.join(ROOT_DIR, "models")
DATA_PROCESSED_DIR = os.path.join(ROOT_DIR, "data", "processed")
REPORTS_DIR = os.path.join(ROOT_DIR, "reports", "figures")
os.makedirs(REPORTS_DIR, exist_ok=True)
os.makedirs(DATA_PROCESSED_DIR, exist_ok=True)


class IndustrialExplainabilityEngine:
    def __init__(self):
        model_path = os.path.join(MODELS_DIR, "best_failure_model.joblib")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model artifact not found at {model_path}.")
        
        self.model = joblib.load(model_path)
        
        (
            self.feature_cols, self.scaler,
            self.X_train, self.y_train, self.X_train_s,
            _, _, _,
            self.X_test, self.y_test, self.X_test_s
        ) = load_feature_splits()

        if hasattr(self.model, "estimator"):
            self.base_estimator = self.model.estimator
        elif hasattr(self.model, "calibrated_classifiers_"):
            self.base_estimator = self.model.calibrated_classifiers_[0].estimator
        else:
            self.base_estimator = self.model

        if isinstance(self.base_estimator, LogisticRegression):
            background = shap.sample(self.X_train_s, min(100, len(self.X_train_s)))
            self.explainer = shap.LinearExplainer(self.base_estimator, background)
            self.eval_data = self.X_test_s
            self.is_linear = True
        else:
            self.explainer = shap.TreeExplainer(self.base_estimator)
            self.eval_data = self.X_test
            self.is_linear = False

    def compute_shap_values(self, X=None):
        data_to_explain = self.eval_data if X is None else X
        explanation = self.explainer(data_to_explain)
        raw_values = explanation.values

        if len(raw_values.shape) == 3 and raw_values.shape[2] == 2:
            shap_matrix = raw_values[:, :, 1]
        else:
            shap_matrix = raw_values

        return explanation, shap_matrix


def generate_global_shap_artifacts():
    print("[1/4] Initializing Explainability Engine...")
    engine = IndustrialExplainabilityEngine()

    print("[2/4] Computing Global SHAP values...")
    eval_data = engine.eval_data
    explanation, shap_vals = engine.compute_shap_values(eval_data)

    # 1. Summary Plot
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_vals, eval_data, feature_names=engine.feature_cols, show=False)
    plt.tight_layout()
    plt.savefig(os.path.join(REPORTS_DIR, "shap_summary_plot.png"), dpi=300, bbox_inches="tight")
    plt.close()

    # 2. Bar Importance
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_vals, eval_data, feature_names=engine.feature_cols, plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig(os.path.join(REPORTS_DIR, "shap_feature_importance.png"), dpi=300, bbox_inches="tight")
    plt.close()

    # 3. Export Mean Importance CSV
    mean_abs_shap = np.mean(np.abs(shap_vals), axis=0)
    df_importance = pd.DataFrame({
        "feature": engine.feature_cols,
        "mean_abs_shap": mean_abs_shap
    }).sort_values(by="mean_abs_shap", ascending=False)
    df_importance.to_csv(os.path.join(MODELS_DIR, "feature_importance_shap.csv"), index=False)

    # 4. Generate & Save per-machine diagnostics parquet
    print("[3/4] Generating machine-level SHAP diagnostics table...")
    shap_cols = [f"shap_{col}" for col in engine.feature_cols]
    df_shap_values = pd.DataFrame(shap_vals, columns=shap_cols)

    # 1. Add machine_id (from test set index/features if available, else sequential)
    if hasattr(engine.X_test, "index"):
        df_shap_values["machine_id"] = engine.X_test.index.values
    else:
        df_shap_values["machine_id"] = np.arange(len(df_shap_values))

    # 2. Add top root cause drivers
    top_driver_indices = np.argmax(np.abs(shap_vals), axis=1)
    df_shap_values["top_root_cause_drivers"] = [engine.feature_cols[i] for i in top_driver_indices]
    df_shap_values["primary_failure_driver"] = df_shap_values["top_root_cause_drivers"]
    
    # 3. Model metrics & actuals
    eval_features = engine.X_test_s if engine.is_linear else engine.X_test
    df_shap_values["failure_probability"] = engine.model.predict_proba(eval_features)[:, 1]
    df_shap_values["actual_failure"] = np.asarray(engine.y_test).ravel()

    # Cost-weighted risk categorization (FN=₹65,000, FP=₹2,500)
    df_shap_values["risk_category"] = pd.cut(
        df_shap_values["failure_probability"],
        bins=[-np.inf, 0.30, 0.70, np.inf],
        labels=["Low", "Medium", "Critical"]
    )

    diag_path = os.path.join(DATA_PROCESSED_DIR, "machine_shap_diagnostics.parquet")
    df_shap_values.to_parquet(diag_path, index=False)