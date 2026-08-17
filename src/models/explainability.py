"""
Precision Manufacturing Plant - Explainability Engine
Generates Global SHAP Summary, Feature Importance artifacts, and Local Explanations.
Supports both Tree-based Ensembles and Linear Models (Logistic Regression).
"""

import os
import sys
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend to prevent GUI thread locks
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

# Ensure src root is accessible for imports
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.dirname(CURRENT_DIR)
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

try:
    from models.trainer import load_feature_splits
except ModuleNotFoundError:
    from trainer import load_feature_splits

MODELS_DIR = os.path.join(os.path.dirname(SRC_DIR), "models")
REPORTS_DIR = os.path.join(os.path.dirname(SRC_DIR), "reports", "figures")
os.makedirs(REPORTS_DIR, exist_ok=True)


class IndustrialExplainabilityEngine:
    def __init__(self):
        # 1. Load trained pipeline artifacts
        model_path = os.path.join(MODELS_DIR, "best_failure_model.joblib")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model artifact not found at {model_path}. Run trainer.py first.")
        
        self.model = joblib.load(model_path)
        
        # 2. Ingest feature splits
        (
            self.feature_cols, self.scaler,
            self.X_train, self.y_train, self.X_train_s,
            _, _, _,
            self.X_test, self.y_test, self.X_test_s
        ) = load_feature_splits()

        # 3. Unwrap base estimator from CalibratedClassifierCV wrapper
        if hasattr(self.model, "estimator"):
            self.base_estimator = self.model.estimator
        elif hasattr(self.model, "calibrated_classifiers_"):
            self.base_estimator = self.model.calibrated_classifiers_[0].estimator
        else:
            self.base_estimator = self.model

        # 4. Route explainer based on underlying model architecture
        if isinstance(self.base_estimator, LogisticRegression):
            # Sample background to keep linear masker fast
            background = shap.sample(self.X_train_s, min(100, len(self.X_train_s)))
            self.explainer = shap.LinearExplainer(self.base_estimator, background)
            self.eval_data = self.X_test_s
            self.is_linear = True
        else:
            self.explainer = shap.TreeExplainer(self.base_estimator)
            self.eval_data = self.X_test
            self.is_linear = False

    def compute_shap_values(self, X=None):
        """Computes SHAP values and handles binary classification dimensions."""
        data_to_explain = self.eval_data if X is None else X
        
        explanation = self.explainer(data_to_explain)
        raw_values = explanation.values

        # If shape is (N, Features, 2), slice out positive failure class (idx 1)
        if len(raw_values.shape) == 3 and raw_values.shape[2] == 2:
            shap_matrix = raw_values[:, :, 1]
        else:
            shap_matrix = raw_values

        return explanation, shap_matrix

    def get_local_machine_attribution(
        self, 
        machine_id: str, 
        df_features: pd.DataFrame, 
        top_n: int = 4
    ) -> Dict[str, Any]:
        """
        Extracts local root-cause breakdown for the latest telemetry reading of a specific machine.
        """
        latest_row = df_features[df_features["machine_id"] == machine_id].sort_values("timestamp").iloc[-1:]
        if latest_row.empty:
            raise ValueError(f"Machine ID {machine_id} not found in feature dataset.")

        X_latest = latest_row[self.feature_cols]
        shap_exp, shap_vals = self.compute_shap_values(X_latest)
        
        # Feature attributions for the single row
        row_shap = shap_vals[0]
        row_feat_vals = X_latest.iloc[0].values

        # Sort features by absolute contribution
        sorted_indices = np.argsort(np.abs(row_shap))[::-1]
        
        top_drivers = []
        for idx in sorted_indices[:top_n]:
            col_name = self.feature_cols[idx]
            top_drivers.append({
                "feature": col_name,
                "feature_value": round(float(row_feat_vals[idx]), 3),
                "shap_attribution": round(float(row_shap[idx]), 4),
                "direction": "Risk Increasing" if row_shap[idx] > 0 else "Risk Reducing"
            })

        base_val = float(self.explainer.expected_value[1] if isinstance(self.explainer.expected_value, (list, np.ndarray)) else self.explainer.expected_value)
        model_output = float(base_val + np.sum(row_shap))

        return {
            "machine_id": machine_id,
            "timestamp": str(latest_row["timestamp"].values[0]),
            "base_expected_value": round(base_val, 4),
            "model_prediction_margin": round(model_output, 4),
            "top_drivers": top_drivers,
            "shap_explanation_object": shap_exp[0]
        }


def generate_global_shap_artifacts():
    print("[1/4] Initializing Explainability Engine & Ingesting Splits...")
    engine = IndustrialExplainabilityEngine()

    print("[2/4] Computing SHAP Explanation Matrix on Test Holdout...")
    # Subsample test set for faster plotting if large
    n_samples = min(500, len(engine.eval_data))
    eval_subset = engine.eval_data[:n_samples]
    
    explanation, shap_vals = engine.compute_shap_values(eval_subset)

    print("[3/4] Generating Global Summary and Importance Plots...")
    # 1. Beeswarm / Summary Plot
    plt.figure(figsize=(12, 8))
    shap.summary_plot(
        shap_vals, 
        eval_subset, 
        feature_names=engine.feature_cols, 
        show=False
    )
    summary_plot_path = os.path.join(REPORTS_DIR, "shap_summary_plot.png")
    plt.tight_layout()
    plt.savefig(summary_plot_path, dpi=300, bbox_inches="tight")
    plt.close()

    # 2. Bar Importance Plot
    plt.figure(figsize=(12, 8))
    shap.summary_plot(
        shap_vals, 
        eval_subset, 
        feature_names=engine.feature_cols, 
        plot_type="bar", 
        show=False
    )
    bar_plot_path = os.path.join(REPORTS_DIR, "shap_feature_importance.png")
    plt.tight_layout()
    plt.savefig(bar_plot_path, dpi=300, bbox_inches="tight")
    plt.close()

    # 3. Export Mean Absolute SHAP Importance to CSV
    mean_abs_shap = np.mean(np.abs(shap_vals), axis=0)
    df_importance = pd.DataFrame({
        "feature": engine.feature_cols,
        "mean_abs_shap": mean_abs_shap
    }).sort_values(by="mean_abs_shap", ascending=False)
    
    importance_csv_path = os.path.join(MODELS_DIR, "feature_importance_shap.csv")
    df_importance.to_csv(importance_csv_path, index=False)

    print("[4/4] Artifacts successfully serialized:")
    print(f"  -> Summary Plot:     {summary_plot_path}")
    print(f"  -> Feature Bar Plot: {bar_plot_path}")
    print(f"  -> Importance CSV:   {importance_csv_path}")


if __name__ == "__main__":
    generate_global_shap_artifacts()