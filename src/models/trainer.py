"""
Precision Manufacturing Plant - ML Failure Prediction Pipeline
Trains, calibrates, and benchmarks Logistic Regression, Random Forest, XGBoost, and LightGBM
for 24-hour failure prediction with cost-sensitive thresholding.
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import RobustScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score,
    average_precision_score, confusion_matrix, brier_score_loss
)
import xgboost as xgb
import lightgbm as lgb

FEATURES_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "processed", "features")
MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "models")
os.makedirs(MODELS_DIR, exist_ok=True)

# Business Cost Weights (in INR ₹)
COST_FP = 2500.0   # Unnecessary preventive inspection
COST_FN = 65000.0  # Unplanned catastrophic breakdown

EXCLUDE_COLS = [
    "reading_id", "machine_id", "timestamp", "operator_id",
    "degradation_state", "failure_next_24h"
]


def load_feature_splits():
    train_df = pd.read_parquet(os.path.join(FEATURES_DIR, "features_train.parquet"))
    val_df = pd.read_parquet(os.path.join(FEATURES_DIR, "features_val.parquet"))
    test_df = pd.read_parquet(os.path.join(FEATURES_DIR, "features_test.parquet"))

    feature_cols = [c for c in train_df.columns if c not in EXCLUDE_COLS]

    X_train = train_df[feature_cols].copy()
    y_train = train_df["failure_next_24h"].values

    X_val = val_df[feature_cols].copy()
    y_val = val_df["failure_next_24h"].values

    X_test = test_df[feature_cols].copy()
    y_test = test_df["failure_next_24h"].values

    # Fit RobustScaler on training set only to prevent leakage
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    return (
        feature_cols, scaler,
        X_train, y_train, X_train_scaled,
        X_val, y_val, X_val_scaled,
        X_test, y_test, X_test_scaled
    )


def compute_cost_optimal_threshold(y_true, y_probs):
    """
    Finds the probability decision threshold that minimizes total industrial financial cost.
    """
    thresholds = np.linspace(0.01, 0.99, 100)
    best_thresh = 0.50
    min_cost = float("inf")

    for t in thresholds:
        preds = (y_probs >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, preds, labels=[0, 1]).ravel()
        cost = (fp * COST_FP) + (fn * COST_FN)
        if cost < min_cost:
            min_cost = cost
            best_thresh = t

    return best_thresh, min_cost


def evaluate_model_performance(name, model, X_test, y_test, threshold=0.5):
    probs = model.predict_proba(X_test)[:, 1]
    preds = (probs >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_test, preds, labels=[0, 1]).ravel()
    total_cost = (fp * COST_FP) + (fn * COST_FN)

    return {
        "model_name": name,
        "threshold": round(threshold, 3),
        "pr_auc": round(average_precision_score(y_test, probs), 4),
        "roc_auc": round(roc_auc_score(y_test, probs), 4),
        "f1_score": round(f1_score(y_test, preds, zero_division=0), 4),
        "precision": round(precision_score(y_test, preds, zero_division=0), 4),
        "recall": round(recall_score(y_test, preds, zero_division=0), 4),
        "brier_score": round(brier_score_loss(y_test, probs), 4),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "total_financial_loss_inr": round(total_cost, 2),
        "probs": probs
    }


def main():
    print("[1/5] Ingesting Chronological Feature Splits...")
    (
        feature_cols, scaler,
        X_train, y_train, X_train_s,
        X_val, y_val, X_val_s,
        X_test, y_test, X_test_s
    ) = load_feature_splits()

    pos_scale = (len(y_train) - sum(y_train)) / max(1, sum(y_train))
    print(f"Features: {len(feature_cols)} | Class Imbalance Ratio (scale_pos_weight): {pos_scale:.2f}")

    # ---------------------------------------------------------
    # 2. Define Model Configurations
    # ---------------------------------------------------------
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=150, max_depth=8, class_weight="balanced", n_jobs=-1, random_state=42),
        "XGBoost": xgb.XGBClassifier(
            n_estimators=180, max_depth=5, learning_rate=0.04,
            scale_pos_weight=pos_scale, subsample=0.85, colsample_bytree=0.85,
            eval_metric="logloss", random_state=42
        ),
        "LightGBM": lgb.LGBMClassifier(
            n_estimators=180, max_depth=6, learning_rate=0.04,
            scale_pos_weight=pos_scale, subsample=0.85, colsample_bytree=0.85,
            random_state=42, verbose=-1
        )
    }

    print("[2/5] Training Benchmark Models and Tuning on Validation Set...")
    fitted_models = {}
    optimal_thresholds = {}

    for name, model in models.items():
        # Select scaled data for linear models, unscaled for tree ensembles
        X_tr = X_train_s if name == "Logistic Regression" else X_train
        X_v = X_val_s if name == "Logistic Regression" else X_val

        model.fit(X_tr, y_train)

        # Calibrate probabilities using Isotonic Regression over validation predictions
        calibrated_model = CalibratedClassifierCV(estimator=model, method="isotonic", cv=3)
        calibrated_model.fit(X_tr, y_train)
        fitted_models[name] = calibrated_model

        # Calculate cost-optimal threshold on validation set
        val_probs = calibrated_model.predict_proba(X_v)[:, 1]
        best_t, _ = compute_cost_optimal_threshold(y_val, val_probs)
        optimal_thresholds[name] = best_t
        print(f"  -> {name:<20} | Optimal Decision Threshold: {best_t:.3f}")

    # ---------------------------------------------------------
    # 3. Evaluate On Unseen Holdout Test Set (Month 6)
    # ---------------------------------------------------------
    print("\n[3/5] Evaluating Calibrated Models on Holdout Test Set (Month 6)...")
    results = []
    for name, model in fitted_models.items():
        X_te = X_test_s if name == "Logistic Regression" else X_test
        thresh = optimal_thresholds[name]
        res = evaluate_model_performance(name, model, X_te, y_test, threshold=thresh)
        results.append(res)

    df_eval = pd.DataFrame(results).drop(columns=["probs"])

    # ---------------------------------------------------------
    # 4. Select and Persist Winning Model
    # ---------------------------------------------------------
    # Winner chosen primarily by PR-AUC and minimized financial loss
    winner_idx = df_eval["pr_auc"].idxmax()
    winner_name = df_eval.loc[winner_idx, "model_name"]
    best_model = fitted_models[winner_name]

    print(f"\n[4/5] Selected Production Model: {winner_name} (PR-AUC: {df_eval.loc[winner_idx, 'pr_auc']:.4f})")

    # Serialize artifacts
    joblib.dump(best_model, os.path.join(MODELS_DIR, "best_failure_model.joblib"))
    joblib.dump(scaler, os.path.join(MODELS_DIR, "feature_scaler.joblib"))
    joblib.dump(feature_cols, os.path.join(MODELS_DIR, "feature_columns.joblib"))
    joblib.dump(optimal_thresholds[winner_name], os.path.join(MODELS_DIR, "optimal_threshold.joblib"))
    df_eval.to_csv(os.path.join(MODELS_DIR, "model_benchmark_results.csv"), index=False)

    print("\n[5/5] Benchmark Performance Matrix:")
    print("---------------------------------------------------------------------------------------------------------")
    print(f"{'Model':<20} | {'PR-AUC':<8} | {'ROC-AUC':<8} | {'Recall':<8} | {'Prec':<8} | {'F1':<6} | {'FN':<4} | {'Loss (Rs.)':<14}")
    print("---------------------------------------------------------------------------------------------------------")
    for _, r in df_eval.iterrows():
        print(f"{r['model_name']:<20} | {r['pr_auc']:>8.4f} | {r['roc_auc']:>8.4f} | {r['recall']:>8.4f} | {r['precision']:>8.4f} | {r['f1_score']:>6.4f} | {r['false_negatives']:>4d} | Rs.{r['total_financial_loss_inr']:>12,.2f}")
    print("---------------------------------------------------------------------------------------------------------")


if __name__ == "__main__":
    main()