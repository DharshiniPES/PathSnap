"""
Bayesian Hyperparameter Optimization for PathSnap.
PES University UE24CS352A: Machine Learning
Uses Optuna with Group-aware Cross-Validation to tune the winning model (XGBoost).
"""

import os
import sys
import time
import joblib
import optuna
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score, log_loss
from xgboost import XGBClassifier

# Silence Optuna info logs during optimization
optuna.logging.set_verbosity(optuna.logging.WARNING)

FEATURE_COLS = [
    "perp_dist", "heading_diff", "cos_heading", "segment_length", "proj_ratio",
    "dist_to_start", "dist_to_end", "veh_speed", "gaussian_emission", "candidate_rank"
]

def main():
    res_dir = r"C:\Users\Dharshini\.gemini\antigravity\scratch\PathSnap\results"
    models_dir = r"C:\Users\Dharshini\.gemini\antigravity\scratch\PathSnap\src\models\saved_models"
    viz_dir = r"C:\Users\Dharshini\.gemini\antigravity\scratch\PathSnap\visualizations"

    print("Loading feature sets for Bayesian optimization...")
    df_train = pd.read_csv(os.path.join(res_dir, "train_features.csv"))
    df_test = pd.read_csv(os.path.join(res_dir, "student_test_features.csv"))

    X_train = df_train[FEATURE_COLS].values
    y_train = df_train["is_match"].values
    groups_train = df_train["trip_id"].values

    X_test = df_test[FEATURE_COLS].values
    y_test = df_test["is_match"].values

    # 3-Fold Group-aware CV for Bayesian search efficiency
    gkf = GroupKFold(n_splits=3)
    splits = list(gkf.split(X_train, y_train, groups=groups_train))

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 80, 200, step=20),
            "max_depth": trial.suggest_int("max_depth", 4, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.03, 0.25, log=True),
            "subsample": trial.suggest_float("subsample", 0.65, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.65, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-2, 10.0, log=True),
            "tree_method": "hist",
            "eval_metric": "logloss",
            "random_state": 42,
            "n_jobs": -1
        }

        fold_scores = []
        for tr_idx, va_idx in splits:
            X_tr, y_tr = X_train[tr_idx], y_train[tr_idx]
            X_va, y_va = X_train[va_idx], y_train[va_idx]

            clf = XGBClassifier(**params)
            clf.fit(X_tr, y_tr)
            preds = clf.predict(X_va)
            fold_scores.append(f1_score(y_va, preds, average="macro"))

        return np.mean(fold_scores)

    print("Running Bayesian Hyperparameter Optimization (15 trials)...")
    t0 = time.time()
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=15)
    opt_duration = time.time() - t0

    best_params = study.best_params
    best_score = study.best_value
    print(f"Optimization completed in {opt_duration:.1f}s!")
    print(f"Best Group-CV Macro F1: {best_score:.4f}")
    print("Best Hyperparameters:", best_params)

    # Train refitted tuned model on full training set
    print("Refitting best tuned model on full training set...")
    best_params.update({"tree_method": "hist", "eval_metric": "logloss", "random_state": 42, "n_jobs": -1})
    tuned_clf = XGBClassifier(**best_params)
    tuned_clf.fit(X_train, y_train)

    # Evaluate tuned model on student test set
    t0 = time.time()
    tuned_probs = tuned_clf.predict_proba(X_test)
    tuned_infer_time = time.time() - t0
    tuned_preds = (tuned_probs[:, 1] >= 0.5).astype(int)

    tuned_macro_f1 = f1_score(y_test, tuned_preds, average="macro")
    tuned_log_loss = log_loss(y_test, tuned_probs[:, 1])

    # Save tuned model
    joblib.dump(tuned_clf, os.path.join(models_dir, "xgboost_tuned.joblib"))

    # Load baseline untuned XGBoost results for comparison
    comp_df = pd.read_csv(os.path.join(res_dir, "model_comparison_table.csv"))
    untuned_row = comp_df[comp_df["Model"] == "XGBoost"].iloc[0]

    # Save tuning comparison record
    tuning_comparison = pd.DataFrame([
        {
            "Model Variant": "Untuned XGBoost (Baseline)",
            "CV Macro F1": untuned_row["CV Macro F1-score"],
            "Student-Test Macro F1": untuned_row["Student-Test Macro F1-score"],
            "Log Loss": untuned_row["Log Loss"],
            "Inference Time (s)": untuned_row["Inference Time (s)"]
        },
        {
            "Model Variant": "Bayesian Tuned XGBoost",
            "CV Macro F1": round(best_score, 4),
            "Student-Test Macro F1": round(tuned_macro_f1, 4),
            "Log Loss": round(tuned_log_loss, 4),
            "Inference Time (s)": round(tuned_infer_time, 4)
        }
    ])
    tuning_csv = os.path.join(res_dir, "bayesian_tuning_comparison.csv")
    tuning_comparison.to_csv(tuning_csv, index=False)
    print("\n=== Untuned vs Tuned Model Performance ===")
    print(tuning_comparison.to_string())

    # Save Optimization History Plot
    trials_df = study.trials_dataframe()
    trials_csv = os.path.join(res_dir, "bayesian_trials_history.csv")
    trials_df.to_csv(trials_csv, index=False)

    plt.figure(figsize=(8, 4.5))
    plt.plot(trials_df["number"] + 1, trials_df["value"], marker="o", color="#2563eb", lw=2)
    plt.title("Bayesian Optimization Trial Convergence (Optuna TPE)", fontweight="bold")
    plt.xlabel("Trial Number")
    plt.ylabel("Validation Macro F1-score")
    plt.grid(True)
    plt.tight_layout()
    opt_plot_path = os.path.join(viz_dir, "bayesian_optimization_trials.png")
    plt.savefig(opt_plot_path, dpi=300)
    plt.close()
    print(f"Saved optimization plot to {opt_plot_path}")

if __name__ == "__main__":
    main()
