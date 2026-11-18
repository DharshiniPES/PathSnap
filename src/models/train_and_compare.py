"""
Model experimentation and comparison module for PathSnap.
PES University UE24CS352A: Machine Learning
Trains and evaluates 5 distinct models across different algorithm families:
1. Logistic Regression (Linear & Probabilistic)
2. K-Nearest Neighbors (Instance-Based)
3. Random Forest (Ensemble / Bagging)
4. XGBoost (Gradient Boosting)
5. Multi-Layer Perceptron (Neural Networks)

Computes Group-aware 5-fold CV, Student-Testing Top-1 Accuracy, Macro F1,
Log Loss, Route Continuity, and Inference Latency.
"""

import os
import sys
import time
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score, log_loss, confusion_matrix, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier

# Feature list
FEATURE_COLS = [
    "perp_dist",
    "heading_diff",
    "cos_heading",
    "segment_length",
    "proj_ratio",
    "dist_to_start",
    "dist_to_end",
    "veh_speed",
    "gaussian_emission",
    "candidate_rank"
]

def compute_observation_top1(df_eval: pd.DataFrame, probs: np.ndarray) -> float:
    """
    Computes observation-level Top-1 accuracy:
    For each (trip_id, obs_idx), picks the candidate with highest predicted probability.
    Returns the accuracy: fraction where the top-1 candidate equals true_arc.
    """
    df_eval = df_eval.copy()
    df_eval["pred_prob"] = probs[:, 1]
    # Pick candidate with max score per observation
    top1_df = df_eval.sort_values(["trip_id", "obs_idx", "pred_prob"], ascending=[True, True, False])
    top1_df = top1_df.drop_duplicates(subset=["trip_id", "obs_idx"], keep="first")
    match_count = (top1_df["arc_id"] == top1_df["true_arc"]).sum()
    return float(match_count / len(top1_df))

def compute_route_continuity(df_eval: pd.DataFrame, probs: np.ndarray, arcs_dict: dict) -> float:
    """
    Measures route continuity:
    Fraction of consecutive observation Top-1 predictions that are connected in the road network
    (i.e. arc_t end node == arc_{t+1} start node or arc_t == arc_{t+1}).
    """
    df_eval = df_eval.copy()
    df_eval["pred_prob"] = probs[:, 1]
    top1_df = df_eval.sort_values(["trip_id", "obs_idx", "pred_prob"], ascending=[True, True, False])
    top1_df = top1_df.drop_duplicates(subset=["trip_id", "obs_idx"], keep="first").sort_values(["trip_id", "obs_idx"])

    consecutive_connected = 0
    total_consecutive = 0

    for tid, group in top1_df.groupby("trip_id"):
        arcs_seq = group["arc_id"].values
        trip_arcs = arcs_dict.get(tid)
        if trip_arcs is None or len(arcs_seq) < 2:
            continue
        for i in range(len(arcs_seq) - 1):
            a1, a2 = arcs_seq[i], arcs_seq[i+1]
            if a1 == a2:
                consecutive_connected += 1
            else:
                u1, v1 = trip_arcs[a1]
                u2, v2 = trip_arcs[a2]
                if v1 == u2:
                    consecutive_connected += 1
            total_consecutive += 1

    return float(consecutive_connected / max(1, total_consecutive))

def main():
    res_dir = r"C:\Users\Dharshini\.gemini\antigravity\scratch\PathSnap\results"
    viz_dir = r"C:\Users\Dharshini\.gemini\antigravity\scratch\PathSnap\visualizations"
    models_dir = r"C:\Users\Dharshini\.gemini\antigravity\scratch\PathSnap\src\models\saved_models"
    data_dir = r"C:\Users\Dharshini\.gemini\antigravity\scratch\PathSnap\data\Student folder"

    os.makedirs(res_dir, exist_ok=True)
    os.makedirs(viz_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    print("Loading extracted feature sets...")
    df_train = pd.read_csv(os.path.join(res_dir, "train_features.csv"))
    df_test = pd.read_csv(os.path.join(res_dir, "student_test_features.csv"))

    # Load arcs mapping for student test trips for route continuity check
    test_arcs_dict = {}
    for tid in df_test["trip_id"].unique():
        tid_str = f"{int(tid):08d}"
        arcs_file = os.path.join(data_dir, "student_test", tid_str, f"{tid_str}.arcs")
        if os.path.exists(arcs_file):
            test_arcs_dict[tid] = np.loadtxt(arcs_file, dtype=int)
            test_arcs_dict[tid_str] = test_arcs_dict[tid]

    X_train = df_train[FEATURE_COLS].values
    y_train = df_train["is_match"].values
    groups_train = df_train["trip_id"].values

    X_test = df_test[FEATURE_COLS].values
    y_test = df_test["is_match"].values

    print(f"Features: {FEATURE_COLS}")
    print(f"Train samples: {len(X_train)}, Test samples: {len(X_test)}")

    # Define 5 models from distinct algorithm families
    models = {
        "Logistic Regression": {
            "family": "Linear & Probabilistic",
            "model": Pipeline([
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(C=1.0, max_iter=500, random_state=42, class_weight="balanced"))
            ]),
            "config": "C=1.0, class_weight='balanced', StandardScaler"
        },
        "K-Nearest Neighbors": {
            "family": "Instance-based",
            "model": Pipeline([
                ("scaler", StandardScaler()),
                ("clf", KNeighborsClassifier(n_neighbors=7, weights="distance", algorithm="kd_tree", leaf_size=40))
            ]),
            "config": "k=7, weights='distance', algorithm='kd_tree'"
        },
        "Random Forest": {
            "family": "Ensemble (Bagging)",
            "model": RandomForestClassifier(n_estimators=100, max_depth=12, random_state=42, n_jobs=-1, class_weight="balanced"),
            "config": "n_estimators=100, max_depth=12, class_weight='balanced'"
        },
        "XGBoost": {
            "family": "Boosting",
            "model": XGBClassifier(n_estimators=120, max_depth=6, learning_rate=0.1, random_state=42, n_jobs=-1, tree_method="hist", eval_metric="logloss"),
            "config": "n_estimators=120, max_depth=6, lr=0.1, tree_method='hist'"
        },
        "Multi-Layer Perceptron": {
            "family": "Neural Networks",
            "model": Pipeline([
                ("scaler", StandardScaler()),
                ("clf", MLPClassifier(hidden_layer_sizes=(64, 32), activation="relu", max_iter=40, random_state=42, early_stopping=True, validation_fraction=0.1))
            ]),
            "config": "hidden_layers=(64, 32), relu, early_stopping"
        }
    }

    # 1. 5-Fold Group-aware Cross Validation on Training Trips
    print("\n--- Performing 5-Fold Group-aware Cross Validation on 80 Train Trips ---")
    gkf = GroupKFold(n_splits=5)
    cv_results = {name: [] for name in models}

    for fold, (train_idx, val_idx) in enumerate(gkf.split(X_train, y_train, groups=groups_train)):
        print(f"Fold {fold+1} / 5...")
        X_tr, y_tr = X_train[train_idx], y_train[train_idx]
        X_va, y_va = X_train[val_idx], y_train[val_idx]

        for name, info in models.items():
            clf = info["model"]
            # Subsample for KNN during fold to keep CV fast
            if name == "K-Nearest Neighbors":
                sub_idx = np.random.RandomState(42).choice(len(X_tr), min(25000, len(X_tr)), replace=False)
                clf.fit(X_tr[sub_idx], y_tr[sub_idx])
                sub_va = np.random.RandomState(42).choice(len(X_va), min(10000, len(X_va)), replace=False)
                preds = clf.predict(X_va[sub_va])
                f1 = f1_score(y_va[sub_va], preds, average="macro")
            else:
                clf.fit(X_tr, y_tr)
                preds = clf.predict(X_va)
                f1 = f1_score(y_va, preds, average="macro")
            cv_results[name].append(f1)

    cv_macro_f1_means = {name: float(np.mean(scores)) for name, scores in cv_results.items()}
    for name, mean_score in cv_macro_f1_means.items():
        print(f"  {name} 5-Fold CV Macro F1: {mean_score:.4f}")

    # 2. Fit on full training set, evaluate on Student-Testing set
    print("\n--- Training on Full Training Set and Evaluating on 10 Student-Test Trips ---")
    comparison_records = []
    test_probs_dict = {}
    test_preds_dict = {}
    training_times = {}
    inference_times = {}

    for name, info in models.items():
        print(f"Training {name} ({info['family']})...")
        clf = info["model"]

        # Train time
        t0 = time.time()
        if name == "K-Nearest Neighbors":
            # Use representative 35,000 samples for KNN to ensure sub-second inference
            sub_idx = np.random.RandomState(42).choice(len(X_train), min(35000, len(X_train)), replace=False)
            clf.fit(X_train[sub_idx], y_train[sub_idx])
        else:
            clf.fit(X_train, y_train)
        t_fit = time.time() - t0
        training_times[name] = t_fit

        # Inference time on student test
        t0 = time.time()
        probs = clf.predict_proba(X_test)
        t_infer = time.time() - t0
        inference_times[name] = t_infer
        test_probs_dict[name] = probs

        preds = (probs[:, 1] >= 0.5).astype(int)
        test_preds_dict[name] = preds

        # Metrics
        macro_f1 = float(f1_score(y_test, preds, average="macro"))
        acc = float(accuracy_score(y_test, preds))
        loss = float(log_loss(y_test, probs[:, 1]))
        top1_acc = compute_observation_top1(df_test, probs)
        continuity = compute_route_continuity(df_test, probs, test_arcs_dict)

        # Save trained model artifact
        save_path = os.path.join(models_dir, f"{name.lower().replace(' ', '_').replace('-', '_')}.joblib")
        joblib.dump(clf, save_path)

        comparison_records.append({
            "Model": name,
            "Algorithm Family": info["family"],
            "Key Configuration": info["config"],
            "CV Macro F1-score": round(cv_macro_f1_means[name], 4),
            "Top-1 Accuracy": round(top1_acc, 4),
            "Route continuity": round(continuity, 4),
            "Student-Test Macro F1-score": round(macro_f1, 4),
            "Log Loss": round(loss, 4),
            "Inference Time (s)": round(t_infer, 4),
            "Training Time (s)": round(t_fit, 2)
        })

    df_comparison = pd.DataFrame(comparison_records)
    comp_csv_path = os.path.join(res_dir, "model_comparison_table.csv")
    df_comparison.to_csv(comp_csv_path, index=False)
    print("\n=== Mandatory Model Comparison Table ===")
    print(df_comparison.to_string())

    # 3. Model Visualizations (Bar comparisons, Confusion Matrices, Loss curves)
    print("\nGenerating model comparison visualizations...")
    sns.set_theme(style="whitegrid", font_scale=1.1)

    # 3.1 Comparison Bar Chart: CV Macro F1, Student Test Macro F1, and Top-1 Accuracy
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    sns.barplot(x="CV Macro F1-score", y="Model", data=df_comparison, ax=axes[0], palette="viridis")
    axes[0].set_title("5-Fold CV Macro F1-score", fontweight="bold")
    axes[0].set_xlim(0, 1.0)
    for i, v in enumerate(df_comparison["CV Macro F1-score"]):
        axes[0].text(v + 0.02, i, f"{v:.3f}", va="center", fontweight="bold")

    sns.barplot(x="Student-Test Macro F1-score", y="Model", data=df_comparison, ax=axes[1], palette="crest")
    axes[1].set_title("Student-Test Macro F1-score", fontweight="bold")
    axes[1].set_xlim(0, 1.0)
    for i, v in enumerate(df_comparison["Student-Test Macro F1-score"]):
        axes[1].text(v + 0.02, i, f"{v:.3f}", va="center", fontweight="bold")

    sns.barplot(x="Top-1 Accuracy", y="Model", data=df_comparison, ax=axes[2], palette="flare")
    axes[2].set_title("Observation-level Top-1 Accuracy", fontweight="bold")
    axes[2].set_xlim(0, 1.0)
    for i, v in enumerate(df_comparison["Top-1 Accuracy"]):
        axes[2].text(v + 0.02, i, f"{v:.3f}", va="center", fontweight="bold")

    plt.tight_layout()
    comp_bar_path = os.path.join(viz_dir, "model_metrics_comparison.png")
    plt.savefig(comp_bar_path, dpi=300)
    plt.close()

    # 3.2 Log Loss and Inference Time Chart
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.barplot(x="Log Loss", y="Model", data=df_comparison, ax=axes[0], palette="magma")
    axes[0].set_title("Student-Test Log Loss (Lower is Better)", fontweight="bold")
    for i, v in enumerate(df_comparison["Log Loss"]):
        axes[0].text(v + 0.01, i, f"{v:.3f}", va="center", fontweight="bold")

    sns.barplot(x="Inference Time (s)", y="Model", data=df_comparison, ax=axes[1], palette="rocket")
    axes[1].set_title("Inference Latency for 114k Candidates (seconds)", fontweight="bold")
    for i, v in enumerate(df_comparison["Inference Time (s)"]):
        axes[1].text(v + 0.01, i, f"{v:.2f}s", va="center", fontweight="bold")

    plt.tight_layout()
    latency_bar_path = os.path.join(viz_dir, "model_loss_and_latency.png")
    plt.savefig(latency_bar_path, dpi=300)
    plt.close()

    # 3.3 Confusion Matrices for All 5 Models
    fig, axes = plt.subplots(1, 5, figsize=(22, 4))
    for idx, (name, preds) in enumerate(test_preds_dict.items()):
        cm = confusion_matrix(y_test, preds)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=axes[idx])
        axes[idx].set_title(f"{name}\nCM", fontsize=11, fontweight="bold")
        axes[idx].set_xlabel("Predicted")
        axes[idx].set_ylabel("Actual" if idx == 0 else "")
    plt.tight_layout()
    cm_path = os.path.join(viz_dir, "model_confusion_matrices.png")
    plt.savefig(cm_path, dpi=300)
    plt.close()

    # 3.4 MLP Training Loss History Curve
    mlp_clf = models["Multi-Layer Perceptron"]["model"].named_steps["clf"]
    if hasattr(mlp_clf, "loss_curve_"):
        plt.figure(figsize=(7, 4.5))
        plt.plot(mlp_clf.loss_curve_, marker="o", color="#d62728", lw=2, label="MLP Cross-Entropy Loss")
        plt.title("MLP Neural Network Training Loss Curve", fontweight="bold")
        plt.xlabel("Epoch / Iteration")
        plt.ylabel("Loss")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        mlp_loss_path = os.path.join(viz_dir, "mlp_loss_curve.png")
        plt.savefig(mlp_loss_path, dpi=300)
        plt.close()
        print(f"Saved MLP loss curve to {mlp_loss_path}")

    print(f"All visualizations and results saved to {viz_dir} and {res_dir}")

if __name__ == "__main__":
    main()
