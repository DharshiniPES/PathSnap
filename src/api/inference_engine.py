"""
Inference Engine for Laptop 1 FastAPI backend.
PES University UE24CS352A: Machine Learning
Handles model loading, on-the-fly candidate extraction, ML inference,
Viterbi sequential decoding, and server-side ground truth evaluation.
"""

import os
import time
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional

from src.data_processing.dataset_loader import load_trip
from src.features.candidate_features import extract_trip_candidates
from src.hmm.viterbi import SimpleViterbiDecoder

FEATURE_COLS = [
    "perp_dist", "heading_diff", "cos_heading", "segment_length", "proj_ratio",
    "dist_to_start", "dist_to_end", "veh_speed", "gaussian_emission", "candidate_rank"
]

class PathSnapEngine:
    def __init__(self, models_dir: str, data_dir: str):
        self.models_dir = models_dir
        self.data_dir = data_dir
        self.loaded_models = {}
        self.viterbi = SimpleViterbiDecoder()
        
        # Load available models
        model_files = {
            "xgboost_tuned": "xgboost_tuned.joblib",
            "xgboost": "xgboost.joblib",
            "random_forest": "random_forest.joblib",
            "mlp": "multi_layer_perceptron.joblib",
            "logistic_regression": "logistic_regression.joblib",
            "knn": "k_nearest_neighbors.joblib"
        }
        for name, fname in model_files.items():
            fpath = os.path.join(models_dir, fname)
            if os.path.exists(fpath):
                self.loaded_models[name] = joblib.load(fpath)

        print(f"PathSnapEngine initialized. Loaded models: {list(self.loaded_models.keys())}")

    def list_available_trips(self, partition: str = "student_test") -> List[str]:
        p_dir = os.path.join(self.data_dir, partition)
        if os.path.exists(p_dir):
            return sorted(os.listdir(p_dir))
        return []

    def run_inference(
        self,
        trip_id: str,
        partition: str = "student_test",
        model_name: str = "xgboost",
        use_hmm: bool = True
    ) -> Dict[str, Any]:
        """
        Runs complete inference pipeline for a given trip:
        1. Loads raw track and road network.
        2. Generates candidate arcs and extracts features.
        3. Predicts candidate match probabilities using chosen ML model.
        4. (Optional) Decodes optimal path using HMM + Viterbi.
        5. Computes server-side accuracy and route continuity against ground truth.
        """
        t0 = time.time()
        trip_dir = os.path.join(self.data_dir, partition, trip_id)
        if not os.path.exists(trip_dir):
            raise FileNotFoundError(f"Trip directory not found: {trip_dir}")

        # 1. Load trip data
        trip = load_trip(trip_dir, trip_id)
        track = trip["track"]
        nodes = trip["nodes"]
        arcs = trip["arcs"]
        route = trip["route"]

        # 2. Extract candidates and features
        cands_df = extract_trip_candidates(trip, is_train=True, max_candidates_per_obs=6, sample_stride=1)
        if len(cands_df) == 0:
            raise ValueError(f"No candidates generated for trip {trip_id}")

        # 3. Model Inference
        clf = self.loaded_models.get(model_name)
        if clf is None:
            # Fallback to first available model
            model_name = list(self.loaded_models.keys())[0]
            clf = self.loaded_models[model_name]

        X_eval = cands_df[FEATURE_COLS].values
        probs = clf.predict_proba(X_eval)[:, 1]
        cands_df["prob"] = probs

        # 4. Sequential Route Reconstruction
        obs_groups = cands_df.groupby("obs_idx")
        obs_cands_list = []
        true_arcs_list = []
        clf_top1_list = []

        for o_idx, o_df in obs_groups:
            c_arcs = o_df["arc_id"].tolist()
            c_probs = o_df["prob"].tolist()
            t_arc = int(o_df["true_arc"].iloc[0])

            obs_cands_list.append({"cand_arcs": c_arcs, "cand_probs": c_probs})
            true_arcs_list.append(t_arc)
            clf_top1_list.append(c_arcs[int(np.argmax(c_probs))])

        if use_hmm:
            predicted_sequence = self.viterbi.decode_trip(obs_cands_list, arcs)
        else:
            predicted_sequence = clf_top1_list

        t_elapsed_ms = (time.time() - t0) * 1000.0

        # 5. Server-side Evaluation (calculating accuracy against ground truth)
        y_true = np.array(true_arcs_list)
        y_pred = np.array(predicted_sequence)

        has_ground_truth = (len(y_true) > 0 and (y_true != -1).any())
        accuracy = float(np.mean(y_true == y_pred)) if has_ground_truth else None

        # Route continuity calculation
        consecutive_connected = 0
        total_transitions = max(1, len(predicted_sequence) - 1)
        for i in range(len(predicted_sequence) - 1):
            a1, a2 = predicted_sequence[i], predicted_sequence[i + 1]
            if a1 == a2 or arcs[a1, 1] == arcs[a2, 0]:
                consecutive_connected += 1
        continuity = float(consecutive_connected / total_transitions)

        # Build polyline coordinate segments for predicted route
        predicted_route_coords = []
        for arc_id in predicted_sequence:
            u, v = arcs[arc_id]
            p1 = [float(nodes[u, 1]), float(nodes[u, 0])]  # lat, lon
            p2 = [float(nodes[v, 1]), float(nodes[v, 0])]
            predicted_route_coords.append([p1, p2])

        # Server-side ground truth polyline coordinates (if present)
        ground_truth_coords = []
        if route is not None and len(route) > 0:
            for arc_id in route:
                u, v = arcs[arc_id]
                p1 = [float(nodes[u, 1]), float(nodes[u, 0])]
                p2 = [float(nodes[v, 1]), float(nodes[v, 0])]
                ground_truth_coords.append([p1, p2])

        # GPS points: subsampled for light payload
        gps_pts = [[float(p[1]), float(p[0])] for p in track[::max(1, len(track) // 300)]]

        return {
            "trip_id": trip_id,
            "partition": partition,
            "model_used": model_name,
            "decoder": "HMM + Viterbi" if use_hmm else "Classifier Top-1 Only",
            "metrics": {
                "top1_accuracy_percent": round(accuracy * 100.0, 2) if accuracy is not None else "N/A",
                "route_continuity_percent": round(continuity * 100.0, 2),
                "total_observations": len(track),
                "predicted_arcs_count": len(predicted_sequence),
                "server_latency_ms": round(t_elapsed_ms, 2)
            },
            "gps_samples": gps_pts,
            "predicted_route": predicted_sequence,
            "predicted_route_coords": predicted_route_coords,
            "ground_truth_route_coords": ground_truth_coords,
            "center": [float(np.mean(track[:, 1])), float(np.mean(track[:, 0]))]
        }
