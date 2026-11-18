"""
Script to extract candidate features across training and student-test partitions.
PES University UE24CS352A: Machine Learning
"""

import os
import sys
import pandas as pd
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.data_processing.dataset_loader import load_trip
from src.features.candidate_features import extract_trip_candidates

def main():
    data_root = r"C:\Users\Dharshini\.gemini\antigravity\scratch\PathSnap\data\Student folder"
    res_dir = r"C:\Users\Dharshini\.gemini\antigravity\scratch\PathSnap\results"
    os.makedirs(res_dir, exist_ok=True)

    train_dir = os.path.join(data_root, "train")
    test_dir = os.path.join(data_root, "student_test")

    train_trips = sorted(os.listdir(train_dir))
    test_trips = sorted(os.listdir(test_dir))

    print(f"Extracting features for 25 training trips (stride=2) for robust Group-aware CV...")
    selected_train_trips = train_trips[:25]
    train_dfs = []
    for tid in tqdm(selected_train_trips, desc="Train trips"):
        trip_path = os.path.join(train_dir, tid)
        trip = load_trip(trip_path, tid)
        df_cands = extract_trip_candidates(trip, is_train=True, max_candidates_per_obs=6, sample_stride=2)
        train_dfs.append(df_cands)

    df_train = pd.concat(train_dfs, ignore_index=True)
    train_csv = os.path.join(res_dir, "train_features.csv")
    df_train.to_csv(train_csv, index=False)
    print(f"Saved training features ({len(df_train)} rows, {df_train['is_match'].sum()} positives) to {train_csv}")

    print(f"Extracting features for all 10 student-test trips (stride=1)...")
    test_dfs = []
    for tid in tqdm(test_trips, desc="Student-Test trips"):
        trip_path = os.path.join(test_dir, tid)
        trip = load_trip(trip_path, tid)
        df_cands = extract_trip_candidates(trip, is_train=True, max_candidates_per_obs=6, sample_stride=1)
        test_dfs.append(df_cands)

    df_test = pd.concat(test_dfs, ignore_index=True)
    test_csv = os.path.join(res_dir, "student_test_features.csv")
    df_test.to_csv(test_csv, index=False)
    print(f"Saved student-test features ({len(df_test)} rows, {df_test['is_match'].sum()} positives) to {test_csv}")

if __name__ == "__main__":
    main()
