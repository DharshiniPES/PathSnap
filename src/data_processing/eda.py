"""
Exploratory Data Analysis (EDA) module for PathSnap.
PES University UE24CS352A: Machine Learning
Analyzes both Training (80 trips) and Student-Test (10 trips) partitions,
computes comprehensive statistics, saves tabular summaries to CSV,
and generates publication-quality figures and Folium maps.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from collections import Counter

# Add parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.data_processing.dataset_loader import load_trip, project_coordinates

def run_eda(data_root: str, output_viz_dir: str, output_res_dir: str):
    os.makedirs(output_viz_dir, exist_ok=True)
    os.makedirs(output_res_dir, exist_ok=True)

    train_dir = os.path.join(data_root, "train")
    test_dir = os.path.join(data_root, "student_test")

    train_trips = sorted(os.listdir(train_dir))
    test_trips = sorted(os.listdir(test_dir))

    print(f"Discovered {len(train_trips)} training trips and {len(test_trips)} student-test trips.")

    # 1. Inventory & Statistics across partitions
    trip_records = []
    sampling_intervals_all = []
    speeds_all = []
    arc_lengths_all = []

    for partition_name, p_dir, trips in [("Training", train_dir, train_trips), ("Student-Testing", test_dir, test_trips)]:
        for tid in trips:
            trip_path = os.path.join(p_dir, tid)
            trip = load_trip(trip_path, tid)

            track = trip["track"]
            nodes = trip["nodes"]
            arcs = trip["arcs"]
            route = trip["route"]
            track_m = trip["track_m"]

            # Missing value check
            missing_pts = int(np.isnan(track).any() or np.isinf(track).any())

            # Sampling intervals & speeds
            dt = np.diff(track[:, 2]) if len(track) > 1 else np.array([])
            sampling_intervals_all.extend(dt.tolist())

            dx = np.diff(track_m[:, 0]) if len(track_m) > 1 else np.array([])
            dy = np.diff(track_m[:, 1]) if len(track_m) > 1 else np.array([])
            distances = np.sqrt(dx**2 + dy**2)
            speeds = np.where(dt > 0, distances / np.maximum(dt, 1e-3), 0.0)
            speeds_all.extend(speeds.tolist())

            # Trajectory length
            traj_length_m = float(np.sum(distances)) if len(distances) > 0 else 0.0
            duration_s = float(track[-1, 2] - track[0, 2]) if len(track) > 1 else 0.0

            # Route length
            route_len_m = 0.0
            if route is not None and len(route) > 0:
                p1_m = trip["nodes_m"][arcs[route, 0]]
                p2_m = trip["nodes_m"][arcs[route, 1]]
                route_arc_lens = np.sqrt(np.sum((p2_m - p1_m)**2, axis=1))
                route_len_m = float(np.sum(route_arc_lens))

            trip_records.append({
                "partition": partition_name,
                "trip_id": tid,
                "gps_points": len(track),
                "duration_seconds": duration_s,
                "trajectory_length_km": traj_length_m / 1000.0,
                "avg_speed_kmh": (np.mean(speeds) * 3.6) if len(speeds) > 0 else 0.0,
                "max_speed_kmh": (np.max(speeds) * 3.6) if len(speeds) > 0 else 0.0,
                "road_nodes": len(nodes),
                "road_arcs": len(arcs),
                "route_arcs": len(route) if route is not None else 0,
                "route_length_km": route_len_m / 1000.0,
                "has_missing_values": bool(missing_pts)
            })

    df_trips = pd.DataFrame(trip_records)
    inventory_csv_path = os.path.join(output_res_dir, "dataset_inventory.csv")
    df_trips.to_csv(inventory_csv_path, index=False)
    print(f"Saved dataset inventory to {inventory_csv_path}")

    # Summary table by partition
    summary = df_trips.groupby("partition").agg({
        "trip_id": "count",
        "gps_points": ["sum", "mean", "min", "max", "median"],
        "duration_seconds": ["mean", "min", "max"],
        "trajectory_length_km": ["mean", "min", "max"],
        "avg_speed_kmh": ["mean", "min", "max"],
        "road_nodes": ["mean", "min", "max"],
        "road_arcs": ["mean", "min", "max"],
        "route_arcs": ["mean", "min", "max"]
    }).round(2)
    summary_csv_path = os.path.join(output_res_dir, "dataset_summary_statistics.csv")
    summary.to_csv(summary_csv_path)
    print("Partition Summary:")
    print(summary)

    # 2. Plotting Sampling Intervals and Velocity Distributions
    sns.set_theme(style="whitegrid", font_scale=1.1)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Sampling intervals (clip for visualization)
    intervals_arr = np.array(sampling_intervals_all)
    exact_1s = np.mean(np.abs(intervals_arr - 1.0) < 0.05) * 100.0

    sns.histplot(intervals_arr[intervals_arr <= 5], bins=50, kde=True, ax=axes[0], color="#1f77b4")
    axes[0].set_title(f"GPS Sampling Interval Distribution\n({exact_1s:.1f}% exactly 1.0s)", fontsize=13, fontweight='bold')
    axes[0].set_xlabel("Time Delta (seconds)")
    axes[0].set_ylabel("Count")

    # Speeds
    speeds_arr = np.array(speeds_all) * 3.6  # convert to km/h
    sns.histplot(speeds_arr[speeds_arr <= 140], bins=60, kde=True, ax=axes[1], color="#2ca02c")
    axes[1].set_title("Vehicle Velocity Distribution (km/h)", fontsize=13, fontweight='bold')
    axes[1].set_xlabel("Speed (km/h)")
    axes[1].set_ylabel("Count")

    plt.tight_layout()
    dist_plot_path = os.path.join(output_viz_dir, "eda_gps_distributions.png")
    plt.savefig(dist_plot_path, dpi=300)
    plt.close()
    print(f"Saved distribution plot to {dist_plot_path}")

    # 3. Plotting Road Network & Trajectory Analysis
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # Boxplot of GPS points per partition
    sns.boxplot(x="partition", y="gps_points", data=df_trips, ax=axes[0], palette="Set2")
    axes[0].set_title("GPS Observations per Trip by Partition", fontsize=13, fontweight='bold')
    axes[0].set_ylabel("Number of GPS Observations")
    axes[0].set_xlabel("")

    # Scatter of Trajectory length vs Road Nodes
    sns.scatterplot(
        x="road_nodes", y="trajectory_length_km", hue="partition", size="route_arcs",
        sizes=(30, 200), data=df_trips, ax=axes[1], palette="Dark2", alpha=0.8
    )
    axes[1].set_title("Road Network Scale vs Trajectory Length", fontsize=13, fontweight='bold')
    axes[1].set_xlabel("Road Network Nodes")
    axes[1].set_ylabel("Trajectory Length (km)")
    axes[1].legend(title="Partition")

    plt.tight_layout()
    analysis_plot_path = os.path.join(output_viz_dir, "eda_dataset_structure.png")
    plt.savefig(analysis_plot_path, dpi=300)
    plt.close()
    print(f"Saved dataset structure plot to {analysis_plot_path}")

    # 4. Mandatory Interactive Folium Map for a representative trip
    sample_tid = "00000000"  # Student test trip 0
    sample_trip_path = os.path.join(test_dir, sample_tid)
    sample_trip = load_trip(sample_trip_path, sample_tid)

    track = sample_trip["track"]
    nodes = sample_trip["nodes"]
    arcs = sample_trip["arcs"]
    route = sample_trip["route"]

    center_lat = float(np.mean(track[:, 1]))
    center_lon = float(np.mean(track[:, 0]))

    fmap = folium.Map(location=[center_lat, center_lon], zoom_start=15, tiles="CartoDB positron")

    # Layer 1: Road network edges (subsampled for quick rendering if dense)
    network_group = folium.FeatureGroup(name="Road Network Segments (Grey)", show=True)
    # Filter arcs close to the trajectory bounding box
    bbox_pad = 0.005
    min_lon, max_lon = np.min(track[:, 0]) - bbox_pad, np.max(track[:, 0]) + bbox_pad
    min_lat, max_lat = np.min(track[:, 1]) - bbox_pad, np.max(track[:, 1]) + bbox_pad

    for arc_idx in range(len(arcs)):
        u, v = arcs[arc_idx]
        p1 = nodes[u]
        p2 = nodes[v]
        if (min_lon <= p1[0] <= max_lon and min_lat <= p1[1] <= max_lat) or \
           (min_lon <= p2[0] <= max_lon and min_lat <= p2[1] <= max_lat):
            folium.PolyLine(
                locations=[[p1[1], p1[0]], [p2[1], p2[0]]],
                color="#888888", weight=1.5, opacity=0.4
            ).add_to(network_group)
    network_group.add_to(fmap)

    # Layer 2: Ground-truth reference route (Green)
    if route is not None and len(route) > 0:
        route_group = folium.FeatureGroup(name="Ground-Truth Route (Green Line)", show=True)
        route_coords = []
        for arc_id in route:
            u, v = arcs[arc_id]
            p1 = nodes[u]
            p2 = nodes[v]
            route_coords.append([p1[1], p1[0]])
            route_coords.append([p2[1], p2[0]])

        folium.PolyLine(
            locations=route_coords,
            color="#00aa00", weight=4, opacity=0.85,
            tooltip=f"Reference Route ({len(route)} traversed arcs)"
        ).add_to(route_group)
        route_group.add_to(fmap)

    # Layer 3: Raw GPS observations (Blue markers)
    gps_group = folium.FeatureGroup(name="Raw GPS Observations (Blue Dots)", show=True)
    for i in range(0, len(track), max(1, len(track) // 300)):  # visualize up to 300 sampled points
        folium.CircleMarker(
            location=[track[i, 1], track[i, 0]],
            radius=3, color="#0055ff", fill=True, fill_color="#0055ff", fill_opacity=0.7,
            popup=f"GPS #{i} @ t={track[i, 2]:.1f}s"
        ).add_to(gps_group)
    gps_group.add_to(fmap)

    folium.LayerControl().add_to(fmap)

    folium_html_path = os.path.join(output_viz_dir, "interactive_map_sample.html")
    fmap.save(folium_html_path)
    print(f"Saved interactive Folium map to {folium_html_path}")

    return df_trips

if __name__ == "__main__":
    data_path = r"C:\Users\Dharshini\.gemini\antigravity\scratch\PathSnap\data\Student folder"
    viz_dir = r"C:\Users\Dharshini\.gemini\antigravity\scratch\PathSnap\visualizations"
    res_dir = r"C:\Users\Dharshini\.gemini\antigravity\scratch\PathSnap\results"
    run_eda(data_path, viz_dir, res_dir)
