"""
Dataset loader and spatial projection utilities for PathSnap.
PES University UE24CS352A: Machine Learning
"""

import os
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional

# WGS84 reference constants
WGS84_A = 6378137.0  # Earth's semi-major axis in meters

def project_coordinates(lon: np.ndarray, lat: np.ndarray, origin_lon: float, origin_lat: float):
    """
    Projects WGS84 geographic coordinates (degrees) to local metric Cartesian coordinates (meters)
    using conformal local equirectangular projection centered at (origin_lon, origin_lat).
    Distortion is < 0.05% over city scale.
    """
    rad = np.pi / 180.0
    x = (lon - origin_lon) * rad * WGS84_A * np.cos(origin_lat * rad)
    y = (lat - origin_lat) * rad * WGS84_A
    return x, y

def unproject_coordinates(x: np.ndarray, y: np.ndarray, origin_lon: float, origin_lat: float):
    """Inverse projection from local metric Cartesian coordinates back to WGS84 degrees."""
    rad = np.pi / 180.0
    lat = origin_lat + y / (rad * WGS84_A)
    lon = origin_lon + x / (rad * WGS84_A * np.cos(origin_lat * rad))
    return lon, lat

def load_trip(trip_dir: str, trip_id: str) -> Dict[str, Any]:
    """
    Loads all raw data files for a single trip:
    - .track: (N_track, 3) -> [longitude, latitude, timestamp_seconds]
    - .nodes: (N_nodes, 2) -> [longitude, latitude]
    - .arcs: (N_arcs, 2) -> [from_node_id, to_node_id]
    - .route: (N_route,) -> sequence of ground-truth arc_id values (if available)
    """
    track_path = os.path.join(trip_dir, f"{trip_id}.track")
    nodes_path = os.path.join(trip_dir, f"{trip_id}.nodes")
    arcs_path = os.path.join(trip_dir, f"{trip_id}.arcs")
    route_path = os.path.join(trip_dir, f"{trip_id}.route")

    if not os.path.exists(track_path):
        raise FileNotFoundError(f"Missing track file: {track_path}")

    # Load track
    track_data = np.loadtxt(track_path)
    if track_data.ndim == 1:
        track_data = track_data.reshape(1, -1)

    # Load nodes
    nodes_data = np.loadtxt(nodes_path) if os.path.exists(nodes_path) else np.empty((0, 2))

    # Load arcs
    arcs_data = np.loadtxt(arcs_path, dtype=int) if os.path.exists(arcs_path) else np.empty((0, 2), dtype=int)
    if arcs_data.ndim == 1 and len(arcs_data) > 0:
        arcs_data = arcs_data.reshape(1, -1)

    # Load route (if available)
    route_data = None
    if os.path.exists(route_path):
        try:
            route_data = np.loadtxt(route_path, dtype=int)
            if route_data.ndim == 0:
                route_data = np.array([int(route_data)])
        except Exception:
            route_data = np.empty(0, dtype=int)

    # Compute origin for local metric projection (center of nodes or track)
    if len(nodes_data) > 0:
        origin_lon = float(np.mean(nodes_data[:, 0]))
        origin_lat = float(np.mean(nodes_data[:, 1]))
    else:
        origin_lon = float(np.mean(track_data[:, 0]))
        origin_lat = float(np.mean(track_data[:, 1]))

    # Local projected coordinates (in meters)
    track_x, track_y = project_coordinates(track_data[:, 0], track_data[:, 1], origin_lon, origin_lat)
    track_m = np.column_stack([track_x, track_y, track_data[:, 2]])

    if len(nodes_data) > 0:
        nodes_x, nodes_y = project_coordinates(nodes_data[:, 0], nodes_data[:, 1], origin_lon, origin_lat)
        nodes_m = np.column_stack([nodes_x, nodes_y])
    else:
        nodes_m = np.empty((0, 2))

    return {
        "trip_id": trip_id,
        "track": track_data,      # lon, lat, time
        "track_m": track_m,      # x (meters), y (meters), time
        "nodes": nodes_data,    # lon, lat
        "nodes_m": nodes_m,      # x (meters), y (meters)
        "arcs": arcs_data,        # from_node, to_node
        "route": route_data,      # ground truth sequence of arc IDs
        "origin_lon": origin_lon,
        "origin_lat": origin_lat
    }
