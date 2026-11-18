"""
Candidate generation and feature extraction module for PathSnap.
PES University UE24CS352A: Machine Learning
"""

import os
import sys
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from typing import Dict, Any, List, Tuple

def extract_trip_candidates(
    trip: Dict[str, Any],
    is_train: bool = True,
    max_candidates_per_obs: int = 8,
    sample_stride: int = 1
) -> pd.DataFrame:
    """
    Generates road segment candidates for each GPS observation and computes
    spatial, kinematic, and topological features.
    """
    track_m = trip["track_m"]
    nodes_m = trip["nodes_m"]
    arcs = trip["arcs"]
    route = trip["route"]
    trip_id = trip["trip_id"]

    num_obs = len(track_m)
    num_arcs = len(arcs)

    if num_obs == 0 or num_arcs == 0:
        return pd.DataFrame()

    p1 = nodes_m[arcs[:, 0]]
    p2 = nodes_m[arcs[:, 1]]
    ab = p2 - p1
    ab_len_sq = np.sum(ab**2, axis=1)
    ab_lens = np.sqrt(ab_len_sq)

    # Arc orientation angle (radians)
    arc_angles = np.arctan2(ab[:, 1], ab[:, 0])

    # True reference arc determination (if route is available)
    true_arc_for_obs = {}
    if route is not None and len(route) > 0:
        rp1 = nodes_m[arcs[route, 0]]
        rp2 = nodes_m[arcs[route, 1]]
        rab = rp2 - rp1
        rab_len_sq = np.maximum(np.sum(rab**2, axis=1), 1e-6)

        # For each observation, find closest arc along reference route
        for i in range(num_obs):
            p = track_m[i, :2]
            ap = p - rp1
            t = np.clip(np.sum(ap * rab, axis=1) / rab_len_sq, 0.0, 1.0)
            proj = rp1 + t[:, None] * rab
            dists = np.sqrt(np.sum((p - proj)**2, axis=1))
            best_idx = np.argmin(dists)
            true_arc_for_obs[i] = int(route[best_idx])

    # Multi-point sampling along arcs for high-recall KDTree indexing
    arc_samples = []
    arc_indices = []
    fractions = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    for frac in fractions:
        pts = p1 + frac * ab
        arc_samples.append(pts)
        arc_indices.append(np.arange(num_arcs))

    all_sampled_pts = np.vstack(arc_samples)
    all_sampled_ids = np.concatenate(arc_indices)

    kdtree = cKDTree(all_sampled_pts)

    # Query KDTree for all observations
    # Subsample observations if sample_stride > 1
    obs_indices = range(0, num_obs, sample_stride)
    query_pts = track_m[obs_indices, :2]

    # Query top-35 nearest sampled points to guarantee >98% candidate recall
    k_query = min(35, len(all_sampled_pts))
    dists_kdtree, neighbors = kdtree.query(query_pts, k=k_query)

    records = []

    for idx_in_subset, obs_idx in enumerate(obs_indices):
        p = track_m[obs_idx, :2]
        t_curr = track_m[obs_idx, 2]

        # Vehicle heading and speed
        if obs_idx > 0:
            prev_p = track_m[obs_idx - 1, :2]
            prev_t = track_m[obs_idx - 1, 2]
            dt = max(t_curr - prev_t, 1e-3)
            dp = p - prev_p
            veh_speed = np.linalg.norm(dp) / dt
            veh_heading = np.arctan2(dp[1], dp[0])
        elif obs_idx + 1 < num_obs:
            next_p = track_m[obs_idx + 1, :2]
            next_t = track_m[obs_idx + 1, 2]
            dt = max(next_t - t_curr, 1e-3)
            dp = next_p - p
            veh_speed = np.linalg.norm(dp) / dt
            veh_heading = np.arctan2(dp[1], dp[0])
        else:
            veh_speed = 0.0
            veh_heading = 0.0

        # Unique candidate arcs from KDTree query
        nbr_ids = neighbors[idx_in_subset]
        cand_arcs = []
        seen = set()
        for nid in nbr_ids:
            arc_id = all_sampled_ids[nid]
            if arc_id not in seen:
                seen.add(arc_id)
                cand_arcs.append(arc_id)

        # Compute exact perpendicular distance for candidates
        cand_p1 = p1[cand_arcs]
        cand_ab = ab[cand_arcs]
        cand_ab_len_sq = np.maximum(ab_len_sq[cand_arcs], 1e-6)
        cand_ap = p - cand_p1
        cand_t = np.clip(np.sum(cand_ap * cand_ab, axis=1) / cand_ab_len_sq, 0.0, 1.0)
        cand_proj = cand_p1 + cand_t[:, None] * cand_ab
        cand_dists = np.sqrt(np.sum((p - cand_proj)**2, axis=1))

        # Sort candidates by perpendicular distance and select top candidates
        sorted_order = np.argsort(cand_dists)
        top_candidates = sorted_order[:max_candidates_per_obs]

        # Ground truth reference arc
        true_arc = true_arc_for_obs.get(obs_idx, -1)

        for rank, c_idx in enumerate(top_candidates):
            arc_id = cand_arcs[c_idx]
            perp_dist = cand_dists[c_idx]
            t_proj = cand_t[c_idx]
            arc_len = ab_lens[arc_id]

            # Heading difference in [-pi, pi] mapped to [0, pi]
            arc_angle = arc_angles[arc_id]
            angle_diff = np.abs(np.arctan2(np.sin(veh_heading - arc_angle), np.cos(veh_heading - arc_angle)))
            cos_heading = np.cos(veh_heading - arc_angle)

            # Node distances
            dist_to_start = np.linalg.norm(p - cand_p1[c_idx])
            dist_to_end = np.linalg.norm(p - p2[arc_id])

            # Theoretical emission Gaussian likelihood
            gaussian_emission = np.exp(- (perp_dist**2) / (2.0 * (10.0**2)))

            label = 1 if (is_train and arc_id == true_arc) else 0

            records.append({
                "trip_id": trip_id,
                "obs_idx": obs_idx,
                "timestamp": t_curr,
                "arc_id": arc_id,
                "perp_dist": perp_dist,
                "heading_diff": angle_diff,
                "cos_heading": cos_heading,
                "segment_length": arc_len,
                "proj_ratio": t_proj,
                "dist_to_start": dist_to_start,
                "dist_to_end": dist_to_end,
                "veh_speed": veh_speed,
                "gaussian_emission": gaussian_emission,
                "candidate_rank": rank + 1,
                "is_match": label,
                "true_arc": true_arc
            })

    return pd.DataFrame(records)
