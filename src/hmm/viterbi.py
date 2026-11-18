"""
HMM and Viterbi decoding for sequential route reconstruction.
PES University UE24CS352A: Machine Learning

Takes candidate predictions from the machine learning classifier
and uses road-network graph connectivity to reconstruct a realistic,
continuous vehicle trajectory.
"""

import numpy as np
from typing import List, Dict, Any, Tuple

class SimpleViterbiDecoder:
    """
    Clean, explainable Hidden Markov Model (HMM) Viterbi decoder.
    
    - Hidden States: Candidate road segments (arcs) at each time step.
    - Emission Probability: Derived from classifier predicted probability.
    - Transition Probability: Derived from road network topological connectivity:
        * Same arc: High probability (vehicle continuing on road).
        * Directly connected arcs (to_node == from_node): High probability (valid road turn).
        * 1-hop neighbor: Moderate probability (short gap / skipped node).
        * Disconnected: Severe penalty (penalizes jumping across parallel streets/buildings).
    """

    def __init__(self, same_arc_prob: float = 0.60, connected_prob: float = 0.35, disconnected_penalty: float = 1e-6):
        self.same_arc_score = np.log(same_arc_prob)
        self.connected_score = np.log(connected_prob)
        self.disconnected_score = np.log(disconnected_penalty)

    def decode_trip(
        self,
        observations_candidates: List[Dict[str, Any]],
        arcs_array: np.ndarray
    ) -> List[int]:
        """
        Runs Viterbi dynamic programming along the trajectory.
        
        Parameters:
        - observations_candidates: List for each GPS observation containing:
            'cand_arcs': list of arc_id integers
            'cand_probs': list of classifier probabilities for each candidate
        - arcs_array: (N_arcs, 2) array where row i has [from_node, to_node]
        
        Returns:
        - List of reconstructed arc_id values, one per GPS observation.
        """
        T = len(observations_candidates)
        if T == 0:
            return []

        # Forward trellis matrices:
        # V[t][j] = best log probability ending in candidate j at time step t
        # backpointer[t][j] = index of candidate at time t-1 that gave best score
        V = []
        backpointers = []

        # 1. Initialization (t = 0)
        first_cands = observations_candidates[0]["cand_arcs"]
        first_probs = observations_candidates[0]["cand_probs"]

        v_0 = np.log(np.maximum(first_probs, 1e-9))
        V.append(v_0)
        backpointers.append([-1] * len(first_cands))

        # Build fast lookup for outgoing arcs from each node
        outgoing_map = {}
        for arc_id in range(len(arcs_array)):
            u = arcs_array[arc_id, 0]
            if u not in outgoing_map:
                outgoing_map[u] = set()
            outgoing_map[u].add(arc_id)

        # 2. Viterbi Forward Recursion (t = 1 to T-1)
        for t in range(1, T):
            prev_cands = observations_candidates[t - 1]["cand_arcs"]
            curr_cands = observations_candidates[t]["cand_arcs"]
            curr_probs = observations_candidates[t]["cand_probs"]

            v_prev = V[t - 1]
            v_curr = np.zeros(len(curr_cands))
            bp_curr = np.zeros(len(curr_cands), dtype=int)

            log_emission = np.log(np.maximum(curr_probs, 1e-9))

            for j, c_curr in enumerate(curr_cands):
                u_curr = arcs_array[c_curr, 0]
                best_score = -1e18
                best_prev_idx = 0

                for i, c_prev in enumerate(prev_cands):
                    v_prev_node = arcs_array[c_prev, 1]

                    # Transition scoring based on road network graph
                    if c_prev == c_curr:
                        # Vehicle stays on same segment
                        trans_score = self.same_arc_score
                    elif v_prev_node == u_curr:
                        # Direct connection: end of prev arc connects to start of curr arc
                        trans_score = self.connected_score
                    elif u_curr in outgoing_map and len(outgoing_map.get(v_prev_node, set()).intersection(outgoing_map.get(u_curr, set()))) > 0:
                        # 1-hop intermediate node
                        trans_score = self.connected_score - 2.0
                    else:
                        # Disconnected / jump across streets
                        trans_score = self.disconnected_score

                    candidate_score = v_prev[i] + trans_score
                    if candidate_score > best_score:
                        best_score = candidate_score
                        best_prev_idx = i

                v_curr[j] = best_score + log_emission[j]
                bp_curr[j] = best_prev_idx

            V.append(v_curr)
            backpointers.append(bp_curr)

        # 3. Backtracking to retrieve best path
        best_last_idx = int(np.argmax(V[-1]))
        reconstructed_path = [observations_candidates[-1]["cand_arcs"][best_last_idx]]

        curr_idx = best_last_idx
        for t in range(T - 1, 0, -1):
            curr_idx = int(backpointers[t][curr_idx])
            reconstructed_path.append(observations_candidates[t - 1]["cand_arcs"][curr_idx])

        # Reverse path to chronological order
        reconstructed_path.reverse()
        return reconstructed_path
