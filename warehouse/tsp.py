"""
Optimal Order Picking Module using Traveling Salesperson Problem (TSP)
via Dynamic Programming with Bitmasking (Held-Karp Algorithm).

Author: Python Engineer & DAA Expert
Project: Smart Warehouse Robot Pathfinding Network
"""

import time
from typing import List, Tuple, Dict, Optional

from warehouse.models import Coordinate, PathResult, TSPResult
from warehouse.astar import AStarNavigator


class TSPOptimizer:
    """
    Solves the Traveling Salesperson Problem (TSP) for order fulfillment.
    
    Given a designated packing station depot and a batch of pick item coordinates,
    this class computes the globally optimal visiting sequence that minimizes
    the robot's total travel distance, starting from the packing station,
    visiting every item location, and returning to the packing station.
    
    Attributes:
        navigator: AStarNavigator instance used to compute exact collision-free
                   shortest paths and distance matrices across warehouse obstacles.
    """

    def __init__(self, navigator: AStarNavigator) -> None:
        """
        Initializes the TSP Optimizer.
        
        Args:
            navigator: AStarNavigator configured for the current warehouse grid.
        """
        self.navigator: AStarNavigator = navigator

    def build_distance_matrix(
        self, nodes: List[Coordinate]
    ) -> Tuple[List[List[float]], List[List[List[Coordinate]]], float]:
        """
        Constructs the pairwise shortest path distance matrix and path registry
        between all target locations using the A* search algorithm.
        
        Optimization:
            Since movement on an undirected grid with symmetric obstacles is symmetric,
            distance(i, j) == distance(j, i) and path(j, i) == reversed(path(i, j)).
            We compute only the upper triangle (N * (N - 1) / 2 calls) instead of N^2,
            reducing A* execution overhead by 50%.
            
        Args:
            nodes: Ordered list of coordinates where index 0 is the packing station
                   and indices 1..N-1 are item pick locations.
                   
        Returns:
            Tuple of:
              - dist_matrix: 2D list of shape [N][N] with shortest travel distances.
              - path_matrix: 2D list of shape [N][N] storing coordinate path lists.
              - elapsed_time: Seconds elapsed during matrix computation.
              
        Complexity Analysis:
            Time Complexity: O(N^2 * (|V| log |V| + |E|))
              Where N is the number of target locations and |V| is total grid cells.
              We execute N*(N-1)/2 A* searches, each bounded by O(|V| log |V|).
            Space Complexity: O(N^2 * L)
              Where L is the average path length (storing intermediate coordinate lists).
        """
        start_time = time.perf_counter()
        n = len(nodes)
        dist_matrix: List[List[float]] = [[0.0 for _ in range(n)] for _ in range(n)]
        path_matrix: List[List[List[Coordinate]]] = [
            [[nodes[i]] if i == j else [] for j in range(n)] for i in range(n)
        ]

        # Compute pairwise distances exploiting symmetry
        for i in range(n):
            for j in range(i + 1, n):
                res: PathResult = self.navigator.find_shortest_path(nodes[i], nodes[j])
                if not res.success:
                    dist_matrix[i][j] = float("inf")
                    dist_matrix[j][i] = float("inf")
                else:
                    dist_matrix[i][j] = res.cost
                    dist_matrix[j][i] = res.cost
                    path_matrix[i][j] = res.path
                    # Reverse path for opposite direction
                    path_matrix[j][i] = list(reversed(res.path))

        elapsed_time = time.perf_counter() - start_time
        return dist_matrix, path_matrix, elapsed_time

    def solve_tsp(
        self, packing_station: Coordinate, item_locations: List[Coordinate]
    ) -> TSPResult:
        """
        Computes the exact optimal picking sequence using Held-Karp Dynamic Programming
        with Bitmasking.
        
        Problem Formulation:
            Let V = {0, 1, ..., N-1} be the set of nodes, where node 0 is the packing
            station depot and nodes 1..N-1 are item pick locations.
            Find a permutation (0, v_1, v_2, ..., v_{N-1}, 0) that minimizes:
                Total Cost = dist(0, v_1) + sum_{i=1}^{N-2} dist(v_i, v_{i+1}) + dist(v_{N-1}, 0)
                
        Dynamic Programming State Formulation:
            - Let S be a subset of vertices visited so far (represented as an integer bitmask).
            - Let u in S be the last visited vertex.
            - dp[mask][u] = Minimum path cost to visit every vertex in subset `mask`,
                            starting at node 0 and currently ending at node `u`.
                            
        Recurrence Relation:
            Base Case:
                dp[1][0] = 0.0  (Bitmask 1 corresponds to {0}; starting at 0 with 0 cost)
                dp[mask][u] = infinity for all other (mask, u)
                
            Transitions:
                For each mask from 1 to 2^N - 1 (with bit 0 set):
                    For each u in mask:
                        For each neighbor v not in mask:
                            next_mask = mask | (1 << v)
                            dp[next_mask][v] = min(dp[next_mask][v], dp[mask][u] + dist[u][v])
                            
            Cycle Closure:
                Optimal Tour Cost = min_{u in {1..N-1}} (dp[(1 << N) - 1][u] + dist[u][0])
                
        Args:
            packing_station: Origin and destination depot coordinate.
            item_locations: List of coordinate locations of items to pick.
            
        Returns:
            TSPResult containing:
              - tour_indices: Ordered indices [0, v_1, ..., v_{N-1}, 0].
              - tour_coordinates: Ordered sequence of grid points.
              - total_distance: Minimum total path distance.
              - subpaths: Segmented A* step-by-step paths between consecutive stops.
              - execution_time_sec: Wall-clock time for DP computation.
              - distance_matrix_time_sec: Time spent generating A* distance matrix.
              - num_targets: Total nodes N.
              - theoretical_operations: Theoretical operation count N^2 * 2^N.
              
        Complexity Analysis:
            Time Complexity:
              - DP Execution: O(N^2 * 2^N)
                Total states = N * 2^N. From each state (mask, u), there are at most N
                possible transitions. Total transitions = sum_{k=1}^N C(N, k) * k * (N - k)
                = N * (N - 1) * 2^(N - 2) = Theta(N^2 * 2^N).
              - Total Algorithm Time (Matrix + DP):
                O(N^2 * (|V| log |V| + |E|) + N^2 * 2^N).
                
            Space Complexity:
              - DP Table & Parent Matrix: O(N * 2^N)
                Required to store the optimal cost and backpointers for tour reconstruction.
        """
        # Node 0 is always the packing station
        all_nodes: List[Coordinate] = [packing_station] + item_locations
        n = len(all_nodes)
        theoretical_ops = (n ** 2) * (2 ** n)

        # Handle Edge Cases
        if n == 1:
            # No items to pick; trivial empty tour
            return TSPResult(
                tour_indices=[0],
                tour_coordinates=[packing_station],
                total_distance=0.0,
                subpaths=[],
                execution_time_sec=0.0,
                distance_matrix_time_sec=0.0,
                num_targets=1,
                theoretical_operations=theoretical_ops
            )

        # Step 1: Precompute Pairwise Distance Matrix via A*
        dist_matrix, path_matrix, matrix_time = self.build_distance_matrix(all_nodes)

        # Step 2: Initialize DP Table and Backtracking Parent Pointers
        dp_start_time = time.perf_counter()
        num_masks = 1 << n  # 2^N

        # dp[mask][u]: minimum distance to cover subset 'mask' ending at vertex 'u'
        dp: List[List[float]] = [[float("inf") for _ in range(n)] for _ in range(num_masks)]
        # parent[mask][u]: stores predecessor vertex for path reconstruction
        parent: List[List[int]] = [[-1 for _ in range(n)] for _ in range(num_masks)]

        # Base case: Robot starts at packing station (node 0) with bitmask 1 (0b00...001)
        dp[1][0] = 0.0

        # Step 3: Populate DP Table using Bitmask Iteration
        # Since adding a new node increases the integer value of mask,
        # natural numeric order (1 to 2^n - 1) strictly satisfies topological order!
        for mask in range(1, num_masks):
            # Only consider states that include the starting packing station (bit 0 must be 1)
            if not (mask & 1):
                continue

            for u in range(n):
                # Node u must be part of current subset
                if not (mask & (1 << u)):
                    continue

                curr_cost = dp[mask][u]
                if curr_cost == float("inf"):
                    continue

                # Transition: extend subset by visiting an unvisited neighbor v
                for v in range(n):
                    if not (mask & (1 << v)):
                        next_mask = mask | (1 << v)
                        cand_cost = curr_cost + dist_matrix[u][v]
                        if cand_cost < dp[next_mask][v]:
                            dp[next_mask][v] = cand_cost
                            parent[next_mask][v] = u

        # Step 4: Find Optimal Cycle Closure returning to packing station (node 0)
        full_mask = num_masks - 1
        min_total_cost = float("inf")
        best_last_node = -1

        for u in range(1, n):
            tour_cost = dp[full_mask][u] + dist_matrix[u][0]
            if tour_cost < min_total_cost:
                min_total_cost = tour_cost
                best_last_node = u

        dp_exec_time = time.perf_counter() - dp_start_time

        # Step 5: Backtrack to reconstruct optimal visit sequence
        tour_indices: List[int] = []
        curr_mask = full_mask
        curr_node = best_last_node

        while curr_node != -1:
            tour_indices.append(curr_node)
            prev_node = parent[curr_mask][curr_node]
            curr_mask = curr_mask ^ (1 << curr_node)
            curr_node = prev_node

        # Reverse tour to start at depot 0: [0, v_1, v_2, ..., v_{N-1}]
        tour_indices.reverse()
        # Append depot 0 to complete closed cycle
        tour_indices.append(0)

        # Step 6: Assemble coordinates and physical A* subpaths
        tour_coordinates = [all_nodes[idx] for idx in tour_indices]
        subpaths: List[List[Coordinate]] = []

        for k in range(len(tour_indices) - 1):
            u_idx = tour_indices[k]
            v_idx = tour_indices[k + 1]
            subpaths.append(path_matrix[u_idx][v_idx])

        return TSPResult(
            tour_indices=tour_indices,
            tour_coordinates=tour_coordinates,
            total_distance=min_total_cost,
            subpaths=subpaths,
            execution_time_sec=dp_exec_time,
            distance_matrix_time_sec=matrix_time,
            num_targets=n,
            theoretical_operations=theoretical_ops
        )
