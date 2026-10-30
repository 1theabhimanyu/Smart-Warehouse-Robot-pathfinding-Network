"""
Point-to-Point Navigation Module using the A* (A-Star) Search Algorithm.

Author: Python Engineer & DAA Expert
Project: Smart Warehouse Robot Pathfinding Network
"""

import heapq
import itertools
import time
from typing import List, Tuple, Dict, Set, Optional

from warehouse.models import Coordinate, PathResult
from warehouse.grid import WarehouseGrid


def manhattan_distance(p1: Coordinate, p2: Coordinate) -> float:
    """
    Computes the standard L1 (Manhattan) distance between two grid coordinates.
    
    Formula:
        h((r1, c1), (r2, c2)) = |r1 - r2| + |c1 - c2|
        
    Mathematical Properties in Orthogonal Grid:
        1. Admissibility: Never overestimates true travel distance because movement
           is strictly orthogonal (cost per step >= 1). Thus, h(n) <= h*(n).
        2. Consistency (Monotonicity): Satisfies triangle inequality:
           h(u) <= c(u, v) + h(v), where c(u, v) = 1 for adjacent aisle cells.
        3. Optimality: Under an admissible and consistent heuristic, A* is guaranteed
           to return the mathematically shortest collision-free path.
           
    Args:
        p1: First coordinate (r1, c1).
        p2: Second coordinate (r2, c2).
        
    Returns:
        Manhattan distance as float.
        
    Complexity:
        Time Complexity: O(1)
        Space Complexity: O(1)
    """
    return float(abs(p1[0] - p2[0]) + abs(p1[1] - p2[1]))


class AStarNavigator:
    """
    Implements the A* graph traversal algorithm to compute the shortest
    collision-free path between two coordinates on a 2D warehouse grid.
    
    Attributes:
        grid: Reference to the WarehouseGrid instance.
    """

    def __init__(self, grid: WarehouseGrid) -> None:
        """
        Initializes the A* Navigator with the target warehouse floor grid.
        
        Args:
            grid: WarehouseGrid defining aisles and obstacle racks.
        """
        self.grid: WarehouseGrid = grid

    def find_shortest_path(self, start: Coordinate, target: Coordinate) -> PathResult:
        """
        Calculates the shortest collision-free path from start to target using A*.
        
        Algorithmic Process:
            1. Validate boundary and navigability of start and target nodes.
            2. Maintain a Priority Queue (Min-Heap) keyed by f(n) = g(n) + h(n):
               - g(n): Exact cost of shortest path from start to node n.
               - h(n): Manhattan distance heuristic estimate from n to target.
               - f(n): Estimated total cost of path passing through n.
            3. Use unique monotonic counter as tie-breaker for identical f-scores
               to ensure strict heap ordering without comparing coordinate tuples.
            4. Maintain closed set (visited) to avoid re-evaluating optimal nodes.
            5. Track parent pointers (came_from) for backtracing the reconstructed path.
            
        Args:
            start: Source coordinate (x1, y1) as (row1, col1).
            target: Destination coordinate (x2, y2) as (row2, col2).
            
        Returns:
            PathResult containing:
              - path: Ordered list of coordinates [(x1, y1), ..., (x2, y2)].
              - cost: Total steps (length of path - 1).
              - nodes_expanded: Count of vertices popped from min-heap.
              - execution_time_sec: Wall-clock execution time in seconds.
              - success: True if a valid path exists, False otherwise.
              
        Complexity Analysis:
            Time Complexity:
              - Worst Case: O(|V| log |V| + |E|) where |V| = H * W (total grid cells),
                and |E| <= 4|V| (4-connected planar grid edges). Each vertex is
                extracted from the binary min-heap at most once due to the consistent
                Manhattan heuristic. Heap push/pop takes O(log |V|).
              - Best Case: O(d log d) where d is the Manhattan distance in an obstacle-free
                open corridor directly toward target.
            
            Space Complexity:
              - Worst Case: O(|V|) auxiliary memory.
                The priority queue (open set), closed set, and came_from mapping
                store at most |V| nodes in the worst-case scenario.
        """
        start_time = time.perf_counter()

        # Step 1: Pre-condition checks
        if not self.grid.is_navigable(start):
            exec_time = time.perf_counter() - start_time
            return PathResult(
                path=[],
                cost=float("inf"),
                nodes_expanded=0,
                execution_time_sec=exec_time,
                success=False
            )

        if not self.grid.is_navigable(target):
            exec_time = time.perf_counter() - start_time
            return PathResult(
                path=[],
                cost=float("inf"),
                nodes_expanded=0,
                execution_time_sec=exec_time,
                success=False
            )

        # Base case: Already at target
        if start == target:
            exec_time = time.perf_counter() - start_time
            return PathResult(
                path=[start],
                cost=0.0,
                nodes_expanded=0,
                execution_time_sec=exec_time,
                success=True
            )

        # Step 2: Initialize Priority Queue and Path Tracking
        # Tie-breaker counter to avoid comparing Coordinate tuples when f-scores collide
        counter = itertools.count()

        # Min-Heap stores: (f_score, tie_breaker, current_coordinate)
        open_heap: List[Tuple[float, int, Coordinate]] = []
        
        # g_score[node]: exact cost of shortest known path from start to node
        g_score: Dict[Coordinate, float] = {start: 0.0}
        
        # came_from[node]: predecessor on optimal path back to start
        came_from: Dict[Coordinate, Coordinate] = {}
        
        # closed_set: nodes that have already been expanded with optimal g_score
        closed_set: Set[Coordinate] = set()

        initial_h = manhattan_distance(start, target)
        heapq.heappush(open_heap, (initial_h, next(counter), start))

        nodes_expanded = 0

        # Step 3: Main A* Search Loop
        while open_heap:
            current_f, _, current_node = heapq.heappop(open_heap)

            # If node has already been finalized via a shorter path, skip
            if current_node in closed_set:
                continue

            nodes_expanded += 1

            # Goal Test
            if current_node == target:
                exec_time = time.perf_counter() - start_time
                reconstructed_path = self._reconstruct_path(came_from, current_node)
                total_cost = g_score[current_node]
                return PathResult(
                    path=reconstructed_path,
                    cost=total_cost,
                    nodes_expanded=nodes_expanded,
                    execution_time_sec=exec_time,
                    success=True
                )

            # Mark current node as finalized in closed set
            closed_set.add(current_node)
            current_g = g_score[current_node]

            # Step 4: Explore orthogonal neighbors
            for neighbor in self.grid.get_neighbors(current_node):
                if neighbor in closed_set:
                    continue

                # Uniform edge weight of 1 step between adjacent grid cells
                tentative_g = current_g + 1.0

                # Relaxation step: if this path to neighbor is strictly superior
                if tentative_g < g_score.get(neighbor, float("inf")):
                    g_score[neighbor] = tentative_g
                    came_from[neighbor] = current_node
                    h_val = manhattan_distance(neighbor, target)
                    f_val = tentative_g + h_val
                    heapq.heappush(open_heap, (f_val, next(counter), neighbor))

        # Target unreachable (blocked by shelf obstacles)
        exec_time = time.perf_counter() - start_time
        return PathResult(
            path=[],
            cost=float("inf"),
            nodes_expanded=nodes_expanded,
            execution_time_sec=exec_time,
            success=False
        )

    def _reconstruct_path(
        self, came_from: Dict[Coordinate, Coordinate], current: Coordinate
    ) -> List[Coordinate]:
        """
        Backtracks parent pointers from target to start to recover the sequence of coordinates.
        
        Args:
            came_from: Dictionary mapping each node to its optimal predecessor.
            current: Destination node.
            
        Returns:
            List of coordinates starting at origin and ending at destination.
            
        Complexity:
            Time Complexity: O(L) where L is the path length (at most |V|).
            Space Complexity: O(L) for the reconstructed path list.
        """
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        path.reverse()
        return path
