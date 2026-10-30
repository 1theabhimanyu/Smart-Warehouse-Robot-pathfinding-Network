"""
Main executable entry point for the Smart Warehouse Robot Pathfinding Network.

Author: Python Engineer & DAA Expert
Project: Smart Warehouse Robot Pathfinding Network
Course: Design and Analysis of Algorithms (DAA)

"""

import sys
import os
import argparse
import time
import math
from typing import List, Tuple

# Ensure package imports resolve cleanly
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from warehouse.simulation import WarehouseSimulation
from warehouse.grid import WarehouseGrid
from warehouse.astar import AStarNavigator
from warehouse.tsp import TSPOptimizer
from warehouse.models import Coordinate


def run_scaling_benchmark() -> None:
    """
    Executes an empirical complexity scaling benchmark comparing empirical runtime
    against theoretical asymptotic Big-O growth curves for both A* and TSP DP.
    
    Academic Demonstration:
        1. A* Search Scaling: Evaluates shortest path across increasing Manhattan distances
           d in {5, 10, 15, 20, 25, 30} on a 30x30 grid.
        2. Held-Karp TSP DP Scaling: Evaluates exact TSP solving time for problem sizes
           N in {4, 6, 8, 10, 12, 14} vs theoretical O(N^2 * 2^N).
    """
    print("\n" + "=" * 90)
    print("   DAA EMPIRICAL VS THEORETICAL COMPLEXITY SCALING BENCHMARK")
    print("=" * 90)

    # -----------------------------------------------------------------
    # Part 1: A* Search Empirical Benchmark
    # -----------------------------------------------------------------
    print("\n[+] PART 1: A* Search Scalability on 30x30 Grid (|V| = 900)")
    grid_bench = WarehouseGrid(30, 30)
    # Add scattered obstacles
    for r in range(5, 25, 4):
        grid_bench.add_shelf_block((r, 5), height=3, width=8)
        grid_bench.add_shelf_block((r, 17), height=3, width=8)

    nav = AStarNavigator(grid_bench)
    test_pairs = [
        ((0, 0), (2, 2)),
        ((0, 0), (5, 4)),
        ((0, 0), (10, 14)),
        ((0, 0), (18, 16)),
        ((0, 0), (24, 26)),
        ((0, 0), (29, 29)),
    ]

    print(f"{'Start -> Goal':<18} | {'Manhattan':<10} | {'Path Cost':<10} | {'Nodes Expanded':<16} | {'Empirical Time':<15}")
    print("-" * 75)
    for start, goal in test_pairs:
        # Ensure navigable
        grid_bench.set_aisle(start)
        grid_bench.set_aisle(goal)
        h = abs(start[0] - goal[0]) + abs(start[1] - goal[1])
        res = nav.find_shortest_path(start, goal)
        time_str = f"{res.execution_time_sec * 1e3:.4f} ms"
        print(f"{str(start):<7} -> {str(goal):<7} | {h:<10} | {res.cost:<10.1f} | {res.nodes_expanded:<16} | {time_str:<15}")

    # -----------------------------------------------------------------
    # Part 2: Held-Karp TSP DP Scaling Benchmark
    # -----------------------------------------------------------------
    print("\n[+] PART 2: Held-Karp TSP DP Scaling across Increasing N")
    print("    Comparing Empirical Time vs Theoretical O(N^2 * 2^N) vs Brute-Force O(N!)")
    tsp_opt = TSPOptimizer(nav)

    # Sample pick candidate coordinates
    candidate_pool: List[Coordinate] = [
        (0, 0), (1, 4), (4, 4), (8, 4), (12, 4), (16, 4),
        (20, 4), (24, 4), (28, 4), (4, 14), (8, 14), (12, 14),
        (16, 14), (20, 14), (24, 14)
    ]
    # Ensure all candidates are navigable
    for pt in candidate_pool:
        grid_bench.set_aisle(pt)

    benchmark_n_values = [4, 6, 8, 10, 12, 14]
    print(f"{'N':<4} | {'Items':<6} | {'DP Time (ms)':<14} | {'Theoretical Ops (N²·2ⁿ)':<25} | {'Brute-Force Ops ((N-1)!)':<25}")
    print("-" * 80)

    for n in benchmark_n_values:
        station = candidate_pool[0]
        items = candidate_pool[1:n]
        res = tsp_opt.solve_tsp(station, items)
        dp_ms = res.execution_time_sec * 1e3
        theoretical_ops = (n ** 2) * (1 << n)
        brute_force_ops = math.factorial(n - 1)
        print(f"{n:<4} | {len(items):<6} | {dp_ms:<14.4f} | {theoretical_ops:<25,} | {brute_force_ops:<25,}")

    print("\nBenchmark Takeaways:")
    print("1. A* Heuristic Efficiency: Expanding only a focused cone of vertices, far below |V|.")
    print("2. Held-Karp Tractability: While N=14 brute force requires 6.2 billion iterations,")
    print("   DP with Bitmasking finishes in ~3.2 million operations, solving in milliseconds.")
    print("=" * 90 + "\n")


def main() -> None:
    """
    Main CLI program entry point.
    """
    parser = argparse.ArgumentParser(
        description="Smart Warehouse Robot Pathfinding Network Simulation (DAA Project)"
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run algorithmic scaling benchmark (Empirical vs Theoretical Big-O)",
    )
    parser.add_argument(
        "--grid-size",
        type=int,
        default=20,
        help="Warehouse grid dimension (default: 20 for 20x20)",
    )

    args = parser.parse_args()

    # Initialize and execute primary simulation
    simulation = WarehouseSimulation(height=args.grid_size, width=args.grid_size)
    simulation.spawn_default_fleet()
    simulation.queue_default_orders()
    simulation.run_simulation()

    # Optional scaling benchmark
    if args.benchmark:
        run_scaling_benchmark()


if __name__ == "__main__":
    main()
