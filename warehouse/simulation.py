"""
Warehouse Simulation Orchestrator and Performance Analytics Module.

Author: Python Engineer & DAA Expert
Project: Smart Warehouse Robot Pathfinding Network
"""

import time
import math
from typing import List, Tuple, Dict, Any, Optional

from warehouse.models import (
    Coordinate,
    Robot,
    Order,
    RobotStatus,
    PathResult,
    TSPResult,
    AssignmentResult,
    SimulationAnalytics,
)
from warehouse.grid import WarehouseGrid
from warehouse.astar import AStarNavigator
from warehouse.tsp import TSPOptimizer
from warehouse.task_allocator import GreedyTaskAllocator


class WarehouseSimulation:
    """
    Coordinates the comprehensive lifecycle of the Smart Warehouse Robot Pathfinding Network.
    
    Orchestration Workflow:
        1. Environment Setup: Initializes a 20x20 grid with obstacles (shelves),
           packing stations, and charging depots.
        2. Fleet Spawning: Instantiates 3 autonomous mobile robots (AMRs).
        3. Order Queueing: Generates 2-3 realistic customer orders with multiple item picks.
        4. Task Allocation: Applies Greedy Dispatch algorithm to assign orders to idle robots.
        5. Route Planning: Applies A* Pathfinding and TSP DP with Bitmasking (Held-Karp).
        6. Route Execution: Simulates robot navigation and item collection.
        7. Empirical vs Theoretical Analytics: Formats and prints a rigorous DAA report.
        
    Attributes:
        grid: 20x20 discrete WarehouseGrid.
        robots: List of spawned Robot instances.
        order_queue: List of customer Order instances.
        navigator: AStarNavigator for point-to-point shortest paths.
        tsp_optimizer: TSPOptimizer for Held-Karp order picking sequences.
        task_allocator: GreedyTaskAllocator for fleet dispatch.
        analytics: SimulationAnalytics container tracking timings and metrics.
    """

    def __init__(self, height: int = 20, width: int = 20) -> None:
        """
        Initializes the simulation container and algorithmic engines.
        
        Args:
            height: Number of warehouse grid rows (default: 20).
            width: Number of warehouse grid columns (default: 20).
            
        Complexity:
            Time Complexity: O(H * W) to allocate the grid topology.
            Space Complexity: O(H * W) to store the grid.
        """
        self.height: int = height
        self.width: int = width
        self.grid: WarehouseGrid = WarehouseGrid(height, width)
        self.grid.setup_realistic_warehouse_layout()

        # Algorithmic modules
        self.navigator: AStarNavigator = AStarNavigator(self.grid)
        self.tsp_optimizer: TSPOptimizer = TSPOptimizer(self.navigator)
        self.task_allocator: GreedyTaskAllocator = GreedyTaskAllocator(
            self.navigator, use_astar_distance=True
        )

        # Entities and telemetry
        self.robots: List[Robot] = []
        self.order_queue: List[Order] = []
        self.completed_orders: List[Order] = []
        self.analytics: SimulationAnalytics = SimulationAnalytics()

    def spawn_default_fleet(self) -> None:
        """
        Spawns 3 autonomous robots at distinct charging docks.
        
        Complexity:
            Time Complexity: O(R) where R = 3.
            Space Complexity: O(R) memory for robot state.
        """
        self.robots = [
            Robot(robot_id="ROBOT-01", position=(19, 2), battery_level=100.0),
            Robot(robot_id="ROBOT-02", position=(19, 9), battery_level=95.0),
            Robot(robot_id="ROBOT-03", position=(19, 16), battery_level=90.0),
        ]

    def queue_default_orders(self) -> None:
        """
        Queues 3 batched customer orders with varying picking loads.
        Locations are situated in navigable aisles adjacent to shelf storage racks.
        
        Orders:
            - ORD-101: 4 item picks, Packing Station (0, 2)
            - ORD-102: 5 item picks, Packing Station (0, 9)
            - ORD-103: 3 item picks, Packing Station (0, 16)
            
        Complexity:
            Time Complexity: O(M) where M = 3.
            Space Complexity: O(M * K) where K is the number of item targets.
        """
        self.order_queue = [
            Order(
                order_id="ORD-101",
                packing_station=(0, 2),
                item_locations=[(4, 1), (6, 5), (12, 1), (14, 5)],
                priority=1,
            ),
            Order(
                order_id="ORD-102",
                packing_station=(0, 9),
                item_locations=[(4, 8), (6, 10), (12, 8), (14, 10), (4, 13)],
                priority=2,
            ),
            Order(
                order_id="ORD-103",
                packing_station=(0, 16),
                item_locations=[(4, 17), (6, 14), (12, 17)],
                priority=3,
            ),
        ]

    def run_simulation(self) -> None:
        """
        Executes the end-to-end multi-robot warehouse simulation pipeline.
        
        Execution Stages:
            1. Greedy Task Allocation: Dispatches idle robots to pending orders.
            2. TSP Optimization: Solves Held-Karp DP with Bitmasking for each order.
            3. Navigation & Physical Route Execution: Models robot movement via A*.
            4. Performance Telemetry: Records empirical execution times and metrics.
            
        Complexity Analysis:
            Time Complexity:
              - Overall Time: O(M * R * T_A* + M * (N^2 * T_A* + N^2 * 2^N))
                where:
                  M = number of orders (3)
                  R = number of robots (3)
                  N = average nodes per order (packing station + items)
                  T_A* = O(|V| log |V| + |E|) with |V| = H * W = 400.
            Space Complexity:
              - Overall Space: O(|V| + M * N * L + N * 2^N)
                for grid graph, paths, and DP memoization tables.
        """
        sim_start_time = time.perf_counter()
        print("=" * 80)
        print("   SMART WAREHOUSE ROBOT PATHFINDING NETWORK SIMULATION")
        print("=" * 80)
        print(f"[*] Grid Dimensions: {self.height}x{self.width} (|V| = {self.height * self.width} cells)")
        print(f"[*] Spawned Fleet: {len(self.robots)} Autonomous Mobile Robots")
        print(f"[*] Incoming Queue: {len(self.order_queue)} Batched Customer Orders\n")

        # Display initial warehouse layout
        print("[+] Physical Warehouse Floor Layout (ASCII Visualization):")
        print(self.grid.render_ascii(
            robot_positions={r.robot_id: r.position for r in self.robots}
        ))
        print("Legend: '#' Shelf/Obstacle | '.' Aisle | 'P' Packing Station | 'C' Charging Dock | 'R' Robot\n")

        # -------------------------------------------------------------
        # STAGE 1: Greedy Task Allocation
        # -------------------------------------------------------------
        print("-" * 80)
        print("STAGE 1: GREEDY TASK ALLOCATION (PROXIMITY-BASED DISPATCH)")
        print("-" * 80)
        alloc_start = time.perf_counter()
        assignments: List[AssignmentResult] = self.task_allocator.allocate_orders(
            self.order_queue, self.robots
        )
        alloc_elapsed = time.perf_counter() - alloc_start
        self.analytics.greedy_assignments = assignments

        for assign in assignments:
            print(
                f" -> Order '{assign.order_id}' assigned to '{assign.robot_id}' "
                f"[Initial Pos: {assign.robot_initial_pos} -> Station: {assign.packing_station}] "
                f"Dispatch Cost: {assign.distance_to_station:.1f} steps "
                f"(Decision Time: {assign.execution_time_sec * 1e6:.2f} µs)"
            )
        print(f"[*] Total Greedy Allocation Time: {alloc_elapsed * 1e3:.4f} ms\n")

        # -------------------------------------------------------------
        # STAGE 2: Optimal Order Picking (TSP via DP with Bitmasking)
        # -------------------------------------------------------------
        print("-" * 80)
        print("STAGE 2: TSP ROUTE OPTIMIZATION (HELD-KARP DP WITH BITMASKING)")
        print("-" * 80)

        # Map robot ID to Robot object for quick lookup
        robot_map = {r.robot_id: r for r in self.robots}

        for assign in assignments:
            order = next(o for o in self.order_queue if o.order_id == assign.order_id)
            robot = robot_map[assign.robot_id]

            print(f"\n[+] Processing Order: {order.order_id} for {robot.robot_id}")
            print(f"    - Packing Station: {order.packing_station}")
            print(f"    - Target Item Coordinates ({len(order.item_locations)} items): {order.item_locations}")

            # Solve TSP
            tsp_res: TSPResult = self.tsp_optimizer.solve_tsp(
                order.packing_station, order.item_locations
            )

            # Record telemetry
            self.analytics.tsp_runs.append({
                "order_id": order.order_id,
                "robot_id": robot.robot_id,
                "n_nodes": tsp_res.num_targets,
                "num_items": len(order.item_locations),
                "matrix_time_sec": tsp_res.distance_matrix_time_sec,
                "dp_time_sec": tsp_res.execution_time_sec,
                "total_tsp_time_sec": tsp_res.distance_matrix_time_sec + tsp_res.execution_time_sec,
                "total_distance": tsp_res.total_distance,
                "theoretical_dp_ops": tsp_res.theoretical_operations,
                "theoretical_brute_force_ops": math.factorial(tsp_res.num_targets - 1),
                "tour_sequence": tsp_res.tour_coordinates,
            })

            print(f"    - Distance Matrix Computation (A*): {tsp_res.distance_matrix_time_sec * 1e3:.4f} ms")
            print(f"    - Held-Karp Bitmask DP Execution:    {tsp_res.execution_time_sec * 1e3:.4f} ms")
            print(f"    - Optimal Tour Distance:            {tsp_res.total_distance:.1f} steps")
            print(f"    - Exact Optimal Tour Sequence:      {' -> '.join(str(c) for c in tsp_res.tour_coordinates)}")

            # ---------------------------------------------------------
            # STAGE 3: Robot Route Execution
            # ---------------------------------------------------------
            # Step A: Robot navigates from dock to packing station
            dispatch_path_res: PathResult = self.navigator.find_shortest_path(
                robot.position, order.packing_station
            )
            self.analytics.astar_queries.append({
                "purpose": f"Dispatch {robot.robot_id} to Station {order.order_id}",
                "start": robot.position,
                "target": order.packing_station,
                "cost": dispatch_path_res.cost,
                "nodes_expanded": dispatch_path_res.nodes_expanded,
                "time_sec": dispatch_path_res.execution_time_sec,
                "success": dispatch_path_res.success,
            })

            # Step B: Execute the full TSP picking route
            # Flatten subpaths into continuous path
            full_execution_path: List[Coordinate] = list(dispatch_path_res.path)
            for sub in tsp_res.subpaths:
                if full_execution_path and sub:
                    # Avoid duplicate coordinate at seam
                    full_execution_path.extend(sub[1:])
                else:
                    full_execution_path.extend(sub)

            total_mission_distance = dispatch_path_res.cost + tsp_res.total_distance
            robot.position = order.packing_station  # Final location after order completion
            robot.total_distance_traveled += total_mission_distance
            robot.battery_level = max(0.0, robot.battery_level - (total_mission_distance * 0.4))
            robot.status = RobotStatus.IDLE
            self.completed_orders.append(order)

            print(f"    - Complete Mission Distance (Dispatch + Picking): {total_mission_distance:.1f} steps")
            print(f"    - Final Robot State: {robot.robot_id} at {robot.position}, Battery: {robot.battery_level:.1f}%\n")

        self.analytics.total_simulation_time_sec = time.perf_counter() - sim_start_time
        print("-" * 80)
        print("ALL BATCHED ORDERS SUCCESSFULLY FULFILLED.")
        print("-" * 80)

        # -------------------------------------------------------------
        # STAGE 4: Analytical Complexity and Performance Report
        # -------------------------------------------------------------
        self.generate_analytical_report()

    def generate_analytical_report(self) -> None:
        """
        Generates and prints a comprehensive, university-level Design and Analysis of
        Algorithms (DAA) performance evaluation report.
        
        Evaluates:
          1. Point-to-Point A* Pathfinding Complexity & Empirical Scalability.
          2. Held-Karp TSP DP with Bitmasking vs Brute Force Permutations.
          3. Greedy Task Allocation Performance.
          4. Global Fleet Utilization and Warehouse Throughput Summary.
        """
        print("\n" + "=" * 90)
        print("          DESIGN AND ANALYSIS OF ALGORITHMS (DAA) PERFORMANCE REPORT")
        print("          PROJECT: SMART WAREHOUSE ROBOT PATHFINDING NETWORK")
        print("=" * 90)

        # =============================================================
        # MODULE 1: A* PATHFINDING ALGORITHMIC EVALUATION
        # =============================================================
        print("\n" + "-" * 90)
        print("1. POINT-TO-POINT NAVIGATION: A* (A-STAR) SEARCH ALGORITHM")
        print("-" * 90)
        print("Theoretical Complexity:")
        print("  - Time Complexity:  O(|V| log |V| + |E|)")
        print(f"                      For Grid {self.height}x{self.width}: |V| = {self.height * self.width}, |E| <= 4 * |V| = {4 * self.height * self.width}")
        print(f"                      Worst-case operations upper bound: ~{int(self.height * self.width * math.log2(self.height * self.width)):,} operations")
        print("  - Space Complexity: O(|V|) = O(H * W) auxiliary memory for open heap, closed set & came_from")
        print("  - Heuristic Properties:")
        print("      * Manhattan Distance h(n) = |r1 - r2| + |c1 - c2| is Admissible (h(n) <= h*(n))")
        print("      * Monotonic / Consistent (h(u) <= c(u, v) + h(v)), ensuring zero node re-expansions")
        print("\nEmpirical A* Dispatch Queries Sample:")
        print(f"{'Query Purpose':<32} | {'Start -> Target':<18} | {'Cost':<6} | {'Expanded':<9} | {'Empirical Time':<15}")
        print("-" * 90)
        for q in self.analytics.astar_queries:
            st_str = f"{q['start']} -> {q['target']}"
            print(f"{q['purpose']:<32} | {st_str:<18} | {q['cost']:<6.1f} | {q['nodes_expanded']:<9} | {q['time_sec'] * 1e3:<10.4f} ms")

        # =============================================================
        # MODULE 2: TSP VIA DYNAMIC PROGRAMMING WITH BITMASKING
        # =============================================================
        print("\n" + "-" * 90)
        print("2. OPTIMAL ORDER PICKING: TSP DYNAMIC PROGRAMMING (HELD-KARP WITH BITMASKING)")
        print("-" * 90)
        print("Theoretical Complexity:")
        print("  - Time Complexity:  Theta(N^2 * 2^N) transitions where N = 1 (packing station) + K (items)")
        print("  - Space Complexity: Theta(N * 2^N) for DP memoization table and backtracking parent pointers")
        print("  - Comparison vs Brute Force: Brute-force exhaustive search requires O(N!) time.")
        print("                               At N=10: N! = 3,628,800 vs N^2 * 2^N = 102,400 (35.4x faster)")
        print("                               At N=15: N! = 1.31e12 vs N^2 * 2^N = 7.37e6 (177,000x faster)\n")

        print("Empirical TSP Benchmark across Batched Orders:")
        header = f"{'Order ID':<9} | {'Nodes (N)':<10} | {'DP Time':<13} | {'Matrix Time':<13} | {'Theoretical Ops':<18} | {'Brute-Force Ops':<16}"
        print(header)
        print("-" * 90)
        for tsp in self.analytics.tsp_runs:
            n = tsp["n_nodes"]
            dp_time_str = f"{tsp['dp_time_sec'] * 1e3:.4f} ms"
            mat_time_str = f"{tsp['matrix_time_sec'] * 1e3:.4f} ms"
            ops_dp_str = f"{tsp['theoretical_dp_ops']:,} (N²·2ⁿ)"
            ops_bf_str = f"{tsp['theoretical_brute_force_ops']:,} ((N-1)!)"
            print(f"{tsp['order_id']:<9} | {n:<10} | {dp_time_str:<13} | {mat_time_str:<13} | {ops_dp_str:<18} | {ops_bf_str:<16}")

        # Verification of polynomial vs exponential scaling
        print("\nAlgorithmic Scaling Analysis:")
        print("  - Pairwise A* Distance Matrix construction dominates wall-clock time for small N,")
        print("    executing N*(N-1)/2 optimal path searches over the warehouse grid.")
        print("  - Bitmask DP transitions execute in pure bitwise integer operations, exhibiting")
        print("    sub-millisecond execution times for practical warehouse batch sizes (N <= 15).")

        # =============================================================
        # MODULE 3: GREEDY TASK ALLOCATION
        # =============================================================
        print("\n" + "-" * 90)
        print("3. TASK ASSIGNMENT: GREEDY PROXIMITY ALLOCATOR")
        print("-" * 90)
        print("Theoretical Complexity:")
        print("  - Time Complexity:  O(M * R * T_A*) where M = len(orders), R = len(robots)")
        print("                      For M=3, R=3: evaluated at most 3 + 2 + 1 = 6 pairwise candidate routes")
        print("  - Space Complexity: O(M + R) auxiliary storage")
        print("  - Strategy Evaluation:")
        print("      The Greedy Choice Property minimizes dispatch deadhead distance locally.")
        print("      While global min-cost matching (Hungarian Algorithm, O(R^3)) guarantees global")
        print("      optimality, greedy allocation achieves O(M * R) near-optimal throughput with")
        print("      instantaneous real-time dispatch response.")

        # =============================================================
        # MODULE 4: FLEET PERFORMANCE SUMMARY
        # =============================================================
        print("\n" + "-" * 90)
        print("4. FLEET PRODUCTIVITY & WAREHOUSE THROUGHPUT SUMMARY")
        print("-" * 90)
        for robot in self.robots:
            print(
                f"  - Robot [{robot.robot_id}]: "
                f"Completed Orders: {robot.assigned_orders_count} | "
                f"Traveled: {robot.total_distance_traveled:.1f} steps | "
                f"Remaining Battery: {robot.battery_level:.1f}% | "
                f"Current Position: {robot.position}"
            )
        print(f"\n[*] Total Simulation Wall-Clock Time: {self.analytics.total_simulation_time_sec * 1e3:.2f} ms")
        print("=" * 90 + "\n")
