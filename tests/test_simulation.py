"""
Unit and Integration Test Suite for Smart Warehouse Robot Pathfinding Network.

Author: Senior Python Engineer & DAA Expert
Project: Smart Warehouse Robot Pathfinding Network
"""

import unittest
import itertools
import math
from typing import List, Tuple

from warehouse.models import (
    Coordinate,
    Robot,
    Order,
    RobotStatus,
    PathResult,
    TSPResult,
)
from warehouse.grid import WarehouseGrid
from warehouse.astar import AStarNavigator, manhattan_distance
from warehouse.tsp import TSPOptimizer
from warehouse.task_allocator import GreedyTaskAllocator
from warehouse.simulation import WarehouseSimulation


class TestManhattanDistance(unittest.TestCase):
    """Verifies properties of the L1 Manhattan distance metric."""

    def test_basic_distance(self):
        p1 = (0, 0)
        p2 = (3, 4)
        self.assertEqual(manhattan_distance(p1, p2), 7.0)

    def test_identity_and_symmetry(self):
        p1 = (5, 8)
        p2 = (12, 3)
        self.assertEqual(manhattan_distance(p1, p1), 0.0)
        self.assertEqual(manhattan_distance(p1, p2), manhattan_distance(p2, p1))


class TestAStarNavigation(unittest.TestCase):
    """Verifies correctness, optimality, and edge cases of the A* search engine."""

    def setUp(self):
        self.grid = WarehouseGrid(10, 10)
        self.nav = AStarNavigator(self.grid)

    def test_start_equals_target(self):
        res = self.nav.find_shortest_path((2, 2), (2, 2))
        self.assertTrue(res.success)
        self.assertEqual(res.cost, 0.0)
        self.assertEqual(res.path, [(2, 2)])

    def test_straight_line_obstacle_free(self):
        res = self.nav.find_shortest_path((0, 0), (0, 5))
        self.assertTrue(res.success)
        self.assertEqual(res.cost, 5.0)
        self.assertEqual(len(res.path), 6)
        self.assertEqual(res.path[0], (0, 0))
        self.assertEqual(res.path[-1], (0, 5))

    def test_obstacle_avoidance(self):
        # Place a vertical wall between col 2 at rows 0..4
        for r in range(5):
            self.grid.set_obstacle((r, 2))

        # Path from (2, 1) to (2, 3) must route around the wall (down to row 5 or up)
        res = self.nav.find_shortest_path((2, 1), (2, 3))
        self.assertTrue(res.success)
        # Verify no coordinate in path intersects an obstacle
        for coord in res.path:
            self.assertEqual(self.grid.grid[coord[0]][coord[1]], 0)
        self.assertEqual(res.path[0], (2, 1))
        self.assertEqual(res.path[-1], (2, 3))

    def test_unreachable_target(self):
        # Completely surround target with obstacles
        target = (5, 5)
        self.grid.set_obstacle((4, 5))
        self.grid.set_obstacle((6, 5))
        self.grid.set_obstacle((5, 4))
        self.grid.set_obstacle((5, 6))

        res = self.nav.find_shortest_path((0, 0), target)
        self.assertFalse(res.success)
        self.assertEqual(res.cost, float("inf"))
        self.assertEqual(res.path, [])


class TestTSPOptimizer(unittest.TestCase):
    """Verifies Held-Karp Dynamic Programming with Bitmasking against Brute Force."""

    def setUp(self):
        self.grid = WarehouseGrid(15, 15)
        self.nav = AStarNavigator(self.grid)
        self.tsp_opt = TSPOptimizer(self.nav)

    def test_single_item_tsp(self):
        station = (0, 0)
        items = [(0, 4)]
        res = self.tsp_opt.solve_tsp(station, items)
        self.assertEqual(res.total_distance, 8.0)
        self.assertEqual(res.tour_coordinates, [(0, 0), (0, 4), (0, 0)])

    def test_zero_items_tsp(self):
        station = (2, 2)
        res = self.tsp_opt.solve_tsp(station, [])
        self.assertEqual(res.total_distance, 0.0)
        self.assertEqual(res.tour_coordinates, [(2, 2)])

    def test_optimality_matches_brute_force(self):
        """Compares DP Bitmask result with exhaustive permutation brute force for N=5."""
        station = (0, 0)
        items = [(0, 3), (3, 3), (3, 0), (1, 1)]
        nodes = [station] + items
        n = len(nodes)

        # Solve with DP
        dp_res = self.tsp_opt.solve_tsp(station, items)

        # Solve with Exhaustive Brute Force
        dist_matrix, _, _ = self.tsp_opt.build_distance_matrix(nodes)
        min_bf_cost = float("inf")
        best_bf_perm = None

        # Permute non-station items
        item_indices = list(range(1, n))
        for perm in itertools.permutations(item_indices):
            tour = [0] + list(perm) + [0]
            cost = sum(dist_matrix[tour[i]][tour[i+1]] for i in range(len(tour) - 1))
            if cost < min_bf_cost:
                min_bf_cost = cost
                best_bf_perm = tour

        self.assertAlmostEqual(dp_res.total_distance, min_bf_cost, places=5)
        self.assertEqual(len(dp_res.tour_indices), n + 1)
        self.assertEqual(dp_res.tour_indices[0], 0)
        self.assertEqual(dp_res.tour_indices[-1], 0)


class TestGreedyTaskAllocator(unittest.TestCase):
    """Verifies greedy dispatch based on proximity."""

    def setUp(self):
        self.grid = WarehouseGrid(10, 10)
        self.nav = AStarNavigator(self.grid)
        self.allocator = GreedyTaskAllocator(self.nav, use_astar_distance=True)

    def test_closest_robot_assigned(self):
        robots = [
            Robot(robot_id="R-FAR", position=(9, 9), status=RobotStatus.IDLE),
            Robot(robot_id="R-CLOSE", position=(0, 1), status=RobotStatus.IDLE),
        ]
        orders = [
            Order(order_id="ORD-01", packing_station=(0, 0), item_locations=[(1, 1)])
        ]

        assignments = self.allocator.allocate_orders(orders, robots)
        self.assertEqual(len(assignments), 1)
        self.assertEqual(assignments[0].robot_id, "R-CLOSE")
        self.assertEqual(robots[1].status, RobotStatus.ASSIGNED)
        self.assertEqual(robots[0].status, RobotStatus.IDLE)


class TestEndToEndSimulation(unittest.TestCase):
    """Verifies that the complete simulation pipeline executes flawlessly."""

    def test_full_pipeline(self):
        sim = WarehouseSimulation(height=20, width=20)
        sim.spawn_default_fleet()
        sim.queue_default_orders()
        # Ensure it runs to completion without exceptions
        sim.run_simulation()
        self.assertEqual(len(sim.completed_orders), 3)
        self.assertGreater(sim.analytics.total_simulation_time_sec, 0.0)


if __name__ == "__main__":
    unittest.main()
