"""
Greedy Task Allocation Module for Autonomous Robot Fleet.

Author: Python Engineer & DAA Expert
Project: Smart Warehouse Robot Pathfinding Network
"""

import time
from typing import List, Tuple, Optional

from warehouse.models import Robot, Order, RobotStatus, AssignmentResult
from warehouse.astar import AStarNavigator, manhattan_distance


class GreedyTaskAllocator:
    """
    Allocates incoming warehouse customer orders to idle autonomous robots
    using a Greedy Optimization strategy based on minimum dispatch distance.
    
    Greedy Choice Property:
        At each assignment step, the algorithm matches an unassigned order to the
        currently idle robot that has the minimum travel distance to the order's
        starting packing station depot. This locally optimal choice minimizes dispatch
        latency and maximizes fleet throughput.
        
    Attributes:
        navigator: AStarNavigator instance for computing exact obstacle-aware
                   navigation distances between robots and packing stations.
        use_astar_distance: If True, uses A* to calculate true obstacle-aware path
                            distance. If False, uses Manhattan distance for O(1) checks.
    """

    def __init__(self, navigator: AStarNavigator, use_astar_distance: bool = True) -> None:
        """
        Initializes the greedy allocator with a pathfinding navigator.
        
        Args:
            navigator: AStarNavigator instance configured with warehouse grid.
            use_astar_distance: Whether to evaluate true A* distance (default True).
        """
        self.navigator: AStarNavigator = navigator
        self.use_astar_distance: bool = use_astar_distance

    def compute_distance(self, robot_pos: Tuple[int, int], station_pos: Tuple[int, int]) -> float:
        """
        Evaluates the distance between a robot and a packing station.
        
        Args:
            robot_pos: Current (row, col) position of robot.
            station_pos: (row, col) position of packing station.
            
        Returns:
            Computed distance as float (or inf if unreachable).
            
        Complexity:
            If A* enabled: O(|V| log |V| + |E|)
            If Manhattan enabled: O(1)
        """
        if self.use_astar_distance:
            res = self.navigator.find_shortest_path(robot_pos, station_pos)
            return res.cost if res.success else float("inf")
        return manhattan_distance(robot_pos, station_pos)

    def allocate_orders(
        self, order_queue: List[Order], available_robots: List[Robot]
    ) -> List[AssignmentResult]:
        """
        Executes the greedy matching algorithm between pending orders and idle robots.
        
        Algorithmic Steps:
            1. Filter the robot fleet to identify currently IDLE units.
            2. Iterate through the queue of incoming orders in FIFO/priority sequence.
            3. For each order, evaluate distance from every remaining idle robot to
               the order's designated packing station.
            4. Select the candidate robot that minimizes this distance (Greedy Choice).
               Tie-breaker: higher battery level or lexicographical robot ID.
            5. Transition chosen robot to ASSIGNED, record assignment, and remove
               from available pool for subsequent orders in this batch.
            6. Repeat until orders are satisfied or no idle robots remain.
            
        Args:
            order_queue: List of unassigned Order objects.
            available_robots: List of Robot objects in the warehouse fleet.
            
        Returns:
            List of AssignmentResult records detailing each allocation.
            
        Complexity Analysis:
            Time Complexity:
              Let M = len(order_queue) and R = len(idle_robots).
              At step i (0 <= i < min(M, R)), there are (R - i) robots evaluated.
              Total distance evaluations = sum_{i=0}^{min(M, R)-1} (R - i) <= M * R.
              - If using A* distance:
                Time = O(M * R * (|V| log |V| + |E|))
              - If using Manhattan heuristic:
                Time = O(M * R)
                
            Space Complexity:
              O(M + R) auxiliary memory to maintain tracking lists and output records.
        """
        alloc_start_time = time.perf_counter()
        assignments: List[AssignmentResult] = []

        # Filter strictly idle robots
        idle_pool: List[Robot] = [r for r in available_robots if r.status == RobotStatus.IDLE]

        for order in order_queue:
            # If no robots are idle, remaining orders must wait in queue
            if not idle_pool:
                break

            best_robot: Optional[Robot] = None
            min_dist: float = float("inf")

            # Evaluate each idle candidate robot against order's packing station
            for robot in idle_pool:
                dist = self.compute_distance(robot.position, order.packing_station)

                if dist < min_dist:
                    min_dist = dist
                    best_robot = robot
                elif dist == min_dist and best_robot is not None:
                    # Tie-breaking: prefer robot with higher battery
                    if robot.battery_level > best_robot.battery_level:
                        best_robot = robot

            # If a reachable robot was found, finalize the greedy assignment
            if best_robot is not None and min_dist < float("inf"):
                elapsed = time.perf_counter() - alloc_start_time

                # Update robot state
                best_robot.status = RobotStatus.ASSIGNED
                best_robot.current_order = order
                best_robot.assigned_orders_count += 1
                order.assigned_robot_id = best_robot.robot_id

                assignment = AssignmentResult(
                    order_id=order.order_id,
                    robot_id=best_robot.robot_id,
                    robot_initial_pos=best_robot.position,
                    packing_station=order.packing_station,
                    distance_to_station=min_dist,
                    execution_time_sec=elapsed
                )
                assignments.append(assignment)

                # Remove assigned robot from pool for subsequent orders
                idle_pool.remove(best_robot)

        return assignments
