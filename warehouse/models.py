"""
Data models and type definitions for the Smart Warehouse Robot Pathfinding Network.

Author: Python Engineer & DAA Expert
Project: Smart Warehouse Robot Pathfinding Network
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Tuple, Dict, Any, Optional

# Type alias for 2D grid coordinates: (row, col)
Coordinate = Tuple[int, int]


class CellType(Enum):
    """Enumeration of cell types in the warehouse grid."""
    AISLE = 0           # Navigable passage
    SHELF = 1           # Impassable storage rack / obstacle
    PACKING_STATION = 2 # Order fulfillment / start-end depot
    CHARGING_DOCK = 3   # Robot idle parking / charging dock


class RobotStatus(Enum):
    """Operational states of a warehouse robot."""
    IDLE = "IDLE"
    ASSIGNED = "ASSIGNED"
    NAVIGATING = "NAVIGATING"
    PICKING = "PICKING"
    RETURNING = "RETURNING"
    MAINTENANCE = "MAINTENANCE"


@dataclass
class PathResult:
    """
    Encapsulates the result of an A* pathfinding search.
    
    Attributes:
        path: List of coordinates from start to target inclusive.
        cost: Total path distance (number of unit steps).
        nodes_expanded: Total number of nodes popped from priority queue.
        execution_time_sec: Wall-clock execution time in seconds.
        success: Boolean flag indicating whether a valid path was found.
    """
    path: List[Coordinate]
    cost: float
    nodes_expanded: int
    execution_time_sec: float
    success: bool = True


@dataclass
class TSPResult:
    """
    Encapsulates the result of TSP Dynamic Programming with Bitmasking.
    
    Attributes:
        tour_indices: Indices of nodes in visit sequence (starting and ending at 0).
        tour_coordinates: Sequence of grid coordinates visited.
        total_distance: Total cost of the closed tour.
        subpaths: List of A* path lists connecting successive tour stops.
        execution_time_sec: Wall-clock execution time of the DP algorithm.
        distance_matrix_time_sec: Wall-clock execution time to compute pairwise A* paths.
        num_targets: Number of locations in TSP instance (including depot).
        theoretical_operations: Estimated theoretical operations (N^2 * 2^N).
    """
    tour_indices: List[int]
    tour_coordinates: List[Coordinate]
    total_distance: float
    subpaths: List[List[Coordinate]]
    execution_time_sec: float
    distance_matrix_time_sec: float
    num_targets: int
    theoretical_operations: int


@dataclass
class Order:
    """
    Represents a warehouse customer order with required pick items.
    
    Attributes:
        order_id: Unique string identifier (e.g., 'ORD-001').
        packing_station: Coordinate of depot where the order starts and ends.
        item_locations: List of coordinate pick targets in the warehouse.
        priority: Priority value (lower is higher priority; default 1).
        assigned_robot_id: ID of the robot assigned to this order, if any.
    """
    order_id: str
    packing_station: Coordinate
    item_locations: List[Coordinate]
    priority: int = 1
    assigned_robot_id: Optional[str] = None


@dataclass
class Robot:
    """
    Represents an autonomous mobile robot (AMR) in the warehouse fleet.
    
    Attributes:
        robot_id: Unique identifier (e.g., 'ROBOT-01').
        position: Current (row, col) grid coordinate.
        status: Current operational status.
        battery_level: Battery percentage (0.0 to 100.0).
        current_order: Active Order object if currently working.
        total_distance_traveled: Lifetime distance accumulated in steps.
        assigned_orders_count: Number of completed or ongoing orders.
    """
    robot_id: str
    position: Coordinate
    status: RobotStatus = RobotStatus.IDLE
    battery_level: float = 100.0
    current_order: Optional[Order] = None
    total_distance_traveled: float = 0.0
    assigned_orders_count: int = 0


@dataclass
class AssignmentResult:
    """
    Result of a greedy task assignment step.
    
    Attributes:
        order_id: ID of assigned order.
        robot_id: ID of assigned robot.
        robot_initial_pos: Starting position of assigned robot.
        packing_station: Order packing station position.
        distance_to_station: Distance from robot to packing station.
        execution_time_sec: Time taken for greedy evaluation.
    """
    order_id: str
    robot_id: str
    robot_initial_pos: Coordinate
    packing_station: Coordinate
    distance_to_station: float
    execution_time_sec: float


@dataclass
class SimulationAnalytics:
    """
    Aggregates performance analytics for the entire simulation run.
    """
    astar_queries: List[Dict[str, Any]] = field(default_factory=list)
    tsp_runs: List[Dict[str, Any]] = field(default_factory=list)
    greedy_assignments: List[AssignmentResult] = field(default_factory=list)
    total_simulation_time_sec: float = 0.0
