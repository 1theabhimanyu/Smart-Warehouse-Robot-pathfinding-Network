"""
Warehouse Grid Representation and Obstacle Management.

Author: Python Engineer & DAA Expert
Project: Smart Warehouse Robot Pathfinding Network
"""

from typing import List, Tuple, Optional, Set, Dict
from warehouse.models import Coordinate, CellType


class WarehouseGrid:
    """
    Models a 2D discrete grid topology representing the physical warehouse floor.
    
    In this model:
      - 0: Navigable aisle (free for robot movement)
      - 1: Impassable shelf rack or physical static obstacle
    
    The coordinate system uses (row, col) indexing:
      - row: 0 to height - 1 (vertical axis, top to bottom)
      - col: 0 to width - 1  (horizontal axis, left to right)
    
    Attributes:
        height: Total number of rows in the grid (e.g., 20).
        width: Total number of columns in the grid (e.g., 20).
        grid: 2D integer matrix where grid[r][c] in {0, 1}.
        packing_stations: List of designated packing station coordinates.
        charging_docks: List of robot depot / charging dock coordinates.
    """

    def __init__(self, height: int = 20, width: int = 20) -> None:
        """
        Initializes an empty warehouse grid with all cells set to navigable aisles (0).
        
        Args:
            height: Number of rows in grid (default: 20).
            width: Number of columns in grid (default: 20).
            
        Complexity:
            Time Complexity: O(height * width) to allocate the 2D array.
            Space Complexity: O(height * width) auxiliary memory for matrix.
        """
        self.height: int = height
        self.width: int = width
        # Initialize full 2D matrix with navigable aisles (0)
        self.grid: List[List[int]] = [[0 for _ in range(width)] for _ in range(height)]
        self.packing_stations: List[Coordinate] = []
        self.charging_docks: List[Coordinate] = []

    def is_in_bounds(self, coord: Coordinate) -> bool:
        """
        Checks whether a coordinate lies within grid boundaries.
        
        Args:
            coord: (row, col) coordinate tuple.
            
        Returns:
            True if 0 <= row < height and 0 <= col < width; False otherwise.
            
        Complexity:
            Time Complexity: O(1)
            Space Complexity: O(1)
        """
        r, c = coord
        return 0 <= r < self.height and 0 <= c < self.width

    def is_navigable(self, coord: Coordinate) -> bool:
        """
        Determines whether a cell can be traversed by a robot.
        
        A cell is navigable if and only if it is within grid bounds
        and its value is 0 (aisle).
        
        Args:
            coord: (row, col) coordinate tuple.
            
        Returns:
            True if cell is traversable; False if impassable (1) or out of bounds.
            
        Complexity:
            Time Complexity: O(1)
            Space Complexity: O(1)
        """
        if not self.is_in_bounds(coord):
            return False
        r, c = coord
        return self.grid[r][c] == 0

    def set_obstacle(self, coord: Coordinate) -> None:
        """
        Marks a specific coordinate as an impassable shelf/obstacle (1).
        
        Args:
            coord: (row, col) coordinate.
        """
        if self.is_in_bounds(coord):
            r, c = coord
            self.grid[r][c] = 1

    def set_aisle(self, coord: Coordinate) -> None:
        """
        Restores a coordinate to a navigable aisle (0).
        
        Args:
            coord: (row, col) coordinate.
        """
        if self.is_in_bounds(coord):
            r, c = coord
            self.grid[r][c] = 0

    def add_shelf_block(self, top_left: Coordinate, height: int, width: int) -> None:
        """
        Places a rectangular block of shelf obstacles into the warehouse grid.
        
        Args:
            top_left: (row, col) of top-left corner of the shelf.
            height: Vertical dimension (in grid cells) of the shelf.
            width: Horizontal dimension (in grid cells) of the shelf.
            
        Complexity:
            Time Complexity: O(height * width)
            Space Complexity: O(1)
        """
        start_r, start_c = top_left
        for r in range(start_r, start_r + height):
            for c in range(start_c, start_c + width):
                self.set_obstacle((r, c))

    def get_neighbors(self, coord: Coordinate) -> List[Coordinate]:
        """
        Retrieves all valid 4-directional navigable neighboring coordinates.
        Movement is restricted to orthogonal Manhattan steps:
        North (-1, 0), South (+1, 0), West (0, -1), East (0, +1).
        
        Args:
            coord: Current (row, col) coordinate.
            
        Returns:
            List of navigable neighboring (row, col) coordinates.
            
        Complexity:
            Time Complexity: O(1) (at most 4 direction checks)
            Space Complexity: O(1) (returns at most 4 items)
        """
        r, c = coord
        # 4-connected orthogonal vectors: (dr, dc)
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        valid_neighbors: List[Coordinate] = []

        for dr, dc in directions:
            neighbor: Coordinate = (r + dr, c + dc)
            if self.is_navigable(neighbor):
                valid_neighbors.append(neighbor)

        return valid_neighbors

    def setup_realistic_warehouse_layout(self) -> None:
        """
        Configures an authentic, industry-standard automated warehouse floor layout:
        - Dimension: 20x20 (or scaled dynamically to self.height x self.width)
        - Northern Edge (Row 0): Main cross-aisle and designated Packing Stations
        - Southern Edge (Row height-1): Maintenance docks and Charging Stations for robots
        - Intermediate thoroughfares: Central horizontal highway for traffic flow
        - Storage Aisles: Parallel shelf racks organized in modular pods with picking aisles
        
        Complexity:
            Time Complexity: O(H * W)
            Space Complexity: O(1) auxiliary
        """
        # Clear any prior obstacles
        self.grid = [[0 for _ in range(self.width)] for _ in range(self.height)]
        
        last_row = self.height - 1
        if self.height == 20 and self.width == 20:
            # Canonical 20x20 Academic Layout
            self.packing_stations = [(0, 2), (0, 9), (0, 16)]
            self.charging_docks = [(19, 2), (19, 9), (19, 16)]

            # Pod Group 1 (Upper Warehouse: Rows 3 to 7)
            self.add_shelf_block((3, 2), height=5, width=2)
            self.add_shelf_block((3, 6), height=5, width=2)
            self.add_shelf_block((3, 11), height=5, width=2)
            self.add_shelf_block((3, 15), height=5, width=2)

            # Pod Group 2 (Lower Warehouse: Rows 11 to 15)
            self.add_shelf_block((11, 2), height=5, width=2)
            self.add_shelf_block((11, 6), height=5, width=2)
            self.add_shelf_block((11, 11), height=5, width=2)
            self.add_shelf_block((11, 15), height=5, width=2)
        else:
            # Dynamically compute columns for docks and stations within valid bounds
            c1 = max(1, min(2, self.width - 2))
            c2 = max(c1 + 2, min(self.width // 2, self.width - 2))
            c3 = max(c2 + 2, min(self.width - 3, self.width - 1))
            cand_cols = sorted(list(dict.fromkeys([c1, c2, c3])))
            while len(cand_cols) < 3 and len(cand_cols) < self.width:
                for c in range(self.width):
                    if c not in cand_cols:
                        cand_cols.append(c)
                        break
                cand_cols = sorted(cand_cols)

            self.packing_stations = [(0, c) for c in cand_cols[:3]]
            self.charging_docks = [(last_row, c) for c in cand_cols[:3]]

            # Dynamic modular storage shelves
            pod_h = max(2, min(5, (self.height - 7) // 2))
            if pod_h > 0 and self.height >= 8:
                r1 = 2
                for sc in [2, 6, 10, 14]:
                    if sc + 1 < self.width:
                        self.add_shelf_block((r1, sc), height=pod_h, width=2)

                r2 = r1 + pod_h + 2
                if r2 + pod_h < self.height - 1:
                    for sc in [2, 6, 10, 14]:
                        if sc + 1 < self.width:
                            self.add_shelf_block((r2, sc), height=pod_h, width=2)

        # Ensure all packing stations and charging docks remain navigable
        for ps in self.packing_stations:
            self.set_aisle(ps)
        for cd in self.charging_docks:
            self.set_aisle(cd)

    def render_ascii(
        self,
        robot_positions: Optional[Dict[str, Coordinate]] = None,
        path: Optional[List[Coordinate]] = None,
        pick_targets: Optional[List[Coordinate]] = None,
        packing_station: Optional[Coordinate] = None,
    ) -> str:
        """
        Renders a formatted ASCII visual map of the warehouse grid.
        
        Symbols:
            '#' : Shelf / Obstacle
            '.' : Navigable Aisle
            'P' : Packing Station
            'C' : Charging Dock
            'R' : Robot Position
            '*' : Item Pick Target
            '+' : Path step
        
        Args:
            robot_positions: Mapping of robot_id to (row, col).
            path: List of coordinates in active navigation path.
            pick_targets: List of target coordinates for order picking.
            packing_station: Specific packing station for the active order.
            
        Returns:
            Multi-line string representation of the grid.
        """
        robot_coords = set(robot_positions.values()) if robot_positions else set()
        path_coords = set(path) if path else set()
        targets = set(pick_targets) if pick_targets else set()

        lines: List[str] = []
        # Header with column indices
        col_header = "    " + "".join(f"{c%10} " for c in range(self.width))
        lines.append(col_header)
        lines.append("   " + "--" * self.width)

        for r in range(self.height):
            row_chars: List[str] = []
            for c in range(self.width):
                coord = (r, c)
                if coord in robot_coords:
                    row_chars.append("R ")
                elif packing_station and coord == packing_station:
                    row_chars.append("P ")
                elif coord in self.packing_stations:
                    row_chars.append("P ")
                elif coord in targets:
                    row_chars.append("* ")
                elif coord in path_coords:
                    row_chars.append("+ ")
                elif coord in self.charging_docks:
                    row_chars.append("C ")
                elif self.grid[r][c] == 1:
                    row_chars.append("# ")
                else:
                    row_chars.append(". ")
            lines.append(f"{r:2d}| " + "".join(row_chars))

        return "\n".join(lines)
