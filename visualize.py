"""
Warehouse Layout and Route Plotting Utility using Matplotlib.

Author: Python Engineer & DAA Expert
Project: Smart Warehouse Robot Pathfinding Network
"""

import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from typing import List, Tuple

from warehouse.simulation import WarehouseSimulation
from warehouse.models import Coordinate


def generate_warehouse_plot(output_filepath: str = "warehouse_simulation_map.png") -> None:
    """
    Renders a 2D graphical representation of the warehouse floor,
    including shelf obstacles, robot positions, and TSP order picking routes.
    
    Args:
        output_filepath: Destination image path.
    """
    sim = WarehouseSimulation(height=20, width=20)
    sim.spawn_default_fleet()
    sim.queue_default_orders()

    # Pre-calculate assignments and TSP routes
    assignments = sim.task_allocator.allocate_orders(sim.order_queue, sim.robots)

    fig, ax = plt.subplots(figsize=(10, 10), dpi=150)
    grid_data = np.zeros((sim.height, sim.width))

    # Fill shelf obstacles
    for r in range(sim.height):
        for c in range(sim.width):
            if sim.grid.grid[r][c] == 1:
                grid_data[r, c] = 1.0

    # Draw grid background
    ax.imshow(grid_data, cmap="Greys", alpha=0.3, origin="upper")

    # Draw grid lines
    ax.set_xticks(np.arange(-0.5, sim.width, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, sim.height, 1), minor=True)
    ax.grid(which="minor", color="lightgray", linestyle="-", linewidth=0.5)
    ax.tick_params(which="minor", size=0)

    # Shelf rack patches
    for r in range(sim.height):
        for c in range(sim.width):
            if sim.grid.grid[r][c] == 1:
                rect = patches.Rectangle(
                    (c - 0.45, r - 0.45), 0.9, 0.9,
                    linewidth=1, edgecolor="#333333", facecolor="#4A5568"
                )
                ax.add_patch(rect)

    # Plot Packing Stations
    for idx, ps in enumerate(sim.grid.packing_stations):
        ax.scatter(ps[1], ps[0], color="#E53E3E", s=200, marker="s", zorder=5, label="Packing Station" if idx == 0 else "")
        ax.text(ps[1], ps[0] - 0.6, f"Station {idx+1}", ha="center", fontsize=8, fontweight="bold", color="#C53030")

    # Plot Charging Docks
    for idx, cd in enumerate(sim.grid.charging_docks):
        ax.scatter(cd[1], cd[0], color="#3182CE", s=200, marker="^", zorder=5, label="Charging Dock" if idx == 0 else "")
        ax.text(cd[1], cd[0] + 0.7, f"Dock {idx+1}", ha="center", fontsize=8, fontweight="bold", color="#2B6CB0")

    # Plot Routes for each assigned robot
    colors = ["#D69E2E", "#805AD5", "#38A169"]
    robot_map = {r.robot_id: r for r in sim.robots}

    for idx, assign in enumerate(assignments):
        order = next(o for o in sim.order_queue if o.order_id == assign.order_id)
        robot = robot_map[assign.robot_id]
        color = colors[idx % len(colors)]

        # Dispatch path
        disp = sim.navigator.find_shortest_path(assign.robot_initial_pos, order.packing_station)
        if disp.path:
            dc_y, dc_x = zip(*disp.path)
            ax.plot(dc_x, dc_y, color=color, linestyle=":", linewidth=2, alpha=0.7, label=f"{robot.robot_id} Dispatch")

        # TSP route
        tsp_res = sim.tsp_optimizer.solve_tsp(order.packing_station, order.item_locations)
        for sub in tsp_res.subpaths:
            if sub:
                sub_y, sub_x = zip(*sub)
                ax.plot(sub_x, sub_y, color=color, linestyle="-", linewidth=2.5, alpha=0.9)

        # Plot pick items
        for item in order.item_locations:
            ax.scatter(item[1], item[0], color=color, s=120, edgecolors="black", marker="o", zorder=6)
            ax.text(item[1] + 0.3, item[0] - 0.2, f"Pick", fontsize=7, color="#1A202C", fontweight="bold")

    ax.set_xlim(-0.5, sim.width - 0.5)
    ax.set_ylim(sim.height - 0.5, -0.5)
    ax.set_title("Smart Warehouse Robot Pathfinding Network\nMulti-Robot TSP Picking Paths & Dispatch", fontsize=13, fontweight="bold", pad=15)
    ax.set_xlabel("Warehouse Grid Column Index (X)", fontsize=10)
    ax.set_ylabel("Warehouse Grid Row Index (Y)", fontsize=10)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.0), fontsize=8)

    plt.tight_layout()
    plt.savefig(output_filepath, bbox_inches="tight")
    plt.close()
    print(f"[+] Graphical warehouse map successfully saved to: {output_filepath}")


if __name__ == "__main__":
    generate_warehouse_plot()
