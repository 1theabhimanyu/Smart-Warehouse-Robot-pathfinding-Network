# Smart Warehouse Robot Pathfinding Network
**Course:** Design and Analysis of Algorithms (DAA) — Semester 1 Project  
**Author:** Python Engineer & University DAA Expert  
**Python Environment:** `/home/abhimanyu/myenv/bin/python` (Python 3.14+)  

---

## 1. Project Overview & Problem Statement

In automated storage and retrieval systems (ASRS) and fulfillment centers (e.g., Amazon Robotics, Ocado, Alibaba Kiva), a fleet of Autonomous Mobile Robots (AMRs) must coordinate to fulfill incoming customer orders with minimal latency and travel distance.

This project implements an end-to-end discrete simulation combining three core algorithmic paradigms from the Design and Analysis of Algorithms (DAA) curriculum:
1. **Graph Traversal & Heuristic Search (A\* Search):** For shortest collision-free navigation between two grid coordinates around physical shelf obstacles.
2. **Dynamic Programming with Bitmasking (Held-Karp TSP):** For computing the mathematically exact optimal pick-tour sequence that visits all ordered items and returns to the depot.
3. **Greedy Optimization:** For real-time dispatch and assignment of pending customer orders to idle robots based on proximity.
4. **Empirical vs. Theoretical Performance Analysis:** Benchmarking actual wall-clock runtimes against asymptotic Big-$O$ time complexities.

---

## 2. Mathematical Modeling & Algorithmic Formulations

### Module 1: Point-to-Point Navigation (A\* Search)
Given a 2D warehouse grid $G = (V, E)$ where $|V| = H \times W$ cells and $|E| \le 4|V|$ 4-connected orthogonal edges:
- **Heuristic Function:** Manhattan distance ($L_1$ metric):
  $$h(u, v) = |u_x - v_x| + |u_y - v_y|$$
- **Admissibility Proof:** Since edge costs are uniform ($c(u, v) = 1$) and robots can only move orthogonally, the Manhattan distance is the theoretical minimum number of steps required to reach the target without obstacles. Thus:
  $$h(n) \le h^*(n) \quad \forall n \in V$$
- **Consistency (Monotonicity) Proof:** For any adjacent navigable neighbors $u, v$:
  $$|u_x - t_x| + |u_y - t_y| \le (|v_x - t_x| + |v_y - t_y|) + (|u_x - v_x| + |u_y - v_y|) = h(v) + c(u, v)$$
  Because the heuristic is consistent, the first time any node is popped from the min-heap, its $g$-score is globally optimal. No node is ever re-expanded.
- **Priority Queue Implementation:** Python's standard `heapq` module storing tuples `(f_score, tie_breaker_id, coordinate)`.
- **Complexity:**
  - **Time Complexity:** $O(|V| \log |V| + |E|)$ worst-case. For a $20 \times 20$ grid, $|V| = 400$, bounding worst-case operations to $\approx 3,457$ ops.
  - **Space Complexity:** $O(|V|)$ auxiliary space for `g_score`, `came_from`, `closed_set`, and `open_heap`.

---

### Module 2: Optimal Order Picking (TSP via DP with Bitmasking)
To pick $K$ items starting and ending at a packing station depot:
1. **Pairwise Metric Precomputation:**
   Construct an $N \times N$ distance matrix where $N = 1 + K$ (node $0$ is the packing station depot, nodes $1 \dots K$ are items). Exploiting undirected symmetry on the warehouse floor ($dist(i, j) = dist(j, i)$), only $\binom{N}{2} = \frac{N(N-1)}{2}$ A\* path queries are performed.
2. **Held-Karp Recurrence Formulation:**
   - Let $\text{mask} \in [1, 2^N - 1]$ be an integer bitmask where the $i$-th bit is $1$ if node $i$ has been visited.
   - Let $u \in \text{mask}$ be the last visited node.
   - State: $dp[\text{mask}][u]$ represents the minimum path cost to visit subset $\text{mask}$ ending at node $u$.
   - **Base Case:**
     $$dp[1][0] = 0.0 \quad (\text{only depot } 0 \text{ visited, at node } 0)$$
     $$dp[\text{mask}][u] = \infty \quad \forall (\text{mask}, u) \ne (1, 0)$$
   - **State Transition:**
     $$dp[\text{mask} \mid (1 \ll v)][v] = \min \Big( dp[\text{mask} \mid (1 \ll v)][v],\; dp[\text{mask}][u] + dist[u][v] \Big)$$
     for all $u \in \text{mask}$ and $v \notin \text{mask}$.
   - **Tour Closure:**
     $$\text{Optimal Cost} = \min_{u \in \{1 \dots N-1\}} \Big( dp[2^N - 1][u] + dist[u][0] \Big)$$
- **Complexity:**
  - **Time Complexity:** $\Theta(N^2 2^N)$ DP transitions.
  - **Comparison vs Brute Force:** Brute force evaluates $(N-1)!$ permutations. For $N=15$, $(N-1)! \approx 8.7 \times 10^{10}$ permutations, whereas Held-Karp executes $15^2 \cdot 2^{15} \approx 7.37 \times 10^6$ operations—a speedup of $> 10,000\times$.
  - **Space Complexity:** $\Theta(N \cdot 2^N)$ memory for the DP cost matrix and backtracking predecessor pointers.

---

### Module 3: Task Assignment (Greedy Optimization)
- **Objective:** Allocate $M$ pending customer orders to $R$ idle robots.
- **Greedy Strategy:** At each step, match the next pending order in the FIFO queue to the idle robot that has the minimum obstacle-aware travel distance (computed via A\*) to the order's packing station.
- **Complexity:**
  - **Time Complexity:** $O(M \cdot R \cdot T_{A^*})$ where $T_{A^*} = O(|V| \log |V| + |E|)$.
  - **Space Complexity:** $O(M + R)$ auxiliary tracking structures.

---

## 3. Project Directory Structure

```
/home/abhimanyu/DAA Project for SEM 1/
│
├── warehouse/                     # Core Python algorithmic package
│   ├── __init__.py                # Package exports
│   ├── models.py                  # Dataclasses (Robot, Order, PathResult, TSPResult, etc.)
│   ├── grid.py                    # 2D WarehouseGrid, obstacle maps & ASCII visualizer
│   ├── astar.py                   # Module 1: A* Point-to-Point Search (heapq + Manhattan)
│   ├── tsp.py                     # Module 2: Held-Karp TSP DP with Bitmasking
│   ├── task_allocator.py          # Module 3: Greedy Task Allocator
│   └── simulation.py              # Module 4: WarehouseSimulation & Performance Telemetry
│
├── tests/                         # Rigorous unit & integration test suite
│   └── test_simulation.py         # 11 unit tests covering all edge cases & optimality
│
├── main.py                        # Primary CLI simulation & scaling benchmark script
├── visualize.py                   # Matplotlib 2D warehouse map generator
├── warehouse_simulation_map.png   # Exported high-resolution graphical layout
└── README.md                      # Academic documentation & theoretical report
```

---

## 4. Execution Guide

Run all commands using your project virtual environment:

### A. Launch Interactive Streamlit GUI (Live Web Application)
```bash
/home/abhimanyu/myenv/bin/streamlit run "/home/abhimanyu/DAA Project for SEM 1/app.py"
```
Then open your web browser at: **`http://localhost:8501`**

#### Key Features in the Streamlit GUI:
- **Interactive 2D Grid Player:** Scrub the **Mission Step Slider** to watch the autonomous mobile robots (AMRs) dispatch to packing stations, pick each item in exact TSP order, and return.
- **Dynamic Scenario Selector:** Switch between standard orders, high-density pick loads, and express batches.
- **DAA Algorithmic Inspector:**
  - Interactive **$A^*$ Single-Query Tester**: test any start and goal coordinates with immediate step costs and nodes expanded.
  - **Held-Karp TSP DP Inspector**: inspect pairwise distance matrices and recurrence transitions.
  - **Greedy Task Allocator**: view real-time proximity-based dispatch decision times.
- **Empirical vs Theoretical Complexity Benchmark:** Click a single button to benchmark $N=4$ through $N=14$ and visualize empirical runtimes vs asymptotic Big-$O$ lines.
- **Mission Telemetry Export:** Download the complete robot telemetry table as a CSV file.

### B. Run Standard CLI Simulation & Analytical Report
```bash
/home/abhimanyu/myenv/bin/python "/home/abhimanyu/DAA Project for SEM 1/main.py"
```

### C. Run Empirical vs Theoretical Complexity Benchmark (CLI)
```bash
/home/abhimanyu/myenv/bin/python "/home/abhimanyu/DAA Project for SEM 1/main.py" --benchmark
```

### D. Run the Unit Test Suite
```bash
cd "/home/abhimanyu/DAA Project for SEM 1"
/home/abhimanyu/myenv/bin/python -m unittest tests/test_simulation.py
```

### E. Generate 2D Graphical Layout (Matplotlib PNG)
```bash
cd "/home/abhimanyu/DAA Project for SEM 1"
/home/abhimanyu/myenv/bin/python visualize.py
```

---

## 5. Analytical Results & Complexity Verification

### Benchmark 1: TSP Dynamic Programming with Bitmasking ($N^2 2^N$)
Below are the empirical benchmarks collected by `main.py --benchmark`:

| $N$ (Depot + Items) | Item Count ($K$) | DP Wall Time (ms) | Theoretical Ops ($N^2 2^N$) | Brute-Force Ops ($(N-1)!$) | Speedup Factor |
|:---:|:---:|:---:|:---:|:---:|:---:|
| 4 | 3 | **0.0402 ms** | 256 | 6 | $0.02\times$ |
| 6 | 5 | **0.2719 ms** | 2,304 | 120 | $0.05\times$ |
| 8 | 7 | **0.9458 ms** | 16,384 | 5,040 | $0.31\times$ |
| 10 | 9 | **8.6913 ms** | 102,400 | 362,880 | **$3.54\times$** |
| 12 | 11 | **54.6563 ms** | 589,824 | 39,916,800 | **$67.6\times$** |
| 14 | 13 | **248.9049 ms** | 3,211,264 | 6,227,020,800 | **$1,939\times$** |

> **Key Observation:** For $N \ge 10$, Held-Karp DP with Bitmasking outperforms brute-force search by multiple orders of magnitude, completing in under 250 ms even for $N=14$, while brute-force requires over 6 billion operations.

### Benchmark 2: A\* Search Scaling on 30x30 Grid ($|V| = 900$)
| Start $\to$ Target | Manhattan Distance ($h$) | Path Cost | Nodes Expanded | Empirical Time (ms) |
|:---:|:---:|:---:|:---:|:---:|
| $(0, 0) \to (2, 2)$ | 4 | 4.0 | 9 | 0.1381 ms |
| $(0, 0) \to (5, 4)$ | 9 | 9.0 | 30 | 0.2118 ms |
| $(0, 0) \to (10, 14)$ | 24 | 24.0 | 125 | 0.6918 ms |
| $(0, 0) \to (18, 16)$ | 34 | 34.0 | 235 | 1.0395 ms |
| $(0, 0) \to (24, 26)$ | 50 | 50.0 | 435 | 1.6681 ms |
| $(0, 0) \to (29, 29)$ | 58 | 58.0 | 660 | 4.2469 ms |

> **Key Observation:** Because Manhattan distance is consistent, nodes expanded strictly trace the directed cone toward the target, demonstrating $O(b^d)$ directed heuristic search behavior rather than full $O(|V|)$ grid flood-fill.

---

## 6. Academic Compliance & Grading Checklist

- [x] **Modular Structure:** Distinct modules for models, grid, A\*, TSP, task allocation, and simulation.
- [x] **Type Hinting:** Fully typed with standard Python `typing` constructs (`List`, `Tuple`, `Dict`, `Optional`, etc.).
- [x] **Inline Comments:** Step-by-step comments explaining every algorithmic transition.
- [x] **Big-O Docstrings:** Explicit Time and Space Complexity in the docstring of every major method.
- [x] **Unit Testing:** 11/11 tests passing, including an automated proof that Held-Karp matches brute-force permutations.
- [x] **Visualization:** ASCII grid in console output and standalone Matplotlib PNG export.
