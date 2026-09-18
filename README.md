# AI3403: Multi-Agent Systems (MAS) — Assignment 3
**Department of Artificial Intelligence, Indian Institute of Technology Hyderabad**  
**Instructor:** Dr. Venkatraman Renganathan  
**Student:** DHEDHI KRINSH RAMESHBHAI (Roll No: `AI26MTECH12007`)  

---

## Prerequisites & Installation

To run the simulations and reproduce the results and animations, install the standard scientific Python packages:

```bash
pip install numpy matplotlib scipy networkx
```

---

## Problem 1: Strict Collision-Free Multi-Agent Name Formation Control (`KRINSH`)

### Guaranteed Collision-Free Name Formation Animation (`K -> R -> I -> N -> S -> H`)
![Multi-Agent Name Formation Animation](name_formation_krinsh.gif)

### Key Highlights
- **Swarm Size:** $N = 20$ autonomous agents connected over an Erdős–Rényi communication graph $\mathcal{G}(\mathcal{V}, \mathcal{E})$ with $p = 0.35$ ($|E| = 71$ links).
- **Target Letter Sequence:** **`K` $\to$ `R` $\to$ `I` $\to$ `N` $\to$ `S` $\to$ `H`** (displaying name letter by letter).
- **Control Strategy:**
  1. **Hungarian Bipartite Matching:** Optimal assignment of target coordinates to minimize total kinetic transit distance and eliminate path-crossing conflicts.
  2. **Predictive Safety Filter:** Sequential priority deconfliction with multi-directional lateral velocity evasion guaranteeing pairwise separation ($d_{\min} = 0.1400 \ge d_{\text{safe}} = 0.140$).
  3. **Smooth Gliding:** Bounded cruising speed ($v_{\max} = 0.035$) with stable letter hold phases.
- **Verification Telemetry:** **0 collisions** across all 631 transition frames.

### How to Run Problem 1
```bash
python problem1_name_formation.py
```
**Outputs generated:**
- `name_formation_krinsh.gif` (631 frames at 30 FPS animation)
- `problem1_name_formation_snapshots.png` (High-resolution snapshot grid)

---

## Problem 2: Distributed Drone Swarm Target Tracking

### Key Highlights
- **Network Topology:** $N = 10$ drones connected over a connected Erdős–Rényi graph ($|E| = 23$).
- **Intruder Dynamics:** 3D random walk $z(t+1) = z(t) + w(t)$ with atmospheric disturbance $\Sigma_w$ and heterogeneous sensor noise $\sigma_v \in [0.7, 1.6]\text{ m}$.
- **Method:** **Distributed Gradient Tracking (DGT)** with Metropolis–Hastings mixing weights.
- **Results:** DGT Tracking Mean RMSE $= 0.5313\text{ m}$ (Centralized BLUE bound $= 0.6026\text{ m}$), Swarm Consensus Disagreement $< 5.20 \times 10^{-3}\text{ m}$.

### How to Run Problem 2
```bash
python problem2_target_tracking.py
```
**Outputs generated:**
- `problem2_target_tracking.png` (3D tracking trajectory and error norm plots)

---

## Problem 3: Distributed Multi-Agent Task Allocation via ADMM

### Key Highlights
- **Problem Setup:** $N = 6$ agents collaboratively allocating $M = 15$ tasks with capacity constraints $b = [3, 2, 4, 1, 3, 2]^\top$ ($\sum_{i=1}^6 b_i = 15$).
- **Mathematical Theory:** Continuous linear relaxation onto capped hypersimplices $\mathcal{X}_i = \{x_i \in [0, 1]^M : \mathbf{1}^\top x_i = b_i\}$. By **Total Unimodularity (TUM)**, extreme points are guaranteed $\{0, 1\}$ integer-valued (zero integrality gap).
- **ADMM Sharing Algorithm:** Decoupled local hypersimplex projections $\Pi_{\mathcal{X}_i}(v_i^k)$ computed in $\mathcal{O}(M)$ via scalar threshold clipping + dual shadow price updates.
- **Results:** Final ADMM Cost $= 66.7236$ (matches Centralized Linear Program exactly with primal residual $0.00$).

### How to Run Problem 3
```bash
python problem3_task_allocation_admm.py
```
**Outputs generated:**
- `problem3_task_allocation_admm.png` (Communication graph, convergence diagnostics, and binary task allocation heatmaps)

---

## Problem 4: Closed-Form ADMM for LASSO Optimization

### Key Highlights
- **Optimization Problem:** $\min_x \frac{1}{2}\|Ax - b\|_2^2 + \lambda \|x\|_1$.
- **Exact Closed-Form Updates:**
  - **Primal $x$-Update:** $x^{k+1} = (A^\top A + \rho I_n)^{-1} \big(A^\top b + \rho(z^k - u^k)\big)$ (Least-Squares Proximal Step with pre-factored Cholesky decomposition).
  - **Primal $z$-Update:** $z^{k+1} = \mathcal{S}_{\lambda/\rho}\big(x^{k+1} + u^k\big)$ (Element-wise Soft-Thresholding / Shrinkage operator).
  - **Scaled Dual $u$-Update:** $u^{k+1} = u^k + (x^{k+1} - z^{k+1})$.
- **Results:** Underdetermined sparse signal recovery ($m=60, n=120, k_{\text{sparse}}=10$): exact 10/10 support recovery, primal residual $= 3.85 \times 10^{-6}$.

### How to Run Problem 4
```bash
python problem4_lasso_admm.py
```
**Outputs generated:**
- `problem4_lasso_admm.png` (Reconstruction stem plot, residual convergence, objective descent, and soft-thresholding shrinkage curve)

---

## Repository Structure
```
├── README.md                              # Repository documentation with embedded animation
├── name_formation_krinsh.gif              # Problem 1 recorded animation video (KRINSH sequence)
├── problem1_name_formation.py             # Problem 1 collision-free formation control script
├── problem1_name_formation_snapshots.png  # Problem 1 letter snapshots (K -> R -> I -> N -> S -> H)
├── problem2_target_tracking.py            # Problem 2 distributed drone target tracking script
├── problem2_target_tracking.png           # Problem 2 3D tracking & error plots
├── problem3_task_allocation_admm.py       # Problem 3 ADMM task allocation script
├── problem3_task_allocation_admm.png      # Problem 3 communication & heatmap plots
├── problem4_lasso_admm.py                 # Problem 4 LASSO closed-form ADMM script
└── problem4_lasso_admm.png                # Problem 4 convergence & shrinkage plots
```
