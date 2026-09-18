# Distributed Multi-Agent Task Allocation via ADMM & Convex Relaxation

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from scipy.optimize import linprog

np.random.seed(42)

# ==============================================================================
# 1. Hypersimplex Projection Routine (Capped Simplex of Capacity b_i)
# ==============================================================================
def project_hypersimplex(v, b, tol=1e-9):
    """
    Solves: min_{x} 0.5 * ||x - v||_2^2
            s.t. sum(x) = b,  0 <= x_j <= 1 for all j in {1, ..., M}
    """
    v = np.array(v, dtype=float)
    low = np.min(v) - 1.0
    high = np.max(v) + 1.0
    
    # Bisection search on the scalar Lagrange multiplier mu
    for _ in range(60):
        mid = 0.5 * (low + high)
        x_cand = np.clip(v - mid, 0.0, 1.0)
        sum_x = np.sum(x_cand)
        
        if abs(sum_x - b) < tol:
            return x_cand
        elif sum_x > b:
            low = mid
        else:
            high = mid
            
    return np.clip(v - 0.5 * (low + high), 0.0, 1.0)

# ==============================================================================
# 2. ADMM Solver for Multi-Agent Task Allocation
# ==============================================================================
def solve_task_allocation_admm(C, b, max_iters=150, rho=2.5):
    """
    Solves the relaxed task allocation problem:
        min_{x_1, ..., x_N} sum_{i=1}^N c_i^T x_i
        s.t. sum_{i=1}^N x_i = 1_M,  x_i in {x in [0, 1]^M : sum(x) = b_i}
    via ADMM Sharing Algorithm (Boyd et al., Section 7).
    """
    N, M = C.shape
    x = np.zeros((N, M))
    # Initialize feasible x_i on each agent's capacity simplex
    for i in range(N):
        x[i, :int(b[i])] = 1.0
        
    lam = np.zeros(M)      # Dual multipliers for coupled task constraint
    hist_obj = []
    hist_prim_res = []
    hist_dual_res = []
    
    for k in range(max_iters):
        x_prev = x.copy()
        x_mean = np.mean(x, axis=0)
        
        # 1. Local Primal x_i Updates
        for i in range(N):
            # Gradient of local quadratic augmented terms
            # v_i = x_i - (1/rho)*(c_i + lam) - (x_mean - (1/N)*1_M)
            v_i = x[i] - (1.0 / rho) * (C[i] + lam) - (x_mean - (1.0 / N) * np.ones(M))
            x[i] = project_hypersimplex(v_i, b[i])
            
        # 2. Average Network Demand
        x_mean_new = np.mean(x, axis=0)
        
        # 3. Dual Multiplier Update
        r_primal = N * x_mean_new - np.ones(M)  # Primal residual: sum(x_i) - 1_M
        lam = lam + rho * r_primal
        
        # 4. Residual Norms & Objective Value
        prim_norm = np.linalg.norm(r_primal)
        dual_norm = rho * N * np.linalg.norm(x_mean_new - x_mean)
        total_cost = np.sum(C * x)
        
        hist_prim_res.append(prim_norm)
        hist_dual_res.append(dual_norm)
        hist_obj.append(total_cost)
        
    return x, hist_obj, hist_prim_res, hist_dual_res

# ==============================================================================
# 3. Problem Setup: Agents, Tasks, Graph, and Cost Matrices
# ==============================================================================
N = 6                # Number of agents
M = 15               # Number of tasks (M > N)
p_er = 0.50          # Erdos-Renyi connectivity probability

# Generate connected graph G(V, E)
while True:
    G = nx.erdos_renyi_graph(N, p_er, seed=np.random.randint(10000))
    if nx.is_connected(G):
        break

# Capacity vector b with sum(b) = M = 15
b_capacities = np.array([3, 2, 4, 1, 3, 2], dtype=int)
assert np.sum(b_capacities) == M, "Sum of capacities must equal total tasks M"

# Cost Matrix 1: Random baseline cost matrix (C_ij > 0)
C_matrix_1 = np.random.uniform(5.0, 30.0, size=(N, M))
# Ensure distinct agent proficiencies across tasks
for i in range(N):
    preferred_tasks = [(i * 2 + k) % M for k in range(3)]
    C_matrix_1[i, preferred_tasks] = np.random.uniform(2.0, 6.0, size=len(preferred_tasks))

# Cost Matrix 2: Alternative cost distribution (to compare task shift behavior)
C_matrix_2 = np.random.uniform(5.0, 30.0, size=(N, M))
for i in range(N):
    specialist_tasks = [((N - 1 - i) * 2 + k) % M for k in range(3)]
    C_matrix_2[i, specialist_tasks] = np.random.uniform(1.5, 5.0, size=len(specialist_tasks))

# Solve with ADMM for both cost matrices
x_sol_1, obj_1, prim_1, dual_1 = solve_task_allocation_admm(C_matrix_1, b_capacities, max_iters=160, rho=3.0)
x_sol_2, obj_2, prim_2, dual_2 = solve_task_allocation_admm(C_matrix_2, b_capacities, max_iters=160, rho=3.0)

# Centralized LP ground truth for Cost Matrix 1
c_flat = C_matrix_1.flatten()
A_eq_task = np.zeros((M, N * M))
for j in range(M):
    for i in range(N):
        A_eq_task[j, i * M + j] = 1.0
b_eq_task = np.ones(M)

A_eq_cap = np.zeros((N, N * M))
for i in range(N):
    A_eq_cap[i, i * M : (i + 1) * M] = 1.0
b_eq_cap = b_capacities.astype(float)

A_eq = np.vstack([A_eq_task, A_eq_cap])
b_eq = np.concatenate([b_eq_task, b_eq_cap])
bounds = [(0, 1) for _ in range(N * M)]

res_lp = linprog(c_flat, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
opt_cost_lp = res_lp.fun

print("===========================================================", flush=True)
print(f"ADMM TASK ALLOCATION REPORT (N = {N} agents, M = {M} tasks):", flush=True)
print(f"Agent Capacities b: {b_capacities.tolist()} (Sum = {np.sum(b_capacities)})", flush=True)
print(f"Cost Matrix 1 - Final ADMM Cost: {obj_1[-1]:.4f} | Optimal LP Cost: {opt_cost_lp:.4f}", flush=True)
print(f"Cost Matrix 1 - Final Primal Residual: {prim_1[-1]:.2e}", flush=True)
print(f"Cost Matrix 2 - Final ADMM Cost: {obj_2[-1]:.4f}", flush=True)
print(f"Cost Matrix 2 - Final Primal Residual: {prim_2[-1]:.2e}", flush=True)
print("===========================================================", flush=True)

# ==============================================================================
# 4. Generate Comprehensive 4-Subplot Visualization
# ==============================================================================
fig, axs = plt.subplots(2, 2, figsize=(15, 11))

# Subplot 1: Communication Graph Topology G(V, E)
ax1 = axs[0, 0]
pos = nx.spring_layout(G, seed=42)
nx.draw_networkx_nodes(G, pos, ax=ax1, node_color='lightblue', node_size=600, edgecolors='black', linewidths=1.5)
nx.draw_networkx_edges(G, pos, ax=ax1, edge_color='royalblue', width=2.0, alpha=0.8)
labels = {i: f"$A_{{{i+1}}}$\n($b_{{{i+1}}}={b_capacities[i]}$)" for i in range(N)}
nx.draw_networkx_labels(G, pos, labels=labels, ax=ax1, font_size=9, font_weight='bold')
ax1.set_title(f"Agent Communication Graph $\mathcal{{G}}(\mathcal{{V}}, \mathcal{{E}})$\n($N={N}$ Agents, $|E|={G.number_of_edges()}$ Edges, Connected)", 
              fontsize=11, fontweight='bold', pad=10)
ax1.axis('off')

# Subplot 2: ADMM Residual & Objective Convergence
ax2 = axs[0, 1]
iters = np.arange(1, len(prim_1) + 1)
ax2.semilogy(iters, prim_1, color='navy', linewidth=2.0, label=r'Primal Residual $\|r^k\|_2 = \|\sum x_i - \mathbf{1}\|_2$')
ax2.semilogy(iters, dual_1, color='crimson', linestyle='--', linewidth=1.8, label=r'Dual Residual $\|s^k\|_2$')
ax2.set_title("ADMM Convergence Diagnostics (Cost Matrix 1)", fontsize=11, fontweight='bold', pad=10)
ax2.set_xlabel("ADMM Iteration $k$", fontweight='bold')
ax2.set_ylabel("Residual Norm (Log Scale)", fontweight='bold')
ax2.legend(loc='upper right', fontsize=8.5)
ax2.grid(True, linestyle='--', alpha=0.4)

# Subplot 3: Task Assignment Heatmap - Cost Matrix 1
ax3 = axs[1, 0]
im1 = ax3.imshow(x_sol_1, cmap='Blues', aspect='auto', interpolation='nearest', vmin=0, vmax=1)
ax3.set_title(f"Task Allocation Heatmap (Cost Matrix 1)\nTotal Optimal Cost = {obj_1[-1]:.2f}", fontsize=11, fontweight='bold', pad=10)
ax3.set_xlabel("Task Index $j \in \{1, \dots, 15\}$", fontweight='bold')
ax3.set_ylabel("Agent Index $i \in \{1, \dots, 6\}$", fontweight='bold')
ax3.set_xticks(np.arange(M))
ax3.set_xticklabels([f"$T_{{{j+1}}}$" for j in range(M)], fontsize=8)
ax3.set_yticks(np.arange(N))
ax3.set_yticklabels([f"$A_{{{i+1}}}$" for i in range(N)], fontsize=9)
plt.colorbar(im1, ax=ax3, fraction=0.046, pad=0.04, label="Assignment Variable $x_{ij} \in [0, 1]$")

# Annotate binary assignments
for i in range(N):
    for j in range(M):
        val = x_sol_1[i, j]
        if val > 0.05:
            ax3.text(j, i, f"{val:.2f}", ha='center', va='center', color='white' if val > 0.6 else 'black', fontsize=7.5, fontweight='bold')

# Subplot 4: Task Assignment Heatmap - Cost Matrix 2 (Demonstrating Allocation Shift)
ax4 = axs[1, 1]
im2 = ax4.imshow(x_sol_2, cmap='Greens', aspect='auto', interpolation='nearest', vmin=0, vmax=1)
ax4.set_title(f"Task Allocation Heatmap (Cost Matrix 2)\nTotal Optimal Cost = {obj_2[-1]:.2f}", fontsize=11, fontweight='bold', pad=10)
ax4.set_xlabel("Task Index $j \in \{1, \dots, 15\}$", fontweight='bold')
ax4.set_ylabel("Agent Index $i \in \{1, \dots, 6\}$", fontweight='bold')
ax4.set_xticks(np.arange(M))
ax4.set_xticklabels([f"$T_{{{j+1}}}$" for j in range(M)], fontsize=8)
ax4.set_yticks(np.arange(N))
ax4.set_yticklabels([f"$A_{{{i+1}}}$" for i in range(N)], fontsize=9)
plt.colorbar(im2, ax=ax4, fraction=0.046, pad=0.04, label="Assignment Variable $x_{ij} \in [0, 1]$")

for i in range(N):
    for j in range(M):
        val = x_sol_2[i, j]
        if val > 0.05:
            ax4.text(j, i, f"{val:.2f}", ha='center', va='center', color='white' if val > 0.6 else 'black', fontsize=7.5, fontweight='bold')

plt.suptitle("AI3403 Assignment 3, Problem 3: Distributed Task Allocation via ADMM and Convex Relaxation", 
             fontsize=13, fontweight='bold', y=0.98)
plt.tight_layout()
plt.savefig("problem3_task_allocation_admm.png", dpi=300)
print("Saved high-res figure: problem3_task_allocation_admm.png", flush=True)
