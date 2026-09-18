# Distributed Drone Target Tracking via Consensus Optimization (DGT & DGD)

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import networkx as nx

# Set random seed for exact reproducibility
np.random.seed(42)

# ==============================================================================
# 1. Simulation Setup & Graph Generation
# ==============================================================================
N = 10               # Number of drone sensor agents (N < 20 as specified)
p_er = 0.45          # Erdos-Renyi edge probability
T_max = 60           # Time horizon steps t in [0, T_max]

# Generate a connected Erdos-Renyi graph G(V, E)
while True:
    G = nx.erdos_renyi_graph(N, p_er, seed=np.random.randint(10000))
    if nx.is_connected(G):
        break

Adj = nx.to_numpy_array(G)
degrees = np.sum(Adj, axis=1)

# Metropolis-Hastings Doubly-Stochastic Weight Matrix W
W = np.zeros((N, N))
for i in range(N):
    for j in range(N):
        if i != j and Adj[i, j] > 0:
            W[i, j] = 1.0 / (max(degrees[i], degrees[j]) + 1.0)
    W[i, i] = 1.0 - np.sum(W[i, :])

print(f"Generated Connected Drone Network: N = {N} agents, |E| = {G.number_of_edges()} edges.", flush=True)

# Fixed drone sensor positions in R^3 around the surveillance zone
drone_positions = np.array([
    [-18.0, -15.0, 15.0],
    [ 16.0, -18.0, 12.0],
    [ 18.0,  15.0, 20.0],
    [-15.0,  18.0, 18.0],
    [  0.0, -20.0, 10.0],
    [ 20.0,   0.0, 16.0],
    [  0.0,  20.0, 22.0],
    [-20.0,   0.0, 14.0],
    [ -8.0,  -8.0, 25.0],
    [  8.0,   8.0, 24.0]
])

# Sensor measurement noise standard deviations (heterogeneous precision)
sigma_v = np.array([0.8, 1.2, 1.0, 1.5, 0.9, 1.4, 1.1, 1.3, 0.7, 1.6])
Sigma_v_inv = [np.eye(3) / (sigma_v[i]**2) for i in range(N)]

# Process noise covariance (random walk disturbance)
sigma_w = 0.45
Sigma_w = (sigma_w**2) * np.eye(3)

# Initial intruder state distribution: z0 ~ N(z_bar_0, Sigma_0)
z_bar_0 = np.array([0.0, 0.0, 8.0])
Sigma_0 = 2.0 * np.eye(3)
z_true_0 = np.random.multivariate_normal(z_bar_0, Sigma_0)

# ==============================================================================
# 2. Simulate True Target Random Walk Trajectory & Sensor Measurements
# ==============================================================================
z_true = np.zeros((T_max + 1, 3))
z_true[0] = z_true_0

# Generate true trajectory: z(t+1) = z(t) + w(t)
for t in range(T_max):
    w_t = np.random.multivariate_normal(np.zeros(3), Sigma_w)
    z_true[t + 1] = z_true[t] + w_t

# Generate noisy sensor measurements: y_i(t) = x_i - z(t) + v_i(t)
# Local target observation: u_i(t) = x_i - y_i(t) = z(t) - v_i(t)
u_meas = np.zeros((T_max + 1, N, 3))
for t in range(T_max + 1):
    for i in range(N):
        v_i = np.random.normal(0, sigma_v[i], size=3)
        y_i = drone_positions[i] - z_true[t] + v_i
        u_meas[t, i] = drone_positions[i] - y_i

# ==============================================================================
# 3. Distributed State Estimation via Distributed Gradient Tracking (DGT)
# ==============================================================================
K_opt = 35           # Number of distributed consensus-gradient iterations per time step
alpha_step = 0.045   # DGT step size

z_est_dgt = np.zeros((T_max + 1, N, 3))
z_central_ml = np.zeros((T_max + 1, 3))

# Centralized Maximum Likelihood / BLUE Information Weights
total_info_matrix = np.sum(Sigma_v_inv, axis=0)
total_info_inv = np.linalg.inv(total_info_matrix)

for t in range(T_max + 1):
    # Centralized ML benchmark: z*(t) = (sum Sigma_i^-1)^-1 * (sum Sigma_i^-1 u_i(t))
    weighted_meas_sum = np.zeros(3)
    for i in range(N):
        weighted_meas_sum += Sigma_v_inv[i] @ u_meas[t, i]
    z_central_ml[t] = total_info_inv @ weighted_meas_sum

    # Local DGT initialization: initialize from previous estimate or local measurement
    if t == 0:
        z_local = u_meas[t].copy()
    else:
        z_local = z_est_dgt[t - 1].copy()

    # Initial local gradient: grad f_i(z_i) = Sigma_v^-1 (z_i - u_i(t))
    grad_local = np.zeros((N, 3))
    for i in range(N):
        grad_local[i] = Sigma_v_inv[i] @ (z_local[i] - u_meas[t, i])
    
    # Initialize gradient tracker s_i^0 = grad f_i(z_i^0)
    s_tracker = grad_local.copy()

    # DGT Inner Iteration Loop
    for k in range(K_opt):
        z_prev = z_local.copy()
        grad_prev = grad_local.copy()

        # Step 1: Primal state consensus mixing and tracking descent
        z_local = W @ z_prev - alpha_step * s_tracker

        # Step 2: Evaluate new local gradients
        for i in range(N):
            grad_local[i] = Sigma_v_inv[i] @ (z_local[i] - u_meas[t, i])

        # Step 3: Gradient tracker consensus mixing and innovation update
        s_tracker = W @ s_tracker + (grad_local - grad_prev)

    z_est_dgt[t] = z_local.copy()

# Average estimated state across all swarm drones: z_hat(t)
z_swarm_mean = np.mean(z_est_dgt, axis=1)

# Estimation error: e(t) = z_hat(t) - z_true(t)
error_dgt = np.linalg.norm(z_swarm_mean - z_true, axis=1)
error_central = np.linalg.norm(z_central_ml - z_true, axis=1)
consensus_dispersion = np.max([np.linalg.norm(z_est_dgt[:, i, :] - z_swarm_mean, axis=1) for i in range(N)], axis=0)

mean_rmse_dgt = np.sqrt(np.mean(error_dgt**2))
mean_rmse_central = np.sqrt(np.mean(error_central**2))
max_consensus_gap = np.max(consensus_dispersion)

print(f"===========================================================", flush=True)
print(f"DISTRIBUTED TARGET TRACKING REPORT:", flush=True)
print(f"Time Horizon: T_max = {T_max} steps", flush=True)
print(f"Distributed Gradient Tracking Mean RMSE: {mean_rmse_dgt:.4f} m", flush=True)
print(f"Centralized Optimal ML Estimator RMSE:   {mean_rmse_central:.4f} m", flush=True)
print(f"Swarm Network Consensus Disagreement:   {max_consensus_gap:.2e} m", flush=True)
print(f"===========================================================", flush=True)

# ==============================================================================
# 4. Generate Comprehensive 4-Subplot Publication Figure
# ==============================================================================
fig = plt.figure(figsize=(16, 12))

# Subplot 1: 3D True vs Estimated Trajectory & Drone Sensor Network
ax1 = fig.add_subplot(2, 2, 1, projection='3d')

# Draw communication network edges between drones
for i in range(N):
    for j in range(i + 1, N):
        if Adj[i, j] > 0:
            ax1.plot([drone_positions[i, 0], drone_positions[j, 0]],
                     [drone_positions[i, 1], drone_positions[j, 1]],
                     [drone_positions[i, 2], drone_positions[j, 2]],
                     color='gray', alpha=0.45, linewidth=0.8, linestyle=':')

# Plot drone sensor nodes
ax1.scatter(drone_positions[:, 0], drone_positions[:, 1], drone_positions[:, 2],
            c='royalblue', s=90, edgecolors='black', label=f'Drone Sensors ($N={N}$)', zorder=5)
for i in range(N):
    ax1.text(drone_positions[i, 0] + 0.8, drone_positions[i, 1] + 0.8, drone_positions[i, 2] + 0.8,
             f"$D_{{{i+1}}}$", fontsize=8, fontweight='bold')

# Plot true and estimated intruder trajectories
ax1.plot(z_true[:, 0], z_true[:, 1], z_true[:, 2],
         color='crimson', linewidth=2.4, label='True Target Trajectory $z(t)$', zorder=3)
ax1.plot(z_swarm_mean[:, 0], z_swarm_mean[:, 1], z_swarm_mean[:, 2],
         color='darkgreen', linewidth=2.0, linestyle='--', label=r'Distributed Estimate $\hat{z}(t)$ (DGT)', zorder=4)
ax1.scatter([z_true[0, 0]], [z_true[0, 1]], [z_true[0, 2]], color='crimson', s=80, marker='o', label='Start $z(0)$')
ax1.scatter([z_true[-1, 0]], [z_true[-1, 1]], [z_true[-1, 2]], color='darkred', s=90, marker='^', label='End $z(T_{\max})$')

ax1.set_title("3D Surveillance Space & Network Topology", fontsize=12, fontweight='bold', pad=10)
ax1.set_xlabel("X (m)", fontweight='bold')
ax1.set_ylabel("Y (m)", fontweight='bold')
ax1.set_zlabel("Z (m)", fontweight='bold')
ax1.legend(loc='upper right', fontsize=8)
ax1.view_init(elev=28, azim=45)

# Subplot 2: Communication Graph Topology G(V, E)
ax2 = fig.add_subplot(2, 2, 2)
pos_2d = nx.spring_layout(G, seed=42)
nx.draw_networkx_nodes(G, pos_2d, ax=ax2, node_color='lightblue', node_size=500, edgecolors='black', linewidths=1.2)
nx.draw_networkx_edges(G, pos_2d, ax=ax2, edge_color='royalblue', width=1.5, alpha=0.7)
nx.draw_networkx_labels(G, pos_2d, ax=ax2, font_size=10, font_weight='bold')
ax2.set_title(f"Erdős–Rényi Communication Network $\mathcal{{G}}(\mathcal{{V}}, \mathcal{{E}})$\n($N={N}$, Connected, Spanning Tree Verified)", 
              fontsize=12, fontweight='bold', pad=10)
ax2.axis('off')

# Subplot 3: Component-Wise Target Tracking (X, Y, Z coordinates vs Time)
ax3 = fig.add_subplot(2, 2, 3)
time_axis = np.arange(T_max + 1)
ax3.plot(time_axis, z_true[:, 0], color='crimson', linewidth=1.8, label=r'True $z_x(t)$')
ax3.plot(time_axis, z_swarm_mean[:, 0], color='crimson', linestyle='--', linewidth=1.6, label=r'Est $\hat{z}_x(t)$')
ax3.plot(time_axis, z_true[:, 1], color='darkblue', linewidth=1.8, label=r'True $z_y(t)$')
ax3.plot(time_axis, z_swarm_mean[:, 1], color='darkblue', linestyle='--', linewidth=1.6, label=r'Est $\hat{z}_y(t)$')
ax3.plot(time_axis, z_true[:, 2], color='darkgreen', linewidth=1.8, label=r'True $z_z(t)$')
ax3.plot(time_axis, z_swarm_mean[:, 2], color='darkgreen', linestyle='--', linewidth=1.6, label=r'Est $\hat{z}_z(t)$')

ax3.set_title("Coordinate-Wise Target Tracking vs Time $t$", fontsize=12, fontweight='bold', pad=10)
ax3.set_xlabel("Discrete Time Step $t$", fontweight='bold')
ax3.set_ylabel("Position (m)", fontweight='bold')
ax3.legend(loc='upper left', ncol=3, fontsize=8)
ax3.grid(True, linestyle='--', alpha=0.4)

# Subplot 4: Estimation Error Norm ||e(t)||_2 & Disagreement
ax4 = fig.add_subplot(2, 2, 4)
ax4.plot(time_axis, error_dgt, color='forestgreen', linewidth=2.0, label=f'DGT Swarm Error $\|e(t)\|_2$ (Mean = {mean_rmse_dgt:.3f} m)')
ax4.plot(time_axis, error_central, color='purple', linestyle=':', linewidth=1.8, label=f'Centralized ML Bound (Mean = {mean_rmse_central:.3f} m)')
ax4.axhline(mean_rmse_dgt, color='forestgreen', linestyle='--', alpha=0.6, label='DGT RMSE Baseline')
ax4.set_title("Tracking Estimation Error $\|e(t)\|_2 = \|\hat{z}(t) - z(t)\|_2$", fontsize=12, fontweight='bold', pad=10)
ax4.set_xlabel("Discrete Time Step $t$", fontweight='bold')
ax4.set_ylabel("Error Norm (m)", fontweight='bold')
ax4.legend(loc='upper right', fontsize=8.5)
ax4.grid(True, linestyle='--', alpha=0.4)

plt.suptitle(r"AI3403 Assignment 3, Problem 2: Distributed Drone Swarm Target Tracking via Consensus Optimization", 
             fontsize=14, fontweight='bold', y=0.98)
plt.tight_layout()
plt.savefig("problem2_target_tracking.png", dpi=300)
print("Saved high-res figure: problem2_target_tracking.png", flush=True)
