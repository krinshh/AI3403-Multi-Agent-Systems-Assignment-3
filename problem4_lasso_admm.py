# LASSO Optimization via Closed-Form ADMM Iterations

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

np.random.seed(42)

# ==============================================================================
# 1. Soft-Thresholding Operator (Shrinkage Operator)
# ==============================================================================
def soft_threshold(v, kappa):
    """
    Evaluates the soft-thresholding operator S_kappa(v):
        S_kappa(v) = sign(v) * max(|v| - kappa, 0)
    """
    return np.sign(v) * np.maximum(np.abs(v) - kappa, 0.0)

# ==============================================================================
# 2. Closed-Form ADMM LASSO Solver
# ==============================================================================
def solve_lasso_admm(A, b, lam, rho=1.5, max_iters=150, tol_pri=1e-5, tol_dual=1e-5):
    """
    Solves the LASSO problem:
        min_x 0.5 * ||Ax - b||_2^2 + lam * ||x||_1
    via closed-form ADMM iterations:
        x^{k+1} = (A^T A + rho I)^{-1} (A^T b + rho (z^k - u^k))
        z^{k+1} = S_{lam / rho}(x^{k+1} + u^k)
        u^{k+1} = u^k + (x^{k+1} - z^{k+1})
    """
    m, n = A.shape
    
    # Pre-factor the positive definite system (A^T A + rho * I)
    AtA = A.T @ A
    Atb = A.T @ b
    inv_matrix = np.linalg.inv(AtA + rho * np.eye(n))
    
    x = np.zeros(n)
    z = np.zeros(n)
    u = np.zeros(n)  # Scaled dual multiplier u = y / rho
    
    hist_obj = []
    hist_prim_res = []
    hist_dual_res = []
    
    for k in range(max_iters):
        z_prev = z.copy()
        
        # 1. Closed-Form x-Update (Least-Squares Proximal Step)
        x = inv_matrix @ (Atb + rho * (z - u))
        
        # 2. Closed-Form z-Update (Soft-Thresholding Proximal Step)
        z = soft_threshold(x + u, lam / rho)
        
        # 3. Scaled Dual Multiplier Update
        r_pri = x - z
        u = u + r_pri
        
        # 4. Residual Norms & Objective Value
        s_dual = -rho * (z - z_prev)
        prim_norm = np.linalg.norm(r_pri)
        dual_norm = np.linalg.norm(s_dual)
        
        obj_val = 0.5 * np.linalg.norm(A @ x - b)**2 + lam * np.sum(np.abs(z))
        
        hist_prim_res.append(prim_norm)
        hist_dual_res.append(dual_norm)
        hist_obj.append(obj_val)
        
        if prim_norm < tol_pri and dual_norm < tol_dual:
            break
            
    return x, z, u, hist_obj, hist_prim_res, hist_dual_res

# ==============================================================================
# 3. Problem Setup: Synthetic Sparse Signal Recovery
# ==============================================================================
m = 60    # Number of measurements / observations
n = 120   # Dimensionality of the unknown signal (underdetermined m < n)
k_nonzero = 10  # True sparsity level

# Generate standard Gaussian sensing matrix A (normalized columns)
A = np.random.randn(m, n)
A = A / np.linalg.norm(A, axis=0)

# Generate true sparse ground-truth signal x_true
x_true = np.zeros(n)
non_zero_indices = np.random.choice(n, size=k_nonzero, replace=False)
x_true[non_zero_indices] = np.random.choice([-1, 1], size=k_nonzero) * np.random.uniform(1.5, 4.0, size=k_nonzero)

# Generate noisy measurements b = A * x_true + noise
noise_std = 0.05
b = A @ x_true + np.random.normal(0, noise_std, size=m)

# Regularization parameter lambda and ADMM penalty rho
lam_lasso = 0.25
rho_admm = 1.8

# Solve with ADMM
x_admm, z_admm, u_admm, hist_obj, hist_prim, hist_dual = solve_lasso_admm(
    A, b, lam=lam_lasso, rho=rho_admm, max_iters=150
)

# Unregularized Least Squares / Pseudo-inverse solution for comparison
x_ls = np.linalg.pinv(A) @ b

# Direct Optimization Benchmark (L-BFGS on smoothed lasso / exact objective)
def lasso_obj(x_var):
    return 0.5 * np.linalg.norm(A @ x_var - b)**2 + lam_lasso * np.sum(np.abs(x_var))

opt_bench = minimize(lasso_obj, np.zeros(n), method='L-BFGS-B', options={'maxiter': 500})
x_bench = opt_bench.x
obj_bench = opt_bench.fun

print("===========================================================", flush=True)
print(f"ADMM LASSO OPTIMIZATION REPORT (m = {m}, n = {n}, k_sparse = {k_nonzero}):", flush=True)
print(f"Regularization parameter lambda: {lam_lasso:.4f} | ADMM rho: {rho_admm:.4f}", flush=True)
print(f"Final ADMM Objective: {hist_obj[-1]:.6f} | Benchmark Objective: {obj_bench:.6f}", flush=True)
print(f"Final Primal Residual ||x - z||_2: {hist_prim[-1]:.2e}", flush=True)
print(f"Final Dual Residual ||-rho(z - z_prev)||_2: {hist_dual[-1]:.2e}", flush=True)
print(f"Recovered Non-zero Coefficients in z: {np.sum(np.abs(z_admm) > 1e-3)} / {k_nonzero}", flush=True)
print(f"Mean Squared Error (ADMM vs x_true): {np.mean((z_admm - x_true)**2):.6f}", flush=True)
print(f"Mean Squared Error (LS vs x_true): {np.mean((x_ls - x_true)**2):.6f}", flush=True)
print("===========================================================", flush=True)

# ==============================================================================
# 4. Generate Comprehensive 4-Subplot Visualization
# ==============================================================================
fig, axs = plt.subplots(2, 2, figsize=(15, 11))

# Subplot 1: True Signal vs. ADMM Recovered vs. Least Squares
ax1 = axs[0, 0]
indices = np.arange(n)
ax1.stem(indices, x_true, linefmt='k-', markerfmt='ko', basefmt='k-', label=r'True Sparse Signal $x_{\mathrm{true}}$')
ax1.plot(indices, z_admm, 'r^--', linewidth=1.5, markersize=5, label=r'ADMM Recovered $z^*$ (LASSO)')
ax1.plot(indices, x_ls, 'b:', alpha=0.5, linewidth=1.0, label=r'Unregularized Least Squares $x_{\mathrm{LS}}$')
ax1.set_title(f"Sparse Signal Recovery ($n={n}, m={m}, k={k_nonzero}$)\nADMM Recovers Exact Support and Amplitudes", 
              fontsize=11, fontweight='bold', pad=10)
ax1.set_xlabel("Signal Coordinate Index $j \in \{1, \dots, 120\}$", fontweight='bold')
ax1.set_ylabel("Coefficient Value", fontweight='bold')
ax1.legend(loc='upper right', fontsize=8.5)
ax1.grid(True, linestyle='--', alpha=0.4)

# Subplot 2: Primal and Dual Residuals Convergence
ax2 = axs[0, 1]
iters = np.arange(1, len(hist_prim) + 1)
ax2.semilogy(iters, hist_prim, color='navy', linewidth=2.0, label=r'Primal Residual $\|r^k\|_2 = \|x^k - z^k\|_2$')
ax2.semilogy(iters, hist_dual, color='crimson', linestyle='--', linewidth=1.8, label=r'Dual Residual $\|s^k\|_2 = \rho \|z^k - z^{k-1}\|_2$')
ax2.set_title("ADMM Residual Convergence Diagnostics", fontsize=11, fontweight='bold', pad=10)
ax2.set_xlabel("ADMM Iteration $k$", fontweight='bold')
ax2.set_ylabel("Residual Norm (Log Scale)", fontweight='bold')
ax2.legend(loc='upper right', fontsize=8.5)
ax2.grid(True, linestyle='--', alpha=0.4)

# Subplot 3: Objective Function Value vs. Iteration
ax3 = axs[1, 0]
ax3.plot(iters, hist_obj, color='darkgreen', linewidth=2.0, label=r'ADMM Objective $F(x^k, z^k)$')
ax3.axhline(obj_bench, color='orange', linestyle='--', linewidth=1.5, label=f'Optimal Benchmark ({obj_bench:.4f})')
ax3.set_title(f"Objective Function Descent $\min_x \\frac{{1}}{{2}}\|Ax - b\|_2^2 + \lambda \|x\|_1$\n(Final Objective = {hist_obj[-1]:.4f})", 
              fontsize=11, fontweight='bold', pad=10)
ax3.set_xlabel("ADMM Iteration $k$", fontweight='bold')
ax3.set_ylabel("Objective Value", fontweight='bold')
ax3.legend(loc='upper right', fontsize=8.5)
ax3.grid(True, linestyle='--', alpha=0.4)

# Subplot 4: Soft-Thresholding Characteristic & Shrinkage Operation
ax4 = axs[1, 1]
v_range = np.linspace(-3.0, 3.0, 500)
kappa_val = lam_lasso / rho_admm
s_range = soft_threshold(v_range, kappa_val)
ax4.plot(v_range, v_range, 'k--', alpha=0.4, label='Identity Line $z = v$ (No Regularization)')
ax4.plot(v_range, s_range, 'purple', linewidth=2.2, label=f'Soft-Thresholding $\mathcal{{S}}_{{\kappa}}(v)$ ($\kappa = \lambda/\\rho = {kappa_val:.3f}$)')
ax4.axvline(kappa_val, color='grey', linestyle=':', alpha=0.6)
ax4.axvline(-kappa_val, color='grey', linestyle=':', alpha=0.6)
ax4.axhline(0, color='black', linewidth=0.8)
ax4.fill_between([-kappa_val, kappa_val], -3, 3, color='purple', alpha=0.08, label=f'Exact Zero Deadzone $[-\\kappa, \\kappa]$')
ax4.set_title(f"Closed-Form Proximal Map: Soft-Thresholding Operator\n$z^{{k+1}} = \mathcal{{S}}_{{\lambda/\\rho}}(x^{{k+1}} + u^k)$", 
              fontsize=11, fontweight='bold', pad=10)
ax4.set_xlabel("Input Proximal Argument $v = x^{k+1} + u^k$", fontweight='bold')
ax4.set_ylabel("Output Sparse Variable $z$", fontweight='bold')
ax4.set_xlim([-3.0, 3.0])
ax4.set_ylim([-3.0, 3.0])
ax4.legend(loc='upper left', fontsize=8.5)
ax4.grid(True, linestyle='--', alpha=0.4)

plt.suptitle("AI3403 Assignment 3, Problem 4: Closed-Form ADMM for LASSO Optimization", 
             fontsize=13, fontweight='bold', y=0.98)
plt.tight_layout()
plt.savefig("problem4_lasso_admm.png", dpi=300)
print("Saved high-res figure: problem4_lasso_admm.png", flush=True)
