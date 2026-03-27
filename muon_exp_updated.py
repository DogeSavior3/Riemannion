import torch
import matplotlib.pyplot as plt
import time
from tqdm import tqdm

from optimizer import RiemannianOptimizerSt
from retraction import polar_retraction, svd_retraction, qr_retraction, geodesic_retraction
from step_algorithm import fixed_step_rule, armijo_step_rule, make_rayleigh_steepest_rule
from transport import ambient_vector_transport
from optimizer import svd_muon_v1, svd_muon_v2

torch.manual_seed(42)
torch.set_default_dtype(torch.float64) 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. EPS-SPECIFIC SETTINGS
plt.rcParams["savefig.format"] = "eps"  # Set default output format
plt.rcParams["ps.useafm"] = True        # Use Adobe Font Metrics (better for EPS)
plt.rcParams["pdf.use14corefonts"] = True  # Compatible core fonts
plt.rcParams["ps.fonttype"] = 42        # Type 42 (TrueType) - best quality
plt.rcParams["pdf.fonttype"] = 42       # For any PDF fallback

# 2. FIGURE SIZE (174mm max width → ~6.85 inches)
width_in = 6                         # 174mm converted to inches
height_in = width_in * 0.52              # 60% of width (adjust ratio as needed)
plt.rcParams["figure.figsize"] = (width_in, height_in)

# # 3. FONT CONFIGURATION (No LaTeX)
# plt.rcParams["text.usetex"] = False
plt.rcParams["text.usetex"] = True
plt.rcParams["font.family"] = "serif"
# # Try these font fallbacks in order:
# plt.rcParams["font.serif"] = ["Times New Roman", "DejaVu Serif", "Liberation Serif"]

# 4. LINE QUALITY
plt.rcParams["lines.linewidth"] = 1.2   # Slightly thicker for vector output
plt.rcParams["lines.markersize"] = 6    # Visible but not oversized

# 5. FONT SIZES (optimized for print)
plt.rcParams["font.size"] = 10          # Base size
plt.rcParams["axes.labelsize"] = 10     # Axis labels
plt.rcParams["xtick.labelsize"] = 9     # Smaller ticks
plt.rcParams["ytick.labelsize"] = 9
plt.rcParams["legend.fontsize"] = 8     # Compact legend text
plt.rcParams["axes.titlesize"] = 10     # Title size
plt.rcParams["lines.markersize"] = 6           # Default marker size
plt.rcParams["lines.markeredgewidth"] = 1.0    # Marker edge thickness

# 6. SAVING PARAMETERS (Call this when saving)
save_kwargs = {
    "format": "pdf",
    "dpi": 300,
    "bbox_inches": "tight",  # Crops whitespace
    "transparent": True  # If you need transparency
}

def rayleigh(A, X):
    return torch.trace(X.T @ (A @ X)) / X.shape[1]

def make_rayleigh_objective(A):
    def objective(X):
        return rayleigh(A, X)

    return objective


def plot_logs(rel_errors, grad_norms, label):
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(width_in, height_in))

    axes[0].plot(rel_errors, label=label)
    axes[0].grid()
    axes[0].set_yscale("log")
    axes[0].set_xlabel("Iteration")
    axes[0].set_ylabel(r"$\frac{|\lambda-\lambda^*|}{|\lambda^*|}$")
    axes[0].set_title("Relative error")

    axes[1].plot(grad_norms, label=label)
    axes[1].grid()
    axes[1].set_yscale("log")
    axes[1].set_xlabel("Iteration")
    axes[1].set_ylabel(r"$\|G\|_F$")
    axes[1].set_title("Gradient norm")

    axes[0].legend()
    axes[1].legend()
    plt.tight_layout()
    plt.show()
def run_experiment(objective, optimizer, X, true_value, max_iter, tol, desc=""):
    rel_errors = []
    grad_norms = []

    start_time = time.perf_counter()

    for _ in tqdm(range(max_iter), desc=desc, leave=False):
        optimizer.zero_grad()

        value = objective(X)
        value.backward()

        optimizer.step()

        grad_norm = optimizer.last_grad_norm
        rel_error = abs(value.item() - true_value) / abs(true_value)

        rel_errors.append(rel_error)
        grad_norms.append(grad_norm)

        if grad_norm < tol:
            break

    elapsed = time.perf_counter() - start_time
    return rel_errors, grad_norms, elapsed

import os

os.environ["OMP_NUM_THREADS"] = "8"
os.environ["MKL_NUM_THREADS"] = "8"

print("torch.get_num_threads()      =", torch.get_num_threads())
print("torch.get_num_interop_threads() =", torch.get_num_interop_threads())
print("OMP_NUM_THREADS =", os.environ.get("OMP_NUM_THREADS"))
print("MKL_NUM_THREADS =", os.environ.get("MKL_NUM_THREADS"))

n = 1200
p_list = [200]
max_iter = 200000
lr = 5e-1
momentum = 0.2
tol = 1e-8

K_n = torch.eye(n, n) * 2.0 + torch.diag(torch.ones(n - 1) * -1.0, 1) + torch.diag(torch.ones(n - 1) * -1.0, -1)
k = torch.arange(1, n + 1)
eigvals = 2 - 2 * torch.cos(torch.pi * k / (n + 1))

fig, axes = plt.subplots(nrows=len(p_list),ncols=2,
    figsize=(width_in, height_in * len(p_list)), squeeze=False,
)

for row_idx, p in enumerate(p_list):
    Q, _ = torch.linalg.qr(torch.randn(n, p))

    objective = make_rayleigh_objective(K_n)
    true_val = torch.mean(eigvals[:p]).item()

    X_svd = Q.clone().detach().requires_grad_(True)
    X_qr = Q.clone().detach().requires_grad_(True)
    X_polar = Q.clone().detach().requires_grad_(True)

    optimizer_svd = RiemannianOptimizerSt(
        [X_svd],
        lr=lr,
        momentum=momentum,
        retraction=svd_retraction,
        step_rule=fixed_step_rule,
        objective=objective,
    )

    optimizer_qr = RiemannianOptimizerSt(
        [X_qr],
        lr=lr,
        momentum=momentum,
        retraction=qr_retraction,
        step_rule=fixed_step_rule,
        objective=objective,
    )

    optimizer_polar = RiemannianOptimizerSt(
        [X_polar],
        lr=lr,
        momentum=momentum,
        retraction=polar_retraction,
        step_rule=fixed_step_rule,
        objective=objective,
    )

    experiments = [
        ("SVD retraction", optimizer_svd, X_svd),
        ("QR retraction", optimizer_qr, X_qr),
        ("Polar retraction", optimizer_polar, X_polar),
    ]

    logs = []
    for label, optimizer, X in experiments:
        rel_errors, grad_norms, elapsed = run_experiment(
            objective=objective,
            optimizer=optimizer,
            X=X,
            true_value=true_val,
            max_iter=max_iter,
            tol=tol,
            desc=f"p={p} | {label}",
        )
        logs.append((f"{label} ({elapsed:.2f}s)", rel_errors, grad_norms))

    ax_err = axes[row_idx, 0]
    ax_grad = axes[row_idx, 1]

    for label, rel_errors, grad_norms in logs:
        ax_err.plot(rel_errors, label=label)
        ax_grad.plot(grad_norms, label=label)

    ax_err.grid()
    ax_err.set_yscale("log")
    ax_err.set_xlabel("Iteration")
    ax_err.set_ylabel(r"$\frac{|\lambda-\lambda^*|}{|\lambda^*|}$")
    ax_err.set_title(f"Relative error, p={p}")

    ax_grad.grid()
    ax_grad.set_yscale("log")
    ax_grad.set_xlabel("Iteration")
    ax_grad.set_ylabel(r"$\|G\|_F$")
    ax_grad.set_title(f"Gradient norm, p={p}")

    ax_err.legend()
    ax_grad.legend()

plt.tight_layout()
plt.show()