import torch
import matplotlib.pyplot as plt
from optimizers import SVDRiemannianGDSt, QRRiemannianGDSt, PolarRiemannianGDSt
from old.search_engines import Rayleigh
import time

torch.manual_seed(42)
torch.set_default_dtype(torch.float64) 
device = "cpu"

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

# Consts
max_iter = 40000
momentum = 0.9
n = 800
K_n = torch.eye(n, n) * 2.0 + torch.diag(torch.ones(n-1) * -1.0, 1) + torch.diag(torch.ones(n-1) * -1.0, -1)

k = torch.arange(1, n + 1)
eigvals = 2 - 2 * torch.cos(torch.pi * k / (n + 1))
tol = 1e-8

p = 100
true_val = torch.mean(eigvals[:p])
Q, _ = torch.linalg.qr(torch.randn(n, p))

fig, axes = plt.subplots(nrows=3, ncols=4, figsize=(width_in * 2, height_in * 3))

fig.suptitle(f'Rayleigh Quotient Optimization: $n={n}$, $p={p}$', 
             fontsize=12, y=0.995, fontweight='bold')

lr = 5e-1

X = Q.clone().detach().requires_grad_(True)
optimizer_Polar = PolarRiemannianGDSt([X], lr, momentum=momentum)
rel_errors = []
grad_norms = []
start_time = time.time()
for it in range(max_iter):
    optimizer_Polar.zero_grad()
    func = Rayleigh(K_n, X)
    func.backward()
    optimizer_Polar.step()

    grad_norm = optimizer_Polar.last_grad_norm
    rel_error = abs(func.item() - true_val) / abs(true_val)

    rel_errors.append(rel_error)
    grad_norms.append(grad_norm)

    if grad_norm < tol:
        break
elapsed_time = time.time() - start_time

axes[0, 0].plot(rel_errors, label=f'Vanilla Polar ({elapsed_time:.3f}s)')
axes[0, 0].grid()
axes[0, 0].set_yscale('log')
axes[0, 0].set_xlabel('Iteration')
axes[0, 0].set_ylabel(r'$\frac{|\lambda - \lambda^*|}{|\lambda^*|}$')
axes[0, 0].set_title('Relative error')
axes[0, 0].legend()

axes[0, 1].plot(grad_norms, label=f'Vanilla Polar ({elapsed_time:.3f}s)')
axes[0, 1].grid()
axes[0, 1].set_yscale('log')
axes[0, 1].set_xlabel('Iteration')
axes[0, 1].set_ylabel(r'$\|G\|_F$')
axes[0, 1].set_title('Grad norm')
axes[0, 1].legend()

X = Q.clone().detach().requires_grad_(True)
optimizer_Polar = PolarRiemannianGDSt([X], lr, momentum=momentum, line_search="steppest")
rel_errors = []
grad_norms = []
start_time = time.time()
for it in range(max_iter):
    optimizer_Polar.zero_grad()
    func = Rayleigh(K_n, X)
    func.backward()
    optimizer_Polar.step()

    grad_norm = optimizer_Polar.last_grad_norm
    rel_error = abs(func.item() - true_val) / abs(true_val)

    rel_errors.append(rel_error)
    grad_norms.append(grad_norm)

    if grad_norm < tol:
        break
elapsed_time = time.time() - start_time

axes[0, 2].plot(rel_errors, label=f'Steppest ({elapsed_time:.3f}s)')
axes[0, 2].grid()
axes[0, 2].set_yscale('log')
axes[0, 2].set_xlabel('Iteration')
axes[0, 2].set_ylabel(r'$\frac{|\lambda - \lambda^*|}{|\lambda^*|}$')
axes[0, 2].set_title('Relative error')
axes[0, 2].legend()

axes[0, 3].plot(grad_norms, label=f'Steppest ({elapsed_time:.3f}s)')
axes[0, 3].grid()
axes[0, 3].set_yscale('log')
axes[0, 3].set_xlabel('Iteration')
axes[0, 3].set_ylabel(r'$\|G\|_F$')
axes[0, 3].set_title('Grad norm')
axes[0, 3].legend()

X = Q.clone().detach().requires_grad_(True)
optimizer_Polar = PolarRiemannianGDSt([X], lr, momentum=momentum, line_search="armijo")
rel_errors = []
grad_norms = []
start_time = time.time()
for it in range(max_iter):
    optimizer_Polar.zero_grad()
    func = Rayleigh(K_n, X)
    func.backward()
    optimizer_Polar.step()

    grad_norm = optimizer_Polar.last_grad_norm
    rel_error = abs(func.item() - true_val) / abs(true_val)

    rel_errors.append(rel_error)
    grad_norms.append(grad_norm)

    if grad_norm < tol:
        break
elapsed_time = time.time() - start_time

axes[1, 0].plot(rel_errors, label=f'Armijo ({elapsed_time:.3f}s)')
axes[1, 0].grid()
axes[1, 0].set_yscale('log')
axes[1, 0].set_xlabel('Iteration')
axes[1, 0].set_ylabel(r'$\frac{|\lambda - \lambda^*|}{|\lambda^*|}$')
axes[1, 0].set_title('Relative error')
axes[1, 0].legend()

axes[1, 1].plot(grad_norms, label=f'Armijo ({elapsed_time:.3f}s)')
axes[1, 1].grid()
axes[1, 1].set_yscale('log')
axes[1, 1].set_xlabel('Iteration')
axes[1, 1].set_ylabel(r'$\|G\|_F$')
axes[1, 1].set_title('Grad norm')
axes[1, 1].legend()

X = Q.clone().detach().requires_grad_(True)
optimizer_Polar = PolarRiemannianGDSt([X], lr, momentum=momentum, line_search="newton-shultz", steps=1)
rel_errors = []
grad_norms = []
start_time = time.time()
for it in range(max_iter):
    optimizer_Polar.zero_grad()
    func = Rayleigh(K_n, X)
    func.backward()
    optimizer_Polar.step()

    grad_norm = optimizer_Polar.last_grad_norm
    rel_error = abs(func.item() - true_val) / abs(true_val)

    rel_errors.append(rel_error)
    grad_norms.append(grad_norm)

    if grad_norm < tol:
        break
elapsed_time = time.time() - start_time

axes[1, 2].plot(rel_errors, label=f'Newton-Shultz ({elapsed_time:.3f}s)')
axes[1, 2].grid()
axes[1, 2].set_yscale('log')
axes[1, 2].set_xlabel('Iteration')
axes[1, 2].set_ylabel(r'$\frac{|\lambda - \lambda^*|}{|\lambda^*|}$')
axes[1, 2].set_title('Relative error')
axes[1, 2].legend()

axes[1, 3].plot(grad_norms, label=f'Newton-Shultz ({elapsed_time:.3f}s)')
axes[1, 3].grid()
axes[1, 3].set_yscale('log')
axes[1, 3].set_xlabel('Iteration')
axes[1, 3].set_ylabel(r'$\|G\|_F$')
axes[1, 3].set_title('Grad norm')
axes[1, 3].legend()

X = Q.clone().detach().requires_grad_(True)
optimizer_Polar = PolarRiemannianGDSt([X], lr, momentum=momentum, line_search="PolarExpress")
rel_errors = []
grad_norms = []
start_time = time.time()
for it in range(max_iter):
    optimizer_Polar.zero_grad()
    func = Rayleigh(K_n, X)
    func.backward()
    optimizer_Polar.step()

    grad_norm = optimizer_Polar.last_grad_norm
    rel_error = abs(func.item() - true_val) / abs(true_val)

    rel_errors.append(rel_error)
    grad_norms.append(grad_norm)

    if grad_norm < tol:
        break
elapsed_time = time.time() - start_time

axes[2, 0].plot(rel_errors, label=f'PolarExpress ({elapsed_time:.3f}s)')
axes[2, 0].grid()
axes[2, 0].set_yscale('log')
axes[2, 0].set_xlabel('Iteration')
axes[2, 0].set_ylabel(r'$\frac{|\lambda - \lambda^*|}{|\lambda^*|}$')
axes[2, 0].set_title('Relative error')
axes[2, 0].legend()

axes[2, 1].plot(grad_norms, label=f'PolarExpress ({elapsed_time:.3f}s)')
axes[2, 1].grid()
axes[2, 1].set_yscale('log')
axes[2, 1].set_xlabel('Iteration')
axes[2, 1].set_ylabel(r'$\|G\|_F$')
axes[2, 1].set_title('Grad norm')
axes[2, 1].legend()

X = Q.clone().detach().requires_grad_(True)
optimizer_Polar = PolarRiemannianGDSt([X], lr, momentum=momentum, line_search="CANS", steps=6)
rel_errors = []
grad_norms = []
start_time = time.time()
for it in range(max_iter):
    optimizer_Polar.zero_grad()
    func = Rayleigh(K_n, X)
    func.backward()
    optimizer_Polar.step()

    grad_norm = optimizer_Polar.last_grad_norm
    rel_error = abs(func.item() - true_val) / abs(true_val)

    rel_errors.append(rel_error)
    grad_norms.append(grad_norm)

    if grad_norm < tol:
        break
elapsed_time = time.time() - start_time

axes[2, 2].plot(rel_errors, label=f'CANS ({elapsed_time:.3f}s)')
axes[2, 2].grid()
axes[2, 2].set_yscale('log')
axes[2, 2].set_xlabel('Iteration')
axes[2, 2].set_ylabel(r'$\frac{|\lambda - \lambda^*|}{|\lambda^*|}$')
axes[2, 2].set_title('Relative error')
axes[2, 2].legend()

axes[2, 3].plot(grad_norms, label=f'CANS ({elapsed_time:.3f}s)')
axes[2, 3].grid()
axes[2, 3].set_yscale('log')
axes[2, 3].set_xlabel('Iteration')
axes[2, 3].set_ylabel(r'$\|G\|_F$')
axes[2, 3].set_title('Grad norm')
axes[2, 3].legend()

plt.savefig("n=800_p=100.pdf")
plt.tight_layout()
plt.show()