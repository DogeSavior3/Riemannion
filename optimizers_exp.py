import torch
import matplotlib.pyplot as plt
from optimizers import SVDRiemannianGDSt, QRRiemannianGDSt, PolarRiemannianGDSt

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
    "format": "eps",
    "dpi": 300,
    "bbox_inches": "tight",  # Crops whitespace
    "transparent": True  # If you need transparency
}

# Consts

max_iter = 100000
lr = 5e-2
momentum = 0.9
n = 100

def Rayleigh(A, X):
	return 1/X.shape[1] * torch.trace(X.T @ (A @ X))

k = torch.arange(1, n + 1)
eigvals = 2 - 2 * torch.cos(torch.pi * k / (n + 1))
K_n = torch.eye(n, n) * 2.0 + torch.diag(torch.ones(n-1) * -1.0, 1) + torch.diag(torch.ones(n-1) * -1.0, -1)

# exps:

for p in [2, 3, 10, 50]:
    rel_errors = {"SVD" : [], "QR" : [], "Polar" : []}
    grad_norms = {"SVD" : [], "QR" : [], "Polar" : []}
    true_val = torch.mean(eigvals[:p])

    Q, _ = torch.linalg.qr(torch.randn(n, p))

    X = Q.clone().detach().requires_grad_(True)
    optimizer_SVD = SVDRiemannianGDSt([X], lr, momentum)

    for it in range(max_iter):
        optimizer_SVD.zero_grad()
        func = Rayleigh(K_n, X)
        func.backward()

        optimizer_SVD.step()

        grad_norm = optimizer_SVD.last_grad_norm
        rel_error = abs(func.item() - true_val) / abs(true_val)

        rel_errors["SVD"].append(rel_error)
        grad_norms["SVD"].append(grad_norm)

        if grad_norm < 1e-14:
            break

    X = Q.clone().detach().requires_grad_(True)
    optimizer_QR = QRRiemannianGDSt([X], lr, momentum)

    for it in range(max_iter):
        optimizer_QR.zero_grad()
        func = Rayleigh(K_n, X)
        func.backward()

        optimizer_QR.step()

        grad_norm = optimizer_QR.last_grad_norm
        rel_error = abs(func.item() - true_val) / abs(true_val)

        rel_errors["QR"].append(rel_error)
        grad_norms["QR"].append(grad_norm)

        if grad_norm < 1e-14:
            break

    X = Q.clone().detach().requires_grad_(True)
    optimizer_Polar = PolarRiemannianGDSt([X], lr, momentum)

    for it in range(max_iter):
        optimizer_Polar.zero_grad()
        func = Rayleigh(K_n, X)
        func.backward()

        optimizer_Polar.step()

        grad_norm = optimizer_Polar.last_grad_norm
        rel_error = abs(func.item() - true_val) / abs(true_val)

        rel_errors["Polar"].append(rel_error)
        grad_norms["Polar"].append(grad_norm)

        if grad_norm < 1e-14:
            break

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(width_in, height_in))
    axes[0].plot(rel_errors["SVD"], label = 'SVD Retraction')
    axes[0].plot(rel_errors["QR"], label = 'QR Retraction')
    axes[0].plot(rel_errors["Polar"], label = 'Polar Retraction')
    axes[0].grid()
    axes[0].set_yscale('log')
    axes[0].set_xlabel('Iteration')
    axes[0].set_ylabel(r'$\frac{|\lambda - \lambda^*|}{|\lambda^*|}$')
    axes[0].set_title('Relative error')
    axes[1].plot(grad_norms["SVD"], label = 'SVD Retraction')
    axes[1].plot(grad_norms["QR"], label = 'QR Retraction')
    axes[1].plot(grad_norms["Polar"], label = 'Polar Retraction')
    axes[1].grid()
    axes[1].set_yscale('log')
    axes[1].set_xlabel('Iteration')
    axes[1].set_ylabel(r'$\|G\|_F$')
    axes[1].set_title('Grad norm')

    plt.legend()
    plt.tight_layout()
    plt.savefig(f'p={p}_all_opt.eps', **save_kwargs)
    plt.show()
