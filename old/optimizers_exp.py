import torch
import matplotlib.pyplot as plt
from optimizers import SVDRiemannianGDSt, QRRiemannianGDSt, PolarRiemannianGDSt
from old.search_engines import Rayleigh
import time
from tqdm import tqdm

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
# Consts
max_iter = 250000
lr_SVD =5e-1
lr_QR = 5e-1
lr_Polar = 5e-1
momentum = 0.9
n = 800
tol = 5e-9

def Rayleigh(A, X1, X2 = None):
    if X2 is None:
        X2 = X1
    return 1/X1.shape[1] * torch.trace(X1.T @ (A @ X2))

K_n = torch.eye(n, device=device) * 2.0 + \
      torch.diag(torch.ones(n-1, device=device) * -1.0, 1) + \
      torch.diag(torch.ones(n-1, device=device) * -1.0, -1)

def Rayleigh_f(X1, X2 = None):
    if X2 is None:
        return Rayleigh(K_n, X)
    return Rayleigh(K_n, X1, X2)

k = torch.arange(1, n + 1, device=device)
eigvals = (2 - 2 * torch.cos(torch.pi * k / (n + 1))).to(device)
LOG_INTERVAL = 1

for p in [100]:
    rel_errors = {"SVD" : [], "QR" : [], "Polar" : []}
    grad_norms = {"SVD" : [], "QR" : [], "Polar" : []}
    times = {"SVD" : 0.0, "QR" : 0.0, "Polar" : 0.0}
    true_val = torch.mean(eigvals[:p])

    Q, _ = torch.linalg.qr(torch.randn(n, p, device=device))


    X = Q.clone().detach().requires_grad_(True)
    optimizer_SVD = SVDRiemannianGDSt([X], lr_SVD, momentum=momentum)

    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    start_time = time.time()
    for it in tqdm(range(max_iter), desc = "SVD_retraction"):
        optimizer_SVD.zero_grad()
        func = Rayleigh(K_n, X)
        func.backward()
        optimizer_SVD.step()

        grad_norm = optimizer_SVD.last_grad_norm

        if it % LOG_INTERVAL == 0 or grad_norm < tol:
            rel_error = abs(func.item() - true_val.item()) / abs(true_val.item())
            rel_errors["SVD"].append(rel_error)
            grad_norms["SVD"].append(grad_norm)

        if grad_norm < tol:
            break
    
    if device.type == 'cuda':
        torch.cuda.synchronize()
    elapsed_time = time.time() - start_time
    times['SVD'] = elapsed_time

    X = Q.clone().detach().requires_grad_(True)
    optimizer_QR = QRRiemannianGDSt([X], lr_QR, momentum=momentum)
    
    if device.type == 'cuda':
        torch.cuda.synchronize()
        
    start_time = time.time()
    for it in tqdm(range(max_iter), desc = "QR_retraction"):
        optimizer_QR.zero_grad()
        func = Rayleigh(K_n, X)
        func.backward()
        optimizer_QR.step()

        grad_norm = optimizer_QR.last_grad_norm
        
        if it % LOG_INTERVAL == 0 or grad_norm < tol:
            rel_error = abs(func.item() - true_val.item()) / abs(true_val.item())
            rel_errors["QR"].append(rel_error)
            grad_norms["QR"].append(grad_norm)

        if grad_norm < tol:
            break
    
    if device.type == 'cuda':
        torch.cuda.synchronize()
    elapsed_time = time.time() - start_time
    times['QR'] = elapsed_time

    X = Q.clone().detach().requires_grad_(True)
    optimizer_Polar = PolarRiemannianGDSt([X], lr_Polar, momentum=momentum)
    
    if device.type == 'cuda':
        torch.cuda.synchronize()
        
    start_time = time.time()
    for it in tqdm(range(max_iter), desc = "Vanilla_polar_retraction"):
        optimizer_Polar.zero_grad()
        func = Rayleigh(K_n, X)
        func.backward()
        optimizer_Polar.step()

        grad_norm = optimizer_Polar.last_grad_norm
        
        if it % LOG_INTERVAL == 0 or grad_norm < tol:
            rel_error = abs(func.item() - true_val.item()) / abs(true_val.item())
            rel_errors["Polar"].append(rel_error)
            grad_norms["Polar"].append(grad_norm)

        if grad_norm < tol:
            break

    if device.type == 'cuda':
        torch.cuda.synchronize()
    elapsed_time = time.time() - start_time
    times['Polar'] = elapsed_time

    iterations = list(range(0, max_iter, LOG_INTERVAL))
    
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(width_in, height_in))
    axes[0].plot(iterations[:len(rel_errors["SVD"])], rel_errors["SVD"])
    axes[0].plot(iterations[:len(rel_errors["QR"])], rel_errors["QR"])
    axes[0].plot(iterations[:len(rel_errors["Polar"])], rel_errors["Polar"])
    axes[0].grid()
    axes[0].set_yscale('log')
    axes[0].set_xlabel('Iteration')
    axes[0].set_ylabel(r'$\frac{|\lambda - \lambda^*|}{|\lambda^*|}$')
    axes[0].set_title('Relative error')
    
    axes[1].plot(iterations[:len(grad_norms["SVD"])], grad_norms["SVD"], label = f'SVD Retraction, ({times["SVD"]:.3f}s)')
    axes[1].plot(iterations[:len(grad_norms["QR"])], grad_norms["QR"], label = f'QR Retraction, ({times["QR"]:.3f}s)')
    axes[1].plot(iterations[:len(grad_norms["Polar"])], grad_norms["Polar"], label = f'Polar Retraction, ({times["Polar"]:.3f}s)')
    axes[1].grid()
    axes[1].set_yscale('log')
    axes[1].set_xlabel('Iteration')
    axes[1].set_ylabel(r'$\|G\|_F$')
    axes[1].set_title('Grad norm')

    plt.legend()
    plt.tight_layout()
    plt.savefig(f'svd,qr,polar,n={n},p={p}.pdf', **save_kwargs)
    plt.show()