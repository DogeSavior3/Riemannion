import torch
torch.manual_seed(42)
torch.set_default_dtype(torch.float64) 
device = "cpu"

def Rayleigh(A, X):
	return 1/X.shape[1] * torch.trace(X.T @ (A @ X))

n=100
K_n = torch.eye(n, n) * 2.0 + torch.diag(torch.ones(n-1) * -1.0, 1) + torch.diag(torch.ones(n-1) * -1.0, -1)

def Rayleigh_f(X):
    return Rayleigh(K_n, X)

def Rayleigh_H(X, H):
    return torch.trace(X.T @ (K_n @ H)) / torch.trace(H.T @ (K_n @ H))

def steppest(X, H):
    return Rayleigh_H(X,H) / Rayleigh_H(H, H)

def polar_retraction(H):
    PolarMatrix = H.T @ H
    Lambda, V = torch.linalg.eigh(PolarMatrix)
    return H @ (V @ torch.diag(1.0 / torch.sqrt(Lambda)) @ V.T)

def armijo_line_search(X, H, fX, fn, lr = 1e-1, c = 1e-4, tau = 0.8, max_iter = 50):
# TODO: Гольштейн
    grad_norm = torch.trace(H.T @ H).item()
    alpha = lr
    for _ in range(max_iter):
        D = X - alpha * H
        X_new = polar_retraction(D)
        # X_new = D
        f_new = fn(X_new)

        if f_new <= fX + c * alpha * grad_norm:
            return X_new

        alpha *= tau

    return X_new
