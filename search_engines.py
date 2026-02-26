import torch

def steppest():
    pass

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
            return alpha, X_new

        alpha *= tau

    return alpha, X_new
