import torch
from polar_express import optimal_composition, PolarExpress
from cans_realization import cans_retraction

def polar_from_matrix(H):
    eigvals, eigvecs = torch.linalg.eigh(H.T @ H)
    inv_sqrt = eigvecs @ torch.diag(1.0 / torch.sqrt(eigvals)) @ eigvecs.T
    return H @ inv_sqrt

def svd_retraction(X, Xi):
    X_new = X + Xi
    U, _, VT = torch.linalg.svd(X_new, full_matrices=False)
    return U @ VT

def qr_retraction(X, Xi):
    X_new = X + Xi
    Q, _ = torch.linalg.qr(X_new)
    return Q

def polar_retraction(X, Xi):
    return polar_from_matrix(X + Xi)

def make_newton_schulz_retraction(steps = 10):

    def newton_schulz_iteration(X, Xi):
        X_new = X + Xi
        X_new = X_new / (X_new.norm(dim=(-2, -1), keepdim=True) * 1.01 + 1e-7)
        for _ in range(steps):
            X_new = 1.5 * X_new - 0.5 * (X_new @ (X_new.T @ X_new))
        return X_new

    return newton_schulz_iteration


def make_cans_retraction(steps = 8):

    def cans_iteration(X, Xi):
        return cans_retraction(X + Xi, steps)

    return cans_iteration


def make_polar_express_retraction(steps = 8, l = 1e-3, safety_factor_eps = 1e-2):

    coeffs_list = optimal_composition(l=l, num_iters=steps, safety_factor_eps=safety_factor_eps)

    def polar_express_iteration(X, Xi):
        return PolarExpress(X + Xi, steps=steps, coeffs_list=coeffs_list)

    return polar_express_iteration

def geodesic_retraction(X, H, t=1.0):
    p = X.shape[1]

    A = X.T @ H
    A = 0.5 * (A - A.T)

    K = H - X @ A
    Q, R = torch.linalg.qr(K)

    Z = torch.zeros((p, p), dtype=X.dtype, device=X.device)
    block = torch.cat([
        torch.cat([A, -R.T], dim=1),
        torch.cat([R,  Z], dim=1),
    ], dim=0)

    MN = torch.linalg.matrix_exp(t * block)[:, :p]

    M = MN[:p, :]
    N = MN[p:, :]

    X_new = X @ M + Q @ N
    return X_new