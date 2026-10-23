import torch


def geodesic_retraction(X, H, t=1.0):
    p = X.shape[1]
    A = X.T @ H
    A = 0.5 * (A - A.T)
    K = H - X @ A
    Q, R = torch.linalg.qr(K)
    Z = torch.zeros((p, p), dtype=X.dtype, device=X.device)
    block = torch.cat([torch.cat([A, -R.T], dim=1), torch.cat([R, Z], dim=1)], dim=0)
    MN = torch.linalg.matrix_exp(t * block)[:, :p]
    M = MN[:p, :]
    N = MN[p:, :]
    X_new = X @ M + Q @ N
    return X_new
