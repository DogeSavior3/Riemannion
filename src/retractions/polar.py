import torch


def polar_from_matrix(H, eps=1e-06):
    eigvals, eigvecs = torch.linalg.eigh(H.T @ H)
    eigvals = torch.clamp(eigvals, min=eps)
    inv_sqrt = eigvecs @ torch.diag(1.0 / torch.sqrt(eigvals)) @ eigvecs.T
    return H @ inv_sqrt


def polar_retraction(X, Xi):
    return polar_from_matrix(X + Xi)
