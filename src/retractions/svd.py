import torch


def svd_retraction(X, Xi):
    X_new = X + Xi
    U, _, VT = torch.linalg.svd(X_new, full_matrices=False)
    return U @ VT
