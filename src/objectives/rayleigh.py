import torch


def rayleigh(A, X):
    return torch.trace(X.T @ (A @ X)) / X.shape[1]


def make_rayleigh_objective(A):
    def objective(X):
        return rayleigh(A, X)

    return objective
