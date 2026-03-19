import torch

def fixed_step_rule(X, H, lr, objective=None, retraction=None):
    return -lr * H

def armijo_step_rule(
    X, H, lr, objective, retraction,
    c: float = 1e-4,
    tau: float = 0.8,
    max_iter: int = 50,
):

    fX = objective(X)
    grad_norm = torch.trace(H.T @ H).item()
    alpha = lr

    for _ in range(max_iter):
        Xi = -alpha * H
        X_next = retraction(X, Xi)
        f_next = objective(X_next)

        if f_next <= fX - c * alpha * grad_norm:
            return Xi

        alpha *= tau

    return -alpha * H

def make_rayleigh_steepest_rule(A):
    def steepest_rule(X, H, lr=None, objective=None, retraction=None):
        alpha = torch.trace(X.T @ (A @ H)) / torch.trace(H.T @ (A @ H))
        return -alpha * H
    return steepest_rule