import torch

def fixed_step_rule(X, H, grad, lr, objective=None, retraction=None):
    return -lr * H

def armijo_step_rule(
    X, H, grad, lr, objective, retraction,
    c: float = 1e-4,
    tau: float = 0.5,
    max_iter: int = 50,
):
    fX = objective(X).item()
    alpha = lr
    arm = -torch.sum(grad * H).item()

    if arm >= 0:
        H = grad
        arm = -torch.sum(grad * grad).item()

    with torch.no_grad():
        for _ in range(max_iter):
            Xi = -alpha * H
            X_next = retraction(X, Xi)
            f_next = objective(X_next).item()

            if f_next <= fX + c * alpha * arm:
                return Xi

            alpha *= tau

    return -alpha * H

def make_rayleigh_steepest_rule(A):
    def steepest_rule(X, H, grad, lr=None, objective=None, retraction=None):
        alpha = torch.trace(X.T @ (A @ H)) / torch.trace(H.T @ (A @ H))
        return -alpha * H
    return steepest_rule