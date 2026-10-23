import torch


def make_rayleigh_steepest_rule(A):

    def steepest_rule(X, H, grad, lr=None, objective=None, retraction=None):
        alpha = torch.trace(X.T @ (A @ H)) / torch.trace(H.T @ (A @ H))
        return -alpha * H

    return steepest_rule
