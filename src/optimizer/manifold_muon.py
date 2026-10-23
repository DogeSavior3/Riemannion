import math

import torch

from src.orthogonalization.msign import msign


@torch.no_grad()
def manifold_muon(W, G, eta=0.1, alpha=0.01, steps=100, tol=1e-6):

    should_tranpose = W.shape[0] < W.shape[1]
    if should_tranpose:
        W = W.T
        G = G.T

    Lambda = -0.25 * (W.T @ G + G.T @ W)

    for step in range(steps):
        A = msign(G + 2 * W @ Lambda)

        H = W.T @ A + A.T @ W

        if torch.norm(H) / math.sqrt(H.numel()) < tol:
            break

        Lambda -= alpha * (1 - step / steps) * H

    new_W = W - eta * A

    new_W = msign(new_W)

    return new_W.T if should_tranpose else new_W
