from itertools import repeat
from math import inf, sqrt
from typing import List

import numpy as np
import torch


def optimal_cubic(l, u):
    alpha = sqrt(3 / (u**2 + l * u + l**2))
    beta = 4 / (2 + l * u * (l + u) * alpha**3)
    return (3 / 2 * alpha * beta, -1 / 2 * alpha**3 * beta)


def optimal_quintic(l, u):
    assert 0 <= l <= u
    if 1 - 5e-06 <= l / u:
        return (15 / 8 / u, -10 / 8 / u**3, 3 / 8 / u**5)
    q = (3 * l + u) / 4
    r = (l + 3 * u) / 4
    E, old_E = (inf, None)
    while not old_E or abs(old_E - E) > 1e-15:
        old_E = E
        LHS = np.array(
            [
                [l, l**3, l**5, 1],
                [q, q**3, q**5, -1],
                [r, r**3, r**5, 1],
                [u, u**3, u**5, -1],
            ]
        )
        a, b, c, E = np.linalg.solve(LHS, np.ones(4))
        q, r = np.sqrt(
            (-3 * b + np.array([-1, 1]) * sqrt(9 * b**2 - 20 * a * c)) / (10 * c)
        )
    return (float(a), float(b), float(c))


def optimal_composition(l, num_iters, safety_factor_eps=0, cushion=0):
    u = 1
    assert 0 <= l <= u
    safety_factor = 1 + safety_factor_eps
    coefficients = []
    for iter in range(num_iters):
        a, b, c = optimal_quintic(max(l, cushion * u), u)
        if cushion * u > l:
            pl = a * l + b * l**3 + c * l**5
            pu = a * u + b * u**3 + c * u**5
            rescaler = 2 / (pl + pu)
            a *= rescaler
            b *= rescaler
            c *= rescaler
        if iter < num_iters - 1:
            a /= safety_factor
            b /= safety_factor**3
            c /= safety_factor**5
        coefficients.append((a, b, c))
        l = a * l + b * l**3 + c * l**5
        u = 2 - l
    return coefficients


@torch.compile
def PolarExpress(G: torch.Tensor, steps: int, coeffs_list: List) -> torch.Tensor:
    assert G.ndim >= 2
    X = G.bfloat16()
    if G.size(-2) > G.size(-1):
        X = X.mT
    X = X / (X.norm(dim=(-2, -1), keepdim=True) * 1.01 + 1e-07)
    hs = coeffs_list[:steps] + list(repeat(coeffs_list[-1], steps - len(coeffs_list)))
    for a, b, c in hs:
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X
    if G.size(-2) > G.size(-1):
        X = X.mT
    return X
