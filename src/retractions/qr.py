import torch


def qr_retraction(X, Xi):
    should_transpose = X.size(-2) < X.size(-1)
    if should_transpose:
        X = X.mT
        Xi = Xi.mT
    Q, R = torch.linalg.qr(X + Xi)
    diag = torch.diagonal(R, dim1=-2, dim2=-1)
    signs = torch.sign(diag)
    signs[signs == 0] = 1.0
    Q = Q * signs.unsqueeze(-2)
    if should_transpose:
        Q = Q.mT
    return Q
