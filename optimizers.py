import torch
from torch.optim import Optimizer

class SVDRiemannianGDSt(Optimizer):

    def __init__(self, params, lr=1e-2):
        defaults = dict(lr=lr)
        super().__init__(params, defaults)
        self.last_grad_norm = 0.0

    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                X = p.data

                H = grad - X @ (X.T @ grad + grad.T @ X) * 0.5
                self.last_grad_norm = torch.norm(H, 'fro').item()
                TxSTgrad = X - lr * H
                U, _, VT = torch.linalg.svd(TxSTgrad, full_matrices=False)
                STgrad = U @ VT

                p.data.copy_(STgrad)

        return loss
    

class QRRiemannianGDSt(Optimizer):

    def __init__(self, params, lr=1e-2):
        defaults = dict(lr=lr)
        super().__init__(params, defaults)
        self.last_grad_norm = 0.0

    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad.data
                X = p.data

                H = grad - X @ (X.T @ grad + grad.T @ X) * 0.5
                self.last_grad_norm = torch.norm(H, 'fro').item()
                TxSTgrad = X - lr * H

                U, _ = torch.linalg.qr(TxSTgrad)
                STgrad = U
                p.data.copy_(STgrad)

        return loss

class PolarRiemannianGDSt(Optimizer):

    def __init__(self, params, lr=1e-2):
        defaults = dict(lr=lr)
        super().__init__(params, defaults)
        self.last_grad_norm = 0.0

    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad.data
                X = p.data

                H = grad - X @ (X.T @ grad + grad.T @ X) * 0.5
                self.last_grad_norm = torch.norm(H, 'fro').item()

                TxSTgrad = - lr * H

                PolarMatrix = torch.eye(TxSTgrad.shape[1]) + TxSTgrad.T @ TxSTgrad
                Lambda, V = torch.linalg.eigh(PolarMatrix)
                Lambda = torch.clamp(Lambda, min=1e-4)

                STgrad = (X + TxSTgrad) @ (V @ torch.diag(1.0/torch.sqrt(Lambda)) @ V.T)
                
                p.data.copy_(STgrad)

        return loss
