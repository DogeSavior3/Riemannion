import torch
from torch.optim import Optimizer

class SVDRiemannianGDSt(Optimizer):

    def __init__(self, params, lr=1e-2, momentum=None):
        defaults = dict(lr=lr, momentum=momentum)
        super().__init__(params, defaults)
        self.last_grad_norm = 0.0

    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            momentum = group['momentum']
            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad.data
                X = p.data
                H = grad - X @ (X.T @ grad + grad.T @ X) * 0.5

                if momentum is not None:
                    if 'momentum_buffer' not in self.state[p]:
                        self.state[p]['momentum_buffer'] = torch.zeros_like(H)

                    h = self.state[p]['momentum_buffer']
                    h_new = h - X @ (X.T @ h + h.T @ X) * 0.5
                    h = momentum * h_new + H
                    self.state[p]['momentum_buffer'] = h
                    TxSTgrad = X - lr * h
                    self.last_grad_norm = torch.norm(h, 'fro').item()
                else:
                    TxSTgrad = X - lr * H
                    self.last_grad_norm = torch.norm(H, 'fro').item()
                
                U, _, VT = torch.linalg.svd(TxSTgrad, full_matrices=False)
                STgrad = U @ VT

                p.data.copy_(STgrad)

        return loss
    

class QRRiemannianGDSt(Optimizer):

    def __init__(self, params, lr=1e-2, momentum=None):
        defaults = dict(lr=lr, momentum=momentum)
        super().__init__(params, defaults)
        self.last_grad_norm = 0.0

    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            momentum = group['momentum']
            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad.data
                X = p.data
                H = grad - X @ (X.T @ grad + grad.T @ X) * 0.5

                if momentum is not None:
                    if 'momentum_buffer' not in self.state[p]:
                        self.state[p]['momentum_buffer'] = torch.zeros_like(H)

                    h = self.state[p]['momentum_buffer']
                    h_new = h - X @ (X.T @ h + h.T @ X) * 0.5
                    h = momentum * h_new + H
                    self.state[p]['momentum_buffer'] = h
                    TxSTgrad = X - lr * h
                    self.last_grad_norm = torch.norm(h, 'fro').item()
                else:
                    TxSTgrad = X - lr * H
                    self.last_grad_norm = torch.norm(H, 'fro').item()

                U, _ = torch.linalg.qr(TxSTgrad)
                STgrad = U
                p.data.copy_(STgrad)

        return loss

class PolarRiemannianGDSt(Optimizer):

    def __init__(self, params, lr=1e-2, momentum=None):
        defaults = dict(lr=lr, momentum=momentum)
        super().__init__(params, defaults)
        self.last_grad_norm = 0.0

    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            momentum = group['momentum']
            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad.data
                X = p.data
                H = grad - X @ (X.T @ grad + grad.T @ X) * 0.5

                if momentum is not None:
                    if 'momentum_buffer' not in self.state[p]:
                        self.state[p]['momentum_buffer'] = torch.zeros_like(H)

                    h = self.state[p]['momentum_buffer']
                    h_new = h - X @ (X.T @ h + h.T @ X) * 0.5
                    h = momentum * h_new + H
                    self.state[p]['momentum_buffer'] = h
                    TxSTgrad = - lr * h
                    self.last_grad_norm = torch.norm(h, 'fro').item()
                else:
                    TxSTgrad = - lr * H
                    self.last_grad_norm = torch.norm(H, 'fro').item()

                PolarMatrix = (X + TxSTgrad).T @ (X + TxSTgrad)
                Lambda, V = torch.linalg.eigh(PolarMatrix)

                STgrad = (X + TxSTgrad) @ (V @ torch.diag(1.0/torch.sqrt(Lambda)) @ V.T)
                
                p.data.copy_(STgrad)

        return loss
