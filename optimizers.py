import torch
from torch.optim import Optimizer
from search_engines import polar_retraction, armijo_line_search, steppest

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

    def __init__(self, params, lr=1e-2, momentum=None, line_search="none", object=None):
        defaults = dict(lr=lr, momentum=momentum, line_search=line_search)
        super().__init__(params, defaults)
        self.last_grad_norm = 0.0
        self.func = object

    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            momentum = group['momentum']
            line_search = group['line_search']
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
                    TxSTgrad = - h
                    self.last_grad_norm = torch.norm(h, 'fro').item()
                else:
                    TxSTgrad = - H
                    self.last_grad_norm = torch.norm(H, 'fro').item()
                
                if line_search == "armijo":
                    X_new = armijo_line_search(X, -TxSTgrad, self.func(X), self.func) # add custom retraction
                    p.data.copy_(X_new)
                elif line_search == "steppest":
                    alpha_opt = steppest(X, H)
                    TxSTgrad *= alpha_opt
                    STgrad = polar_retraction(X + TxSTgrad)
                    p.data.copy_(STgrad)
                else:
                    TxSTgrad *= lr
                    STgrad = polar_retraction(X + TxSTgrad)
                    p.data.copy_(STgrad)

        return loss
