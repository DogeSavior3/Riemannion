import torch
from torch.optim import Optimizer
from search_engines import polar_retraction, armijo_line_search, steppest, newton_schulz_polar, Rayleigh, Rayleigh_steppest, Rayleigh_armijo
from polar_express import optimal_composition, PolarExpress
from cans_realization import cans_retraction

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

    def __init__(self, params, lr=1e-2, momentum=None, line_search="none", object=None, steps = 8):
        defaults = dict(lr=lr, momentum=momentum, line_search=line_search, steps=steps)
        super().__init__(params, defaults)
        self.last_grad_norm = 0.0
        self.func = object
        if line_search == "PolarExpress":
            self.coeffs_list = optimal_composition(l=1e-3, num_iters=steps, safety_factor_eps=1e-2)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            momentum = group['momentum']
            line_search = group['line_search']
            steps = group['steps']
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
                    STgrad = armijo_line_search(X, -TxSTgrad, Rayleigh_armijo(X), Rayleigh_armijo) # add custom retraction
                elif line_search == "steppest":
                    alpha_opt = steppest(X, H)
                    TxSTgrad *= alpha_opt
                    STgrad = polar_retraction(X + TxSTgrad)
                elif line_search == "PolarExpress":
                    TxSTgrad *= lr
                    STgrad = PolarExpress(X + TxSTgrad, steps=steps, coeffs_list=self.coeffs_list)
                elif line_search == "newton_schulz":
                    TxSTgrad *= lr
                    STgrad = newton_schulz_polar(X + TxSTgrad, steps)
                elif line_search == "CANS":
                    TxSTgrad *= lr
                    STgrad = cans_retraction(X + TxSTgrad, steps)
                else:
                    TxSTgrad *= lr
                    STgrad = polar_retraction(X + TxSTgrad)

                p.data.copy_(STgrad)

        return loss
