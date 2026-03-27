import torch
from math import sqrt
from torch.optim import Optimizer
from step_algorithm import fixed_step_rule
from transport import projection_transport

def stiefel_tangent_projection(X, G):
    return G - X @ ((X.T @ G + G.T @ X) * 0.5)

def svd_muon_v1(X, M):
    U, S, VT = torch.linalg.svd(M, full_matrices=False)
    # scale = S.sum()
    scale = torch.norm(S, 'fro')
    # scale = S[0]
    # scale = sqrt(X.shape[0])
    # scale = 1
    M_muon = (U @ VT) * scale
    return stiefel_tangent_projection(X, M_muon)

def svd_muon_v2(X, M, eps=1e-12):
    U, _, VT = torch.linalg.svd(M, full_matrices=False)
    D = U @ VT
    D = stiefel_tangent_projection(X, D)
    D = D / (torch.linalg.matrix_norm(D, ord=2) + eps)
    return 0.1 * D

class RiemannianOptimizerSt(Optimizer):
    def __init__(
        self, params, lr=1e-2,
        retraction=None,
        step_rule=fixed_step_rule,
        tangent_projection=stiefel_tangent_projection,
        vector_transport=projection_transport,
        momentum=None,
        objective=None,
        muon=False,
        muon_method=svd_muon_v1,
        temp_flag=False
    ):

        defaults = dict(
            lr=lr,
            momentum=momentum,
            muon=muon,
        )
        super().__init__(params, defaults)

        self.retraction = retraction
        self.step_rule = step_rule
        self.tangent_projection = tangent_projection
        self.vector_transport = vector_transport
        self.objective = objective
        self.muon_method = muon_method
        self.last_grad_norm = 0.0

        self.iteration = 0
        self.temp_flag = temp_flag

    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        self.iteration += 1

        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            muon = group["muon"]

            for p in group["params"]:
                if p.grad is None:
                    print(p.grad)
                    continue

                X = p.data
                G = p.grad.data

                H = self.tangent_projection(X, G)

                if momentum is not None:
                    state = self.state[p]
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(H)
                        

                    cum_grad = state["momentum_buffer"]
                    if self.iteration > 1:
                        cum_grad = self.vector_transport(X, X, cum_grad)
                    cum_grad = momentum * cum_grad + H

                    # if muon:
                        # if not self.temp_flag or self.iteration >= 400:
                        #     cum_grad = self.muon_method(X, cum_grad)
                        # cum_grad = self.muon_method(X, cum_grad)

                    state["momentum_buffer"] = cum_grad
                    if muon:
                        TxSTGrad = self.muon_method(X, cum_grad)
                    else:
                        TxSTGrad = cum_grad
                else:
                    TxSTGrad = H

                self.last_grad_norm = torch.norm(TxSTGrad, p="fro").item()

                Xi = self.step_rule(
                    X=X, H=TxSTGrad, lr=lr,
                    objective=self.objective,
                    retraction=self.retraction
                )

                X_new = self.retraction(X, Xi)
                p.data.copy_(X_new)

        return loss