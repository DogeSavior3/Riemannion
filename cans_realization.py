# this code is based on repository
# https://github.com/JunLi-Galios/Optimization-on-Stiefel-Manifold-via-Cayley-Transform/blob/master/utils.py
import torch
from torch.optim.optimizer import Optimizer, required
import numpy as np
import random

episilon = 1e-8


def explicit3(A, B):
    e = ((A ** 2 + A * B + B ** 2) / 3) ** 0.5
    a = 2 / (2 * e ** 3 + A ** 2 * B + B ** 2 * A)
    p = [a * (A ** 2 + A * B + B ** 2), -a]
    err = (2 * e ** 3 - A ** 2 * B - B ** 2 * A) / (2 * e ** 3 + A ** 2 * B + B ** 2 * A)
    return p, err


def cans_retraction(X, iters=1):
    # polar retraction based on CANS orthogonalization

    # norm = (torch.norm(X) ** 2 - X.shape[-2] + 1) ** 0.5
    norm = (X.norm(dim=(-2, -1), keepdim=True) * 1.01 + 1e-7)
    X /= norm
    left = 1.0 / norm
    right = 1.0
    for _ in range(iters):
        coefs, err = explicit3(left, right)
        XXT = X @ X.mT
        XXTX = XXT @ X
        c1, c3 = coefs
        X = c1 * X + c3 * XXTX
        left, right = 1 - err, 1 + err

    return X


def qr_retraction(tan_vec):  # tan_vec, p-by-n, p <= n
    [p, n] = tan_vec.size()
    tan_vec.t_()
    q, r = torch.qr(tan_vec)
    d = torch.diag(r, 0)
    ph = d.sign()
    q *= ph.expand_as(q)
    q.t_()
    return q


class CANS_SGD(Optimizer):
    r"""This optimizer updates variables with two different routines
        based on the boolean variable 'stiefel'.

        If stiefel is True, the variables will be updated by SGD-G proposed
        as decorrelated weight matrix.

        If stiefel is False, the variables will be updated by SGD.
        This routine was taken from https://github.com/pytorch/pytorch/blob/master/torch/optim/sgd.py.

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups

        -- common parameters
        lr (float): learning rate
        momentum (float, optional): momentum factor (default: 0)
        stiefel (bool, optional): whether to use SGD-G (default: False)

        -- parameters in case stiefel is False
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        dampening (float, optional): dampening for momentum (default: 0)
        nesterov (bool, optional): enables Nesterov momentum (default: False)

        -- parameters in case stiefel is True
        omega (float, optional): orthogonality regularization factor (default: 0)
        grad_clip (float, optional): threshold for gradient norm clipping (default: None)
    """

    def __init__(self, params, lr=required, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False, stiefel=True, omega=0,
                 grad_clip=None, use_qr=False, retraction_iters=1):
        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov,
                        stiefel=stiefel, omega=0, grad_clip=grad_clip,
                        retraction_iters=retraction_iters, use_qr=use_qr)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super(CANS_SGD, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(CANS_SGD, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            momentum = group['momentum']
            stiefel = group['stiefel']

            for p in group['params']:
                if p.grad is None:
                    continue

                X = p.data.view(p.size()[0], -1)
                X = X / (X.norm(dim=1, keepdim=True) + 1e-7)
                X = X.T
                if stiefel and X.size()[1] <= X.size()[0]:  # nxp, n>=p

                    weight_decay = group['weight_decay']
                    dampening = group['dampening']
                    nesterov = group['nesterov']

                    rand_num = random.randint(1, 101)
                    if rand_num == 1:
                        X = qr_retraction(X.T).T

                    g = p.grad.data.view(p.size()[0], -1)
                    lr = group['lr']

                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        param_state['momentum_buffer'] = torch.zeros(g.T.size())
                        if p.is_cuda:
                            param_state['momentum_buffer'] = param_state['momentum_buffer'].cuda()
                    V = param_state['momentum_buffer']  # this is momentum
                    V = momentum * V - g.T  # update momentum
                    # project V to tangent space (get Riemannian gradient)
                    VX = V.T @ X
                    V_new = V - 0.5 * X @ (VX + VX.T)
                    # retraction
                    if group['use_qr']:
                        p_new = qr_retraction((X + lr * V_new).T)
                    else:
                        p_new = cans_retraction(X + lr * V_new, iters=group['retraction_iters']).T
                    # check_identity(p_new.T) ###
                    p.data.copy_(p_new.view(p.size()))
                    V.copy_(V_new)

                else:
                    d_p = p.grad.data
                    if weight_decay != 0:
                        d_p.add_(weight_decay, p.data)
                    if momentum != 0:
                        param_state = self.state[p]
                        if 'momentum_buffer' not in param_state:
                            buf = param_state['momentum_buffer'] = d_p.clone()
                        else:
                            buf = param_state['momentum_buffer']
                            buf.mul_(momentum).add_(1 - dampening, d_p)
                        if nesterov:
                            d_p = d_p.add(momentum, buf)
                        else:
                            d_p = buf

                    p.data.add_(-group['lr'], d_p)

        return loss


class CANS_Adam(Optimizer):
    r"""This optimizer updates variables with two different routines
        based on the boolean variable 'grassmann'.

        If grassmann is True, the variables will be updated by Adam-G proposed
        in 'Riemannian approach to batch normalization'.

        If grassmann is False, the variables will be updated by SGD.
        This routine was taken from https://github.com/pytorch/pytorch/blob/master/torch/optim/sgd.py.


    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups

        -- common parameters
        lr (float): learning rate
        momentum (float, optional): momentum factor (default: 0)
        grassmann (bool, optional): whether to use Adam-G (default: False)

        -- parameters in case grassmann is False
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        dampening (float, optional): dampening for momentum (default: 0)
        nesterov (bool, optional): enables Nesterov momentum (default: False)

        -- parameters in case grassmann is True
        beta2 (float, optional): the exponential decay rate for the second moment estimates (defulat: 0.99)
        epsilon (float, optional): a small constant for numerical stability (default: 1e-8)
        omega (float, optional): orthogonality regularization factor (default: 0)
        grad_clip (float, optional): threshold for gradient norm clipping (default: None)
    """

    def __init__(self, params, lr=required, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False,
                 grassmann=False, beta2=0.99, epsilon=1e-8, omega=0,
                 grad_clip=None, retraction_iters=1, use_qr=False):
        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov,
                        grassmann=grassmann, beta2=beta2, epsilon=epsilon, omega=0, grad_clip=grad_clip,
                        retraction_iters=retraction_iters, use_qr=use_qr)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super(CANS_Adam, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(CANS_Adam, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            stiefel = group['stiefel']

            for p in group['params']:
                if p.grad is None:
                    continue

                beta1 = group['momentum']
                beta2 = group['beta2']
                epsilon = group['epsilon']

                X = p.data.view(p.size()[0], -1)
                X = X / (X.norm(dim=1, keepdim=True) + 1e-7)
                X = X.T
                if stiefel and X.size()[1] <= X.size()[0]:
                    rand_num = random.randint(1, 101)
                    if rand_num == 1:
                        X = qr_retraction(X.T).T

                    g = p.grad.data.view(p.size()[0], -1)

                    param_state = self.state[p]
                    if 'm_buffer' not in param_state:
                        size = p.size()
                        param_state['m_buffer'] = torch.zeros([int(np.prod(size[1:])), size[0]])
                        param_state['v_buffer'] = torch.zeros([1])
                        if p.is_cuda:
                            param_state['m_buffer'] = param_state['m_buffer'].cuda()
                            param_state['v_buffer'] = param_state['v_buffer'].cuda()

                        param_state['beta1_power'] = beta1
                        param_state['beta2_power'] = beta2

                    m = param_state['m_buffer']
                    v = param_state['v_buffer']
                    beta1_power = param_state['beta1_power']
                    beta2_power = param_state['beta2_power']

                    mnew = beta1 * m + (1.0 - beta1) * g.T
                    vnew = beta2 * v + (1.0 - beta2) * (torch.norm(g) ** 2)

                    mnew_hat = mnew / (1 - beta1_power)
                    vnew_hat = vnew / (1 - beta2_power)

                    MX = mnew_hat.T @ X
                    mnew = mnew_hat - 0.5 * X @ (MX + MX.T)
                    lr = group['lr']
                    if group['use_qr']:
                        p_new = qr_retraction((X - lr * mnew / vnew_hat.add(epsilon).sqrt()).T)
                    else:
                        p_new = cans_retraction(X - lr * mnew / vnew_hat.add(epsilon).sqrt(),
                                                iters=group['retraction_iters']).T
                    # check_identity(p_new.T) ###
                    p.data.copy_(p_new.view(p.size()))
                    mnew = mnew * (1 - beta1_power)
                    m.copy_(mnew)
                    v.copy_(vnew)

                    param_state['beta1_power'] *= beta1
                    param_state['beta2_power'] *= beta2

                else:
                    momentum = group['momentum']
                    weight_decay = group['weight_decay']
                    dampening = group['dampening']
                    nesterov = group['nesterov']
                    d_p = p.grad.data
                    if weight_decay != 0:
                        d_p.add_(weight_decay, p.data)
                    if momentum != 0:
                        param_state = self.state[p]
                        if 'momentum_buffer' not in param_state:
                            buf = param_state['momentum_buffer'] = d_p.clone()
                        else:
                            buf = param_state['momentum_buffer']
                            buf.mul_(momentum).add_(1 - dampening, d_p)
                        if nesterov:
                            d_p = d_p.add(momentum, buf)
                        else:
                            d_p = buf

                    p.data.add_(-group['lr'], d_p)

        return loss
