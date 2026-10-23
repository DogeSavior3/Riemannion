def fixed_step_rule(X, H, grad, lr, objective=None, retraction=None):
    return -lr * H
