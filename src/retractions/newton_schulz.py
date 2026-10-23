def make_newton_schulz_retraction(steps=10):

    def newton_schulz_iteration(X, Xi):
        X_new = X + Xi
        X_new = X_new / (X_new.norm(dim=(-2, -1), keepdim=True) * 1.01 + 1e-07)
        for _ in range(steps):
            X_new = 1.5 * X_new - 0.5 * (X_new @ (X_new.T @ X_new))
        return X_new

    return newton_schulz_iteration
