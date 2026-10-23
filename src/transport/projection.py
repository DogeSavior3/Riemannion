def projection_transport(X, X_old, M):
    return M - X @ ((X.T @ M + M.T @ X) * 0.5)
