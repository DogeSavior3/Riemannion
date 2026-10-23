def explicit3(A, B):
    e = ((A**2 + A * B + B**2) / 3) ** 0.5
    a = 2 / (2 * e**3 + A**2 * B + B**2 * A)
    p = [a * (A**2 + A * B + B**2), -a]
    err = (2 * e**3 - A**2 * B - B**2 * A) / (2 * e**3 + A**2 * B + B**2 * A)
    return (p, err)


def cans_retraction(X, iters=1):
    norm = X.norm(dim=(-2, -1), keepdim=True) * 1.01 + 1e-07
    X /= norm
    left = 1.0 / norm
    right = 1.0
    for _ in range(iters):
        coefs, err = explicit3(left, right)
        XXT = X @ X.mT
        XXTX = XXT @ X
        c1, c3 = coefs
        X = c1 * X + c3 * XXTX
        left, right = (1 - err, 1 + err)
    return X
