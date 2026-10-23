from src.orthogonalization.cans import cans_retraction


def make_cans_retraction(steps=8):

    def cans_iteration(X, Xi):
        return cans_retraction(X + Xi, steps)

    return cans_iteration
