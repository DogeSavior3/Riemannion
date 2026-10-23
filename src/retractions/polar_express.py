from src.orthogonalization.polar_express import PolarExpress, optimal_composition


def make_polar_express_retraction(steps=8, l=0.001, safety_factor_eps=0.01):
    coeffs_list = optimal_composition(
        l=l, num_iters=steps, safety_factor_eps=safety_factor_eps
    )

    def polar_express_iteration(X, Xi):
        return PolarExpress(X + Xi, steps=steps, coeffs_list=coeffs_list)

    return polar_express_iteration
