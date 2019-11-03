import numpy as np


def repeated_outer_product(*xi):
    r"""
    Compute the other product
    ... math ::
        x_1 \otimes \cdots \otimes x_K
    which is a `n1 x ... x nK`.
    """
    res = 1.
    for x in xi:
        res = np.multiply.outer(res, x)
    return res

def partial_repeated_outer_product(*xi, k: int):
    r"""
    Compute the partial outer product with the same dimensionality
    by replacing entry `k` by a vector of 1s.
    """
    ones_k = np.ones_like(xi[k])
    xi = list(xi)
    xi[k] = ones_k
    return repeated_outer_product(*xi)

