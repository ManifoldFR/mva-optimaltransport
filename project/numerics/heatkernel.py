import numpy as np
import numba as nb
from scipy.stats import norm, multivariate_normal
from typing import Tuple


class HeatKernel(object):
    r"""
    The heat kernel over a grid in :math:`\mathbb{R}^d`.
    
    Args:
        grid: grid points e.g. produced by np.meshgrid
        t: time interval of the kernel
        dim (:obj:`int`, optional): explicit dimension of the space
    """
    
    def __init__(self, t: float, grid: np.ndarray, dim=None):
        if dim is not None:
            self.dim = dim
        else:
            self.dim = grid.shape[-1]
        self.grid = grid
        if dim == 1 and grid.shape[-1] != dim:
            self.kernel = norm.pdf(x=grid, scale=t**(0.5))
        else:
            self.kernel = multivariate_normal.pdf(
                x=grid, cov=t*np.eye(self.dim))


class FastHeatKernel(object):
    r"""
    This variant of :class:`HeatKernel` stores the marginal kernels instead.

    Args:
        grid: spatial grid points
        dim (:obj:`int`, optional): explicit dimension
    """

    def __init__(self, t: float, grid: Tuple[np.ndarray], dim=None):
        import math
        if dim is not None:
            self.dim = dim
        else:
            self.dim = len(grid)
        self.grid = grid
        rv = norm(scale=math.sqrt(t))
        self.kernels = [
            rv.pdf(x=grid[i]) for i in range(self.dim)
        ]


