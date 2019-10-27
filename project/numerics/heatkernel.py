import numpy as np
import numba as nb
from scipy.stats import multivariate_normal


class HeatKernel:
    r"""
    The heat kernel over a grid in :math:`\mathbb{R}^d`.
    
    Args
        grid: grid points e.g. produced by np.meshgrid
        t: time interval of the kernel
    """
    
    def __init__(self, t: float, *grid):
        self.dim = len(grid)
        self.grid = grid
        self._mvn = multivariate_normal(cov=t*np.eye(self.dim))
        grid_tensorized = np.stack(grid, -1)
        self.kernel = self._mvn.pdf(grid_tensorized)
