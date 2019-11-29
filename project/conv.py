import numpy as np
import cupy as cp

from typing import List, Any

XpArray = Any[np.ndarray, cp.ndarray]


class Convolver:
    r"""Utility class for efficiently computing multi-marginal convolution as a tensor
    contraction to `R`.
    
    .. math::
        R_{i_1,\ldots,i_N} = \prod_k P_{i_k,i_{k+1}}    
    """
    
    def __init__(self, kernel_op):
        """
        Args
            kernel_op: marginal kernel operator P to apply
        """
        super().__init__()
        self.kernel_op = kernel_op

    def _partial_convolution(self, potentials: Any[List[np.ndarray], List[cp.ndarray]], idx: int) -> XpArray:
        """Perform partial convolution with the N marginals.
        idx: which index to leave out
        """
        
        n_marg = len(potentials)  # no. of marginals
        K = self.kernel_op
        
        print("Forward conv")
        A = 1
        # iterate convolution
        for j in range(idx):
            print("marginal %d" % j)
            A = K(A * potentials[j])
        
        print("Backward conv")
        B = 1
        for j in range(n_marg - idx - 1):
            print("marginal %d" % (n_marg-j-1))
            B = K(B * potentials[n_marg-j-1])
        return A * B

