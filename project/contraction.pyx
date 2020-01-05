# cython: infer_types=True
# cython: boundscheck=False
# cython: wraparound=False
# cython: nonecheck=False
"""Message-passing algorithm for computing the tensor contraction."""
cimport cython
import numpy as np
cimport numpy as np
from numpy cimport ndarray

import scipy.sparse as sps
from scipy.sparse import linalg as splin

from utils import laplacian


cdef class KernelOp:
    cdef ndarray[double] call(self, ndarray x):
        pass
    
    def __call__(self, ndarray x):
        return self.call(x)


cdef class FactoredKernel(KernelOp):
    """Factorized kernel for two dimensions."""
    def __init__(self, ndarray K1, ndarray K2):
        self.K1 = K1
        self.K2 = K2

    cdef ndarray[double] call(self, ndarray x):
        return np.dot(self.K2 @ x, self.K1)


cdef class GeodesicKernel(KernelOp):
    cdef public laplacian
    cdef public solver
    cdef size_t num_steps
    cdef size_t nx
    cdef size_t ny

    def __init__(self, ndarray mask, double dx, double dy, double gamma, size_t L=10):
        """Geodesic distance kernel, inspired by Peyré "Entropic Wasserstein
        Gradient Flows" [2015]. Applies to a state vector `x` by integrating the heat equation
        starting with initial state `x` for a given number of time steps up to a given time.
        
        Args:
            mask: domain obstacle mask to define the Laplacian operator
            gamma: integration max time

        """
        self.nx = mask.shape[0]
        self.ny = mask.shape[1]
        cdef ndarray mat = laplacian.noflux_laplacian_2d(
            mask, dx, dy).reshape(self.nx*self.ny, 5)
        self.laplacian = laplacian.assemble_matrix(mat, self.nx, self.ny)
        self.num_steps = L
        A_ = sps.identity(self.nx*self.ny) - gamma / L * self.laplacian
        A_ = A_.tocsc()
        self.solver = splin.factorized(A_)

    cdef ndarray[double] call(self, ndarray x):
        cdef ndarray z = x.copy().ravel()
        for i in range(self.num_steps):
            z[:] = self.solver(z)
        return z.reshape(self.nx, self.ny)


cpdef ndarray[double] compute_message(list arrs, size_t idx, KernelOp op):
    r"""Perform partial convolution with the N marginals.

    This function computes the tensor contraction:
    .. math::
        R_{i_1,\ldots,i_N} \prod_{k=1}^N a_{i_k}
    
    Args:
        a_s: List of potentials.
        idx: Index of the marginal we want to compute.
        op: Operation to perform on the marginals.
    """
    cdef size_t n_marg = len(arrs)  # no. of marginals
    cdef shape = arrs[0].shape
    cdef ndarray A = np.ones(shape)
    cdef size_t j
    for j in range(idx):
        A[:] = op.call(A * arrs[j])
    
    cdef ndarray B = np.ones(shape)
    for j in range(n_marg - idx - 1):
        B[:] = op.call(B * arrs[n_marg-j-1])
    return A * B


cpdef list compute_marginals(list arrs, KernelOp op):
    r"""Compute the marginals `a[k] * op(a[0],...,a[k-1],a[k+1],...a[n-1])`.
    Complexity scales with `len(arrs)**2`.

    Args:
        arrs: List of dual potentials.

    Returns:
        List of marginals.
    """
    cdef size_t n_marg = len(arrs)
    cdef list result = list()
    cdef size_t k
    cdef ndarray marg
    for k in range(n_marg):
        marg = arrs[k] * compute_message(arrs, k, op)
        result.append(marg)

    return result
