# cython: infer_types=True
# cython: boundscheck=False
# cython: wraparound=False
# cython: nonecheck=False
"""Message-passing algorithm for computing the tensor contraction."""
cimport cython
import numpy as np
cimport numpy as np
from numpy cimport ndarray


cdef class KernelOp:
    cpdef ndarray[double] call(self, ndarray x):
        pass


cdef class FactoredKernel(KernelOp):
    """Factorized kernel for two dimensions."""
    cdef ndarray K1, K2
    
    def __init__(self, ndarray K1, ndarray K2):
        self.K1 = K1
        self.K2 = K2

    cpdef ndarray[double] call(self, ndarray x):
        return np.dot(np.dot(self.K2, x), self.K1)


cpdef ndarray[double] compute_message(list arrs, int idx, KernelOp op):
    r"""Perform partial convolution with the N marginals.

    This function computes the tensor contraction:
    .. math::
        R_{i_1,\ldots,i_N} \prod_{k=1}^N a_{i_k}
    
    Args:
        a_s: List of potentials.
        idx: Index of the marginal we want to compute.
        op: Operation to perform on the marginals.
    """
    cdef Py_ssize_t n_marg = len(arrs)  # no. of marginals
    shape = arrs[0].shape
    cdef ndarray A = np.ones(shape)
    cdef Py_ssize_t j
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
    cdef Py_ssize_t n_marg = len(arrs)
    cdef list result = list()
    cdef int k
    cdef ndarray marg
    for k in range(n_marg):
        marg = arrs[k] * compute_message(arrs, k, op)
        result.append(marg)

    return result
