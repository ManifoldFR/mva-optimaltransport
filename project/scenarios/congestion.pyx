# cython: infer_types=True
# cython: boundscheck=False
# cython: wraparound=False
# cython: nonecheck=False
"""Functions for the crowd congestion scenarios e.g. proximal operator."""
cimport cython
import numpy as np 
cimport numpy as np
from numpy cimport ndarray
from contraction cimport compute_message, KernelOp


cpdef ndarray prox_operator(ndarray mes, ndarray mask, double congest_max, ndarray psi, double epsilon=1.):
    r"""Proximal operator on the set of hard congestion and obstacle constraints,
    and a potential :math:`\Psi`."""
    return np.minimum(mes * np.exp(-psi/epsilon),
                      congest_max) * (1-mask)


cdef double TAU = 1e-30


def multi_sinkhorn(list arrays, KernelOp op, ndarray rho_0, 
                   ndarray mask, double congest_max, ndarray psi,
                   double epsilon=1.):
    """Multimarginal sinkhorn"""
    cdef size_t n_marg = len(arrays)  # no. of marginals
    cdef ndarray zero_arr = np.zeros_like(rho_0)
    cdef ndarray conv
    cdef ndarray numer
    conv = compute_message(arrays, 0, op)
    arrays[0] = rho_0 / (conv + TAU)
    
    cdef size_t k

    for k in range(1, n_marg-1):
        conv = compute_message(arrays, k, op)
        numer = prox_operator(conv, mask, congest_max, zero_arr, epsilon)
        arrays[k] = numer / (conv + TAU)
    
    conv = compute_message(arrays, n_marg-1, op)
    numer = prox_operator(conv, mask, congest_max, psi, epsilon)
    arrays[n_marg - 1] = numer / (conv + TAU)

