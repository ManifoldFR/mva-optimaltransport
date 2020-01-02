# cython: infer_types=True
# cython: boundscheck=False
# cython: wraparound=False
# cython: nonecheck=False
import numpy as np
cimport numpy as np
from numpy cimport ndarray


cpdef double hilbert_metric(ndarray u, ndarray v, double epsilon=1e-30):
    """Compute the Hilbert metric between the two vectors."""
    cdef ndarray diff = np.log(u+epsilon) - np.log(v+epsilon)
    return np.max(diff) - np.min(diff)


cpdef double hilbert_metric_chained(list a1, list a2):
    """Compute the chained Hilbert metric as in the article "Generalized incompressible
    flows, multi-marginal transport and Sinkhorn algorithm. 2018." by Benamou et al.

    .. math::
        d_H(a_1, \ldots, a_n) = \sum_i d_H(a_i, a_{i+1})
    
    """
    cdef double result = 0.
    cdef size_t i
    cdef size_t n_marg = len(a1)
    for i in range(n_marg):
        result += hilbert_metric(a1[i], a2[i])
    return result
