"""Message-passing algorithm for computing the tensor contraction."""
cimport cython
import numpy as np
cimport numpy as np
from numpy cimport ndarray


cpdef compute_message(list a_s, int idx, object op):
    """Perform partial convolution with the N marginals.

    This function computes the tensor contraction
    
    Args:
        a_s: List of poitentials.
        idx: Index of the marginal we want to compute.
    """
    n_marg = len(a_s)  # no. of marginals
    A = np.ones_like(a_s[0])
    # iterate convolution
    for j in range(idx):
        A[:] = op(A * a_s[j])
    
    B = np.ones_like(a_s[0])
    for j in range(n_marg - idx - 1):
        B[:] = op(B * a_s[n_marg-j-1])
    return A * B
