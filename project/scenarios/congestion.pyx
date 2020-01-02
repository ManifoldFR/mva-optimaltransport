# cython: infer_types=True
# cython: boundscheck=False
# cython: wraparound=False
# cython: nonecheck=False
"""Functions for the crowd congestion scenarios e.g. proximal operator.
"""
import numpy as np 
cimport numpy as np
from numpy cimport ndarray


cpdef ndarray prox_operator(ndarray mes, ndarray mask, double congest_max, ndarray psi, double epsilon=1.):
    r"""Proximal operator on the set of hard congestion and obstacle constraints,
    and a potential :math:`\Psi`."""
    return np.minimum(mes * np.exp(-psi/epsilon),
                      congest_max) * (1-mask)
