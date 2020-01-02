"""Functions for the crowd congestion scenarios e.g. proximal operator.
"""
import numpy as np 
from numpy import ndarray


def prox_operator(mes: ndarray, mask: ndarray, congest_max: float,
                  psi: ndarray, epsilon: float=1.):
    r"""Proximal operator on the set of hard congestion and obstacle constraints,
    and a potential :math:`\Psi`."""
    return np.minimum(mes * np.exp(-psi/epsilon),
                      congest_max) * (1-mask)
