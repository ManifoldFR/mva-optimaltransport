# cython: nonecheck=False
# cython: boundscheck=False
# cython: wraparound=False
# cython: initializedcheck=False
"""Cython implementation of the Fast Sweeping algorithm for solving
the Eikonal equation."""
cimport cython
import numpy as np
cimport numpy as np
from numpy cimport ndarray

from libc.math cimport sqrt, fmin, fabs

ctypedef np.npy_bool bool_t

cdef double _solve_algebraic_update(double a, double b, double rhs) nogil:
    r"""Solve the algebraic equation

    .. math:: [(x - a)^{+} + (x - b)^{+}]^2 = rhs^2

    Args:
        rhs: Root of the right-hand side `f[i,j] * h`.
    """
    if fabs(a - b) >= rhs:
        return fmin(a, b) + rhs
    else:
        return 0.5 * (a + b + sqrt(2 * rhs * rhs - (a - b) * (a - b)))


@cython.boundscheck(False)
cpdef void update_grid(double[:, :] f, double h, double[:, :] C,
                       bool_t reverse_x, bool_t reverse_y):
    r"""Perform sweeping update.
    
    Args:
        f (array): right-hand side of the Eikonal equation.
        h (double): grid step size.
        C (array): grid to update.

    Result:
        C must contain the updated array.
    """
    cdef Py_ssize_t n = C.shape[0]
    cdef Py_ssize_t m = C.shape[1]

    cdef Py_ssize_t i, j
    cdef double uxmin, uymin, ubar

    cdef int Imin, Jmin, Imax, Jmax, Istep, Jstep
    if reverse_x:
        Imin = n-1
        Imax = -1
        Istep = -1
    else:
        Imin = 0
        Imax = n
        Istep = 1

    if reverse_y:
        Jmin = m-1
        Jmax = -1
        Jstep = -1
    else:
        Jmin = 0
        Jmax = m
        Jstep = 1

    for i in range(Imin, Imax, Istep):
        for j in range(Jmin, Jmax, Jstep):
            if i == 0:
                uxmin = C[i+1,j]
            elif i == n-1:
                uxmin = C[i-1,j]
            else:
                uxmin = fmin(C[i-1,j], C[i+1,j])

            if j == 0:
                uymin = C[i,j+1]
            elif j == m-1:
                uymin = C[i,j-1]
            else:
                uymin = fmin(C[i,j-1], C[i,j+1])

            ubar = _solve_algebraic_update(
                uxmin, uymin, h*f[i,j])
            C[i,j] = fmin(ubar, C[i,j])

@cython.boundscheck(False)
cpdef void init_grid(double[:,:] C, const bool_t[:,:] target_mask,
                     double init_value=1e5) nogil:
    r"""
    Args
        C (double[:,:]): grid to initialize
        target_mask (bool_t[:,:]): target set
        init_value (double): initialization value

    """
    cdef Py_ssize_t n = C.shape[0]
    cdef Py_ssize_t m = C.shape[1]
    
    for i in range(n):
        for j in range(m):
            # In the domain, set target set
            # and non-target values
            if target_mask[i, j]:
                C[i, j] = 0.
            else:
                C[i, j] = init_value

@cython.boundscheck(False)
cpdef ndarray[double,ndim=2] fast_sweep(
    double[:,:] f, const double h, const bool_t[:,:] target_mask,
    const int iters, double init_value=10.0):
    r"""Fast-sweeping method.
    
    Args:
        f (double[:,:]): speed field.
        h (double): grid size step.
        target_mask (bool_t[:,:]): target set.
        iters (int): number of iterates.
    
    Returns:
        The array of geodesic distances to the mask
        under the velocity f.
    """
    cdef int ny, nx
    ny = f.shape[0]
    nx = f.shape[1]
    cdef ndarray[double, ndim=2] C_owned = np.ones((ny, nx))
    cdef double[:,::1] C_view = C_owned
    init_grid(C_view, target_mask, init_value)
    cdef int k
    
    for k in range(iters):
        update_grid(f, h, C_view, False, False)
        update_grid(f, h, C_view, True, False)
        update_grid(f, h, C_view, True, True)
        update_grid(f, h, C_view, False, True)
    return C_owned
