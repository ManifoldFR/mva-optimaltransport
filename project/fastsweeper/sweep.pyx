"""
Cython implementation of the Fast Sweeping algorithm for solving
the Eikonal equation.
"""
cimport cython
import numpy as np
cimport numpy as np
from numpy import ndarray
from libc.math cimport sqrt, fmin

ctypedef np.npy_bool bool_t

cpdef _solve_algebraic_update(double a, double b, double rhs):
    r"""
    Solve the algebraic equation
    .. math:: [(x - a)^{+} + (x - b)^{+}]^2 = rhs^2

    Args
        rhs: root of rhs (f[i,j] * h)
    """
    if abs(a - b) >= rhs:
        return fmin(a, b) + rhs
    else:
        return 0.5 * (a + b + sqrt(2 * rhs * rhs - (a - b) * (a - b)))


@cython.wraparound(False)
@cython.boundscheck(False)
cpdef void update_grid(double[:, ::1] f, double h, double[:, ::1] C,
                       bool_t reverse_x, bool_t reverse_y):
    r"""
    Args
        f (array): right-hand side of the HJ eqn
        h (double): grid step size
        C (array): grid to update

    Result
        C must contain the updated array.
    """
    cdef Py_ssize_t n = C.shape[0]
    cdef Py_ssize_t m = C.shape[1]

    cdef Py_ssize_t i, j
    cdef double uxmin, uymin, ubar

    x_range = range(n)
    if reverse_x:
        x_range = reversed(x_range)
    y_range = range(m)
    if reverse_y:
        y_range = reversed(y_range)

    for i in x_range:

        for j in y_range:
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

@cython.wraparound(False)
@cython.boundscheck(False)
cpdef void init_grid(double[:,::1] C, bool_t[:,::] target_mask, double init_value=1e5):
    r"""
    Args
        C (double[:,:]): grid to initialize
        target_mask (bool_t[:,:]): target set
        init_value (double): initialization value

    """
    cdef Py_ssize_t n = C.shape[0]
    cdef Py_ssize_t m = C.shape[0]

    for i in range(n):
        for j in range(m):
            # In the domain, set target set
            # and non-target values
            if target_mask[i, j]:
                C[i, j] = 0.
            else:
                C[i, j] = init_value
            
@cython.wraparound(False)
@cython.boundscheck(False)
cpdef void fast_sweep(double[:, ::1] f, double h, bool_t[:,::] target_mask,
                      int iters, double[:,::1] C, init_value=10):
    r"""
    Args
        f (double[:,:]): speed field
        h (double): grid size step
        target_mask (bool_t[:,:]): target set
        iters (int): number of iterates
        C (double[:,:]): grid to initialize

    """
    init_grid(C, target_mask, init_value)

    cdef int k

    for k in range(iters):
        update_grid(f, h, C, False, False)
        update_grid(f, h, C, True, False)
        update_grid(f, h, C, True, True)
        update_grid(f, h, C, False, True)
