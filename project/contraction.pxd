cimport numpy as np
from numpy cimport ndarray

cdef class KernelOp:
    cdef ndarray[double] call(self, ndarray x)

cdef class FactoredKernel(KernelOp):
    """Factorized kernel for two dimensions."""
    cdef ndarray K1, K2
    
    cdef ndarray[double] call(self, ndarray x)

cpdef ndarray[double] compute_message(list arrs, size_t idx, KernelOp op)

