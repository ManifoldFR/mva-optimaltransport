import numpy as np
from numpy import ndarray
import scipy.sparse as sps


def noflux_laplacian_2d(mask: ndarray,
                        dx: float, dy: float):
    nx, ny = mask.shape
    matrx = np.zeros((nx, ny, 5))
    # last 2 dims: center, left, up, right, down

    for i in range(nx):
        for j in range(ny):
            matrx[i, j, 0] = -2./dx**2 - 2./dy**2
            matrx[i, j, 1] = 1./dx**2  # left
            matrx[i, j, 2] = 1./dy**2  # up
            matrx[i, j, 3] = 1./dx**2  # right
            matrx[i, j, 4] = 1./dy**2  # down
    for i in range(nx):
        for j in range(ny):
            fill_masked_stencil(mask, i, j, dx, dy, matrx[i, j])

    return matrx


def fill_masked_stencil(mask: ndarray, i: int, j: int, dx, dy, stencil):
    nx, ny = mask.shape
    if mask[i, j]:
        stencil[:] = 0.
    elif i > 0 and mask[i-1, j]:
        stencil[1] = stencil[1]+stencil[3]
    elif i < nx-1 and mask[i+1, j]:
        stencil[3] = stencil[3]+stencil[1]
    elif j > 0 and mask[i, j-1]:
        stencil[2] = stencil[2]+stencil[4]
    elif j < ny-1 and mask[i, j+1]:
        stencil[4] = stencil[4]+stencil[2]


if __name__ == "__main__":
    nx = ny = 11
