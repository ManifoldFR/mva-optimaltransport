"""No-flux Laplacian on domains represented by rectangular grids
with masks."""
import numpy as np
from numpy import ndarray
import scipy.sparse as sps


def noflux_laplacian_2d(mask: ndarray,
                        dx: float, dy: float):
    nx, ny = mask.shape
    matrx = np.zeros((nx, ny, 5))
    # stencil: 0 center, 1 left, 2 right, 3 up, 4 down
    for i in range(nx):
        for j in range(ny):
            _fill_masked_stencil(mask, i, j, dx, dy, matrx[i, j])

    return matrx


def _fill_masked_stencil(mask: ndarray, i: int, j: int, dx, dy, stencil):
    # stencil: 0 center, 1 left, 2 right, 3 up, 4 down
    nx, ny = mask.shape
    neigh = _get_neighbors(mask, i, j)
    # print("Neighbors of", (i,j),"\t",neigh, end=' ')
    
    stencil[0] = -2./dx**2 - 2./dy**2
    stencil[1] = 1./dx**2  # left
    stencil[2] = 1./dx**2  # right
    stencil[3] = 1./dy**2  # up
    stencil[4] = 1./dy**2  # down
    
    # Edge is boundary when mask value is different from neighbor
    # in that case impose Neumann boundary condition
    # on the edge
    if not neigh[0]:
        # left edge crosses boundary
        stencil[2] += stencil[1]
        stencil[1] = 0
    if not neigh[1]:
        # right edge crosses boundary
        stencil[1] += stencil[2]
        stencil[2] = 0
    if not neigh[2]:
        # up node crosses boundary
        stencil[4] += stencil[3]
        stencil[3] = 0  # kill up edge
    if not neigh[3]:
        # down node crosses boundary
        stencil[3] += stencil[4]
        stencil[4] = 0  # kill down edge
    # print("stencil:", stencil/1000)


def _get_neighbors(mask: ndarray, i: int, j: int):
    nx, ny = mask.shape
    neigh = np.ones((4,), dtype=int)
    if i==0 or (i > 0 and mask[i-1, j] != mask[i, j]):
        # left edge crosses boundary
        neigh[0] = 0
    if i==nx-1 or (i < nx-1 and mask[i+1, j] != mask[i, j]):
        # right edge crosses boundary
        neigh[1] = 0
    if j==ny-1 or (j < ny-1 and mask[i, j+1] != mask[i, j]):
        # up node crosses boundary
        neigh[2] = 0
    if j==0 or (j > 0 and mask[i, j-1] != mask[i, j]):
        # down node crosses boundary
        neigh[3] = 0
    return neigh


def assemble_matrix(mat_rshpd, nx, ny):
    """
    
    Args:
        mat_rshpd: flattened array of stencils
    """
    offsets = [0, ny, -ny, -1, 1]
    mat_sparse = sps.spdiags(mat_rshpd.T, offsets, nx*ny, nx*ny, format='csr')
    return mat_sparse
