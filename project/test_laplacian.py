import numpy as np

from utils import laplacian
from utils.plotting import plot_domain, send_zero_transparent

import matplotlib.pyplot as plt

import scipy.sparse as sps
from scipy.sparse import linalg 


plt.rcParams['figure.figsize'] = (4, 4)
plt.rcParams['figure.dpi'] = 100
plt.rcParams["savefig.dpi"] = 160
plt.rcParams['text.usetex'] = True


nx = ny = 101
xmax = 1.
ymax = 1.
dx = xmax / (nx-1)
dy = ymax / (ny-1)
print("dx:", dx, "dy:", dy)
xar = np.linspace(0, xmax, nx)
yar = np.linspace(0, ymax, ny)
xg, yg = np.meshgrid(xar, yar)

extent = [0, xar.max(), 0, yar.max()]


mask = np.zeros((nx, ny), dtype=bool)
b_size = 2
mask[:, :b_size] = True; mask[:b_size, :] = True
mask[-b_size:, :] = True; mask[:, -b_size:] = True
mask[:] |= (np.abs(xg - 0.6) < 0.08) & (yg < 0.5)
mask[:] |= (np.abs(xg - 0.7) < 0.08) & (yg > 0.8)
mask[:] |= (xg < 0.2) & (np.abs(yg - 0.5) < 0.04)

domain_img = np.zeros((ny, nx, 4))
domain_img[mask, 3] = 1.

plot_domain(domain_img, extent=extent)
plt.tight_layout()



dt = 1e-4
print("dt:", dt)

diff_coe = 40.
print("CFL:", diff_coe*dt*(1/dx**2 + 1/dy**2))
mat = diff_coe * dt * laplacian.noflux_laplacian_2d(mask, dx, dy)
mat_rshpd = mat.reshape((nx*ny, 5))


# layout is center, left, right, up, down
# left and down dead
print(mask[0,0])
print(mat[0,0])

# 
print(mask[3,0])
print(mat[3,0])

# up-node dead
print(mask[0, 2])
print(mat[0, 2])
# down edge dead

# down is dead
print(mask[2,0])
print(mat[2,0])

from utils.laplacian import assemble_matrix

mat_sparse = assemble_matrix(mat_rshpd, nx, ny)


plt.figure(figsize=(4,4))
plt.imshow(mat_sparse.toarray()[nx-5:2*nx+5, nx-5:2*nx+5], cmap=plt.cm.binary)
plt.tight_layout()
plt.close()

# Run diffusion

initial_distrib = np.exp(-((xg-.4)**2 + (yg-.6)**2)/.009)
initial_distrib[mask] = 0.
initial_distrib /= initial_distrib.sum()
initial_distrib_flat = initial_distrib.ravel()

fig = plt.figure(figsize=(8,4))
plt.subplot(1,2,1)
plt.imshow(initial_distrib, cmap=plt.cm.Blues, extent=extent, origin='lower')
plot_domain(domain_img, alpha=.2, extent=extent)
plt.title("Initial distribution $t=0$")


# Implicit stepping

A_ = sps.identity(nx*nx, format='dia') - mat_sparse
A_ = A_.tocsc()
## solver for sparse systems Ax = b using cached LU factors
solver = linalg.factorized(A_)
N_t = 10
T_f = N_t * dt

import time

t_a = time.time()
res_ = [initial_distrib_flat]

for _ in range(N_t):
    next_iter = solver(res_[-1])
    res_.append(next_iter)
print("Elapsed time (pre-LU'd):", time.time() - t_a)


state_final = res_[-1].reshape(nx, ny)
print("Total mass:", state_final.sum())
print("Mass outside:", state_final[mask].sum())

plt.subplot(1,2,2)
plt.imshow(state_final, cmap=plt.cm.Blues,
           extent=extent, origin='lower')
plot_domain(domain_img, alpha=.2)
plt.title("Diffusion at time $t=%.2e$\n($%d$ time steps)" % (T_f, N_t))
plt.tight_layout()
plt.show()

fig.savefig("images/laplacian_example.png")
