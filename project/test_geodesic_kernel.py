import numpy as np

from utils import laplacian
from utils.plotting import plot_domain, send_zero_transparent

import matplotlib.pyplot as plt

from contraction import GeodesicKernel

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
mask[:, :b_size] = True
mask[:b_size, :] = True
mask[-b_size:, :] = True
mask[:, -b_size:] = True
mask[:] |= (np.abs(xg - 0.6) < 0.08) & (yg < 0.5)
mask[:] |= (np.abs(xg - 0.7) < 0.08) & (yg > 0.8)
mask[:] |= (xg < 0.2) & (np.abs(yg - 0.5) < 0.04)

domain_img = np.zeros((ny, nx, 4))
domain_img[mask, 3] = 1.

# Geodesic kernel

gamma = 0.01  # final time
gk = GeodesicKernel(mask, dx, dy, gamma)

beta = 0.01
x0 = [0.2, 0.8]
init_distrib = np.exp(-((xg - x0[0])**2 + (yg - x0[1])**2) / beta)
init_distrib[mask] = 0.
init_distrib /= init_distrib.sum()

result = [init_distrib]
result.append(gk(result[-1]))

plt.subplot(1, 2, 1)
plt.imshow(init_distrib, cmap=plt.cm.Blues, extent=extent, origin='lower')
plot_domain(domain_img, alpha=.2, extent=extent)

plt.subplot(1, 2, 2)
plt.imshow(result[-1], cmap=plt.cm.Blues, extent=extent, origin='lower')
plot_domain(domain_img, alpha=.2, extent=extent)

plt.tight_layout()
plt.show()



