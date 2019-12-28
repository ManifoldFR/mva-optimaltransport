import numpy as np
import time
from fastsweeper.sweep import update_grid

n = 251
xar = np.linspace(0, 1, n)
dx = xar[1] - xar[0]
f_ = np.ones((n, n))
dist_matrix = (xar[:, None] - xar[None, :]) ** 2

C = np.exp(-0.5 * dist_matrix)
print("Initial C", C)

t = time.time()
# Perform updates
for _ in range(10):
    update_grid(f_, dx, C, False, False)

print("Updated C", C)
elapsed = time.time() - t

print(f"Elapsed time: {elapsed:.3e}s")
