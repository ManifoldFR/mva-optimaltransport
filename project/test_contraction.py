import numpy as np
from scipy.stats import norm
from scipy.spatial import distance
from contraction import compute_message, compute_marginals, FactoredKernel

nx = ny = 21
xar = np.linspace(0, 1, nx)
xg, yg = np.meshgrid(xar, xar)

N = 20
dt = 1. / N
cost_mat1 = distance.cdist(xar[:, None], xar[:, None])
K1 = norm.pdf(cost_mat1, scale=dt ** 0.5)

kernel = FactoredKernel(K1, K1)

a_s = [np.ones((nx, ny)) for _ in range(N)]

## Test compute_message:

mess0 = compute_message(a_s, 0, kernel)
print("Message shape:", mess0.shape)

# Test compute marginals
import time

ta = time.time()
marginals = compute_marginals(a_s, kernel)
print("Elapsed time:", time.time() - ta)
