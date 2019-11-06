import numpy as np
from sweep import _solve_algebraic_update

def print_check(a, b, rhs, x):
    print(f"Input  (a, b, rhs) = {(a,b,rhs)})")
    print(f"Result x = {x}")


a = 1.5
b = 1
rhs = 0.5

## Theoretical answer:
# Since |a-b| = 0.5 = rhs,
# then x = min(a,b) + rhs = 1 + 0.5 = 1.5
x = _solve_algebraic_update(a, b, rhs)
print_check(a, b, rhs, x)
print()

import math
a = 0.1
b = 0.7
rhs = math.sqrt(0.5)

# We have |a - b| < rhs
# Expected result is x = 0.8
x = _solve_algebraic_update(a, b, rhs)
print_check(a, b, rhs, x)
print()

#### TEST GRID UPDATE ####
import time
from sweep import update_grid

n = 251
xar = np.linspace(0, 1, n)
dx = xar[1] - xar[0]
f_ = np.ones((n, n))
dist_matrix = (xar[:, None] - xar[None, :]) ** 2

C = np.exp(-0.5 * dist_matrix)
print("Initial C")

t = time.time()
update_grid(f_, dx, C, False, False)

print("Updated C")
elapsed = time.time() - t

print(f"Elapsed time: {elapsed:.3e}s")

#### 
