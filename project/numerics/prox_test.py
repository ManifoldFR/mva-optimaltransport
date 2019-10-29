"""
Test for using CVXPY to compute KL-proximal operators.
"""
import numpy as np
import cvxpy as cvx
import matplotlib.pyplot as plt

print("CVXPY version:", cvx.__version__)


nx = 21
xar = np.linspace(-1, 1, nx)
xgrid, ygrid = np.meshgrid(xar, xar)

sigma = 0.5

## Gaussian density
rho = np.exp(-0.5 * (xgrid**2 + ygrid**2) / sigma**2)
rho /= rho.sum()

extent = [-1, 1, -1, 1]

rho_max = cvx.Parameter()
z = cvx.Variable(shape=rho.shape, nonneg=True, name="z")  # implicit z>=0


## We want to project rho on the space of measures under rho_max
## w.r.t. the Kullback-Leibler divergence

constraints = [z <= rho_max,] # do not saturate rho_max
# the reference measure rho is already normalized so the KL prox will be normalized

obj = cvx.sum(cvx.kl_div(z, rho))
prob = cvx.Problem(cvx.Minimize(obj),
                   constraints=constraints)

z_solutions = []
ratio_values = [0.9, 1.8, 2.8]
rho_values = []

# Iterate over values of rho_max
for ratio in ratio_values:
    rho_max.value = ratio * 1. / np.prod(rho.shape)
    rho_values.append(rho_max.value)
    print("rho_max:", rho_max.value)
    prob.solve()

    print("Problem status:", prob.status)
    print("Optimal problem value:", prob.value)

    z_value = np.array(z.value)
    print("1-norm of sol.: %.3e" % np.sum(z_value))  # sanity check for normalization
    z_solutions.append(z_value)
    print()

num_plots = 1 + len(ratio_values)
fig: plt.Figure = plt.figure(figsize=(1+4*num_plots//2,9), dpi=80)

n_cols = num_plots // 2
plt.subplot(2, n_cols, 1)
plt.imshow(rho, origin='lower', interpolation='bilinear',
           extent=extent)
plt.title("Initial measure $\\rho$")

for i in range(len(ratio_values)):
    plt.subplot(2, n_cols, i+2)
    z_value = z_solutions[i]
    plt.imshow(z_value, origin='lower', interpolation='bilinear',
            extent=extent,
            vmax=rho.max())
    title = r"Projection $z = \mathrm{prox}^{KL}_{\leq M}(\rho)$"
    title += "\n$\\rho_{\\mathrm{max}}="+f"{rho_values[i]:.3e}$"
    plt.title(title)

fig.subplots_adjust(right=0.88)
cbar_ax = fig.add_axes([0.9, 0.15, 0.01, 0.7])
plt.colorbar(cax=cbar_ax)

plt.show()
