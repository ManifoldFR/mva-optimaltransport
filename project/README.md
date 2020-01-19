# Project: An Entropy minimization approach for computing variational Mean-Field games

This code contains an implementation of the paper [An Entropy minimization approach for computing variational Mean-Field games](https://hal.archives-ouvertes.fr/hal-01848370v3) by Benamou et al. [2019] as well as a few numerical tests. It is written in Python and Cython and is ready to use and extend with other heat kernels, cost functions and constraints.

![Geodesic kernel transport](images/geodesic_room3/transport.png)

Find the associated project report [here](../report/report.pdf).


## Building

I wrote multiple Cython modules for performance-critical code, such as computing solutions to the Eikonal equation or message-passing for computing the multi-marginal tensor contraction. This requires that the Cython compiler and setuptools be installed.
Build the Cython code using
```bash
python setup.py build_ext --inplace
```

## Modules

### Core

Tensor contractions with respect to the discrete Wiener measure are handled using a message passing algorithm (see [contraction.pyx](contraction.pyx)). The algorithm is generic and modular with the heat kernel (corresponding to the 2-marginal of the Wiener measure) being a sublass of the `KernelOp` class.


#### Laplacian heat kernel

We extend the framework of the initial paper by replacing the heat kernel in the Wiener measure by a geodesic distance kernel approximated using the Laplacian operator (as suggested in Peyré [2015] for JKO flows, following Crane et al. [2013]). Functions for computing the Laplacian as a sparse matrix are provided in the module [utils/laplacian.py](utils/laplacian.py).
![Adapted Laplacian for 2D grid with mask](images/laplacian_example.png)


### Proximal operators and Sinkhorn

The Sinkhorn iterations require to define the proximal operator for the set of costs and constraints. This is not fully modular. See [scenarios/congestion.pyx](scenarios/congestion.pyx) for an example on crowd congestion written in Cython.



## References

* Jean-David Benamou, Guillaume Carlier, Simone Marino, Luca Nenna. An entropy minimization approach to second-order variational mean-field games. 2019. ⟨hal-01848370v3⟩
* Crane, Keenan, Clarisse Weischedel, and Max Wardetzky. “_Geodesics in Heat_”. ACM Transactions on Graphics 32.5 (2013): 1–11. Crossref. Web.
* G. Peyré. _Entropic Approximation of Wasserstein Gradient Flows_. SIAM Journal on Imaging Sciences, 8(4), pp. 2323–2351, 2015. 
