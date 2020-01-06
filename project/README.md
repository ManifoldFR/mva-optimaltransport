# Project: An Entropy minimization approach for computing variational Mean-Field games

## Building


I wrote multiple Cython modules for performance-critical code, such as computing solutions to the Eikonal equation or message-passing for computing the multi-marginal tensor contraction.

Build it using
```bash
python setup.py build_ext --inplace
```

## Modules

### Core

Tensor contractions with respect to the Wiener measure using are handled using message passing algorithms in [contraction.pyx](contraction.pyx). They are generic and heat kernel defining the 2-marginal can be defined by sub-classing the `KernelOp` class.

![Geodesic kernel transport](images/geodesic_room3/transport.png)

### Laplacian with obstacles

We extend the framework of the initial paper by replacing the heat kernel in the Wiener measure by a geodesic distance kernel approximated using the Laplacian operator (as suggested in Peyré [2015] for JKO flows, following Crane et al. [2013]). Functions for computing the Laplacian as a sparse matrix are provided in the module [utils/laplacian.py](utils/laplacian.py).

![Adapted Laplacian for 2D grid with mask](images/laplacian_example.png)

## References

* Crane, Keenan, Clarisse Weischedel, and Max Wardetzky. “_Geodesics in Heat_”. ACM Transactions on Graphics 32.5 (2013): 1–11. Crossref. Web.
* G. Peyré. _Entropic Approximation of Wasserstein Gradient Flows_. SIAM Journal on Imaging Sciences, 8(4), pp. 2323–2351, 2015. 
