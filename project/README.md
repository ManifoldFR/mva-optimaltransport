# Project: An Entropy minimization approach for computing variational Mean-Field games


## Building


I wrote multiple Cython modules for performance-critical code, such as computing solutions to the Eikonal equation or message-passing for computing the multi-marginal tensor contraction.

Build it using
```bash
python setup.py build_ext --inplace
```

## Laplacian

Idea for an extension: replace the heat kernel in the Wiener measure by a geodesic distance kernel approached using the Laplacian operator. Functions for computing the Laplacian as a sparse matrix are provided in the module [utils/laplacian.py](utils/laplacian.py).

![](images/laplacian_example.png)

* Crane, Keenan, Clarisse Weischedel, and Max Wardetzky. “_Geodesics in Heat_”. ACM Transactions on Graphics 32.5 (2013): 1–11. Crossref. Web.
* G. Peyré. _Entropic Approximation of Wasserstein Gradient Flows_. SIAM Journal on Imaging Sciences, 8(4), pp. 2323–2351, 2015. 
