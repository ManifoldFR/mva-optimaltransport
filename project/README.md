# Project: An Entropy minimization approach for computing variational Mean-Field games


## Building


I wrote multiple Cython modules:
* one implementing the Fast Sweeping Method to solve the Eikonal equation to obtain a geodesic distance map on a grid (it is useful for modelling the crowd congestion problem)  
* one implementing a message-passing algorithm to compute the contraction by the Gibbs kernel (see report)
Build it using
```bash
python setup.py build_ext --inplace
```

