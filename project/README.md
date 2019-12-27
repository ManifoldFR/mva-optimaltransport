# Project: An Entropy minimization approach for computing variational Mean-Field games


## Building


I wrote a Cython module implementing the Fast Sweeping Method to solve the Eikonal equation to obtain a geodesic distance map on a grid (it is useful for modelling the crowd congestion problem). Build it using
```bash
python setup.py build_ext --inplace
```

