# Project: An Entropy minimization approach for computing variational Mean-Field games


## Building


I wrote multiple Cython modules for performance-critical code, such as computing solutions to the Eikonal equation or message-passing for computing the multi-marginal tensor contraction.

Build it using
```bash
python setup.py build_ext --inplace
```

