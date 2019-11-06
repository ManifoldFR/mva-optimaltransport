from distutils.core import setup
from Cython.Build import cythonize

name = "Fast sweeping algorithm for HJ equation"

setup(
    name=name,
    ext_modules=cythonize('fastsweeper/sweep.pyx',
                          language_level=3),
)
