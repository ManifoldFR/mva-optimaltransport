from distutils.core import setup
from Cython.Build import cythonize
import numpy

name = "Fast sweeping algorithm for HJ equation"

setup(
    name=name,
    ext_modules=cythonize('fastsweeper/sweep.pyx',
                          language_level=3),
    include_dirs=[numpy.get_include()]
)
