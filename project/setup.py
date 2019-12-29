from setuptools import setup
from Cython.Build import cythonize
import numpy

name = "Fast sweeping algorithm for HJ equation"

cy_exts = ['fastsweeper/*.pyx', '*.pyx']

setup(
    name=name,
    ext_modules=cythonize(
        cy_exts, language_level=3,
        compiler_directives={'embedsignature': True}),
    include_dirs=[numpy.get_include()],
    zip_safe=False
)
