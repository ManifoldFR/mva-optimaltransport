from setuptools import setup
from Cython.Build import cythonize
import numpy

name = "SMFG_toolbox"

cy_exts = ['utils/*.pyx', '*.pyx']

setup(
    ext_modules=cythonize(
        cy_exts, language_level=3,
        compiler_directives={'embedsignature': True}),
    include_dirs=[numpy.get_include()],
    zip_safe=False
)
