'''
Not currently using this (using pyximport instead), but it's here for debugging because pyximport does not give us
access to the c file.
'''

from distutils.core import setup
from Cython.Build import cythonize
import numpy

setup(
    ext_modules=cythonize("compute_cluster_masks.pyx"),
    include_dirs=[numpy.get_include()]
)
