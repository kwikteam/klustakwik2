'''
The numerics module contains all code that is to be compiled. At the moment there are two backends,
the Cython one (cylib) and the Numba one (numbalib). Only the Cython one is complete. Later, when a decision is made
about what to use, we will refactor this code and remove the multiple backends (which isn't maintainable).
'''

from .cylib import *
