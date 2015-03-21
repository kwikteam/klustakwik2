import numpy
import pyximport
 
pyximport.install(setup_args={'include_dirs':numpy.get_include()})
