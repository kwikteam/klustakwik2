'''
This tool can be used to run setup.py inplace.

Not currently using this (using pyximport instead), but it's here for debugging because pyximport does not give us
access to the c file.
'''

import os
os.system('python setup.py build_ext --inplace')
