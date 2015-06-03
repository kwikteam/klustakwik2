'''
This file is used to build the Cython libraries inplace on development systems.
'''

import os

os.chdir('..')
os.system('python setup.py build_ext --inplace')
