'''
This file is used to build the Cython libraries inplace on development systems.
'''

import os

curdir = os.getcwd()
try:
    base, _ = os.path.split(__file__)
    os.chdir(os.path.join(base, '..'))
    os.system('python setup.py build_ext --inplace')
finally:
    os.chdir(curdir)
