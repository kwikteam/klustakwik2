'''
This file is used to build the Cython libraries inplace on development systems.
'''

import os

os.chdir('..')

if os.name=='nt':
    os.system('python setup.py build_ext --inplace --compiler=msvc')
else:
    os.system('python setup.py build_ext --inplace')

