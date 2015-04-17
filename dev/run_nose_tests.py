'''
Run all the tests using nose. Exits with error code 1 if a test failed.
'''
import sys

import klustakwik2

if not klustakwik2.test():  # If the test fails, exit with a non-zero error code
    sys.exit(1)
