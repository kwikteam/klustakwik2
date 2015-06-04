'''
Run all the tests using nose. Exits with error code 1 if a test failed.
'''
import sys

from klustakwik2.tests import run

if not run():  # If the test fails, exit with a non-zero error code
    sys.exit(1)
