import os

try:
    input = raw_input
except NameError:
    pass

input('This will upload a new version of KlustaKwik2 to PyPI, press return to continue ')
# upload to pypi
os.chdir('../')
os.system('python setup.py sdist --formats=zip,gztar,bztar upload')
