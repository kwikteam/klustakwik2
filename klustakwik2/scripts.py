'''
Utilities for scripts
'''

import sys

__all__ = ['parse_args']

def parse_args(num_args, msg):
    if len(sys.argv)<=num_args:
        print msg
        exit(1)
    params = {}
    for spec in sys.argv[num_args+1:]:
        name, val = spec.split('=')
        name = name.strip()
        val = eval(val.strip())
        params[name] = val
    return sys.argv[1:num_args+1], params

