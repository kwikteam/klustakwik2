'''
Utilities for scripts
'''

import sys

__all__ = ['parse_args']

def parse_args(num_args, allowed_params, msg):
    msg += '\nAllowed arguments and default values:\n'
    for k, v in allowed_params.iteritems():
        msg += '\n    %s = %s' % (k, v)
    if len(sys.argv)<=num_args:
        print msg
        exit(1)
    params = {}
    for spec in sys.argv[num_args+1:]:
        name, val = spec.split('=')
        name = name.strip()
        val = eval(val.strip())
        params[name] = val
    for k in params.keys():
        if k not in allowed_params:
            print msg
            exit(1)
    for k, v in allowed_params.iteritems():
        if k not in params:
            params[k] = v
    return sys.argv[1:num_args+1], params
