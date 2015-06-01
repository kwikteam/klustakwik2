'''
Utilities for scripts
'''

import sys
from six import iteritems

__all__ = ['parse_args']

def parse_args(num_args, allowed_params, msg, string_args=set()):
    msg += '\nAllowed arguments and default values:\n'
    for k, v in iteritems(allowed_params):
        msg += '\n    %s = %s' % (k, v)
    if len(sys.argv)<=num_args:
        print(msg)
        exit(1)
    params = {}
    for spec in sys.argv[num_args+1:]:
        name, val = spec.split('=')
        if name not in string_args:
            if val.lower()=='true':
                val = 'True'
            elif val.lower()=='false':
                val = 'False'
            val = eval(val)
        params[name] = val
    for k in list(params.keys()):
        if k not in allowed_params:
            print(msg)
            exit(1)
    for k, v in iteritems(allowed_params):
        if k not in params:
            params[k] = v
    return sys.argv[1:num_args+1], params
