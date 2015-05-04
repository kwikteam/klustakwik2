'''
Some tools for debugging
'''
from numpy import *

__all__ = ['dump_covariance_matrices', 'dump_variable']

def covariance_matrix_dump_callback(kk):
    for cluster, cov in enumerate(kk.covariance):
        msg = 'Covariance matrix for cluster %d:\n' % cluster
        msg += 'Unmasked: %s\nMasked: %s\n' % (cov.unmasked, cov.masked)
        msg += 'Block:\n%s\nDiagonal:\n%s\n' % (cov.block, cov.diagonal)
        kk.log('debug', msg, suffix='covariance_matrices')


def dump_covariance_matrices(kk):
    kk.register_callback(covariance_matrix_dump_callback, 'end_M_step')
    
    
class DumpVariableCallback(object):
    def __init__(self, varname, iscode, suffix):
        if not iscode:
            self.varcode = 'kk.%s' % varname
        else:
            self.varcode = varname
        self.suffix = suffix
        self.ns = {}
        exec 'from numpy import *' in self.ns
    def __call__(self, kk):
        self.ns['kk'] = kk
        obj = eval(self.varcode, self.ns)
        msg = str(obj)
        if '\n' in msg:
            msg = '\n'+msg
        kk.log('debug', msg, suffix=self.suffix)
    
    
def dump_variable(kk, varname, slot='end_iteration', suffix=None, iscode=False):
    if suffix is None:
        suffix = varname
    kk.register_callback(DumpVariableCallback(varname, iscode, suffix), slot=slot)
