'''
Some tools for debugging
'''
from numpy import *
import time

__all__ = ['dump_covariance_matrices', 'dump_variable', 'dump_all', 'dump_timings']

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


class DumpAllCallback(object):
    def __init__(self, slot):
        self.slot = slot
    def __call__(self, kk, *args, **kwds):
        kk.log('debug', 'Dumping variables from slot '+self.slot, suffix=self.slot)
        for i, arg in enumerate(args):
            self.dump_var(kk, 'arg'+str(i), arg)
        for k, v in kwds.iteritems():
            self.dump_var(kk, k, v)
    def dump_var(self, kk, name, val):
        msg = name+' = '
        if isinstance(val, ndarray):
            msg = msg+'\n'+str(val)
        else:
            msg = msg+str(val)
        kk.log('debug', msg, suffix=self.slot)


def dump_all(kk, slot):
    kk.register_callback(DumpAllCallback(slot), slot=slot)
    

class SectionTimer(object):
    def __init__(self, method_name, filter=None, name=None):
        self.t_total = 0.0
        self.num_calls = 0
        self.method_name = method_name
        if filter is None:
            filter = lambda kk: False
        self.filter = filter
        if name is None:
            name = 'time_'+method_name
        self.name = name
    def start(self, kk):
        if kk.name:
            return
        if self.filter(kk):
            return
        self.t_start = time.time()
    def end(self, kk):
        if kk.name:
            return
        if self.filter(kk):
            return
        this_time = time.time()-self.t_start
        self.t_total += this_time
        self.num_calls += 1
        mean_time = self.t_total/self.num_calls
        kk.log('debug', 'This call: %.2f ms. Average: %.2f ms' % (this_time*1000, mean_time*1000),
               suffix=self.name)
        

def dump_timings(kk, method_name):
    section_timer = SectionTimer(method_name)
    kk.register_callback(section_timer.start, slot='start_'+method_name)
    kk.register_callback(section_timer.end, slot='end_'+method_name)
    