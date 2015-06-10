'''
Some tools for debugging
'''
from numpy import *
import time
import subprocess
import os
from six import iteritems, exec_

__all__ = ['dump_covariance_matrices', 'dump_variable', 'dump_all', 'get_kk_version']


def get_kk_version(version):
    curdir = os.getcwd()
    filedir, _ = os.path.split(__file__)
    os.chdir(filedir)
    try:
        fnull = open(os.devnull, 'w')
        version = 'git-'+subprocess.check_output(['git', 'describe', '--abbrev=8', '--dirty',
                                                       '--always', '--tags'], stderr=fnull).strip().decode('ascii')
    except:
        pass
    os.chdir(curdir)
    return version


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
        exec_('from numpy import *', self.ns)
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
        for k, v in iteritems(kwds):
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
