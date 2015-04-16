'''
Cluster data starting from a fet/fmask pair and outputting a clu

Usage:
    python cluster_fet_fmask_to_clu basefname shanknum [param=val, ...]
    
Do not put any spaces in each param=val.
'''

from numpy import *
from klustakwik2 import *

import sys
import time

if __name__=='__main__':
    (fname, shank), params = parse_args(2, __doc__)

    log_to_file(fname+'.klg.'+shank, 'debug')
    log_suppress_hierarchy('klustakwik', inclusive=False)

    start_time = time.time()
    raw_data = load_fet_fmask_to_raw(fname, shank)
    log_message('debug', 'Loading data from .fet and .fmask file took %.2f s' % (time.time()-start_time))
    data = raw_data.to_sparse_data()
    
    log_message('info', 'Number of spikes in data set: '+str(data.num_spikes))
    log_message('info', 'Number of unique masks in data set: '+str(data.num_masks))

    kk = KK(data, **params)
    
    kk.cluster(kk.mask_starts)
    clusters = kk.clusters
    savetxt(fname+'.clu.'+shank, clusters, '%d', header=str(amax(clusters)), comments='')
    