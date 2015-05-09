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
    script_params = default_parameters.copy()
    script_params.update(
        drop_last_n_features=0,
        save_clu_every=None,
        run_monitoring_server=False,
        )
    (fname, shank), params = parse_args(2, script_params, __doc__.strip()+'\n')
    
    drop_last_n_features = params.pop('drop_last_n_features')
    save_clu_every = params.pop('save_clu_every')
    run_monitoring_server = params.pop('run_monitoring_server')

    log_to_file(fname+'.klg.'+shank, 'debug')
    log_suppress_hierarchy('klustakwik', inclusive=False)

    start_time = time.time()
    raw_data = load_fet_fmask_to_raw(fname, shank, drop_last_n_features=drop_last_n_features)
    log_message('debug', 'Loading data from .fet and .fmask file took %.2f s' % (time.time()-start_time))
    data = raw_data.to_sparse_data()
    
    log_message('info', 'Number of spikes in data set: '+str(data.num_spikes))
    log_message('info', 'Number of unique masks in data set: '+str(data.num_masks))

    kk = KK(data, **params)
    
    if save_clu_every is not None:
        kk.register_callback(SaveCluEvery(fname, shank, save_clu_every))
    if run_monitoring_server:
        kk.register_callback(MonitoringServer)
    
    kk.cluster(kk.mask_starts)
    clusters = kk.clusters
    savetxt(fname+'.clu.'+shank, clusters, '%d', header=str(amax(clusters)), comments='')
    