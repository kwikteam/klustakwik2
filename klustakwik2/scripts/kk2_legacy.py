'''
Cluster data starting from a fet/fmask pair and outputting a clu

Usage:
    kk2_legacy basefname shanknum [param=val, ...]
    
Do not put any spaces in each param=val.
'''

from numpy import *
from klustakwik2 import *

import sys
import time

def main():
    script_params = default_parameters.copy()
    script_params.update(
        drop_last_n_features=0,
        save_clu_every=None,
        run_monitoring_server=False,
        save_all_clu=False,
        debug=True,
        start_from_clu=None,
        use_noise_cluster=True,
        use_mua_cluster=True,
        subset_schedule=None,
        )
    (fname, shank), params = parse_args(2, script_params, __doc__.strip()+'\n',
                                        string_args=set(['start_from_clu']))
    
    drop_last_n_features = params.pop('drop_last_n_features')
    save_clu_every = params.pop('save_clu_every')
    run_monitoring_server = params.pop('run_monitoring_server')
    save_all_clu = params.pop('save_all_clu')
    debug = params.pop('debug')
    num_starting_clusters = params.pop('num_starting_clusters')
    start_from_clu = params.pop('start_from_clu')
    use_noise_cluster = params.pop('use_noise_cluster')
    use_mua_cluster = params.pop('use_mua_cluster')
    subset_schedule = params.pop('subset_schedule')
    
    if subset_schedule is not None:
        if save_clu_every is not None:
            print('Note that intermediate clu files will only be saved for the last subset operation.')
        if not use_noise_cluster:
            print('Must use noise cluster when using subsetting')
            exit(1)

    if debug:
        log_to_file(fname+'.klg.'+shank, 'debug')
    else:
        log_to_file(fname+'.klg.'+shank, 'info')
    log_suppress_hierarchy('klustakwik', inclusive=False)

    start_time = time.time()
    raw_data = load_fet_fmask_to_raw(fname, shank, drop_last_n_features=drop_last_n_features)
    log_message('debug', 'Loading data from .fet and .fmask file took %.2f s' % (time.time()-start_time))
    data = raw_data.to_sparse_data()
    
    log_message('info', 'Number of spikes in data set: '+str(data.num_spikes))
    log_message('info', 'Number of unique masks in data set: '+str(data.num_masks))

    kk = KK(data, use_noise_cluster=use_noise_cluster, use_mua_cluster=use_mua_cluster, **params)
    
    if save_clu_every is not None:
        kk.register_callback(SaveCluEvery(fname, shank, save_clu_every, save_all=save_all_clu))
    if run_monitoring_server:
        kk.register_callback(MonitoringServer())
    
    if start_from_clu is None:
        if subset_schedule is None:
            kk.cluster_mask_starts()
        else:
            kk.cluster_with_subset_schedule(num_starting_clusters, subset_schedule)
    else:
        clusters = load_clu(start_from_clu)
        kk.cluster_from(clusters)
    clusters = kk.clusters
    save_clu(kk, fname, shank)
    
if __name__=='__main__':
    main()
    