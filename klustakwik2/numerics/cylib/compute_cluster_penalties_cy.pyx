#cython: boundscheck=False
#cython: wraparound=False
#cython: cdivision=True
#cython: infer_types=True

import numpy
cimport numpy

from cython cimport integral, floating

from libc.math cimport log

cpdef do_compute_penalties(
            floating[:] cluster_penalty,
            integral[:] num_spikes,
            integral[:] clusters,
            floating penalty_k,
            floating penalty_k_log_n,
            floating[:] float_num_unmasked,
            ):
    cdef integral p, c, total_num_spikes, num_clusters
    cdef floating num_unmasked, num_params, mean_params
    total_num_spikes = len(clusters)
    num_clusters = len(cluster_penalty)
    for p in xrange(total_num_spikes):
        c = clusters[p]
        num_unmasked = float_num_unmasked[p]
        num_params = num_unmasked*(num_unmasked+1)/2.0+num_unmasked+1.0
        cluster_penalty[c] += num_params
        num_spikes[c] += 1
    for c in xrange(num_clusters):
        if num_spikes[c]==0:
            num_spikes[c] = 1
        mean_params = cluster_penalty[c]/num_spikes[c]
        cluster_penalty[c] = (penalty_k*mean_params*2+
                              penalty_k_log_n*mean_params*log(total_num_spikes)/2)
