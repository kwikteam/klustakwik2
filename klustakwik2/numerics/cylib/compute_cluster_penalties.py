from .compute_cluster_penalties_cy import do_compute_penalties
from numpy import zeros, amax

__all__ = ['compute_penalties']

def compute_penalties(kk, clusters):
        if clusters is None:
            clusters = kk.clusters
        num_clusters = amax(clusters)+1
        cluster_penalty = zeros(num_clusters)
        ustart = kk.data.unmasked_start
        uend = kk.data.unmasked_end
        penalty_k = kk.penalty_k
        penalty_k_log_n = kk.penalty_k_log_n
        float_num_unmasked = kk.data.float_num_unmasked
        num_spikes = zeros(num_clusters, dtype=int)

        do_compute_penalties(cluster_penalty, num_spikes, clusters,
                             penalty_k, penalty_k_log_n, float_num_unmasked)

        return cluster_penalty
    