from .m_step_cy import *

__all__ = ['compute_cluster_means', 'compute_covariance_matrices']

def get_diagonal(x):
    '''
    Return a writeable view of the diagonal of x
    '''
    return x.reshape(-1)[::x.shape[0]+1]


def compute_cluster_means(kk):
    data = kk.data
    num_clusters = len(kk.num_cluster_members)
    num_features = kk.num_features    
    
    cluster_mean = numpy.zeros((num_clusters, num_features))
    num_added = numpy.zeros((num_clusters, num_features), dtype=int)

    do_compute_cluster_means(
                kk.clusters, data.unmasked, data.unmasked_start, data.unmasked_end,
                data.features, data.values_start, data.values_end,
                cluster_mean, num_added,
                kk.num_cluster_members, data.noise_mean,
                kk.mua_point, kk.prior_point)
                        
    return cluster_mean


def compute_covariance_matrices(kk):
    data = kk.data
    num_cluster_members = kk.num_cluster_members
    num_clusters = len(num_cluster_members)
    num_features = kk.num_features

    for cluster in xrange(1, num_clusters):
        cov = kk.covariance[cluster]
        block = cov.block
        block_diagonal = get_diagonal(block)

        spikes_in_cluster = kk.spikes_in_cluster
        spikes_in_cluster_offset = kk.spikes_in_cluster_offset
        spike_indices = spikes_in_cluster[spikes_in_cluster_offset[cluster]:spikes_in_cluster_offset[cluster+1]]

        f2m = numpy.zeros(num_features)
        ct = numpy.zeros(num_features)
        
        if cluster==1:
            point = kk.mua_point
            do_var_accum_mua(spike_indices, kk.cluster_mean[cluster, :], cov.unmasked, block,
                             data.unmasked, data.unmasked_start, data.unmasked_end,
                             data.features, data.values_start, data.values_end,
                             f2m, ct, data.correction_terms, num_features,
                             )
        else:
            point = kk.prior_point
            do_var_accum(spike_indices, kk.cluster_mean[cluster, :], cov.unmasked, block,
                         data.unmasked, data.unmasked_start, data.unmasked_end,
                         data.features, data.values_start, data.values_end,
                         f2m, ct, data.correction_terms, num_features,
                         )
        
        # Add prior
        block_diagonal[:] += point*data.noise_variance[cov.unmasked]
        cov.diagonal[:] += point*data.noise_variance[cov.masked]
        
        # Normalise
        factor = 1.0/(num_cluster_members[cluster]+point-1)
        cov.block *= factor
        cov.diagonal *= factor
