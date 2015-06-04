from .m_step_cy import *
from six.moves import range

__all__ = ['compute_cluster_mean', 'compute_covariance_matrix']

def get_diagonal(x):
    '''
    Return a writeable view of the diagonal of x
    '''
    return x.reshape(-1)[::x.shape[0]+1]


def compute_cluster_mean(kk, cluster):
    data = kk.data
    num_clusters = len(kk.num_cluster_members)
    num_features = kk.num_features
    
    cluster_mean = numpy.zeros(num_features)
    num_added = numpy.zeros(num_features, dtype=int)
    spikes = kk.get_spikes_in_cluster(cluster)

    prior = 0
    if cluster==kk.mua_cluster:
        prior = kk.mua_point
    elif cluster>=kk.num_special_clusters:
        prior = kk.prior_point

    do_compute_cluster_mean(
                spikes, data.unmasked, data.unmasked_start, data.unmasked_end,
                data.features, data.values_start, data.values_end,
                cluster_mean, num_added, data.noise_mean, prior)

    return cluster_mean


def compute_covariance_matrix(kk, cluster, cluster_mean, cov):
    if cluster<kk.first_gaussian_cluster:
        return
    data = kk.data
    num_cluster_members = kk.num_cluster_members
    num_clusters = len(num_cluster_members)
    num_features = kk.num_features

    block = cov.block
    block_diagonal = get_diagonal(block)

    spikes_in_cluster = kk.spikes_in_cluster
    spikes_in_cluster_offset = kk.spikes_in_cluster_offset
    spike_indices = spikes_in_cluster[spikes_in_cluster_offset[cluster]:spikes_in_cluster_offset[cluster+1]]

    f2m = numpy.zeros(num_features)
    ct = numpy.zeros(num_features)
    
    if kk.use_mua_cluster and cluster==kk.mua_cluster:
        point = kk.mua_point
        do_var_accum_mua(spike_indices, cluster_mean,
                         kk.data.noise_mean, kk.data.noise_variance,
                         cov.unmasked, block,
                         data.unmasked, data.unmasked_start, data.unmasked_end,
                         data.features, data.values_start, data.values_end,
                         f2m, ct, data.correction_terms, num_features,
                         )
    else:
        point = kk.prior_point
        do_var_accum(spike_indices, cluster_mean,
                     kk.data.noise_mean, kk.data.noise_variance,
                     cov.unmasked, block,
                     data.unmasked, data.unmasked_start, data.unmasked_end,
                     data.features, data.values_start, data.values_end,
                     f2m, ct, data.correction_terms, num_features,
                     )
        
    # add correction term for diagonal
    cov.diagonal[:] += len(spike_indices)*data.noise_variance[cov.masked]
    
    # Add prior
    block_diagonal[:] += point*data.noise_variance[cov.unmasked]
    cov.diagonal[:] += point*data.noise_variance[cov.masked]
    
    # Normalise
    factor = 1.0/(num_cluster_members[cluster]+point-1)
    cov.block *= factor
    cov.diagonal *= factor
