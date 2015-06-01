import numpy
import numba
from six.moves import range

def compute_cluster_means(kk):
    num_clusters = len(kk.num_cluster_members)
    num_features = kk.num_features
    data = kk.data

    cluster_mean = numpy.zeros((num_clusters, num_features))
    num_added = numpy.zeros((num_clusters, num_features), dtype=int)

    _compute_cluster_means(kk.clusters,
                           data.unmasked, data.unmasked_start, data.unmasked_end,
                           data.features, data.values_start, data.values_end,
                           cluster_mean, num_added,
                           kk.mua_point, kk.prior_point,
                           data.noise_mean, kk.num_cluster_members)

    return cluster_mean

@numba.jit(nopython=True)
def _compute_cluster_means(
            clusters,
            unmasked, ustart, uend,
            features, vstart, vend,
            cluster_mean, num_added,
            mua_point, prior_point,
            noise_mean, num_cluster_members,
            ):

    num_clusters = len(num_cluster_members)
    num_features = cluster_mean.shape[1]

    for p in range(len(clusters)):
        c = clusters[p]
        num_unmasked = uend[p]-ustart[p]
        for i in range(num_unmasked):
            j = unmasked[ustart[p]+i]
            k = vstart[p]+i
            cluster_mean[c, j] += features[k]
            num_added[c, j] += 1

    for cluster in range(num_clusters):
        if num_cluster_members[cluster]==0:
            continue
        prior = 0
        if cluster==1:
            prior = mua_point
        elif cluster>=2:
            prior = prior_point
        for i in range(num_features):
            cluster_mean[cluster, i] += noise_mean[i]*(num_cluster_members[cluster]-num_added[cluster, i])
            cluster_mean[cluster, i] += prior*noise_mean[i]
            cluster_mean[cluster, i] /= num_cluster_members[cluster]+prior
