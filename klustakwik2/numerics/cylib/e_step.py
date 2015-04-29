from numpy import *
from .e_step_cy import *

__all__ = ['compute_log_p_and_assign']

def compute_log_p_and_assign(kk, cluster, inv_cov_diag, log_root_det, chol):
    num_clusters = len(kk.num_cluster_members)
    num_features = kk.num_features
    num_spikes = kk.num_spikes

    weight = kk.weight
    log_addition = log_root_det-log(weight[cluster])+0.5*log(2*pi)*num_features

    data = kk.data
    unmasked = data.unmasked
    ustart = data.unmasked_start
    uend = data.unmasked_end
    features = data.features
    vstart = data.values_start
    vend = data.values_end
    
    f2cm = numpy.zeros(num_features)
    root = numpy.zeros(num_features)
    
    log_p_best = kk.log_p_best
    log_p_second_best = kk.log_p_second_best
    noise_mean = data.noise_mean
    cluster_mean = kk.cluster_mean
    correction_terms = data.correction_terms
    
    clusters = kk.clusters
    clusters_second_best = kk.clusters_second_best
    old_clusters = kk.old_clusters
    full_step = kk.full_step

    num_skipped = do_log_p_assign_computations(
                                  noise_mean, cluster_mean, correction_terms,
                                  log_p_best, log_p_second_best,
                                  clusters, clusters_second_best, old_clusters,
                                  full_step,
                                  inv_cov_diag, weight,
                                  unmasked, ustart, uend, features, vstart, vend,
                                  root, f2cm, num_features, num_spikes, log_addition, cluster,
                                  chol.block, chol.diagonal, chol.masked, chol.unmasked,
                                  )
    
    return num_skipped
