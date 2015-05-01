from numpy import *
from .e_step_cy import *
import multiprocessing

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
    
    
    log_p_best = kk.log_p_best
    log_p_second_best = kk.log_p_second_best
    noise_mean = data.noise_mean
    cluster_mean = kk.cluster_mean
    correction_terms = data.correction_terms
    
    clusters = kk.clusters
    clusters_second_best = kk.clusters_second_best
    old_clusters = kk.old_clusters
    full_step = kk.full_step
    
    n_cpu = multiprocessing.cpu_count()
    f2cm_multiple = numpy.zeros(num_features*n_cpu)
    root_multiple = numpy.zeros(num_features*n_cpu)
    cluster_log_p = numpy.zeros(num_spikes)

    num_skipped = do_log_p_assign_computations(
                                  noise_mean, cluster_mean, correction_terms,
                                  log_p_best, log_p_second_best,
                                  clusters, clusters_second_best, old_clusters,
                                  full_step,
                                  inv_cov_diag, weight,
                                  unmasked, ustart, uend, features, vstart, vend,
                                  root_multiple, f2cm_multiple, num_features, num_spikes, log_addition, cluster,
                                  chol.block, chol.diagonal, chol.masked, chol.unmasked,
                                  n_cpu,
                                  cluster_log_p,
                                  )
    
    return num_skipped
