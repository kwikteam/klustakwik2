from numpy import *
from .e_step_cy import *
import multiprocessing
from six import itervalues

__all__ = ['compute_log_p_and_assign']

def compute_log_p_and_assign(kk, cluster, weight, inv_cov_diag, log_root_det, chol, cluster_mean,
                             only_evaluate_current_clusters, n_cpu=None):
    num_clusters = len(kk.num_cluster_members)
    num_features = kk.num_features
    num_spikes = kk.num_spikes

    log_addition = log_root_det-log(weight)+0.5*log(2*pi)*num_features

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
    noise_variance = data.noise_variance
    correction_terms = data.correction_terms

    clusters = kk.clusters
    clusters_second_best = kk.clusters_second_best
    old_clusters = kk.old_clusters
    full_step = kk.full_step

    if n_cpu is None:
        n_cpu = multiprocessing.cpu_count()
    f2cm_multiple = numpy.zeros(num_features*n_cpu)
    root_multiple = numpy.zeros(num_features*n_cpu)
    cluster_log_p = numpy.zeros(num_spikes)

    if only_evaluate_current_clusters:
        candidates = kk.quick_step_candidates[cluster]
        num_spikes = len(candidates)
    elif full_step:
        candidates = zeros(0, dtype=int)
    else:
        candidates = kk.quick_step_candidates[cluster]
        num_spikes = len(candidates)

    do_log_p_assign_computations(
                                  noise_mean, noise_variance, cluster_mean, correction_terms,
                                  log_p_best, log_p_second_best,
                                  clusters, clusters_second_best, old_clusters,
                                  full_step,
                                  inv_cov_diag,
                                  unmasked, ustart, uend, features, vstart, vend,
                                  root_multiple, f2cm_multiple,
                                  num_features, num_spikes, log_addition, cluster,
                                  chol.block, chol.diagonal, chol.masked, chol.unmasked,
                                  n_cpu,
                                  cluster_log_p,
                                  candidates,
                                  only_evaluate_current_clusters,
                                  )

    if only_evaluate_current_clusters:
        kk.log_p_best[candidates] = cluster_log_p[candidates]
    elif full_step and kk.collect_candidates and hasattr(kk, 'old_log_p_best') and kk.full_step_every>1:
        max_quick_step_candidates = min(kk.max_quick_step_candidates,
            kk.max_quick_step_candidates_fraction*kk.num_spikes*kk.num_clusters_alive)
        candidates, = (cluster_log_p-kk.old_log_p_best<=kk.dist_thresh).nonzero()
        kk.quick_step_candidates[cluster] = array(candidates, dtype=int)
        num_candidates = sum(len(v) for v in itervalues(kk.quick_step_candidates))
        if num_candidates>max_quick_step_candidates:
            kk.collect_candidates = False
            kk.quick_step_candidates.clear()
            kk.force_next_step_full = True
            if num_candidates>kk.max_quick_step_candidates:
                kk.log('info', 'Ran out of storage space for quick step, try increasing '
                               'max_quick_step_candidates if this happens often.')
            else:
                kk.log('debug', 'Exceeded quick step point fraction, next step '
                                'will be full')

    return kk.num_spikes-num_spikes
