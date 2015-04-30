'''
Some tools for debugging
'''

__all__ = ['dump_covariance_matrices', 'dump_cluster_counts', 'dump_cluster_means']

def covariance_matrix_dump_callback(kk):
    for cluster, cov in enumerate(kk.covariance):
        msg = 'Covariance matrix for cluster %d:\n' % cluster
        msg += 'Unmasked: %s\nMasked: %s\n' % (cov.unmasked, cov.masked)
        msg += 'Block:\n%s\nDiagonal:\n%s\n' % (cov.block, cov.diagonal)
        kk.log('debug', msg, suffix='covariance_matrices')


def dump_covariance_matrices(kk):
    kk.register_callback(covariance_matrix_dump_callback, 'end_M_step')


def cluster_count_dump_callback(kk):
    kk.log('debug', 'Cluster counts: %s' % kk.num_cluster_members, suffix='cluster_counts')


def dump_cluster_counts(kk, slot='end_EC_steps'):
    kk.register_callback(cluster_count_dump_callback, slot=slot)


def cluster_mean_dump_callback(kk):
    kk.log('debug', 'Cluster means: %s' % kk.cluster_mean, suffix='cluster_mean')


def dump_cluster_means(kk):
    kk.register_callback(cluster_mean_dump_callback, 'end_M_step')
    