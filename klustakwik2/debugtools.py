'''
Some tools for debugging
'''

__all__ = ['dump_covariance_matrices']

def covariance_matrix_dump_callback(kk):
    for cluster, cov in enumerate(kk.covariance):
        msg = 'Covariance matrix for cluster %d:\n' % cluster
        msg += 'Unmasked: %s\nMasked: %s\n' % (cov.unmasked, cov.masked)
        msg += 'Block:\n%s\nDiagonal:\n%s\n' % (cov.block, cov.diagonal)
        kk.log('debug', msg, suffix='covariance_matrices')

def dump_covariance_matrices(kk):
    kk.register_callback(covariance_matrix_dump_callback, 'end_M_step')
