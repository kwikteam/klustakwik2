'''
Wrapper for distributing work over multiple machines.

This can be used to abstract distribution using multiprocessing, IPython, etc.
'''

__all__ = ['Distributer', 'MockDistributer']

class Distributer(object):
    '''
    Base class for distributing KK work
    '''
    def __init__(self):
        pass

    def start(self, kk):
        raise NotImplementedError

    def iteration(self, clusters, cluster_order, full_step, only_evaluate_current_clusters):
        raise NotImplementedError

    def iteration_results(self):
        raise NotImplementedError


class MockDistributer(Distributer):
    '''
    Class for mocking distributed computation in one process to test code.
    '''
    def __init__(self, N):
        self.N = N

    def start(self, kk):
        self.kk_copies = [kk.copy_without_callbacks() for _ in range(self.N)]

    def iteration(self, clusters, cluster_order, full_step, only_evaluate_current_clusters):
        for kk_copy in self.kk_copies:
            kk_copy.clusters = clusters.copy()
            kk_copy.full_step = full_step
            kk_copy.reindex_clusters()
            kk_copy.prepare_for_MEC_steps(only_evaluate_current_clusters=only_evaluate_current_clusters)
        for i, cluster in enumerate(cluster_order):
            kk_copy = self.kk_copies[i%self.N]
            kk_copy.MEC_steps_cluster(cluster, only_evaluate_current_clusters=only_evaluate_current_clusters)

    def iteration_results(self):
        for kk_copy in self.kk_copies:
            res = dict(
                log_p_best=kk_copy.log_p_best,
                log_p_second_best=kk_copy.log_p_second_best,
                clusters=kk_copy.clusters,
                clusters_second_best=kk_copy.clusters_second_best,
                )
            yield res
