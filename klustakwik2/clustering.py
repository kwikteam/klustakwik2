from numpy import *
from numpy.linalg import LinAlgError
from numpy.random import randint
from itertools import izip

from .logger import log_message

from .mask_starts import mask_starts
from .linear_algebra import BlockPlusDiagonalMatrix
from .default_parameters import default_parameters

from .cylib.compute_cluster_masks import accumulate_cluster_mask_sum
# from .cylib.m_step import compute_cluster_means, compute_covariance_matrix
from .cylib.m_step import compute_covariance_matrix
from .numbalib.m_step import compute_cluster_means
from .cylib.e_step import compute_log_p_and_assign

import time

__all__ = ['KK']

def get_diagonal(x):
    '''
    Return a writeable view of the diagonal of x
    '''
    return x.reshape(-1)[::x.shape[0]+1]


def add_slots(meth):
    def new_meth(self, *args, **kwds):
        self.run_callbacks('start_'+meth.__name__)
        res = meth(self, *args, **kwds)
        self.run_callbacks('end_'+meth.__name__)
        return res
    new_meth.__name__ = meth.__name__
    new_meth.__doc__ = meth.__doc__
    return new_meth

        
class KK(object):
    def __init__(self, data, callbacks=None, name='',
                 **params):
        self.name = name
        if callbacks is None:
            callbacks = {}
        self.callbacks = callbacks
        self.data = data
        self.params = params
        actual_params = default_parameters.copy()
        for k, v in params.iteritems():
            if k not in default_parameters:
                raise ValueError("There is no parameter "+k)
            actual_params[k] = v
        for k, v in actual_params.iteritems():
            setattr(self, k, v)
            if name=='':
                self.log('info', '%s = %s' % (k, v), suffix='initial_parameters')

    def register_callback(self, callback, slot='end_iteration'):
        if slot not in self.callbacks:
            self.callbacks[slot] = []
        self.callbacks[slot].append(callback)
        
    def run_callbacks(self, slot):
        if slot in self.callbacks:
            for callback in self.callbacks[slot]:
                callback(self)
            
    def log(self, level, msg, suffix=None):
        if suffix is not None:
            if self.name=='':
                name = suffix
            else:
                name = self.name+'.'+suffix
        else:
            name = self.name
        log_message(level, msg, name=name)
            
    def copy(self, name='kk_copy'):
        if self.name:
            sep = '.'
        else:
            sep = ''
        return KK(self.data, name=self.name+sep+name,
                  callbacks=self.callbacks,
                  **self.params)
        
    def subset(self, spikes, name='kk_subset'):
        newdata = self.data.subset(spikes)
        if self.name:
            sep = '.'
        else:
            sep = ''
        return KK(newdata, name=self.name+sep+name,
                  callbacks=self.callbacks,
                  **self.params)
    
    def initialise_clusters(self, clusters):
        self.clusters = clusters
        self.old_clusters = -1*ones(len(self.clusters), dtype=int)
        self.reindex_clusters()
    
    def cluster(self, num_starting_clusters):
        self.log('info', 'Clustering full data set of %d points' % self.data.num_spikes)
        clusters = mask_starts(self.data, num_starting_clusters)
        self.cluster_from(clusters)
        
    def cluster_from(self, clusters, recurse=True, allow_assign_to_noise=True):
        self.initialise_clusters(clusters)
        return self.CEM(recurse=recurse, allow_assign_to_noise=allow_assign_to_noise)
    
    def prepare_for_CEM(self):
        self.full_step = True
        self.current_iteration = 0
    
    def CEM(self, recurse=True, allow_assign_to_noise=True):        
        self.prepare_for_CEM()

        score = old_score = 0.0
                
        #while self.current_iteration==0: # for debugging
        while self.current_iteration<=self.max_iterations:
            self.log('debug', 'Starting iteration %d' % self.current_iteration)
            self.log('debug', 'Starting M-step')
            self.M_step()
            self.log('debug', 'Finished M-step')
            self.log('debug', 'Starting EC-steps')
            self.EC_steps(allow_assign_to_noise=allow_assign_to_noise)
            self.log('debug', 'Finished EC-steps')
            self.log('debug', 'Starting compute_cluster_penalties')
            self.compute_cluster_penalties()
            self.log('debug', 'Finished compute_cluster_penalties')
            if recurse:
                self.log('debug', 'Starting consider_deletion')
                self.consider_deletion()
                self.log('debug', 'Finished consider_deletion')
            self.log('debug', 'Starting compute_score')
            old_score, score = score, self.compute_score()
            self.log('debug', 'Finished compute_score')
            
            num_changed = sum(self.clusters!=self.old_clusters)
            
            self.current_iteration += 1
    
            last_step_full = self.full_step
            # TODO: add this back in when we have num_changed, etc.
            self.full_step = (num_changed>self.num_changed_threshold*self.num_spikes or
                              num_changed==0 or
                              self.current_iteration % self.full_step_every == 0 or
                              score > old_score) 

            self.reindex_clusters()
    
            QF_id = {True:'F', False:'Q'}[self.full_step]
            self.log('info', 'Iteration %d%s: %d clusters, %d changed, '
                             'score=%f' % (self.current_iteration, QF_id, self.num_clusters_alive,
                                           num_changed, score))

            # TODO: save current progress

            # Try splitting
            did_split = False
            if recurse and self.split_every>0:
                if (self.current_iteration==self.split_first or
                    (self.current_iteration>self.split_first and
                     self.current_iteration-self.split_first%self.split_every==self.split_every-1) or
                    (num_changed==0 and last_step_full)):
#                 if True:
                    did_split = self.try_splits()
               
            self.run_callbacks('end_iteration')     
                    
            if num_changed==0 and last_step_full and not did_split:
                self.log('info', 'No points changed, previous step was full and did not split, '
                                 'so finishing.')
                break
        else:
            # ran out of iterations
            self.log('info', 'Number of iterations exceeded maximum %d' % self.max_iterations)
            
        return score

    @add_slots    
    def M_step(self):
        # eliminate any clusters with 0 members, compute the list of spikes
        # in each cluster, compute the cluster masks and allocate space for
        # covariance matrices
        self.reindex_clusters()
        self.compute_cluster_masks()

        num_cluster_members = self.num_cluster_members
        num_clusters = self.num_clusters_alive
        num_features = self.num_features
        
        # Normalize by total number of points to give class weight
        denom = float(self.num_spikes+self.noise_point+self.mua_point+
                                      self.prior_point*(num_clusters-2))
        self.weight = weight = (num_cluster_members+self.prior_point)/denom
        # different calculation for clusters 0 and 1
        weight[0] = (num_cluster_members[0]+self.noise_point)/denom
        weight[1] = (num_cluster_members[1]+self.mua_point)/denom
        
        # Compute means for each cluster
        # Note that we do this densely at the moment, might want to switch
        # that to a sparse structure later
        self.cluster_mean = compute_cluster_means(self)
        
        # Compute covariance matrices
        for cluster in xrange(1, num_clusters):
            if cluster==1:
                point = self.mua_point
            else:
                point = self.prior_point
            compute_covariance_matrix(self, cluster)
            cov = self.covariance[cluster]
            block_diagonal = get_diagonal(cov.block)
            
            # Add prior
            block_diagonal[:] += point*self.data.noise_variance[cov.unmasked]
            cov.diagonal[:] += point*self.data.noise_variance[cov.masked]
            
            # Normalise
            factor = 1.0/(num_cluster_members[cluster]+point-1)
            cov.block *= factor
            cov.diagonal *= factor
                    
    @add_slots    
    def EC_steps(self, allow_assign_to_noise=True):
        if not allow_assign_to_noise:
            cluster_start = 2
        else:
            cluster_start = 1

        num_spikes = self.num_spikes
        num_clusters = self.num_clusters_alive
        num_features = self.num_features
        
        weight = self.weight

        self.old_clusters = self.clusters
        self.clusters = -ones(num_spikes, dtype=int)
        self.clusters_second_best = -ones(num_spikes, dtype=int)
        self.log_p_best = inf*ones(num_spikes)
        self.log_p_second_best = inf*ones(num_spikes)
        num_skipped = 0
        
        # start with cluster 0 - uniform distribution over space
        # because we have normalized all dims to 0...1, density will be 1.
        if allow_assign_to_noise:
            self.clusters[:] = 0
            self.log_p_best[:] = -log(weight[0])
        
        clusters_to_kill = []
        
        for cluster in xrange(cluster_start, num_clusters):
            cov = self.covariance[cluster]
            try:
                chol = cov.cholesky()
            except LinAlgError:
                clusters_to_kill.append(cluster) # todo: we don't actually do anything with this...
                continue

            # LogRootDet is given by log of product of diagonal elements
            log_root_det = sum(log(chol.diagonal))+sum(log(chol.block.diagonal()))

            # compute diagonal of inverse of cov matrix
            inv_cov_diag = zeros(num_features)
            basis_vector = zeros(num_features)
            for i in xrange(num_features):
                basis_vector[i] = 1.0
                root = chol.trisolve(basis_vector)
                inv_cov_diag[i] = sum(root**2)
                basis_vector[i] = 0.0
                
            compute_log_p_and_assign(self, cluster, inv_cov_diag, log_root_det, chol)

        # we've reassigned clusters so we need to recompute the partitions, but we don't want to
        # reindex yet because we may reassign points to different clusters and we need the original
        # cluster numbers for that
        self.partition_clusters()
    
    @add_slots    
    def compute_cluster_penalties(self):
        num_cluster_members = self.num_cluster_members
        num_clusters = self.num_clusters_alive
        self.cluster_penalty = cluster_penalty = zeros(num_clusters)
        sic = self.spikes_in_cluster
        sico = self.spikes_in_cluster_offset
        ustart = self.data.unmasked_start
        uend = self.data.unmasked_end
        penalty_k = self.penalty_k
        penalty_k_log_n = self.penalty_k_log_n
        for cluster in xrange(num_clusters):
            curspikes = sic[sico[cluster]:sico[cluster+1]]
            num_spikes = len(curspikes)
            if num_spikes>0:
                num_unmasked = uend[curspikes]-ustart[curspikes]
                num_params = sum(num_unmasked*(num_unmasked+1)/2+num_unmasked+1)
                mean_params = float(num_params)/num_spikes
                cluster_penalty[cluster] = penalty_k*mean_params*2+penalty_k_log_n*mean_params*log(mean_params)/2
    
    @add_slots    
    def consider_deletion(self):
        num_cluster_members = self.num_cluster_members
        num_clusters = self.num_clusters_alive
        sic = self.spikes_in_cluster
        sico = self.spikes_in_cluster_offset
        log_p_best = self.log_p_best
        log_p_second_best = self.log_p_second_best
        
        deletion_loss = zeros(num_clusters)
        I = arange(self.num_spikes)
        add.at(deletion_loss, self.clusters, log_p_second_best-log_p_best)
        candidate_cluster = 2+argmin((deletion_loss-self.cluster_penalty)[2:])
        loss = deletion_loss[candidate_cluster]
        delta_pen = self.cluster_penalty[candidate_cluster]
        
        if loss<0:
            # delete this cluster
            num_points_in_candidate = sico[candidate_cluster+1]-sico[candidate_cluster]
            self.log('info', 'Deleting cluster {cluster} ({numpoints} points): lose {lose} but '
                             'gain {gain}'.format(cluster=candidate_cluster,
                                                  numpoints=num_points_in_candidate,
                                                  lose=deletion_loss[candidate_cluster],
                                                  gain=delta_pen))
            # reassign points
            cursic = sic[sico[candidate_cluster]:sico[candidate_cluster+1]]
            self.clusters[cursic] = self.clusters_second_best[cursic]
            self.log_p_best[cursic] = self.log_p_second_best[cursic]
            # recompute penalties
            self.compute_cluster_penalties()
            
        # at this point we have invalidated the partitions, so to make sure we don't miss
        # something, we wipe them out here
        self.invalidate_partitions()
        # we've also invalidated the second best log_p and clusters
        self.log_p_second_best = None
        self.clusters_second_best = None

    @add_slots    
    def compute_score(self):
        penalty = sum(self.cluster_penalty)
        raw = sum(self.log_p_best)
        score = raw+penalty
        self.log('debug', 'compute_score: raw %f + penalty %f = %f' % (raw, penalty, score))
        self.run_callbacks('end_compute_score')
        return score

    @property
    def num_spikes(self):
        return self.data.num_spikes
    
    @property
    def num_features(self):
        return self.data.num_features#
    
    @property
    def num_clusters_alive(self):
        return len(self.num_cluster_members)
    
    def reindex_clusters(self):
        '''
        Remove any clusters with 0 members (except for clusters 0 and 1),
        and recompute the list of spikes in each cluster. After this function is
        run, you can use the attributes:
        
        - num_cluster_members (of length the number of clusters)
        - spikes_in_cluster, spikes_in_cluster_offset

        spikes_in_cluster[spikes_in_cluster_offset[c]:spikes_in_cluster_offset[c+1]] will be in the indices
        of all the spikes in cluster c. 
        '''
        num_cluster_members = array(bincount(self.clusters), dtype=int)
        I = num_cluster_members>0
        I[0:2] = True # we keep clusters 0 and 1
        remapping = hstack((0, cumsum(I)))[:-1]
        self.clusters = remapping[self.clusters]
        if hasattr(self, 'clusters_second_best'):
            del self.clusters_second_best
        self.partition_clusters()
        
    def partition_clusters(self):
        self.num_cluster_members = num_cluster_members = array(bincount(self.clusters), dtype=int)
        I = array(argsort(self.clusters), dtype=int)
        y = self.clusters[I]
        n = amax(y)+2
        J = searchsorted(y, arange(n))
        self.spikes_in_cluster = I
        self.spikes_in_cluster_offset = J
    
    def invalidate_partitions(self):
        self.num_cluster_members = None
        self.spikes_in_cluster = None
        self.spikes_in_cluster_offset = None
        
    def get_spikes_in_cluster(self, cluster):
        sic = self.spikes_in_cluster
        sico = self.spikes_in_cluster_offset
        return sic[sico[cluster]:sico[cluster+1]]
        
    @add_slots    
    def compute_cluster_masks(self):
        '''
        Computes the masked and unmasked indices for each cluster based on the
        masks for each point in that cluster. Allocates space for covariance
        matrices.
        '''
        num_clusters = self.num_clusters_alive
        num_features = self.num_features
        
        # Compute the sum of 
        cluster_mask_sum = zeros((num_clusters, num_features))
        cluster_mask_sum[:2, :] = -1 # ensure that clusters 0 and 1 are masked
        # Use efficient version
        accumulate_cluster_mask_sum(self, cluster_mask_sum)
        
        # Compute the masked and unmasked sets
        self.cluster_masked_features = []
        self.cluster_unmasked_features = []
        self.covariance = []
        for cluster in xrange(num_clusters):
            curmask = cluster_mask_sum[cluster, :]
            unmasked, = (curmask>=self.points_for_cluster_mask).nonzero()
            masked, = (curmask<self.points_for_cluster_mask).nonzero()
            unmasked = array(unmasked, dtype=int)
            masked = array(masked, dtype=int)
            self.cluster_masked_features.append(masked)
            self.cluster_unmasked_features.append(unmasked)
            self.covariance.append(BlockPlusDiagonalMatrix(masked, unmasked))
            
    @add_slots    
    def try_splits(self):
        did_split = False
        num_clusters = self.num_clusters_alive
        
        self.log('info', 'Trying to split clusters')
        
        self.log('debug', 'Computing score before splitting')
        score = self.compute_score()
        
        for cluster in xrange(2, num_clusters):
            if num_clusters>=self.max_possible_clusters:
                self.log('info', 'No more splitting, already at maximum number of '
                                 'clusters' % self.max_possible_clusters)
                return did_split
            
            spikes_in_cluster = self.get_spikes_in_cluster(cluster)
            if len(spikes_in_cluster)==0:
                continue
            K2 = self.subset(spikes_in_cluster, name='split_candidate')
            # at this point in C++ code we look for an unused cluster, but here we can just
            # use num_clusters+1
            self.log('debug', 'Trying to split cluster %d' % cluster)
            # initialise with current clusters, do not allow creation of new clusters
            K2.max_possible_clusters = 3
            clusters = full(len(spikes_in_cluster), 2, dtype=int)
            unsplit_score = K2.cluster_from(clusters, recurse=False, allow_assign_to_noise=False)
            # initialise randomly, allow for one additional cluster
            K2.max_possible_clusters = 4
            clusters = randint(2, 4, size=len(spikes_in_cluster))
            if len(unique(clusters))!=2: # todo: better way of handling this?
                continue
            split_score = K2.cluster_from(clusters, recurse=False, allow_assign_to_noise=False)
            
            if K2.num_clusters_alive<3:
                # todo: logging
                continue
            
            if split_score>=unsplit_score:
                # todo: logging
                continue
                
            # todo: always split bimodal
#             if (AlwaysSplitBimodal)
#             {
#                 DidSplit = 1;
#                 Output("\n We are always splitting bimodal clusters so it's getting split into cluster %d.\n", (int)UnusedCluster);
#                 p2 = 0;
#                 for (p = 0; p < nPoints; p++)
#                 {
#                     if (Class[p] == c)
#                     {
#                         if (K2.Class[p2] == 2) Class[p] = c;
#                         else if (K2.Class[p2] == 3) Class[p] = UnusedCluster;
#                         else Error("split should only produce 2 clusters\n");
#                         p2++;
#                     }
#                     ClassAlive[Class[p]] = 1;
#                 }
#             }

            # will splitting improve the score in the whole data set?
            K3 = self.copy(name='split_evaluation')
            K3.prepare_for_CEM()
            clusters = self.clusters.copy()
            I3 = (K2.clusters==3)
            clusters[spikes_in_cluster[I3]] = num_clusters # next available cluster
            K3.initialise_clusters(clusters)
            K3.M_step()
            K3.EC_steps() # todo: original code omits C step - a problem?
            K3.compute_cluster_penalties()
            new_score = K3.compute_score()
            # todo: logging
            if new_score<score:
                did_split = True
                self.clusters = K3.clusters
                num_clusters += 1
            else:
                pass
            
        return did_split
