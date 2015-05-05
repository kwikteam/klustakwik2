from numpy import *
from numpy.linalg import LinAlgError
from numpy.random import randint
from itertools import izip
import hashlib

from .logger import log_message

from .mask_starts import mask_starts
from .linear_algebra import BlockPlusDiagonalMatrix
from .default_parameters import default_parameters

from .numerics import (accumulate_cluster_mask_sum, compute_cluster_means, compute_covariance_matrices,
                       compute_log_p_and_assign)

import time

__all__ = ['KK']

class PartitionError(Exception):
    pass


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
                 use_noise_cluster=True, use_mua_cluster=True,
                 **params):
        self.name = name
        if callbacks is None:
            callbacks = {}
        self.callbacks = callbacks
        self.data = data
        self.cluster_hashes = set()
        # user parameters
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
        # Assignment of special clusters
        self.num_special_clusters = 0
        self.first_gaussian_cluster = 0
        self.special_clusters = {}
        self.use_noise_cluster = use_noise_cluster
        self.use_mua_cluster = use_mua_cluster
        if use_noise_cluster:
            self.noise_cluster = self.special_clusters['noise'] = self.num_special_clusters
            self.num_special_clusters += 1
            self.first_gaussian_cluster += 1
        else:
            self.noise_cluster = -2
        if use_mua_cluster:
            self.mua_cluster = self.special_clusters['mua'] = self.num_special_clusters
            self.num_special_clusters += 1
        else:
            self.mua_cluster = -2

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
            
    def copy(self, name='kk_copy', **additional_params):
        if self.name:
            sep = '.'
        else:
            sep = ''
        params = self.params.copy()
        params.update(**additional_params)
        return KK(self.data, name=self.name+sep+name,
                  callbacks=self.callbacks,
                  **params)
        
    def subset(self, spikes, name='kk_subset', **additional_params):
        newdata = self.data.subset(spikes)
        if self.name:
            sep = '.'
        else:
            sep = ''
        params = self.params.copy()
        params.update(**additional_params)
        return KK(newdata, name=self.name+sep+name,
                  callbacks=self.callbacks,
                  **params)
    
    def initialise_clusters(self, clusters):
        self.clusters = clusters
        self.old_clusters = -1*ones(len(self.clusters), dtype=int)
        self.reindex_clusters()
    
    def cluster(self, num_starting_clusters):
        self.log('info', 'Clustering full data set of %d points, %d features' % (self.data.num_spikes,
                                                                                 self.data.num_features))
        clusters = mask_starts(self.data, num_starting_clusters, self.num_special_clusters)
        self.cluster_from(clusters)
        
    def cluster_from(self, clusters, recurse=True):
        self.initialise_clusters(clusters)
        return self.CEM(recurse=recurse)
    
    def prepare_for_CEM(self):
        self.full_step = True
        self.current_iteration = 0
    
    def CEM(self, recurse=True):        
        self.prepare_for_CEM()

        score = score_raw = score_penalty = None
        
        iterations_until_next_split = self.split_first
        tried_splitting_to_escape_cycle_hashes = set()
                
        while self.current_iteration<self.max_iterations:
            self.log('debug', 'Starting iteration %d' % self.current_iteration)
            self.log('debug', 'Starting M-step')
            self.M_step()
            self.log('debug', 'Finished M-step')
            self.log('debug', 'Starting EC-steps')
            self.EC_steps()
            self.log('debug', 'Finished EC-steps')
            self.log('debug', 'Starting compute_cluster_penalties')
            self.compute_cluster_penalties()
            self.log('debug', 'Finished compute_cluster_penalties')
            if recurse:
                self.log('debug', 'Starting consider_deletion')
                self.consider_deletion()
                self.log('debug', 'Finished consider_deletion')
            self.log('debug', 'Starting compute_score')
            old_score = score
            old_score_raw = score_raw
            old_score_penalty = score_penalty
            score, score_raw, score_penalty = self.compute_score()
            self.log('debug', 'Finished compute_score')
            
            num_changed = sum(self.clusters!=self.old_clusters)
            
            self.current_iteration += 1
    
            last_step_full = self.full_step
            self.full_step = (num_changed>self.num_changed_threshold*self.num_spikes or
                              num_changed==0 or
                              self.current_iteration % self.full_step_every == 0 or
                              (old_score is not None and score > old_score)) 

            self.reindex_clusters()
    
            QF_id = {True:'F', False:'Q'}[self.full_step]
            msg = 'Iteration %d%s: %d clusters, %d changed, score=%f' % (self.current_iteration, QF_id,
                                                                         self.num_clusters_alive, num_changed, score)
            if old_score is not None:
                msg += ' (decreased by %f)' % (old_score-score)
            self.log('info', msg)
            if old_score is not None:
                msg = 'Change in scores: raw=%f, penalty=%f, total=%f'  % (old_score_raw-score_raw,
                                                                           old_score_penalty-score_penalty,
                                                                           old_score-score)
                self.log('debug', msg)

            # Splitting logic
            iterations_until_next_split -= 1
            if num_changed==0 and last_step_full:
                self.log('info', 'No points changed and last step was full, so trying to split.')
                iterations_until_next_split = 0
                
            # Cycle detection/breaking
            cluster_hash = hashlib.sha1(self.clusters.view(uint8)).hexdigest()
            if cluster_hash in self.cluster_hashes and num_changed>0:
                if recurse:
                    if cluster_hash in tried_splitting_to_escape_cycle_hashes:
                        self.log('error', 'Cycle detected! Already tried attempting to break out '
                                          'by splitting, so abandoning.')
                        break
                    else:
                        self.log('warning', 'Cycle detected! Attempting to break out by splitting.')
                        iterations_until_next_split = 0
                    tried_splitting_to_escape_cycle_hashes.add(cluster_hash)
                else:
                    self.log('error', 'Cycle detected! Splitting is not enabled, so abandoning.')
                    break
            self.cluster_hashes.add(cluster_hash)

            # Try splitting
            did_split = False
            if recurse and iterations_until_next_split<=0:
                did_split = self.try_splits()
                iterations_until_next_split = self.split_every
               
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
        denom = self.num_spikes+self.prior_point*(num_clusters-self.num_special_clusters)
        if self.use_noise_cluster:
            denom += self.noise_point
        if self.use_mua_cluster:
            denom += self.mua_point
        denom = float(denom)
        self.weight = weight = (num_cluster_members+self.prior_point)/denom
        # different calculation for special clusters
        if self.use_noise_cluster:
            weight[self.noise_cluster] = (num_cluster_members[self.noise_cluster]+self.noise_point)/denom
        if self.use_mua_cluster:
            weight[self.mua_cluster] = (num_cluster_members[self.mua_cluster]+self.mua_point)/denom
        
        # Compute means for each cluster
        # Note that we do this densely at the moment, might want to switch
        # that to a sparse structure later
        self.cluster_mean = compute_cluster_means(self)
        
        # Compute covariance matrices
        compute_covariance_matrices(self)

                    
    @add_slots    
    def EC_steps(self):
        cluster_start = self.num_special_clusters
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
        
        if self.use_noise_cluster:
            # start with cluster 0 - uniform distribution over space
            # because we have normalized all dims to 0...1, density will be 1.
            self.clusters[:] = self.noise_cluster
            self.log_p_best[:] = -log(weight[self.noise_cluster])
        
        clusters_to_kill = []
        
        for cluster in xrange(cluster_start, num_clusters):
            cov = self.covariance[cluster]
            try:
                chol = cov.cholesky()
            except LinAlgError:
                self.log('warning', 'Linear algebra error on cluster '+str(cluster))
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
        float_num_unmasked = self.data.float_num_unmasked
        for cluster in xrange(num_clusters):
            curspikes = sic[sico[cluster]:sico[cluster+1]]
            num_spikes = len(curspikes)
            if num_spikes>0:
                num_unmasked = float_num_unmasked[curspikes]
                num_params = sum(num_unmasked*(num_unmasked+1)/2+num_unmasked+1)
                mean_params = float(num_params)/num_spikes
                cluster_penalty[cluster] = penalty_k*mean_params*2+penalty_k_log_n*mean_params*log(num_spikes)/2
    
    @add_slots    
    def consider_deletion(self):
        num_cluster_members = self.num_cluster_members
        num_clusters = self.num_clusters_alive
        if num_clusters<=self.num_special_clusters:
            self.log('info', 'Not enough clusters to try deletion')
            return
        sic = self.spikes_in_cluster
        sico = self.spikes_in_cluster_offset
        log_p_best = self.log_p_best
        log_p_second_best = self.log_p_second_best
        
        deletion_loss = zeros(num_clusters)
        I = arange(self.num_spikes)
        add.at(deletion_loss, self.clusters, log_p_second_best-log_p_best)
        candidate_cluster = self.num_special_clusters+argmin((deletion_loss-self.cluster_penalty)[self.num_special_clusters:])
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
        return score, raw, penalty

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
        I[0:self.num_special_clusters] = True # we keep special clusters
        remapping = hstack((0, cumsum(I)))[:-1]
        self.clusters = remapping[self.clusters]
        if hasattr(self, 'clusters_second_best'):
            del self.clusters_second_best
        self.partition_clusters()
        
    def partition_clusters(self):
        try:
            if self.num_special_clusters>0:
                self.num_cluster_members = num_cluster_members = array(bincount(self.clusters,
                                                                                minlength=self.num_special_clusters),
                                                                       dtype=int)
            else:
                self.num_cluster_members = num_cluster_members = array(bincount(self.clusters), dtype=int)
        except ValueError:
            raise PartitionError
        I = array(argsort(self.clusters), dtype=int)
        y = self.clusters[I]
        n = amax(y)
        if n<self.num_special_clusters-1:
            n = self.num_special_clusters-1
        n += 2
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
        # Use efficient version
        accumulate_cluster_mask_sum(self, cluster_mask_sum)
        cluster_mask_sum[:self.num_special_clusters, :] = -1 # ensure that special clusters are masked
        
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
        score, _, _ = self.compute_score()
        
        for cluster in xrange(self.first_gaussian_cluster, num_clusters):
            if num_clusters>=self.max_possible_clusters:
                self.log('info', 'No more splitting, already at maximum number of '
                                 'clusters' % self.max_possible_clusters)
                return did_split
            
            spikes_in_cluster = self.get_spikes_in_cluster(cluster)
            if len(spikes_in_cluster)==0:
                continue
            K2 = self.subset(spikes_in_cluster, name='split_candidate',
                             use_noise_cluster=False, use_mua_cluster=False)
            # at this point in C++ code we look for an unused cluster, but here we can just
            # use num_clusters+1
            self.log('debug', 'Trying to split cluster %d' % cluster)
            # initialise with current clusters, do not allow creation of new clusters
            K2.max_possible_clusters = 1
            clusters = full(len(spikes_in_cluster), 0, dtype=int)
            try:
                unsplit_score = K2.cluster_from(clusters, recurse=False)
            except PartitionError:
                # todo: logging
                self.log('error', 'Partitioning error on split, K2.clusters = %s' % K2.clusters)
                continue
            # initialise randomly, allow for one additional cluster
            K2.max_possible_clusters = 2
            clusters = randint(0, 2, size=len(spikes_in_cluster))
            if len(unique(clusters))!=2: # todo: better way of handling this?
                continue
            try:
                split_score = K2.cluster_from(clusters, recurse=False)
            except PartitionError:
                # todo: logging
                self.log('error', 'Partitioning error on split, K2.clusters = %s' % K2.clusters)
                continue
            
            if K2.num_clusters_alive==0: # todo: can this happen?
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
            I1 = (K2.clusters==1)
            clusters[spikes_in_cluster[I1]] = num_clusters # next available cluster
            K3.initialise_clusters(clusters)
            K3.M_step()
            K3.EC_steps() # todo: original code omits C step - a problem?
            K3.compute_cluster_penalties()
            new_score, _, _ = K3.compute_score()
            # todo: logging
            if new_score<score:
                did_split = True
                self.clusters = K3.clusters
                num_clusters += 1
            else:
                pass
            
        return did_split
