from numpy import *
from numpy.linalg import LinAlgError
from numpy.random import randint
from six import iteritems
from six.moves import range
import hashlib
from random import shuffle

from .logger import log_message

from .mask_starts import mask_starts
from .linear_algebra import BlockPlusDiagonalMatrix
from .default_parameters import default_parameters

from .numerics import (accumulate_cluster_mask_sum, compute_cluster_mean,
                       compute_covariance_matrix,
                       compute_log_p_and_assign, compute_penalties)

import time

__all__ = ['KK']

class PartitionError(Exception):
    pass


class section(object):
    def __init__(self, kk, name, *args, **kwds):
        self.kk = kk
        self.name = name
        self.args = args
        self.kwds = kwds
        if not hasattr(kk, '_section_timings_t_total'):
            kk._section_timings_t_total = {}
            kk._section_timings_num_calls = {}
        if name not in kk._section_timings_t_total:
            kk._section_timings_t_total[name] = 0.0
            kk._section_timings_num_calls[name] = 0
    def __enter__(self):
        self.kk.run_callbacks('start_'+self.name, *self.args, **self.kwds)
        self.t_start = time.time()
    def __exit__(self, type, value, traceback):
        this_time = time.time()-self.t_start
        self.kk.run_callbacks('end_'+self.name, *self.args, **self.kwds)
        if self.kk.name:
            self.kk.log('debug', 'This call: %.2f ms.' % (this_time*1000),
                        suffix='timing.'+self.name)
            return
        name = self.name
        st_t = self.kk._section_timings_t_total
        st_n = self.kk._section_timings_num_calls
        st_t[name] += this_time
        st_n[name] += 1
        mean_time = st_t[name]/st_n[name]
        self.kk.log('debug', 'This call: %.2f ms. Average: %.2f ms. Total: %.2f s. '
                             'Num calls: %d' % (this_time*1000, mean_time*1000,
                                                st_t[name], st_n[name]),
                    suffix='timing.'+self.name)


def add_slots(meth):
    def new_meth(self, *args, **kwds):
        with section(self, meth.__name__, *args, **kwds):
            res = meth(self, *args, **kwds)
        return res
    new_meth.__name__ = meth.__name__
    new_meth.__doc__ = meth.__doc__
    return new_meth


class KK(object):
    '''
    Main object used for clustering data

    This documentation is temporary, just to highlight the parts of the interface that should
    not change and can be used externally:

    * kk.special_clusters: a dict mapping special cluster names to their associated cluster index.
    * kk.num_special_clusters: the number of special clusters. This is also the cluster index of the
      first normal cluster.
    * Initialisation KK(data, use_noise_cluster=True, use_mua_cluster=True, **params)
    * Method kk.cluster_mask_starts(num_starting_clusters) will cluster from mask starts.
    * Method kk.cluster_from(clusters) will cluster from the given array of cluster assignments.
    * kk.clusters (after clustering) is the array of cluster assignments.
    * Method kk.register_callback(callback, slot=None) register a callback function that will be
      called at the given slot (see code for slot names), by default at the end of each iteration
      of the algorithm. Callback functions are normally called as f(kk), but some slots will
      defined additional arguments and keyword arguments.
    '''
    def __init__(self, data, callbacks=None, name='',
                 is_subset=False, is_copy=False,
                 map_log_to_debug=False,
                 **params):
        self.name = name
        if callbacks is None:
            callbacks = {}
        self.callbacks = callbacks
        self.data = data
        self.cluster_hashes = set()
        self.is_subset = is_subset
        self.is_copy = is_copy
        self.map_log_to_debug = map_log_to_debug
        # user parameters
        show_params = name=='' and not is_subset and not is_copy
        self.params = params
        actual_params = default_parameters.copy()
        for k, v in iteritems(params):
            if k not in default_parameters:
                raise ValueError("There is no parameter "+k)
            actual_params[k] = v
        for k, v in iteritems(actual_params):
            setattr(self, k, v)
            if show_params:
                self.log('info', '%s = %s' % (k, v), suffix='initial_parameters')
        use_noise_cluster = self.use_noise_cluster
        use_mua_cluster = self.use_mua_cluster
        self.all_params = actual_params
        # Assignment of special clusters
        self.num_special_clusters = 0
        self.first_gaussian_cluster = 0
        self.special_clusters = {}
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

    def run_callbacks(self, slot, *args, **kwds):
        if slot in self.callbacks:
            for callback in self.callbacks[slot]:
                callback(self, *args, **kwds)

    def log(self, level, msg, suffix=None):
        if self.map_log_to_debug:
            level = 'debug'
        if suffix is not None:
            if self.name=='':
                name = suffix
            else:
                name = self.name+'.'+suffix
        else:
            name = self.name
        log_message(level, msg, name=name)

    def copy(self, name='kk_copy',
             use_noise_cluster=None, use_mua_cluster=None,
             **additional_params):
        if self.name:
            sep = '.'
        else:
            sep = ''
        if use_noise_cluster is None:
            use_noise_cluster = self.use_noise_cluster
        if use_mua_cluster is None:
            use_mua_cluster = self.use_mua_cluster
        params = self.params.copy()
        params.update(**additional_params)
        if use_noise_cluster is not None:
            params['use_noise_cluster'] = use_noise_cluster
        if use_mua_cluster is not None:
            params['use_mua_cluster'] = use_mua_cluster
        return KK(self.data, name=self.name+sep+name,
                  callbacks=self.callbacks,
                  is_copy=True,
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
                  is_subset=True,
                  **params)

    def subset_features(self, feature_indices, name='', **additional_params):
        newdata, spikes = self.data.subset_features(feature_indices)
        if self.name:
            sep = '.'
        else:
            sep = ''
        params = self.params.copy()
        params.update(**additional_params)
        return KK(newdata, name=self.name+sep+name,
                  callbacks=self.callbacks,
                  is_subset=True,
                  **params), spikes

    def initialise_clusters(self, clusters):
        self.clusters = clusters
        self.old_clusters = -1*ones(len(self.clusters), dtype=int)
        self.reindex_clusters()

    def cluster_with_subset_schedule(self, num_starting_clusters, schedule):
        schedule = array(schedule)
        if schedule[-1]!=1:
            raise ValueError("Last value in subset schedule must be 1")
        if (diff(schedule)<=0).any():
            raise ValueError("Subset schedule must be strictly increasing")
        if (schedule<=0).any():
            raise ValueError("All subset schedule fractions must be between 0 and 1")
        spikes = arange(self.num_spikes)
        shuffle(spikes)
        I_end = unique(array(self.num_spikes*schedule, dtype=int))
        clusters = mask_starts(self.data, num_starting_clusters, self.num_special_clusters)
        clusters = clusters[spikes]
        clusters[I_end[0]:] = 0 # assign to noise, they will quickly get reassigned
        for i, i_end in enumerate(I_end):
            self.log('info', 'Subsetting stage %d of %d, %d points of %d' % (i+1, len(I_end), i_end,
                                                                             self.num_spikes))
            if i==0:
                split_first = self.split_first
            else:
                split_first = 0
            if i_end<len(spikes):
                kk_sub = self.subset(spikes[:i_end], name='', split_first=split_first,
                                     break_fraction=self.subset_break_fraction,
                                     use_noise_cluster=self.use_noise_cluster,
                                     use_mua_cluster=self.use_mua_cluster,
                                     )
            else:
                kk_sub = self.copy(name='', split_first=split_first)
                clusters_new = zeros(self.num_spikes, dtype=int)
                clusters_new[spikes] = clusters
                clusters = clusters_new
            kk_sub.cluster_from(clusters[:i_end].copy())
            clusters[:i_end] = kk_sub.clusters
        self.clusters = kk_sub.clusters
        self.reindex_clusters()

    def cluster_mask_starts(self):
        clusters = mask_starts(self.data, self.num_starting_clusters, self.num_special_clusters)
        self.cluster_from(clusters)

    def cluster_from(self, clusters, recurse=True, score_target=-inf):
        self.log('info', 'Clustering data set of %d points, %d features' % (self.data.num_spikes,
                                                                            self.data.num_features))
        self.initialise_clusters(clusters)
        return self.iterate(recurse=recurse, score_target=score_target)
    
    def prepare_for_iterate(self):
        self.full_step = True
        self.current_iteration = 0
        self.force_next_step_full = False
        self.score_history = []
    
    def iterate(self, recurse=True, score_target=-inf):        
        self.prepare_for_iterate()

        score = score_raw = score_penalty = None

        iterations_until_next_split = self.split_first
        tried_splitting_to_escape_cycle_hashes = set()

        self.log('info', 'Starting iteration 0 with %d clusters' % self.num_clusters_alive)

        while self.current_iteration<self.max_iterations:
            if self.force_next_step_full:
                self.force_next_step_full = False
                self.full_step = True
                self.log('debug', 'Next step forced to be full')
            self.log('debug', 'Starting iteration %d' % self.current_iteration)
            if self.full_step:
                self.log('debug', 'This step is a full step')
            else:
                self.log('debug', 'This step is a quick step')
            self.MEC_steps()
            self.compute_cluster_penalties()
            # only delete after a full step to simplify quick steps
            if recurse and self.full_step and self.consider_cluster_deletion:
                self.consider_deletion()
            old_score = score
            old_score_raw = score_raw
            old_score_penalty = score_penalty
            score, score_raw, score_penalty = self.compute_score()
            self.score_history.append((score, score_raw, score_penalty))
            
            clusters_changed, = (self.clusters!=self.old_clusters).nonzero()
            clusters_changed = array(clusters_changed, dtype=int)
            num_changed = len(clusters_changed)
            if num_changed and not self.full_step and self.full_step_every>1:
                # add these changed clusters to all the candidate sets
                num_candidates = 0
                max_quick_step_candidates = min(self.max_quick_step_candidates,
                    self.max_quick_step_candidates_fraction*self.num_spikes*self.num_clusters_alive)
                with section(self, 'quick_step_union'):
                    for cluster, candidates in list(self.quick_step_candidates.items()):
                        candidates = union1d(candidates, clusters_changed)
                        self.quick_step_candidates[cluster] = candidates
                        num_candidates += len(candidates)
                        if num_candidates>max_quick_step_candidates:
                            self.quick_step_candidates = dict()
                            self.force_next_step_full = True
                            if num_candidates>self.max_quick_step_candidates:
                                self.log('info', 'Ran out of storage space for quick step, try increasing '
                                                 'max_quick_step_candidates if this happens often.')
                            else:
                                self.log('debug', 'Exceeded quick step point fraction, next step '
                                                  'will be full')
                            break

            self.run_callbacks('scores', score=score, score_raw=score_raw,
                               score_penalty=score_penalty, old_score=old_score,
                               old_score_raw=old_score_raw, old_score_penalty=old_score_penalty,
                               num_changed=num_changed,
                               )

            self.current_iteration += 1

            QF_id = {True:'F', False:'Q'}[self.full_step]
            msg = 'Iteration %d%s: %d clusters, %d changed, score=%f' % (self.current_iteration, QF_id,
                                                                         self.num_clusters_alive,
                                                                         num_changed, score)

            last_step_full = self.full_step
            self.full_step = (num_changed>self.num_changed_threshold*self.num_spikes or
                              num_changed==0 or
                              self.current_iteration % self.full_step_every == 0 or
                              (old_score is not None and score > old_score))
            if not hasattr(self, 'old_log_p_best'):
                self.full_step = True

            self.reindex_clusters()

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

            if num_changed<self.break_fraction*self.num_spikes and last_step_full:
                self.log('info', 'Number of points changed below break fraction, so finishing.')
                break

            if score<score_target:
                self.log('info', 'Reached score target, so finishing.')
        else:
            # ran out of iterations
            self.log('info', 'Number of iterations exceeded maximum %d' % self.max_iterations)

        return score
    
    @add_slots
    def MEC_steps(self, only_evaluate_current_clusters=False):
        # eliminate any clusters with 0 members, compute the list of spikes
        # in each cluster, compute the cluster masks and allocate space for
        # covariance matrices
        self.reindex_clusters()
        # Computes the masked and unmasked indices for each cluster based on the
        # masks for each point in that cluster. Allocates space for covariance
        # matrices.
        num_clusters = self.num_clusters_alive
        num_features = self.num_features
        num_cluster_members = self.num_cluster_members
        cluster_start = self.num_special_clusters
        num_spikes = self.num_spikes

        # Weight computations
        denom = self.num_spikes+self.prior_point*(num_clusters-self.num_special_clusters)
        if self.use_noise_cluster:
            denom += self.noise_point
        if self.use_mua_cluster:
            denom += self.mua_point
        denom = float(denom)
        if self.use_noise_cluster:
            noise_weight = (num_cluster_members[self.noise_cluster]+self.noise_point)/denom
        
        # Arrays that will be used in E-step part
        if only_evaluate_current_clusters:
            self.clusters_second_best = zeros(0, dtype=int)
            self.log_p_best = empty(num_spikes)
            self.log_p_second_best = empty(0)
        elif self.full_step:
            self.old_clusters = self.clusters
            self.clusters = -ones(num_spikes, dtype=int)
            self.clusters_second_best = -ones(num_spikes, dtype=int)
            if hasattr(self, 'log_p_best'):
                self.old_log_p_best = self.log_p_best
            self.log_p_best = inf*ones(num_spikes)
            self.log_p_second_best = inf*ones(num_spikes)
        else:
            self.old_clusters = self.clusters.copy()

        num_skipped = 0

        if self.full_step and self.use_noise_cluster and not only_evaluate_current_clusters:
            # start with cluster 0 - uniform distribution over space
            # because we have normalized all dims to 0...1, density will be 1.
            self.clusters[:] = self.noise_cluster
            self.log_p_best[:] = -log(noise_weight)

        if only_evaluate_current_clusters:
            self.quick_step_candidates = dict()
            for cluster in range(num_clusters):
                self.quick_step_candidates[cluster] = self.get_spikes_in_cluster(cluster)
            self.collect_candidates = False
        elif self.full_step:
            self.quick_step_candidates = dict()
            self.collect_candidates = True
        else:
            self.collect_candidates = False
        
        clusters_to_kill = []
        
        for cluster in range(num_clusters):
            # Compute the sum of fmasks
            if cluster<self.num_special_clusters:
                cluster_mask_sum = full(num_features, -1.0) # ensure that special clusters are masked
            else:
                cluster_mask_sum = zeros(num_features)
                accumulate_cluster_mask_sum(self, cluster_mask_sum, self.get_spikes_in_cluster(cluster))

            # Compute the masked and unmasked sets
            unmasked, = (cluster_mask_sum>=self.points_for_cluster_mask).nonzero()
            masked, = (cluster_mask_sum<self.points_for_cluster_mask).nonzero()
            unmasked = array(unmasked, dtype=int)
            masked = array(masked, dtype=int)
            cov = BlockPlusDiagonalMatrix(masked, unmasked)

            ########### M step ########################################################
        
            # Normalize by total number of points to give class weight
            if cluster==self.noise_cluster:
                weight = noise_weight
            elif cluster==self.mua_cluster:
                weight = (num_cluster_members[self.mua_cluster]+self.mua_point)/denom
            else:
                weight = (num_cluster_members[cluster]+self.prior_point)/denom
        
            # Compute means for each cluster
            # Note that we do this densely at the moment, might want to switch
            # that to a sparse structure later
            cluster_mean = compute_cluster_mean(self, cluster)        
            # Compute covariance matrices
            compute_covariance_matrix(self, cluster, cluster_mean, cov)
            
            ########### EC steps ######################################################
            
            if cluster<self.num_special_clusters:
                continue   
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
            for i in range(num_features):
                basis_vector[i] = 1.0
                root = chol.trisolve(basis_vector)
                inv_cov_diag[i] = sum(root**2)
                basis_vector[i] = 0.0

            self.run_callbacks('e_step_before_main_loop', cholesky=chol, cluster=cluster,
                               inv_cov_diag=inv_cov_diag)
                
            compute_log_p_and_assign(self, cluster, weight, inv_cov_diag, log_root_det, chol,
                                     cluster_mean, only_evaluate_current_clusters, self.num_cpus)
            
            self.run_callbacks('e_step_after_main_loop')

        # we've reassigned clusters so we need to recompute the partitions, but we don't want to
        # reindex yet because we may reassign points to different clusters and we need the original
        # cluster numbers for that
        self.partition_clusters()

    @add_slots
    def compute_cluster_penalties(self, clusters=None):
        cluster_penalties = compute_penalties(self, clusters)
        if clusters is None:
            self.cluster_penalty = cluster_penalties
        return cluster_penalties

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

        score, score_raw, score_penalty = self.compute_score()
        candidate_cluster = -1
        improvement = -inf
        for cluster in range(self.num_special_clusters, num_clusters):
            new_clusters = self.clusters.copy()
            # reassign points
            cursic = sic[sico[cluster]:sico[cluster+1]]
            new_clusters[cursic] = self.clusters_second_best[cursic]
            # compute penalties if we reassigned this
            penalties = self.compute_cluster_penalties(new_clusters)
            new_score = score_raw+deletion_loss[cluster]+sum(penalties)
            cur_improvement = score-new_score # we want improvement to be a positive value
            if cur_improvement>improvement:
                improvement = cur_improvement
                candidate_cluster = cluster

        if improvement>0:
            # delete this cluster
            num_points_in_candidate = sico[candidate_cluster+1]-sico[candidate_cluster]
            self.log('info', 'Deleting cluster {cluster} ({numpoints} points): improves score '
                             'by {improvement}'.format(cluster=candidate_cluster,
                                                       numpoints=num_points_in_candidate,
                                                       improvement=improvement))
            # reassign points
            cursic = sic[sico[candidate_cluster]:sico[candidate_cluster+1]]
            self.clusters[cursic] = self.clusters_second_best[cursic]
            self.log_p_best[cursic] = self.log_p_second_best[cursic]
            # at this point we have invalidated the partitions, so to make sure we don't miss
            # something, we wipe them out here
            self.partition_clusters()
            self.compute_cluster_penalties() # and recompute the penalties
            # we've also invalidated the second best log_p and clusters
            self.log_p_second_best = None
            self.clusters_second_best = None
            # and we will need to do a full step next time
            self.force_next_step_full = True

    @add_slots
    def compute_score(self):
        penalty = sum(self.cluster_penalty)
        raw = sum(self.log_p_best)
        score = raw+penalty
        self.log('debug', 'compute_score: raw %f + penalty %f = %f' % (raw, penalty, score))
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

    @add_slots
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
        total_clusters = sum(I)
        if hasattr(self, '_total_clusters') and total_clusters<self._total_clusters:
            self.force_next_step_full = True
            if hasattr(self, 'clusters_second_best'):
                del self.clusters_second_best
        self._total_clusters = total_clusters
        self.partition_clusters()

    def partition_clusters(self, clusters=None):
        if clusters is None:
            clusters = self.clusters
            assign_to_self = True
        else:
            assign_to_self = False
        try:
            if self.num_special_clusters>0:
                num_cluster_members = array(bincount(clusters, minlength=self.num_special_clusters), dtype=int)
            else:
                num_cluster_members = array(bincount(clusters), dtype=int)
        except ValueError:
            raise PartitionError
        I = array(argsort(clusters), dtype=int)
        y = clusters[I]
        n = amax(y)
        if n<self.num_special_clusters-1:
            n = self.num_special_clusters-1
        n += 2
        J = searchsorted(y, arange(n))
        if assign_to_self:
            self.num_cluster_members = num_cluster_members
            self.spikes_in_cluster = I
            self.spikes_in_cluster_offset = J
        else:
            return I, J, num_cluster_members

    def invalidate_partitions(self):
        self.num_cluster_members = None
        self.spikes_in_cluster = None
        self.spikes_in_cluster_offset = None

    def get_spikes_in_cluster(self, cluster):
        sic = self.spikes_in_cluster
        sico = self.spikes_in_cluster_offset
        return sic[sico[cluster]:sico[cluster+1]]
        
    @add_slots
    def try_splits(self):
        did_split = False
        num_clusters = self.num_clusters_alive

        self.log('info', 'Trying to split clusters')

        score_ref = None

        self.reindex_clusters()

        for cluster in range(self.num_special_clusters, num_clusters):
            if num_clusters>=self.max_possible_clusters:
                self.log('info', 'No more splitting, already at maximum number of '
                                 'clusters: %d' % self.max_possible_clusters)
                return did_split

            spikes_in_cluster = self.get_spikes_in_cluster(cluster)
            if len(spikes_in_cluster)==0:
                continue

            with section(self, 'split_candidate'):
                if self.max_split_iterations is not None:
                    max_iter = self.max_split_iterations
                else:
                    max_iter = self.max_iterations

                K2 = self.subset(spikes_in_cluster, name='split_candidate',
                                 use_noise_cluster=False, use_mua_cluster=False,
                                 max_iterations=max_iter,
                                 map_log_to_debug=True,
                                 )
                # at this point in C++ code we look for an unused cluster, but here we can just
                # use num_clusters+1
                self.log('debug', 'Trying to split cluster %d containing '
                                  '%d points' % (cluster, len(spikes_in_cluster)))
                # initialise with current clusters, do not allow creation of new clusters
                K2.max_possible_clusters = 1
                clusters = full(len(spikes_in_cluster), 0, dtype=int)
                try:
                    unsplit_score = K2.cluster_from(clusters, recurse=False)
                except PartitionError:
                    self.log('error', 'Partitioning error on split, K2.clusters = %s' % K2.clusters)
                    continue
                self.run_callbacks('split_k2_1', cluster=cluster, K2=K2,
                                   unsplit_score=unsplit_score)
                # initialise randomly, allow for one additional cluster
                K2.max_possible_clusters = 2
                clusters = randint(0, 2, size=len(spikes_in_cluster))
                if amax(clusters)!=1:
                    continue

                if self.fast_split:
                    score_target = unsplit_score
                else:
                    score_target = -inf

                try:
                    split_score = K2.cluster_from(clusters, recurse=False,
                                                  score_target=score_target)
                except PartitionError:
                    self.log('error', 'Partitioning error on split, K2.clusters = %s' % K2.clusters)
                    continue
                self.run_callbacks('split_k2_2', cluster=cluster, K2=K2, split_score=split_score,
                                   unsplit_score=unsplit_score)

                if K2.num_clusters_alive==0:
                    self.log('error', 'No clusters alive in K2')
                    continue

                if split_score>=unsplit_score:
                    self.log('debug', 'Score after (%f) splitting worse than before (%f), '
                                      'so not splitting' % (split_score, unsplit_score))
                    continue

            if self.always_split_bimodal:
                self.log('debug', 'Always splitting bimodal clusters, so splitting cluster '
                                  '%d into %d' % (cluster, num_clusters))
                clusters = self.clusters.copy()
                I1 = (K2.clusters==1)
                clusters[spikes_in_cluster[I1]] = num_clusters # next available cluster
                did_split = True
                self.clusters = clusters
                self.reindex_clusters()
                num_clusters = self.num_clusters_alive
                continue

            with section(self, 'split_evaluation'):
                # will splitting improve the score in the whole data set?
                K3 = self.copy(name='split_evaluation', map_log_to_debug=True)
                clusters = self.clusters.copy()

                if score_ref is None:
                    K3.initialise_clusters(clusters)
                    K3.prepare_for_iterate()
                    K3.MEC_steps(only_evaluate_current_clusters=True)
                    K3.compute_cluster_penalties()
                    score_ref, _, _ = K3.compute_score()

                I1 = (K2.clusters==1)
                clusters[spikes_in_cluster[I1]] = num_clusters # next available cluster

                K3.initialise_clusters(clusters)
                K3.prepare_for_iterate()
                K3.MEC_steps(only_evaluate_current_clusters=True)
                K3.compute_cluster_penalties()
                score_new, _, _ = K3.compute_score()

            if score_new<score_ref:
                self.log('debug', 'Score improved after splitting, so splitting cluster '
                                  '%d into %d' % (cluster, num_clusters))
                did_split = True
                self.clusters = K3.clusters.copy()
                self.reindex_clusters()
                num_clusters = self.num_clusters_alive
                score_ref = score_new
            else:
                self.log('debug', 'Score got worse after splitting')

        # if we split, should make the next step full
        if did_split:
            self.force_next_step_full = True
            self.log('info', 'Split into %d clusters' % num_clusters)

        return did_split
