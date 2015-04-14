from numpy import *
from numpy.linalg import LinAlgError
from itertools import izip

from .mask_starts import mask_starts
from .linear_algebra import BlockPlusDiagonalMatrix

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

default_parameters = dict(
     prior_point=1,
     mua_point=1,
     noise_point=1,
     points_for_cluster_mask=10,
     dist_thresh=log(1000),
     penalty_k=0.0,
     penalty_k_log_n=1.0,
     max_iterations=10000,
     num_changed_threshold=0.05,
     full_step_every=20,
     split_first=20,
     split_every=40,
     )                          

class KK(object):
    def __init__(self, data, **params):        
        self.data = data
        self.params = params
        actual_params = default_parameters.copy()
        actual_params.update(**params)
        for k, v in actual_params.iteritems():
            setattr(self, k, v)
        
    def subset(self, spikes):
        newdata = self.data.subset(spikes)
        return KK(newdata, **self.params)
    
    def cluster(self, num_starting_clusters):
        self.clusters = mask_starts(self.data, num_starting_clusters)
        self.old_clusters = -1*ones(len(self.clusters), dtype=int)
        self.reindex_clusters()
        self.CEM()
    
    def CEM(self, recurse=True):
        
        score = old_score = 0.0
        
        self.full_step = True
        self.current_iteration = 0
        
        while self.current_iteration==0: # for debugging
        #while self.current_iteration<=self.max_iterations:
            self.M_step()
            self.EC_steps()
            self.compute_cluster_penalties()
            if recurse:
                self.consider_deletion()
            old_score, score = score, self.compute_score()
            
            num_changed = sum(self.clusters!=self.old_clusters)
            
            self.current_iteration += 1
    
            last_step_full = self.full_step
            # TODO: add this back in when we have num_changed, etc.
            self.full_step = (num_changed>self.num_changed_threshold*self.num_spikes or
                              num_changed==0 or
                              self.current_iteration % self.full_step_every == 0 or
                              score > old_score) 
    
            # TODO: save current progress

            # Try splitting
            did_split = False
            if recurse and self.split_every>0:
                if (self.current_iteration==self.split_first or
                    (self.current_iteration>self.split_first or
                     self.current_iteration-self.split_first%self.split_every==self.split_every-1 or
                     (num_changed==0 and last_step_full))):
                    
                    did_split = self.try_splits()
                    
            if num_changed==0 and last_step_full and not did_split:
                break 
    
    def M_step(self):
        # eliminate any clusters with 0 members, compute the list of spikes
        # in each cluster, compute the cluster masks and allocate space for
        # covariance matrices
        self.reindex_clusters()
        self.compute_cluster_masks()

        num_cluster_members = self.num_cluster_members
        num_clusters = len(self.num_cluster_members)
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
        compute_cluster_means(self)
        start = time.time()
        self.cluster_mean = compute_cluster_means(self)
        print time.time()-start
        
        # Compute covariance matrices
        for cluster in xrange(2, num_clusters):
            compute_covariance_matrix(self, cluster)
            cov = self.covariance[cluster]
            block_diagonal = get_diagonal(cov.block)
            
            # Add prior
            block_diagonal[:] += self.prior_point*self.data.noise_variance[cov.unmasked]
            cov.diagonal[:] += self.prior_point*self.data.noise_variance[cov.masked]
            
            # Normalise
            factor = 1.0/(num_cluster_members[cluster]+self.prior_point-1)
            cov.block *= factor
            cov.diagonal *= factor
                    
    def EC_steps(self, allow_assign_to_noise=True):
        if not allow_assign_to_noise:
            cluster_start = 2
        else:
            cluster_start = 1

        num_spikes = self.num_spikes
        num_clusters = len(self.num_cluster_members)
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
                clusters_to_kill.append(cluster)
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
    
    def compute_cluster_penalties(self):
        num_cluster_members = self.num_cluster_members
        num_clusters = len(self.num_cluster_members)
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
    
    def consider_deletion(self):
        num_cluster_members = self.num_cluster_members
        num_clusters = len(self.num_cluster_members)
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
            
    def compute_score(self):
        score = sum(self.cluster_penalty)
        score += sum(self.log_p_best)
        return score

    @property
    def num_spikes(self):
        return self.data.num_spikes
    
    @property
    def num_features(self):
        return self.data.num_features
    
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
        
    def compute_cluster_masks(self):
        '''
        Computes the masked and unmasked indices for each cluster based on the
        masks for each point in that cluster. Allocates space for covariance
        matrices.
        '''
        num_clusters = len(self.num_cluster_members)
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
            
    def try_splits(self):
        did_split = False
        num_clusters = len(self.num_cluster_members)
        
        if num_clusters>=self.max_possible_clusters-2:
            print "Won't try splitting - already at maximum number of clusters"
            return False
        
#     if (!AlwaysSplitBimodal)
#     {
#         if (KK_split == NULL)
#         {
#             KK_split = new KK(*this);
#         }
#         else
#         {
#             // We have to clear these to bypass the debugging checks
#             // in precomputations.cpp
#             KK_split->Unmasked.clear();
#             KK_split->UnmaskedInd.clear();
#             KK_split->SortedMaskChange.clear();
#             KK_split->SortedIndices.clear();
#             // now we treat it as empty
#             KK_split->ConstructFrom(*this);
#         }
#     }
#     //KK &K3 = *KK_split;
# #define K3 (*KK_split)

        score = self.compute_score()
        
        for cluster in xrange(2, num_clusters):
            pass
# 
#         // set up K2 structure to contain points of this cluster only
# 
#         vector<integer> SubsetIndices;
#         for(p=0; p<nPoints; p++)
#             if(Class[p]==c)
#                 SubsetIndices.push_back(p);
#         if(SubsetIndices.size()==0)
#             continue;
# 
#         if (K2_container)
#         {
#             // We have to clear these to bypass the debugging checks
#             // in precomputations.cpp
#             K2_container->Unmasked.clear();
#             K2_container->UnmaskedInd.clear();
#             K2_container->SortedMaskChange.clear();
#             K2_container->SortedIndices.clear();
#             //K2_container->AllVector2Mean.clear();
#             // now we treat it as empty
#             K2_container->ConstructFrom(*this, SubsetIndices);
#         }
#         else
#         {
#             K2_container = new KK(*this, SubsetIndices);
#         }
#         //KK K2(*this, SubsetIndices);
#         KK &K2 = *K2_container;
# 
#         // find an unused cluster
#         UnusedCluster = -1;
#         for(c2=2; c2<MaxPossibleClusters; c2++)
#         {
#              if (!ClassAlive[c2])
#              {
#                  UnusedCluster = c2;
#                  break;
#              }
#         }
#         if (UnusedCluster==-1)
#         {
#             Output("No free clusters, abandoning split");
#             return DidSplit;
#         }
# 
#         // do it
#         if (Verbose >= 1) Output("\n Trying to split cluster %d (%d points) \n", (int)c, (int)K2.nPoints);
#         K2.nStartingClusters=3; // (3 = 1 clusters + 2 unused noise/MUA cluster)
#         UnsplitScore = K2.CEM(NULL, 0, 1, false);
#         K2.nStartingClusters=4; // (4 = 2 clusters + 2 unused noise/MUA cluster)
#         SplitScore = K2.CEM(NULL, 0, 1, false);
# 
#         // Fix by Michael Zugaro: replace next line with following two lines
#         // if(SplitScore<UnsplitScore) {
#         if(K2.nClustersAlive<3) Output("\n Split failed - leaving alone\n");
#         if((SplitScore<UnsplitScore)&&(K2.nClustersAlive>=3)) {
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
#             else
#             {
#                 // will splitting improve the score in the whole data set?
# 
#                 // assign clusters to K3
#                 for (c2 = 0; c2 < MaxPossibleClusters; c2++) K3.ClassAlive[c2] = 0;
#                 //   Output("%d Points in class %d in KKobject K3 ", (int)c2, (int)K3.nClassMembers[c2]);
#                 p2 = 0;
#                 for (p = 0; p < nPoints; p++)
#                 {
#                     if (Class[p] == c)
#                     {
#                         if (K2.Class[p2] == 2) K3.Class[p] = c;
#                         else if (K2.Class[p2] == 3) K3.Class[p] = UnusedCluster;
#                         else Error("split should only produce 2 clusters\n");
#                         p2++;
#                     }
#                     else K3.Class[p] = Class[p];
#                     K3.ClassAlive[K3.Class[p]] = 1;
#                 }
#                 K3.Reindex();
# 
#                 // compute scores
# 
#                 K3.MStep();
#                 K3.EStep();
#                 //Output("About to compute K3 class penalties");
#                 if (UseDistributional) K3.ComputeClassPenalties(); //SNK Fixed bug: Need to compute the cluster penalty properly, cluster penalty is only used in UseDistributional mode
#                 NewScore = K3.ComputeScore();
#                 Output("\nSplitting cluster %d changes total score from " SCALARFMT " to " SCALARFMT "\n", (int)c, Score, NewScore);
# 
#                 if (NewScore < Score)
#                 {
#                     DidSplit = 1;
#                     Output("\n So it's getting split into cluster %d.\n", (int)UnusedCluster);
#                     // so put clusters from K3 back into main KK struct (K1)
#                     for (c2 = 0; c2 < MaxPossibleClusters; c2++) ClassAlive[c2] = K3.ClassAlive[c2];
#                     for (p = 0; p < nPoints; p++) Class[p] = K3.Class[p];
#                 }
#                 else
#                 {
#                     Output("\n So it's not getting split.\n");
#                 }
#             }
#         }
#     }
#     return DidSplit;
# #undef K3
