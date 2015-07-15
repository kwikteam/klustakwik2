'''
Default clustering parameters
'''

from numpy import log

default_parameters = dict(
     prior_point=1,
     mua_point=2,
     noise_point=1,
     points_for_cluster_mask=100,
     penalty_k=0.0,
     penalty_k_log_n=1.0,
     max_iterations=1000,
     num_starting_clusters=500,
     use_noise_cluster=True,
     use_mua_cluster=True,
     num_changed_threshold=0.05,
     full_step_every=1,
     split_first=20,
     split_every=40,
     max_possible_clusters=1000,
     dist_thresh=log(10000.0),
     max_quick_step_candidates=100000000, # this uses around 760 MB RAM
     max_quick_step_candidates_fraction=0.4,
     always_split_bimodal=False,
     subset_break_fraction=0.01,
     break_fraction=0.0,
     fast_split=False,
     max_split_iterations=None,
     consider_cluster_deletion=True,
     num_cpus=None,
     )
