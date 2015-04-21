'''
Default clustering parameters
'''

default_parameters = dict(
     prior_point=1,
     mua_point=2,
     noise_point=1,
     points_for_cluster_mask=10,
     penalty_k=0.0,
     penalty_k_log_n=1.0,
     max_iterations=1000,
     num_changed_threshold=0.05,
     full_step_every=20,
     split_first=20,
     split_every=40,
     max_possible_clusters=1000,
     mask_starts=500,
     )
