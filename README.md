# klustakwik2

## Installation instructions

Install Python using the [Anaconda distribution](http://continuum.io/downloads).

Install KlustaKwik using ``pip install klustakwik2``.

On Windows, you will need a copy of MS Visual Studio Express 2008 for Python, available for free
download [here](http://stackoverflow.com/questions/26140192/microsoft-visual-c-compiler-for-python-2-7).

## Usage

To cluster a pair of files ``name.fet.n``, ``name.fmask.n`` run the command:

    kk2_legacy name n
    
This will generate a ``name.klg.n`` and ``name.clu.n`` file.

You can specify additional options to this script. The major ones are explained below:

* **max_iterations=1000**: the maximum number of full iterations to carry out before giving up.
* **max_possible_clusters=1000**: the maximum number of clusters to output at the end. At the
  moment, this option limits RAM usage but in a future version there will be no such limit.
* **drop_last_n_features=0**: don't use the last N features of the fet file for clustering.
* **save_clu_every=None**: how often (in minutes) to save a temporary .clu file.
* **num_starting_clusters=500**: how many initial clusters to use (should probably be higher than
  the number of final clusters you expect to find).
* **start_from_clu=None**: if specified, start the clustering from a previously saved .clu file.
* **use_mua_cluster=True**: whether or not to use "black hole" MUA cluster variant of the
  algorithm. This is designed to improve clustering on noisier data sets, but you can turn it off
  if you are having problems with it. Note that if this is on, cluster 2 will be the MUA cluster,
  and the first 'normal' cluster will be cluster 3.
