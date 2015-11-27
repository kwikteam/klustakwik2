klustakwik2
-----------

.. image:: https://travis-ci.org/kwikteam/klustakwik2.svg?branch=master
    :target: https://travis-ci.org/kwikteam/klustakwik2

Installation instructions
=========================

Install Python using the `Anaconda distribution <http://continuum.io/downloads>`_. You will
need to install the packages numpy, scipy, cython and nose. For Windows, Python 2.7 might be a better option than
3.x.

On all platforms, KlustaKwik can be installed using ``pip install klustakwik2`` (from source) or ``conda install -c kwikteam klustakwik2`` (precompiled binary). The default installation options
are as follows:

* **Linux**: Multithreading on by default.
* **Windows**: Multithreading on by default if MSVC installed, otherwise off. Note that under the Anaconda distribution,
  multithreading will be off by default, but this can be easily resolved if you have MSVC installed, see below.
* **Mac**: Multithreading off by default.

To override these options, install from source (see below).

Multithreading with Anaconda on Windows
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The Anaconda distribution installs its own compiler that doesn't support OpenMP for multithreading and uses this by
default instead of the MSVC compiler, which does support OpenMP. To disable the Anaconda compiler, simply run the
following command before installing KlustaKwik.

    conda remove libpython

Installing from source
~~~~~~~~~~~~~~~~~~~~~~

Download the source, either from one of the source distributions `on PyPI <https://pypi.python.org/pypi/klustakwik2>`_
or get the latest version `from GitHub <https://github.com/kwikteam/klustakwik2>`_. Run one of the following commands
from a command prompt in this directory. For default options:

    python setup.py install

To force multithreading to be on:

    python setup.py install --with-openmp

To force multithreading to be off:

    python setup.py install --no-openmp

Windows
~~~~~~~

Install a precompiled Windows binary with Anaconda in an anaconda package:
    conda install -c kwikteam klustakwik2

If you wish to compile from source, the instructions are a bit more complicated:

Using Python 2.7, you will need a copy of MS Visual Studio Express 2008 for Python, available for free
download `here <http://www.microsoft.com/en-us/download/details.aspx?id=44266>`_. Python 3.x might require a different
version of Visual Studio, we haven't tested this.

Download the source as above. Open a command prompt in the
directory where you downloaded and extracted the files. If you installed Python for all users, then you will need
admin rights on this command prompt. To get this in Windows, press the Windows key, type "cmd", right click on
"cmd.exe" and click "Run as administrator".

Now run the commands as in the section on installing from source above.

Mac
~~~

It is possible to install a version of ``gcc`` that allows for multithreading. TODO: details.

Usage
=====

To cluster a pair of files ``name.fet.n``, ``name.fmask.n`` run the command:

    kk2_legacy name n
    
This will generate a ``name.klg.n`` and ``name.clu.n`` file. Note that the first time you run it,
it will generate a whole lot of warnings and compiler output: ignore this, it is normal.

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
