from klustakwik2 import *
from pylab import *
import time
import cPickle as pickle
import os

fname, shank = '../temp/testsmallish', 4

log_to_file(fname+'.klg', 'debug')

if os.path.exists(fname+'.pickle'):
    start_time = time.time()
    data = pickle.load(open(fname+'.pickle', 'rb'))
    print 'load from pickle:', time.time()-start_time
else:
    start_time = time.time()
    raw_data = loat_fet_fmask_to_raw(fname, shank)
    print 'load_fet_fmask_to_raw:', time.time()-start_time
    start_time = time.time()
    data = raw_data.to_sparse_data()
    print 'raw_data.to_sparse_data():', time.time()-start_time
    pickle.dump(data, open(fname+'.pickle', 'wb'), -1)
    
print 'Number of spikes:', data.num_spikes
print 'Number of unique masks:', data.num_masks

kk = KK(data)

kk.cluster(100)
