from klustakwik2 import *
from pylab import *
import time
import cPickle as pickle
import os

fname, shank = '../temp/testsmallish', 4

# if os.path.exists(fname+'.pickle'):
#     start = time.time()
#     data = pickle.load(open(fname+'.pickle', 'rb'))
#     print 'Loading time (from pickle):', time.time()-start
# else:
start = time.time()
data = load_fet_fmask(fname, shank)
print 'Loading time (from text):', time.time()-start
pickle.dump(data, open(fname+'.pickle', 'wb'), -1)

start = time.time()
data.do_initial_precomputations()
print 'Initial precomputation time:', time.time()-start

kk = KK(data)

kk.cluster(num_starting_clusters=100)
