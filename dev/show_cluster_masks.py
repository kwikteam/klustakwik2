from pylab import *
from klustakwik2 import *
import time

if __name__=='__main__':
    c = MonitoringClient()
    ion()
    while True:
        c.pause()
        cluster_unmasked_features = c.evaluate('kk.cluster_unmasked_features')
        num_features = c.evaluate('kk.num_features')
        num_clusters = len(cluster_unmasked_features)
        c.go()
     
        maskimg = []
        for cluster, unmasked in enumerate(cluster_unmasked_features):
            row = zeros(num_features)
            row[unmasked] = 1
            maskimg.append(row)
        maskimg = array(maskimg)
        imshow(maskimg, origin='lower left', aspect='auto', interpolation='nearest')
        gray()
        show()
        time.sleep(1)
