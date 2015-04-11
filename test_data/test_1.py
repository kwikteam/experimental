import operator
from pprint import pprint
from timeit import default_timer

import numpy as np
import matplotlib.pyplot as plt

from vispy.app import run

import phy
from phy.cluster.manual.session import Session

phy.debug()

# path = '/home/cyrille/spikesorting/1_simple120sec/test_hybrid_120sec.kwik'  # HDD
# path = '/home/cyrille/1_simple120sec/test_hybrid_120sec.kwik'  # SSD
path = '/data/spikesorting/3_nick_full/20141202_all.kwik'

session = Session()

# t0 = default_timer()
session.open(path)


# spike_clusters = session.model.spike_clusters[::10]
# spike_samples = session.model.spike_samples[::10]

# from phy.stats.ccg import correlograms
# # Compute the correlograms.
# ccgs = correlograms(spike_samples,
#                     spike_clusters,
#                     binsize=20,
#                     winsize_bins=51,
#                     )
# print(spike_samples.dtype)
# print(spike_clusters.dtype)
# exit()


# print(session.model.recordings)
# print(session.model.spike_recordings)
# print(session.model._recording_offsets)
# plt.plot(session.model.spike_samples)
# plt.show()
# exit()

# print(default_timer() - t0)


# Cluster counts.
# session.cluster_store.disk_store.clear()
x = session.clustering.cluster_counts
print(session.model.recordings)
cl = (sorted(x.items(), key=operator.itemgetter(1))[::-1][:50])
clu = [_[0] for _ in cl][1:5]

# print(clu)
session.select(clu)
# print(session.model.n_spikes)
# print(session.model.n_recordings)
# print(session.clustering.n_clusters)
# session.select([3, 4, 5, 6])

session.show_correlograms()
# session.show_features()
# session.show_waveforms()

run()

session.close()
