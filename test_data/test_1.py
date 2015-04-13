import operator
from pprint import pprint
from timeit import default_timer

import numpy as np
import matplotlib.pyplot as plt

from vispy.app import run

import phy
from phy.cluster.manual.session import Session

phy.debug()

# path = '/home/cyrille/1_simple120sec/test_hybrid_120sec.kwik'
path = '/data/spikesorting/4_20141130/20141130_all.kwik'

session = Session()

session.open(path)

# Cluster counts.
x = session.clustering.cluster_counts
cl = (sorted(x.items(), key=operator.itemgetter(1))[::-1][:50])
clu = [_[0] for _ in cl][1:4]

print(clu)
session.select(clu)

session.show_correlograms()
session.show_features()
session.show_waveforms()

run()

session.close()
