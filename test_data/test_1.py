import operator
from pprint import pprint
from timeit import default_timer

import numpy as np
import matplotlib.pyplot as plt

from vispy.app import run

import phy
from phy.cluster.manual.session import Session

phy.set_level('info')

# path = '/home/cyrille/1_simple120sec/test_hybrid_120sec.kwik'
path = '/data/spikesorting/4_20141130/20141130_all.kwik'

session = Session()

session.open(path)

# Cluster counts.
x = session.clustering.cluster_counts
cl = (sorted(x.items(), key=operator.itemgetter(1))[::-1])
clu = [_[0] for _ in cl][1:]

session.select(clu)
# session.show_waveforms()

# session.select([9, 12, 24])
# session.select([9, 24, 12])
# session.show_correlograms()
# session.show_waveforms()
session.show_traces()
# session.show_features()

run()

session.close()
