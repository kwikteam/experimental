import operator
from pprint import pprint
from timeit import default_timer

import numpy as np
import matplotlib.pyplot as plt

from vispy.app import run

import phy
from phy.cluster.manual import Session

phy.set_level('info')

path = '/data/spikesorting/1_simple120sec/test_hybrid_120sec.kwik'
# path = '/data/spikesorting/4_20141130/20141130_all.kwik'
# path = '/data/spikesorting/for_shabnam_20150429/manual_sort_test.kwik'

session = Session(path)
session.model.describe()

# Cluster counts.
# x = session.clustering.cluster_counts
# cl = (sorted(x.items(), key=operator.itemgetter(1))[::-1])
# clu = [_[0] for _ in cl][1:5]

session.show_gui()

# This is only necessary when using VisPy canvases exclusively, not the Qt GUI.
# run()

session.close()
