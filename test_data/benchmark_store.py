from __future__ import print_function

import os
import os.path as op
import shutil
from pprint import pprint
from timeit import default_timer


import h5py
import numpy as np
from numpy.testing import assert_allclose as ac

import phy
from phy.cluster.manual.store import DiskStore
from phy.io.h5 import open_h5
from phy.cluster.manual._utils import _spikes_per_cluster
from phy.utils.array import _index_of

phy.debug()

_store_path = '_store'
n_spikes = 200000
n_channels = 200
n_clusters = 100


# Generate the dataset.
def _gen_arr():
    arr = np.random.rand(n_spikes, n_channels).astype(np.float32)
    with open_h5('test', 'w') as f:
        f.write('/test', arr)


def _gen_spike_clusters():
    sc = np.random.randint(size=n_spikes, low=0, high=n_clusters)
    with open_h5('sc', 'w') as f:
        f.write('/sc', sc)


def _load_spike_clusters():
    with open_h5('sc', 'r') as f:
        return f.read('/sc')[...]


def _reset_store():
    for path in (_store_path, '_flat'):
        if op.exists(path):
            shutil.rmtree(path)
        os.mkdir(path)


# _gen_spike_clusters()
# _gen_arr()

f = open_h5('test', 'r')

sc = _load_spike_clusters()
arr = f.read('/test')
spikes = np.arange(n_spikes)
spc = _spikes_per_cluster(spikes, sc)


def _flat_file(cluster):
    return op.join('_flat', str(cluster))


def _free_cache():
    os.system('sync')
    os.system('sudo sh -c "sync; echo 3 > /proc/sys/vm/drop_caches"')


# @profile
def _gen_store_1(chunk_size):
    _reset_store()
    _free_cache()

    t0 = default_timer()
    # chunk_size = 10000
    # print("chunks")
    for i in range(n_spikes // chunk_size):
        # print(i, end='\r')
        a, b = i * chunk_size, (i + 1) * chunk_size

        # Load a chunk from HDF5.
        assert isinstance(arr, h5py.Dataset)
        sub_arr = arr[a:b]
        assert isinstance(sub_arr, np.ndarray)
        sub_sc = sc[a:b]
        sub_spikes = np.arange(a, b)

        # Split the spikes.
        sub_spc = _spikes_per_cluster(sub_spikes, sub_sc)

        # Go through the clusters.
        clusters = sorted(sub_spc.keys())
        for cluster in clusters:
            idx = _index_of(sub_spc[cluster], sub_spikes)

            # Save part of the array to a binary file.
            with open(_flat_file(cluster), 'ab') as f:
                sub_arr[idx].tofile(f)
    # print()

    ds = DiskStore(_store_path)

    # Next, put the flat binary files back to HDF5.
    # print("flat to HDF5")
    for cluster in range(n_clusters):
        # print(cluster, end='\r')
        data = np.fromfile(_flat_file(cluster),
                           dtype=np.float32).reshape((-1, n_channels))
        ds.store(cluster, data=data)
    print("time", default_timer() - t0)
    # print()

    # Test.
    cluster = 0
    arr2 = ds.load(cluster, 'data')

    ac(arr[spc[cluster], :], arr2)


chunk_sizes = (100, 1000, 10000, 100000)
chunk_sizes = (10000,)

for chunk_size in chunk_sizes:
    print(chunk_size)
    _gen_store_1(chunk_size)
    print()


f.close()
