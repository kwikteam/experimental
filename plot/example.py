import numpy as np
import h5py
from vispy import app
from signals import SignalsCanvas

filename = '/data/spikesorting/nick128_sorted/20141009_all_AdjGraph.raw.kwd'
f = h5py.File(filename)
data = f['/recordings/0/data']

nchannels = data.shape[1]
nsamples = 20000

data = data[:nsamples,:].astype(np.float32).T
data *= 5./np.abs(data).max()

c = SignalsCanvas(data)
c.show()
app.run()

f.close()
