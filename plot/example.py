import math
import numpy as np
import h5py

from vispy import app, keys
from signals import SignalsVisual
from panzoomcanvas import PanZoomCanvas

filename = '/data/spikesorting/nick128_sorted/20141009_all_AdjGraph.raw.kwd'

def load_data(filename, sample_start=0, sample_stop=20000):
    with h5py.File(filename) as f:
        data = f['/recordings/0/data']

        # Data selection.
        data = data[sample_start:sample_stop,:].astype(np.float32).T

        # Data normalization.
        data -= data.mean(axis=1)[:, None]
        data *= 1. / np.abs(data).max()
    return data

c = PanZoomCanvas()
c.signals = SignalsVisual(load_data(filename))

@c.connect
def on_mouse_wheel(event):
    if event.modifiers == (keys.CONTROL,):
        sign = np.sign(event.delta[1])
        c.signals.signal_scale = np.clip(c.signals.signal_scale * 1.2 ** sign,
                                         1e-2, 1e2)

app.run()

