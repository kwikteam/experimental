import math
import numpy as np
import h5py

from vispy import app, keys
from signals import SignalsVisual
from panzoomcanvas import PanZoomCanvas

filename = '/data/spikesorting/nick128_sorted/20141009_all_AdjGraph.raw.kwd'
with h5py.File(filename) as f:
    data = f['/recordings/0/data']

    data = data[:20000,:].astype(np.float32).T
    data -= data.mean(axis=1)[:, None]
    data *= 1. / np.abs(data).max()

c = PanZoomCanvas()
c.signals = SignalsVisual(data)

@c.connect
def on_mouse_wheel(event):
    if event.modifiers == (keys.CONTROL,):
        sign = np.sign(event.delta[1])
        c.signals.signal_scale = np.clip(c.signals.signal_scale * 1.2 ** sign,
                                         1e-2, 1e2)

# c.show()
app.run()

