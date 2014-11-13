import math
import numpy as np
import h5py

from vispy import app, keys
from signals import SignalsVisual
from panzoomcanvas import PanZoomCanvas


def load_data(filename, sample_start=0, sample_stop=20000):
    with h5py.File(filename) as f:
        data = f['/recordings/0/data']

        # Data selection.
        # WARNING: beware the transposition (necessary to load contiguous
        # data on the GPU, but can be worked around later with an index buffer)
        data = data[sample_start:sample_stop,:].astype(np.float32).T

        # Data normalization.
        data -= data.mean(axis=1)[:, None]
        data *= 1. / np.abs(data).max()
    return data


def get_data_size(filename):
    with h5py.File(filename) as f:
        data = f['/recordings/0/data']
        return data.shape


class Pager(object):
    def __init__(self, nsamples_total=None, nsamples_page=20000):
        self.nsamples_total = nsamples_total
        self.nsamples_page = nsamples_page
        self._page_index = 0
        self._page_max = self._to_page(self.nsamples_total - 1)

    def _to_page(self, sample):
        return sample // self.nsamples_page

    @property
    def page_index(self):
        return self._page_index

    @page_index.setter
    def page_index(self, value):
        self._page_index = np.clip(value, 0, self._page_max)

    @property
    def page_max(self):
        return self._page_max

    # @property
    def bounds(self):
        return (self.nsamples_page * self._page_index,
                self.nsamples_page * (self._page_index + 1))

    def next(self):
        # if self._page_index >= self._page_max - 1:
        #     raise ValueError("The last page has been reached.")
        self._page_index = min(self._page_index + 1, self._page_max - 1)

    def previous(self):
        # if self._page_index <= 0:
        #     raise ValueError("The first page has been reached.")
        self._page_index = max(self._page_index - 1, 0)

    def from_time(self, time):
        self.from_sample(int(time * 20000))

    def from_sample(self, sample):
        self.page_index = self._to_page(sample)


class DataLoader(object):
    def __init__(self, filename):
        self.filename = filename
        self.nsamples_total, self.nchannels = get_data_size(filename)
        self.pager = Pager(nsamples_total=self.nsamples_total,
                           nsamples_page=20000)
        self._load()

    def _load(self):
        start, stop = self.pager.bounds()
        self.data = load_data(self.filename, start, stop)
        return self.data

    def next(self):
        self.pager.next()
        return self._load()

    def previous(self):
        self.pager.previous()
        return self._load()

    def first(self):
        self.pager.page_index = 0
        return self._load()

    def end(self):
        self.pager.page_index = self.pager.page_max - 1
        return self._load()

    def from_time(self, time):
        self.pager.from_time(time)
        return self._load()

    def from_sample(self, sample):
        self.pager.from_sample(sample)
        return self._load()


if __name__ == '__main__':

    filename = '/data/spikesorting/nick128_sorted/20141009_all_AdjGraph.raw.kwd'

    c = PanZoomCanvas()
    c.signals = SignalsVisual(load_data(filename))

    loader = DataLoader(filename)

    @c.connect
    def on_mouse_wheel(event):
        if event.modifiers == (keys.CONTROL,):
            sign = np.sign(event.delta[1])
            c.signals.signal_scale = np.clip(c.signals.signal_scale*1.2**sign,
                                             1e-2, 1e2)

    @c.connect
    def on_key_press(event):
        if event.key == 'Left':
            c.signals.data = loader.previous()
            c.update()
        elif event.key == 'right':
            c.signals.data = loader.next()
            c.update()
        elif event.key == 'Home':
            c.signals.data = loader.first()
            c.update()
        elif event.key == 'End':
            c.signals.data = loader.last()
            c.update()

    app.run()

