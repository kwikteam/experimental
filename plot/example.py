import numpy as np
from vispy import app
from signals import show_raw_data

if __name__ == '__main__':

    filename = '/data/spikesorting/nick128_sorted/20141009_all_AdjGraph.raw.kwd'
    show_raw_data(filename, nchannels=128)
    app.run()
