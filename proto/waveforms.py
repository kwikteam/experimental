"""
TODO
* use vispy transforms for box placement
* refactor probe renormalization
* refactor data baking
* add more interactivity
* add masks, alpha, depth
* support sparse structure
* use ST instead of PanZoom
"""

import math

import numpy as np

from vispy import gloo
from vispy import app
from vispy.visuals import Visual
from vispy.visuals.shaders import ModularProgram, Function, Variable

from panzoomcanvas import PanZoomCanvas

"""
sparse

a_box: (cluster, channel)       0..nclusters-1, 0..nchannels-1    \sum_spike nchannels_spike
repeated nsamples

a_cluster
a_channel

"""


class Waveforms(Visual):
    VERT_SHADER = """
    attribute float a_data;  // -1..1
    attribute float a_time;  // -1..1
    attribute vec2 a_box;  // 0..(nclusters-1, nchannels-1)

    uniform float nclusters;
    uniform float nchannels;
    uniform vec2 u_data_scale;
    uniform sampler2D u_channel_pos;
    uniform sampler2D u_cluster_color;

    varying vec3 v_color;
    varying vec2 v_box;

    vec2 get_box_pos(vec2 box) {  // box = (cluster, channel)
        vec2 box_pos = texture2D(u_channel_pos,
                                 vec2(box.y / (nchannels - 1.), .5)).xy;
        box_pos = 2. * box_pos - 1.;
        box_pos.x += .1 * (box.x - .5 * (nclusters - 1.));
        return box_pos;
    }

    vec3 get_color(float cluster) {
        return texture2D(u_cluster_color,
                         vec2(cluster / (nclusters - 1.), .5)).xyz;
    }

    void main() {
        vec2 pos = u_data_scale * vec2(a_time, a_data);  // -1..1
        vec2 box_pos = get_box_pos(a_box);
        v_color = get_color(a_box.x);
        v_box = a_box;
        gl_Position = vec4($transform(pos + box_pos), 0., 1.);
    }
    """

    FRAG_SHADER = """
    varying vec3 v_color;
    varying vec2 v_box;

    void main() {
        if ((fract(v_box.x) > 0.) || (fract(v_box.y) > 0.))
            discard;
        gl_FragColor = vec4(v_color, 1.);
    }
    """

    def __init__(self, **kwargs):
        super(Waveforms, self).__init__(**kwargs)

        a_data = .25 * np.random.randn(nspikes, nchannels,
                                       nsamples).astype(np.float32)

        cluster_colors = np.random.uniform(size=(nclusters, 3),
                                           low=.5, high=.9).astype(np.float32)

        # WARNING: texture values for channel positions must be in [0,1]
        # and will be rescaled to [-1,1]
        # channel_positions = np.zeros((nchannels, 2)).astype(np.float32)
        # channel_positions[:, 0] = .5
        # channel_positions[:, 1] = np.linspace(0.1, .9, nchannels)

        spike_clusters = np.random.randint(size=nspikes,
                                           low=0,
                                           high=nclusters).astype(np.float32)

        # nwaveforms = sum_spike nchannels_spike
        # TODO: update this for sparse
        nchannels_per_spike = nchannels * np.ones(nspikes, dtype=np.int32)
        channels_per_spike = np.tile(np.arange(nchannels).astype(np.float32),
                                     nspikes)

        nwaveforms = np.sum(nchannels_per_spike)

        a_time = np.tile(np.linspace(-1., 1.,
                                     nsamples).astype(np.float32),
                         nwaveforms)

        a_cluster = np.repeat(spike_clusters, nchannels_per_spike * nsamples)
        a_channel = np.repeat(channels_per_spike, nsamples)
        a_box = np.c_[a_cluster, a_channel]

        u_channel_pos = np.dstack((channel_positions.reshape((1,
                                                              nchannels, 2)),
                                   np.zeros((1, nchannels, 1),
                                            dtype=np.float32)))

        u_cluster_color = cluster_colors.reshape((1, nclusters, 3))

        self.program = ModularProgram(self.VERT_SHADER, self.FRAG_SHADER)
        self.program['a_data'] = a_data
        self.program['a_time'] = a_time
        self.program['a_box'] = a_box
        self.program['nclusters'] = nclusters
        self.program['nchannels'] = nchannels
        self.program['u_data_scale'] = (.03, .02)
        self.program['u_channel_pos'] = gloo.Texture2D(u_channel_pos,
                                                      wrapping='clamp_to_edge')
        self.program['u_cluster_color'] = gloo.Texture2D(u_cluster_color)

        gloo.set_state(clear_color='black', blend=True,
                       blend_func=('src_alpha', 'one_minus_src_alpha'))

    @property
    def box_scale(self):
        return self.program['u_data_scale']

    @box_scale.setter
    def box_scale(self, value):
        assert isinstance(value, tuple) and len(value) == 2
        self.program['u_data_scale'] = value
        self.update()

    def draw(self, event):
        self.program.draw('line_strip')

class WaveformView(PanZoomCanvas):
    def __init__(self, **kwargs):
        super(WaveformView, self).__init__(**kwargs)
        self.waveforms = Waveforms()

    def on_key_press(self, event):
        super(WaveformView, self).on_key_press(event)
        if event.key == '+':
            u, v = self.waveforms.box_scale
            self.waveforms.box_scale = (u, v*1.1)
        if event.key == '-':
            u, v = self.waveforms.box_scale
            self.waveforms.box_scale = (u, v/1.1)

if __name__ == '__main__':



    channel_positions = np.array([  (35, 310),
                                    (-34, 300),
                                    (33, 290),
                                    (-32, 280),
                                    (31, 270),
                                    (-30, 260),
                                    (29, 250),
                                    (-28, 240),
                                    (27, 230),
                                    (-26, 220),
                                    (25, 210),
                                    (-24, 200),
                                    (23, 190),
                                    (-22, 180),
                                    (21, 170),
                                    (-20, 160),
                                    (19, 150),
                                    (-18, 140),
                                    (17, 130),
                                    (-16, 120),
                                    (15, 110),
                                    (-14, 100),
                                    (13, 90),
                                    (-12, 80),
                                    (11, 70),
                                    (-10, 60),
                                    (9, 50),
                                    (-8, 40),
                                    (7, 30),
                                    (-6, 20),
                                    (5, 10),
                                    (0, 0)], dtype=np.float32)

    channel_positions -= channel_positions.min(axis=0)
    channel_positions /= channel_positions.max(axis=0)
    channel_positions = .2 + .6 * channel_positions

    nclusters = 5
    nchannels = 32
    nsamples = 50
    nspikes = 100

    c = WaveformView()
    c.show()
    app.run()
