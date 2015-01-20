import math

import numpy as np

from vispy import gloo
from vispy import app
from vispy.visuals import Visual
from vispy.visuals.shaders import ModularProgram, Function, Variable

from panzoomcanvas import PanZoomCanvas

class Waveforms(Visual):
    VERT_SHADER = """
    attribute float a_data;  // -1..1
    attribute float a_time;  // -1..1
    attribute float a_cluster;  // 0..(nclusters-1)
    attribute float a_channel;  // 0..(nchannels-1)

    uniform float nclusters;
    uniform float nchannels;
    //uniform float nsamples;
    uniform vec2 u_data_scale;
    uniform sampler2D u_channel_pos;
    uniform sampler2D u_cluster_color;

    varying vec3 v_color;
    varying vec2 v_clu_channel;

    vec2 get_box_pos(float cluster, float channel) {
        vec2 box_pos = texture2D(u_channel_pos,
                                 vec2(channel / (nchannels - 1.), .5)).xy;
        box_pos = 2. * box_pos - 1.;
        box_pos.x += .1 * (cluster - .5 * (nclusters - 1.));
        return box_pos;
    }

    vec3 get_color(float cluster) {
        return texture2D(u_cluster_color,
                         vec2(cluster / (nclusters - 1.), .5)).xyz;
    }

    void main() {
        vec2 pos = u_data_scale * vec2(a_time, a_data);  // -1..1
        vec2 box_pos = get_box_pos(a_cluster, a_channel);
        v_color = get_color(a_cluster);
        v_clu_channel = vec2(a_cluster, a_channel);
        gl_Position = vec4($transform(pos + box_pos), 0., 1.);
    }
    """

    FRAG_SHADER = """
    varying vec3 v_color;
    varying vec2 v_clu_channel;

    void main() {
        if ((fract(v_clu_channel.x) > 0.) || (fract(v_clu_channel.y) > 0.))
            discard;
        gl_FragColor = vec4(v_color, 1.);
    }
    """

    def __init__(self, **kwargs):
        super(Waveforms, self).__init__(**kwargs)

        nclusters = 3
        nchannels = 32
        nsamples = 40
        nspikes = 100

        a_data = .25 * np.random.randn(nspikes, nchannels,
                                       nsamples).astype(np.float32)

        cluster_colors = np.random.uniform(size=(nclusters, 3),
                                           low=.5, high=.9).astype(np.float32)

        # WARNING: texture values for channel positions must be in [0,1]
        # and will be rescaled to [-1,1]
        channel_positions = np.zeros((nchannels, 2)).astype(np.float32)
        channel_positions[:, 0] = .5
        channel_positions[:, 1] = np.linspace(0.1, .9, nchannels)

        spike_clusters = np.random.randint(size=nspikes,
                                           low=0,
                                           high=nclusters).astype(np.float32)

        a_time = np.tile(np.linspace(-1., 1.,
                                     nsamples).astype(np.float32),
                         nchannels * nspikes)

        a_cluster = np.repeat(spike_clusters, nchannels * nsamples)

        a_channel = np.tile(np.repeat(np.arange(nchannels).astype(np.float32),
                                       nsamples), nspikes)

        u_channel_pos = np.dstack((channel_positions.reshape((1,
                                                              nchannels, 2)),
                                   np.zeros((1, nchannels, 1),
                                            dtype=np.float32)))

        u_cluster_color = cluster_colors.reshape((1, nclusters, 3))

        self.program = ModularProgram(self.VERT_SHADER, self.FRAG_SHADER)
        self.program['a_data'] = a_data
        self.program['a_time'] = a_time
        self.program['a_cluster'] = a_cluster
        self.program['a_channel'] = a_channel
        self.program['nclusters'] = nclusters
        self.program['nchannels'] = nchannels
        self.program['u_data_scale'] = (.03, .02)
        self.program['u_channel_pos'] = gloo.Texture2D(u_channel_pos,
                                                      wrapping='clamp_to_edge')
        self.program['u_cluster_color'] = gloo.Texture2D(u_cluster_color)

        gloo.set_state(clear_color='black', blend=True,
                       blend_func=('src_alpha', 'one_minus_src_alpha'))

    def draw(self, event):
        self.program.draw('line_strip')

class WaveformView(PanZoomCanvas):
    def __init__(self, **kwargs):
        super(WaveformView, self).__init__(**kwargs)
        self.waveforms = Waveforms()

if __name__ == '__main__':
    c = WaveformView()
    c.show()
    app.run()
