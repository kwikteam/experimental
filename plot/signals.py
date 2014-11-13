from vispy import gloo
from vispy import app
from vispy.visuals import Visual
from vispy.visuals.shaders import ModularProgram, Function, Variable
import numpy as np
import os.path as op
import math

X_TRANSFORM = """
float get_x(float x_index) {
    // 'x_index' is between 0 and nsamples.
    return -1 + 2 * x_index / ($nsamples-1);
}
"""

Y_TRANSFORM = """
float get_y(float y_index, float sample) {
    // 'y_index' is between 0 and nsignals.
    float a = $scale / $nsignals;
    float b = -1 + 2 * (y_index + .5) / $nsignals;

    return a * sample + b;
}
"""

DISCRETE_CMAP = """
vec3 get_color(float index) {
    float x = (index + .5) / $ncolors;
    return texture2D($colormap, vec2(x, .5)).rgb;
}
"""

class SignalsVisual(Visual):
    VERTEX_SHADER = """
    attribute float a_position;

    attribute vec2 a_index;
    varying vec2 v_index;

    uniform float u_nsignals;
    uniform float u_nsamples;

    void main() {
        vec2 position = vec2($get_x(a_index.y),
                             $get_y(a_index.x, a_position));
        gl_Position = vec4($panzoom(position), 0.0, 1.0);

        v_index = a_index;
    }
    """

    FRAGMENT_SHADER = """
    varying vec2 v_index;

    void main() {
        gl_FragColor = vec4($get_color(v_index.x), 1.);

        // Discard vertices between two signals.
        if ((fract(v_index.x) > 0.))
            discard;
    }
    """

    def __init__(self, data):
        super(SignalsVisual, self).__init__()

        self.program = ModularProgram(self.VERTEX_SHADER, self.FRAGMENT_SHADER)

        nsignals, nsamples = data.shape
        # nsamples, nsignals = data.shape

        self._data = data

        a_index = np.c_[np.repeat(np.arange(nsignals), nsamples),
                      np.tile(np.arange(nsamples), nsignals)] \
                .astype(np.float32)

        # Doesn't seem to work nor to be very efficient.
        # indices = nsignals * np.arange(nsamples)
        # indices = indices[None, :] + np.arange(nsignals)[:, None]
        # indices = indices.flatten().astype(np.uint32)
        # self._ibuffer = gloo.IndexBuffer(indices)

        self._buffer = gloo.VertexBuffer(data.reshape(-1, 1))
        self.program['a_position'] = self._buffer
        self.program['a_index'] = a_index

        x_transform = Function(X_TRANSFORM)
        x_transform['nsamples'] = nsamples
        self.program.vert['get_x'] = x_transform

        y_transform = Function(Y_TRANSFORM)
        y_transform['scale'] = Variable('uniform float u_signal_scale', 5.)
        y_transform['nsignals'] = nsignals
        self.program.vert['get_y'] = y_transform
        self._y_transform = y_transform

        colormap = Function(DISCRETE_CMAP)
        cmap = np.random.uniform(size=(1, nsignals, 3), low=.5, high=.9) \
               .astype(np.float32)
        colormap['colormap'] = Variable('uniform sampler2D u_colormap',
                                        gloo.Texture2D(cmap))
        colormap['ncolors'] = nsignals
        self.program.frag['get_color'] = colormap

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, value):
        self._data = value
        self._buffer.set_subdata(value.reshape(-1, 1))
        self.update()

    @property
    def signal_scale(self):
        return self._y_transform['scale'].value

    @signal_scale.setter
    def signal_scale(self, value):
        self._y_transform['scale'].value = value
        self.update()

    def draw(self):
        self.program.draw('line_strip')
