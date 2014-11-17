import numpy as np
import os.path as op
import math

from vispy import gloo
from vispy import app, keys
from vispy.visuals import Visual
from vispy.visuals.shaders import ModularProgram, Function, Variable

from loader import DataLoader
from panzoomcanvas import PanZoomCanvas

X_TRANSFORM = """
float get_x(float x_index) {
    // 'x_index' is between 0 and nsamples.
    return -1. + 2. * x_index / (float($nsamples) - 1.);
}
"""

Y_TRANSFORM = """
float get_y(float y_index, float sample) {
    // 'y_index' is between 0 and nsignals.
    float a = float($scale) / float($nsignals);
    float b = -1. + 2. * (y_index + .5) / float($nsignals);

    return a * sample + b;
}
"""

DISCRETE_CMAP = """
vec3 get_color(float index) {
    float x = (index + .5) / float($ncolors);
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
        gl_Position = vec4($transform(position), 0.0, 1.0);

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

        self._program = ModularProgram(self.VERTEX_SHADER, self.FRAGMENT_SHADER)

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
        self._program['a_position'] = self._buffer
        self._program['a_index'] = a_index

        x_transform = Function(X_TRANSFORM)
        x_transform['nsamples'] = nsamples
        self._program.vert['get_x'] = x_transform

        y_transform = Function(Y_TRANSFORM)
        y_transform['scale'] = Variable('uniform float u_signal_scale', 5.)
        y_transform['nsignals'] = nsignals
        self._program.vert['get_y'] = y_transform
        self._y_transform = y_transform

        colormap = Function(DISCRETE_CMAP)
        cmap = np.random.uniform(size=(1, nsignals, 3), low=.5, high=.9) \
               .astype(np.float32)
        tex = gloo.Texture2D((cmap * 255).astype(np.uint8))
        colormap['colormap'] = Variable('uniform sampler2D u_colormap', tex)
        colormap['ncolors'] = nsignals
        self._program.frag['get_color'] = colormap

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

    def draw(self, transform_system):
        self._program.draw('line_strip')


class RawDataView(PanZoomCanvas):
    def __init__(self, filename=None, page_duration=1., nchannels=None,
                 **kwargs):
        if 'position' not in kwargs:
            kwargs['position'] = (400, 300)
        if 'size' not in kwargs:
            kwargs['size'] = (800,600)
        super(RawDataView, self).__init__(**kwargs)

        self.loader = DataLoader(filename, page_duration=page_duration,
                            nchannels=nchannels)

        self.signals = SignalsVisual(self.loader.data)

    def on_mouse_wheel(self, event):
        super(RawDataView, self).on_mouse_wheel(event)
        if event.modifiers == (keys.CONTROL,):
            sign = np.sign(event.delta[1])
            self.signals.signal_scale = np.clip(self.signals.signal_scale \
                                                *1.2**sign,
                                                1e-2, 1e2)

    def on_key_press(self, event):
        super(RawDataView, self).on_key_press(event)
        if event.key == 'Left':
            self.signals.data = self.loader.previous()
            self.update()
        elif event.key == 'right':
            self.signals.data = self.loader.next()
            self.update()
        elif event.key == 'Home':
            self.signals.data = self.loader.first()
            self.update()
        elif event.key == 'End':
            self.signals.data = self.loader.last()
            self.update()


def show_raw_data(filename, page_duration=1., nchannels=None, **kwargs):
    view = RawDataView(filename=filename, page_duration=page_duration,
                       nchannels=nchannels, **kwargs)
    view.show()
