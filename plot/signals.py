from vispy import gloo
from vispy import app
from vispy.visuals import Visual
from vispy.visuals.shaders import ModularProgram, Function, Variable
import numpy as np
import os.path as op
import math

class SignalsVisual(Visual):
    VERTEX_SHADER = """
    attribute float a_position;

    attribute vec2 a_index;
    varying vec2 v_index;

    uniform float u_size;
    uniform float u_n;

    attribute vec3 a_color;
    varying vec4 v_color;

    void main() {
        float nrows = u_size;

        float x = -1 + 2*a_index.y / (u_n-1);
        vec2 position = vec2(x, a_position);

        float a = 1./nrows;
        float b = -1 + 2*(a_index.x+.5) / nrows;
        vec2 position_tr = vec2(position.x, a*5*position.y+b);

        gl_Position = vec4($panzoom(position_tr), 0.0, 1.0);

        v_color = vec4(a_color, 1.);
        v_index = a_index;
    }
    """

    FRAGMENT_SHADER = """
    varying vec4 v_color;
    varying vec2 v_index;

    void main() {
        gl_FragColor = v_color;

        // Discard vertices between two signals.
        if ((fract(v_index.x) > 0.))
            discard;
    }
    """

    def __init__(self, signals):
        self.program = ModularProgram(self.VERTEX_SHADER, self.FRAGMENT_SHADER)

        m, n = signals.shape
        y = signals
        color = np.repeat(np.random.uniform(size=(m, 3), low=.5, high=.9),
                              n, axis=0).astype(np.float32)
        index = np.c_[np.repeat(np.arange(m), n),
                      np.tile(np.arange(n), m)].astype(np.float32)

        self.program['a_position'] = y.reshape(-1, 1)
        self.program['a_color'] = color
        self.program['a_index'] = index

        self.program['u_size'] = m
        self.program['u_n'] = n

    def draw(self):
        self.program.draw('line_strip')
