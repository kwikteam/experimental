from vispy import gloo
from vispy import app
from vispy.scene import Visual
import numpy as np
import math

class SignalsVisual(Visual):
    VERTEX_SHADER = """
    #version 120
    attribute float a_position;

    attribute vec2 a_index;
    varying vec2 v_index;

    uniform vec2 u_scale;
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
        gl_Position = vec4(u_scale * vec2(position.x, a*position.y+b), 0.0, 1.0);
        
        v_color = vec4(a_color, 1.);
        v_index = a_index;
    }
    """

    FRAGMENT_SHADER = """
    #version 120

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
        self.program = gloo.Program(self.VERTEX_SHADER, self.FRAGMENT_SHADER)

        m, n = signals.shape
        y = signals
        color = np.repeat(np.random.uniform(size=(m, 3), low=.5, high=.9),
                              n, axis=0).astype(np.float32)
        index = np.c_[np.repeat(np.arange(m), n),
                      np.tile(np.arange(n), m)].astype(np.float32)

        self.program['a_position'] = y.reshape(-1, 1)
        self.program['a_color'] = color
        self.program['a_index'] = index
        self.program['u_scale'] = (1., 1.)
        self.program['u_size'] = m
        self.program['u_n'] = n
    
    def draw(self):
        self.program.draw('line_strip')

        
if __name__ == '__main__':
    m = 16
    n = 100
    y = np.random.randn(m, n).astype(np.float32)
    y /= np.abs(y).max()
    v = SignalsVisual(y)

    c = app.Canvas(keys='interactive')

    @c.connect
    def on_initialize(event):
        gloo.set_state(clear_color='black', blend=True, 
                       blend_func=('src_alpha', 'one_minus_src_alpha'))

    @c.connect
    def on_resize(event):
        c.width, c.height = event.size
        gloo.set_viewport(0, 0, c.width, c.height)
        
    @c.connect
    def on_mouse_wheel(event):
        dx = np.sign(event.delta[1]) * .05
        scale_x, scale_y = v.program['u_scale']     
        scale_x_new, scale_y_new = (scale_x * math.exp(2.5*dx),) * 2
        v.program['u_scale'] = (max(1, scale_x_new), max(1, scale_y_new))
        c.update()
        
    @c.connect
    def on_draw(event):
        gloo.clear()
        v.draw()

    c.show()
    app.run()
