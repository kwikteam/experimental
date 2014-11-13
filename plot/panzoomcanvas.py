from vispy import gloo
from vispy import app
from vispy.visuals import Visual
from vispy.visuals.shaders import ModularProgram, Function, Variable
import numpy as np
import os.path as op
import math

PAN_ZOOM_FUNC = """
vec2 pan_zoom(vec2 position) {
    return $scale * (position + $pan);
}
"""

class PanZoomCanvas(app.Canvas):
    def __init__(self):
        super(PanZoomCanvas, self).__init__(keys='interactive', show=True)
        self._visuals = []

        self._pan_zoom = Function(PAN_ZOOM_FUNC)
        self._pan_zoom['pan'] = Variable('uniform vec2 u_pan', (0., 0.))
        self._pan_zoom['scale'] = Variable('uniform vec2 u_scale', (1., 1.))

    def on_initialize(self, event):
        gloo.set_state(clear_color='black', blend=True,
                       blend_func=('src_alpha', 'one_minus_src_alpha'))

    def on_resize(self, event):
        self.width, self.height = event.size
        gloo.set_viewport(0, 0, self.width, self.height)

    def _normalize(self, x_y):
        x, y = x_y
        w, h = float(self.width), float(self.height)
        return x/(w/2.)-1., y/(h/2.)-1.

    def bounds(self):
        pan_x, pan_y = self._pan_zoom['pan'].value
        scale_x, scale_y = self._pan_zoom['scale'].value
        xmin = -1 / scale_x - pan_x
        xmax = +1 / scale_x - pan_x
        ymin = -1 / scale_y - pan_y
        ymax = +1 / scale_y - pan_y
        return (xmin, ymin, xmax, ymax)

    def on_mouse_move(self, event):
        if event.is_dragging and not event.modifiers:
            x0, y0 = self._normalize(event.press_event.pos)
            x1, y1 = self._normalize(event.last_event.pos)
            x, y = self._normalize(event.pos)
            dx, dy = x - x1, -(y - y1)
            button = event.press_event.button

            pan_x, pan_y = self._pan_zoom['pan'].value
            scale_x, scale_y = self._pan_zoom['scale'].value

            if button == 1:
                self._pan_zoom['pan'].value = (pan_x+dx/scale_x,
                                               pan_y+dy/scale_y)
            elif button == 2:
                scale_x_new, scale_y_new = (scale_x * math.exp(2.5*dx),
                                            scale_y * math.exp(2.5*dy))
                self._pan_zoom['scale'].value = (scale_x_new, scale_y_new)
                self._pan_zoom['pan'].value = (pan_x -
                                         x0 * (1./scale_x - 1./scale_x_new),
                                         pan_y +
                                         y0 * (1./scale_y - 1./scale_y_new))
            self.update()

    def on_mouse_wheel(self, event):
        if not event.modifiers:
            dx = np.sign(event.delta[1])*.05
            x0, y0 = self._normalize(event.pos)
            pan_x, pan_y = self._pan_zoom['pan'].value
            scale_x, scale_y = self._pan_zoom['scale'].value
            scale_x_new, scale_y_new = (scale_x * math.exp(2.5*dx),
                                        scale_y * math.exp(2.5*dx))
            self._pan_zoom['scale'].value = (scale_x_new, scale_y_new)
            self._pan_zoom['pan'].value = (pan_x -
                                     x0 * (1./scale_x - 1./scale_x_new),
                                     pan_y +
                                     y0 * (1./scale_y - 1./scale_y_new))
            self.update()

    def on_key_press(self, event):
        if event.key == 'R':
            self._pan_zoom['scale'].value = (1., 1.)
            self._pan_zoom['pan'].value = (0., 0.)
            self.update()

    def add_visual(self, name, value):
        value.program.vert['panzoom'] = self._pan_zoom
        value.events.update.connect(self.update)
        self._visuals.append(value)

    def __setattr__(self, name, value):
        if isinstance(value, Visual):
            self.add_visual(name, value)
        super(PanZoomCanvas, self).__setattr__(name, value)

    @property
    def visuals(self):
        return self._visuals

    def on_draw(self, event):
        gloo.clear()
        for visual in self.visuals:
            visual.draw()
