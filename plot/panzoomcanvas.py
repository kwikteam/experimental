from vispy import gloo
from vispy import app
from vispy.visuals import Visual
from vispy.visuals.transforms import PanZoomTransform, TransformSystem
from vispy.visuals.shaders import Variable
import numpy as np
import os.path as op
import math


class PanZoomCanvas(app.Canvas):
    def __init__(self, **kwargs):
        super(PanZoomCanvas, self).__init__(keys='interactive',
                                            show=True, **kwargs)
        self._visuals = []

        self._pz = PanZoomTransform()
        self._pz.pan = Variable('uniform vec2 u_pan', (0, 0))
        self._pz.zoom = Variable('uniform vec2 u_zoom', (1, 1))

        self._tr = TransformSystem(self)

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
        pan_x, pan_y = self._pz.pan
        zoom_x, zoom_y = self._pz.zoom
        xmin = -1 / zoom_x - pan_x
        xmax = +1 / zoom_x - pan_x
        ymin = -1 / zoom_y - pan_y
        ymax = +1 / zoom_y - pan_y
        return (xmin, ymin, xmax, ymax)

    def on_mouse_move(self, event):
        if event.is_dragging and not event.modifiers:
            x0, y0 = self._normalize(event.press_event.pos)
            x1, y1 = self._normalize(event.last_event.pos)
            x, y = self._normalize(event.pos)
            dx, dy = x - x1, -(y - y1)
            button = event.press_event.button

            pan_x, pan_y = self._pz.pan
            zoom_x, zoom_y = self._pz.zoom

            if button == 1:
                self._pz.pan = (pan_x + dx/zoom_x,
                                pan_y + dy/zoom_y)
            elif button == 2:
                zoom_x_new, zoom_y_new = (zoom_x * math.exp(2.5 * dx),
                                          zoom_y * math.exp(2.5 * dy))
                self._pz.zoom = (zoom_x_new, zoom_y_new)
                self._pz.pan = (pan_x - x0 * (1./zoom_x - 1./zoom_x_new),
                                pan_y + y0 * (1./zoom_y - 1./zoom_y_new))
            self.update()

    def on_mouse_wheel(self, event):
        if not event.modifiers:
            dx = np.sign(event.delta[1])*.05
            x0, y0 = self._normalize(event.pos)
            pan_x, pan_y = self._pz.pan
            zoom_x, zoom_y = self._pz.zoom
            zoom_x_new, zoom_y_new = (zoom_x * math.exp(2.5 * dx),
                                      zoom_y * math.exp(2.5 * dx))
            self._pz.zoom = (zoom_x_new, zoom_y_new)
            self._pz.pan = (pan_x - x0 * (1./zoom_x - 1./zoom_x_new),
                            pan_y + y0 * (1./zoom_y - 1./zoom_y_new))
            self.update()

    def on_key_press(self, event):
        if event.key == 'R':
            self._pz.zoom = (1., 1.)
            self._pz.pan = (0., 0.)
            self.update()

    def add_visual(self, name, value):
        value._program.vert['transform'] = self._pz
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
            visual.draw(self._tr)
