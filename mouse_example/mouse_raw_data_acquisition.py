#
#  Copyright 2016 Luigi Stammati <luigi dot stammati at uniroma2 dot it>
#
#  This file is part of FEBA
#
#  FEBA is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  FEBA is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with FEBA.  If not, see <http://www.gnu.org/licenses/>.
#

from matplotlib import pyplot
import time


class MouseEventHandler:
    def __init__(self):
        """
        Open a windows to allow mouse raw data acquisition for test.
        data will be stored in the raw_data list attribute
        :return:
        """
        self.fig = pyplot.figure()
        ax = self.fig.add_subplot(111)
        ax.plot()
        self.raw_data = []
        self.listener_ids = self.start_listeners()

        pyplot.show()

    def mouse_down_handler(self, event):

        print "mouse down"
        button_map = {
            1: "mouse-left-down",
            2: "mouse-middle-down",
            3: "mouse-right-down"
        }

        raw_event = dict()

        raw_event["event_type"] = button_map[event.button]
        raw_event["x"] = event.xdata
        raw_event["y"] = event.ydata
        raw_event["timestamp"] = time.time()

        valid = True
        for at in raw_event:
            valid = valid and raw_event[at] is not None

        if valid:
            self.raw_data.append(raw_event)
            print "raw event added %s" % raw_event

    def mouse_up_handler(self, event):

        print "mouse up"
        button_map = {
            1: "mouse-left-up",
            2: "mouse-middle-up",
            3: "mouse-right-up"
        }

        raw_event = dict()

        raw_event["event_type"] = button_map[event.button]
        raw_event["x"] = event.xdata
        raw_event["y"] = event.ydata
        raw_event["timestamp"] = time.time()

        valid = True
        for at in raw_event:
            valid = valid and raw_event[at] is not None

        if valid:
            self.raw_data.append(raw_event)
            print "raw event added %s" % raw_event

    def mouse_move_handler(self, event):

        print "Move"
        raw_event = dict()

        raw_event["event_type"] = "move"
        raw_event["x"] = event.xdata
        raw_event["y"] = event.ydata
        raw_event["timestamp"] = time.time()

        valid = True
        for at in raw_event:
            valid = valid and raw_event[at] is not None

        if valid:
            self.raw_data.append(raw_event)
            print "raw event added %s" % raw_event

    def start_listeners(self):
        print "listeners start"
        md_id = self.fig.canvas.mpl_connect('button_press_event', self.mouse_down_handler)
        mu_id = self.fig.canvas.mpl_connect('button_release_event', self.mouse_up_handler)
        mm_id = self.fig.canvas.mpl_connect('motion_notify_event', self.mouse_move_handler)

        return [md_id, mu_id, mm_id]

    def close_listeners(self):
        if not not self.listener_ids:
            for id in self.listener_ids:
                self.fig.canvas.mpl_disconnect(id)

