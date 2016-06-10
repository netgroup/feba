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

import feba
import numpy

mouse_schema = {
  "title": "mouse raw data",
  "type": "object",
  "properties": {
    "x": {
      "type": "number"
    },
    "y": {
      "type": "number"
    },
    "timestamp": {
      "type": "number"
    },
    "event_type": {
      "type": "string",
      "enum": ["move", "mouse-left-down", "mouse-left-up", "mouse-right-down", "mouse-right-up", "scroll"]
    }
  },
  "required": [
    "x",
    "y",
    "timestamp",
    "event_type"
  ]
}


class MouseAction(feba.Action):

    def __init__(self, data_sequence=None, level=None):
        super(MouseAction, self).__init__(data_sequence=data_sequence, schema=mouse_schema, level=level)


class LeftClick(MouseAction):

    def __init__(self, data_sequence=None):
        super(LeftClick, self).__init__(data_sequence=data_sequence, level=1)

    def check_pattern(self, sequence, index=None, partial=False):
        super(LeftClick, self).check_pattern(sequence, index)
        try:
            # TODO configurable threshold
            duration_threshold = 5.0
            if partial:
                # partial validation
                if index is not None:
                    if index == 0:
                        cond = sequence[index].event_type == "mouse-left-down"
                    else:
                        cond = sequence[index].event_type == "move"
                else:
                    raise Exception("Index must be specified in the partial validation")
            else:
                if index is not None:
                    if index == 0:
                        cond = sequence[index].event_type == "mouse-left-down"
                    elif index == -1 or index == len(sequence) - 1:
                        cond = sequence[index].event_type == "mouse-left-up"
                    else:
                        cond = sequence[index].event_type == "move"
                else:
                    time1 = sequence[0].timestamp
                    time2 = sequence[-1].timestamp
                    cond = time2 - time1 < duration_threshold

            return cond
        except Exception, e:
            raise e


class RightClick(MouseAction):
    def __init__(self, data_sequence=None):
        super(RightClick, self).__init__(data_sequence, level=1)

    def check_pattern(self, sequence, index=None, partial=False):
        super(RightClick, self).check_pattern(sequence, index)
        try:
            # TODO configurable threshold
            duration_threshold = 5.0
            if partial:
                # partial validation
                if index is not None:
                    if index == 0:
                        cond = sequence[index].event_type == "mouse-right-down"
                    else:
                        cond = sequence[index].event_type == "move"
                else:
                    raise Exception("Index must be specified in the partial validation")
            else:
                if index is not None:
                    if index == 0:
                        cond = sequence[index].event_type == "mouse-right-down"
                    elif index == -1 or index == len(sequence) - 1:
                        cond = sequence[index].event_type == "mouse-right-up"
                    else:
                        cond = sequence[index].event_type == "move"
                else:
                    time1 = sequence[0].timestamp
                    time2 = sequence[-1].timestamp
                    cond = time2 - time1 < duration_threshold

            return cond
        except Exception, e:
            raise e


class DoubleClick(MouseAction):

    def __init__(self, data_sequence=None):
        super(DoubleClick, self).__init__(data_sequence=data_sequence, level=2)

    def check_pattern(self, sequence, index=None, partial=False):
        super(DoubleClick, self).check_pattern(sequence, index)
        try:
            # TODO configurable threshold
            duration_threshold = 5.0
            if partial:
                # partial validation
                if index is not None:
                    if index == 0:
                        cond = isinstance(sequence[index], LeftClick)
                    else:
                        cond = False
                else:
                    raise Exception("Index must be specified in the partial validation")
            else:
                if index is not None:
                    if index == 0:
                        cond = isinstance(sequence[index], LeftClick)
                    elif index == -1 or index == len(sequence) - 1:
                        cond = isinstance(sequence[index], LeftClick)
                    else:
                        cond = False
                else:
                    if len(sequence) == 2:

                        edge_data1 = sequence[0].get_edge_raw_data()
                        edge_data2 = sequence[1].get_edge_raw_data()

                        first_click_end_time = edge_data1[1].timestamp
                        last_click_start_time = edge_data2[0].timestamp

                        cond = last_click_start_time - first_click_end_time < duration_threshold
                    else:
                        cond = False

            return cond
        except Exception, e:
            raise e


class DragAndDrop(MouseAction):

    def __init__(self, data_sequence=None):
        super(DragAndDrop, self).__init__(data_sequence, level=1)

    def check_pattern(self, sequence, index=None, partial=False):
        super(DragAndDrop, self).check_pattern(sequence, index)
        try:
            # TODO configurable threshold
            duration_threshold = 5.0
            if partial:
                # partial validation
                if index is not None:
                    if index == 0:
                        cond = sequence[index].event_type == "mouse-left-down"
                    else:
                        cond = sequence[index].event_type == "move"
                else:
                    raise Exception("Index must be specified in the partial validation")
            else:
                if index is not None:
                    if index == 0:
                        cond = sequence[index].event_type == "mouse-left-down"
                    elif index == -1 or index == len(sequence) - 1:
                        cond = sequence[index].event_type == "mouse-left-up"
                    else:
                        cond = sequence[index].event_type == "move"
                else:
                    time1 = sequence[0].timestamp
                    time2 = sequence[-1].timestamp
                    cond = time2 - time1 > duration_threshold

            return cond
        except Exception, e:
            raise e

    def extract_features(self):
        super(DragAndDrop, self).extract_features()

        def compute_statistics(basic_params):
            """
            compute minimum, maximum, mean, standard deviation and max-min of each list of basic params keys
            :param basic_params: dict, each element is a numpy array of numbers
            :return: output_features: dict,# key used in output features is the param name followed by
            the label of the operation made on (e.g "velocity_avg")

            """

            output_features = {}
            for kk in basic_params:
                # type check
                if isinstance(basic_params[kk], list):
                    # convert to numpy array
                    basic_param = numpy.array(basic_params[kk])
                elif isinstance(basic_params[kk], numpy.ndarray):
                    basic_param = basic_params[kk]
                else:
                    raise Exception("invalid inumpyut parameter: '%s' is not a list or a numpy array" % kk)

                # average (without considering NaN values)
                f_key = '_'.join([kk, "avg"])
                output_features[f_key] = numpy.nanmean(basic_param)
                # min (without considering NaN values)
                min_value = numpy.nanmin(basic_param)
                f_key = '_'.join([kk, "min"])
                output_features[f_key] = min_value
                # max (without considering NaN values)
                max_value = numpy.nanmax(basic_param)
                f_key = '_'.join([kk, "max"])
                output_features[f_key] = max_value
                # standard deviation
                f_key = '_'.join([kk, "std"])
                output_features[f_key] = numpy.nanstd(basic_param)
                # range, computed as max - min
                f_key = '_'.join([kk, "range"])
                output_features[f_key] = abs(max_value - min_value)
                '''
                # number of NaN values
                f_key = '_'.join([kk, "NaN"])
                output_features[f_key] = numpy.sum(numpy.isnan(basic_param))
                '''

            return output_features
        try:
            # check if raw data sequence isn't empty
            if not self.data_sequence:
                raise Warning("raw data sequence is empty, features cannot be extracted")
            else:
                # compute values between each pair of raw data
                if len(self.data_sequence) > 1:
                    bas_params = {}

                    # x coordinates
                    xs = [p.x for p in self.data_sequence]
                    # y coordinates
                    ys = [p.y for p in self.data_sequence]
                    # timestamps
                    timestamps = [p.timestamp for p in self.data_sequence]

                    bas_params["x"] = xs
                    bas_params["y"] = ys

                    deltas_x = numpy.diff(xs)
                    deltas_y = numpy.diff(ys)
                    deltas_t = numpy.diff(timestamps)

                    if 0 in deltas_t:
                        raise ZeroDivisionError("time intervals cannot be zero")

                    # velocity toward the x axis
                    velocity_x = abs(deltas_x / deltas_t)
                    # velocity toward the y axis
                    velocity_y = abs(deltas_y / deltas_t)

                    # add to basic params
                    bas_params["velocity_x"] = velocity_x
                    bas_params["velocity_y"] = velocity_y

                    # angle with x axis
                    angles = numpy.arctan2(deltas_y, deltas_x)
                    bas_params["angles"] = angles

                    # compute average, variability, min, max and range of values and use it as features
                    features = compute_statistics(bas_params)

                    # add features
                    for key in features:
                        self.add_feature(key, features[key])

        except Exception, e:
            raise e


class Move(MouseAction):
    def __init__(self, data_sequence=None):
        super(Move, self).__init__(data_sequence, level=1)

    def check_pattern(self, sequence, index=None, partial=False):
        super(Move, self).check_pattern(sequence, index)
        try:
            # TODO configurable threshold
            duration_threshold = 5.0
            if partial:
                # partial validation
                if index is not None:
                    if index == 0:
                        cond = sequence[index].event_type == "move"
                    else:
                        cond = sequence[index].event_type == "move"
                        time1 = sequence[index - 1].timestamp
                        time2 = sequence[index].timestamp
                        cond = cond and (time2 - time1 < duration_threshold)
                else:
                    raise Exception("Index must be specified in the partial validation")
            else:
                if index is not None:
                    if index == 0:
                        cond = sequence[index].event_type == "move"
                    else:
                        cond = sequence[index].event_type == "move"
                        time1 = sequence[index - 1].timestamp
                        time2 = sequence[index].timestamp
                        cond = cond and (time2 - time1 < duration_threshold)
                else:
                    cond = True
            return cond
        except Exception, e:
            raise e

    def extract_features(self):
        super(Move, self).extract_features()

        def compute_statistics(basic_params):
            """
            compute minimum, maximum, mean, standard deviation and max-min of each list of basic params keys
            :param basic_params: dict, each element is a numpy array of numbers
            :return: output_features: dict,# key used in output features is the param name followed by
            the label of the operation made on (e.g "velocity_avg")

            """

            output_features = {}
            for kk in basic_params:
                # type check
                if isinstance(basic_params[kk], list):
                    # convert to numpy array
                    basic_param = numpy.array(basic_params[kk])
                elif isinstance(basic_params[kk], numpy.ndarray):
                    basic_param = basic_params[kk]
                else:
                    raise Exception("invalid inumpyut parameter: '%s' is not a list or a numpy array" % kk)

                # average (without considering NaN values)
                f_key = '_'.join([kk, "avg"])
                output_features[f_key] = numpy.nanmean(basic_param)
                # min (without considering NaN values)
                min_value = numpy.nanmin(basic_param)
                f_key = '_'.join([kk, "min"])
                output_features[f_key] = min_value
                # max (without considering NaN values)
                max_value = numpy.nanmax(basic_param)
                f_key = '_'.join([kk, "max"])
                output_features[f_key] = max_value
                # standard deviation
                f_key = '_'.join([kk, "std"])
                output_features[f_key] = numpy.nanstd(basic_param)
                # range, computed as max - min
                f_key = '_'.join([kk, "range"])
                output_features[f_key] = abs(max_value - min_value)
                '''
                # number of NaN values
                f_key = '_'.join([kk, "NaN"])
                output_features[f_key] = numpy.sum(numpy.isnan(basic_param))
                '''

            return output_features
        try:
            # check if raw data sequence isn't empty
            if not self.data_sequence:
                raise Warning("raw data sequence is empty, features cannot be extracted")
            else:
                # compute values between each pair of raw data
                if len(self.data_sequence) > 1:
                    bas_params = {}

                    # x coordinates
                    xs = [p.x for p in self.data_sequence]
                    # y coordinates
                    ys = [p.y for p in self.data_sequence]
                    # timestamps
                    timestamps = [p.timestamp for p in self.data_sequence]

                    deltas_x = numpy.diff(xs)
                    deltas_y = numpy.diff(ys)
                    deltas_t = numpy.diff(timestamps)

                    if 0 in deltas_t:
                        raise ZeroDivisionError("time intervals cannot be zero")

                    # velocity toward the x axis
                    velocity_x = abs(deltas_x / deltas_t)
                    # velocity toward the y axis
                    velocity_y = abs(deltas_y / deltas_t)

                    # add to basic params
                    bas_params["velocity_x"] = velocity_x
                    bas_params["velocity_y"] = velocity_y

                    # angle with x axis
                    angles = numpy.arctan2(deltas_y, deltas_x)
                    bas_params["angles"] = angles

                    # compute average, variability, min, max and range of values and use it as features
                    features = compute_statistics(bas_params)

                    # add features
                    for key in features:
                        self.add_feature(key, features[key])

        except Exception, e:
            raise e


class MouseActionExtractor(feba.ActionExtractor):
    def __init__(self):
        super(MouseActionExtractor, self).__init__([LeftClick, DragAndDrop, Move, RightClick, DoubleClick])


class MouseFeatureExtractor(feba.FeaturesExtractor):
    def __init__(self, actions):
        super(MouseFeatureExtractor, self).__init__(actions)
