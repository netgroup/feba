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

import warlock
import numpy


class RawDataFactory(object):
    def __init__(self, schema):
        super(RawDataFactory, self).__init__()
        self.schema = schema

    def get_raw_data_class(self):
        try:
            RawData=warlock.model_factory(self.schema)
            return RawData
        except Exception, e:
            raise e


class FeaturesHandler(object):
    def __init__(self):
        super(FeaturesHandler, self).__init__()

    def add_feature(self, name, value):
        setattr(self, name, value)

    def get_feature_names(self):
        try:
            feature_names = [key for (key, value) in sorted(self.__dict__.items()) if not key.startswith("_")]
            return feature_names
        except Exception, e:
            raise e

    def get_features_values(self):
        try:
            feature_values = [value for (key, value) in sorted(self.__dict__.items()) if not key.startswith("_")]
            return feature_values
        except Exception, e:
            raise e

    def get_features(self, prefix=None):
        """

        :param prefix: string will be appended to all feature names
        :return: dict, keys are feature names, values are feature values
        """
        try:
            # features = {key: value for (key, value) in self.__dict__ if not key.startswith("_")}
            features = self.__dict__
            new_features = {}
            if prefix is not None:
                # rename and add to features dict
                if isinstance(prefix, str):
                    for old_key in features:
                        new_key = prefix + old_key
                        new_features[new_key] = features[old_key]
                else:
                    raise Warning("prefix must be a string")
            return new_features
        except Exception, e:
            raise e


class Action(object):
    """
    Abstract concept of action. It has a generic sequence of basic elements
    and a method to extract features from those. Features are put into a FeaturesHandler object
    """

    def __init__(self, data_sequence=None, level=None, schema=None):
        """

        :param data_sequence:
        :param level: Specifies the hierarchy level of the action. Level 1 indicates that data sequence is a list of raw data.
        Level k indicates that data sequence is made up of actions whose max level can be k.
        :param schema: json schema specifies the structure of raw data
        :return:
        """
        super(Action, self).__init__()
        try:
            if schema is not None:
                # set the factory to retrieve the Raw data class based on the json schema given
                self.raw_data_factory = RawDataFactory(schema=schema)
            if data_sequence is not None:
                self.set_data_sequence(data_sequence)
            else:
                if level is None:
                    # TODO implicit set of the level even if the sequence is not given
                    raise AttributeError("If data sequence is not specified, the level must be given")
                else:
                    self.level = level

            self.features_handler = FeaturesHandler()

            # used by the add raw data method
            self.tmp_data_buffer = []
        except Exception, e:
            raise e

    def validate(self, sequence):
        """
        This method check if the raw data sequence is in compliance with the pattern that define the action.
        :param sequence:
        :return: True if the validation successes. False otherwise
        """
        try:

            partial_validation = True
            # check the condition over each raw data point and return true if the sequence is partially valid
            for index in range(len(sequence)):
                partial_validation = partial_validation and self.check_pattern(sequence, index=index, partial=True)

            complete_validation = True
            # check the condition over each raw data point and return true if the sequence is completely valid
            for index in range(len(sequence)):
                complete_validation = complete_validation and self.check_pattern(sequence, index=index)
            complete_validation = complete_validation and self.check_pattern(sequence)

            # check the condition over each raw data point without consider the last point
            complete_validation_wl = True
            # check the condition over each raw data point except the last one and return true if the sequence is partially valid
            sub_sequence = [p for p in sequence[:-1]]
            for index in range(len(sub_sequence)):
                complete_validation_wl = complete_validation_wl and self.check_pattern(sub_sequence, index=index)
            complete_validation_wl = complete_validation_wl and self.check_pattern(sub_sequence)

            """
            # validate the level
            if self.level > 1:
                condition = condition and self.level == max(ac.level for ac in sequence) + 1
            """

            validation_response = {
                "partial_validation": partial_validation,
                "complete_validation": complete_validation,
                "complete_validation_wl": complete_validation_wl
            }

            return validation_response

        except Exception, e:
            raise e

    def check_pattern(self, sequence, index=None, partial=False):
        """
        This method should specify the condition that each single data along with the whole sequence must comply with
        :param partial:
        :param sequence: the sequence of data to be checked
        :param index: the index of the single data to be check. If None, just the condition on the whole sequece are checked
        :return:
        """
        return True

    def add_data(self, rd):
        """
        This method add a single raw data point to a buffer and check if the condition that defines the action is valid.
        :param rd: dict with raw data attributes or a single action object
        :return: string. A status code:
            'ready' -> sequence is valid to be added
            'ready_wl' -> sequence is valid but without the last point
            'wait' -> sequence is valid until now, but not yet to be added
            'invalid' -> sequence is not valid and won't be added
        """
        try:
            # RESPONSE STATUS:
            # ready -> sequence is valid to be added
            # ready_wl -> sequence is valid but without the last point
            # wait -> sequence is valid until now, but not yet to be added
            # invalid -> sequence is not valid and won't be added

            RESPONSE_STATUS_READY = "ready"
            RESPONSE_STATUS_WAIT = "wait"
            RESPONSE_STATUS_READY_WL = "ready_wl"
            RESPONSE_STATUS_INVALID = "invalid"

            buffer_sequence_status = RESPONSE_STATUS_WAIT

            if self.level == 1:
                if getattr(self, "raw_data_factory"):
                    # add the point to the tmp buffer and than check the condition
                    RD = self.raw_data_factory.get_raw_data_class()
                    rd_point = RD(rd)
                    self.tmp_data_buffer.append(rd_point)
                else:
                    raise Exception("A raw data json schema is needed")
            else:
                self.tmp_data_buffer.append(rd)

            if len(self.tmp_data_buffer) > 1:

                validation_resp = self.validate(sequence=self.tmp_data_buffer)
                if not validation_resp["partial_validation"] and not validation_resp["complete_validation"] and not validation_resp["complete_validation_wl"]:
                    buffer_sequence_status = RESPONSE_STATUS_INVALID
                elif validation_resp["partial_validation"]:
                    buffer_sequence_status = RESPONSE_STATUS_WAIT
                elif validation_resp["complete_validation"]:
                    buffer_sequence_status = RESPONSE_STATUS_READY
                elif validation_resp["complete_validation_wl"]:
                    buffer_sequence_status = RESPONSE_STATUS_READY_WL
            else:
                # check just if the first point can be valid
                partial_validation = self.check_pattern(sequence=self.tmp_data_buffer, index=0, partial=True)
                buffer_sequence_status = RESPONSE_STATUS_WAIT if partial_validation else RESPONSE_STATUS_INVALID

            return buffer_sequence_status

        except Exception, e:
            raise e

    def reset_buffer(self):
        """
        Empty the tmp buffer used to collect raw data points by the add method
        :return:
        """
        del self.tmp_data_buffer[:]

    def extract_features(self):
        """
        Extract features from the data_sequence and put them into the features_handler
        using the add features_method
        :return:
        """

        def _compute_statistics(basic_params):
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
                f_key = '_'.join([kk, "delta_avg"])
                output_features[f_key] = numpy.nanmean(basic_param)
                # min (without considering NaN values)
                min_value = abs(numpy.nanmin(basic_param))
                f_key = '_'.join([kk, "delta_min"])
                output_features[f_key] = min_value
                # max (without considering NaN values)
                max_value = abs(numpy.nanmax(basic_param))
                f_key = '_'.join([kk, "delta_max"])
                output_features[f_key] = max_value
                # standard deviation
                f_key = '_'.join([kk, "delta_std"])
                output_features[f_key] = numpy.nanstd(basic_param)

            return output_features

        try:
            # check if raw data sequence isn't empty
            if not self.data_sequence:
                raise Warning("raw data sequence is empty, features cannot be extracted")
            if len(self.data_sequence) > 1:
                attr_names = self.get_raw_data_attributes()
                if self.level == 1:
                    # lower level action
                    # iterate over the attributes
                    basic_par = {}
                    for key in attr_names:
                        # the first raw datais used as reference to identity the type of attribute key
                        if isinstance(getattr(self.data_sequence[0], key), (int, float, long)):
                            attr = [getattr(p, key) for p in self.data_sequence]
                            deltas_attr = numpy.diff(attr)

                            basic_par[key] = deltas_attr

                            # range, computed as last value - first value
                            range_key = '_'.join([key, "range"])
                            edge_data = self.get_edge_raw_data()
                            range_value = abs(getattr(edge_data[1], key) - getattr(edge_data[0], key))
                            self.add_feature(range_key, range_value)

                    if len(self.data_sequence) > 2:
                        # compute average, variability, min, max and range of values and use it as features
                        basic_features = _compute_statistics(basic_params=basic_par)

                        # add features
                        for bf in basic_features:
                            self.add_feature(bf, basic_features[bf])

                else:
                    basic_par = {}
                    for attr_n in attr_names:
                        # last edge data used to check the type. Can be improved
                        if isinstance(getattr(self.last_edge_data, attr_n), (int, float, long)):

                            # Compute the difference between the edge values of each actions in the sequence
                            first_edges = [getattr(p.get_edge_raw_data()[0], attr_n) for p in self.data_sequence]
                            last_edges = [getattr(p.get_edge_raw_data()[1], attr_n) for p in self.data_sequence]
                            # last edge value of action N - first edge value of action N-1
                            deltas_edge_attr = [fe - le for (fe, le) in zip(first_edges[1:], last_edges[:len(last_edges) - 1])]

                            basic_par['_'.join([attr_n, "cross"])] = deltas_edge_attr if len(deltas_edge_attr) > 1 else deltas_edge_attr[0]

                            # Compute the range
                            # range, computed as last value - first value
                            range_key = '_'.join([attr_n, "range"])
                            edge_data = self.get_edge_raw_data()
                            range_value = abs(getattr(edge_data[1], attr_n) - getattr(edge_data[0], attr_n))
                            # add the feature
                            self.add_feature(range_key, range_value)

                    if len(self.data_sequence) > 2:
                        # compute average, variability, min, max and range of values and use it as features
                        basic_features = _compute_statistics(basic_params=basic_par)
                    else:
                        # take just the values
                        basic_features = basic_par

                    # add features
                    for bf in basic_features:
                        self.add_feature(bf, basic_features[bf])

                    # call the extract feature method of each lower level action
                    for lower_ac in self.data_sequence:
                        lower_ac.extract_features()

        except Exception, e:
            raise e

    def get_raw_data_attributes(self):
        """
        :return: list of the names of raw data attributes at lower level
        """
        try:
            attr = []
            if hasattr(self, "level") and hasattr(self, "data_sequence"):
                if self.level == 1:
                    # TODO better retrieving of attribute names
                    #  __original is the internal variable used by warlock lib
                    for key, value in self.data_sequence[0].__original__.iteritems():
                        attr.append(key)
                else:
                    attr = self.data_sequence[0].get_raw_data_attributes()
            else:
                raise Exception("attributes do not exist")
            return attr
        except Exception, e:
            raise e

    def add_feature(self, name, value):
        self.features_handler.add_feature(name=name, value=value)

    def get_features(self, prefix=None):
        """

        :param prefix:
        :return:
        """
        ###
        # feature name convention:
        # "l[action-level]_[action index in the sequence]_[feature name]
        # e.g l2_1_F1
        #
        ###

        if prefix is None:
            # root prefix
            prefix = str(0)

        features = self.features_handler.get_features(prefix=prefix + "_")

        if self.level != 1:
            for rds_idx in range(len(self.data_sequence)):

                current_prefix = prefix + "." + str(rds_idx)
                lower_lev_features = self.data_sequence[rds_idx].get_features(prefix=current_prefix)

                # add to features dict
                for f in lower_lev_features:
                    features[f] = lower_lev_features[f]

        return features

    def get_edge_raw_data(self):
        """
        Get the two edge raw data point of the action
        :return: list of two elements containing the first and last edge raw data ordered
        """
        try:
            edge_data = []
            if hasattr(self, "first_edge_data") and hasattr(self, "last_edge_data"):
                edge_data.append(self.first_edge_data)
                edge_data.append(self.last_edge_data)
            return edge_data
        except Exception, e:
            raise e

    def set_data_sequence(self, data_sequence):
        try:
            if isinstance(data_sequence, list):
                # if raw data sequence is a list, check if it is a list of Action instances
                if all(isinstance(item, Action) for item in data_sequence):
                    setattr(self, "data_sequence", data_sequence)
                    # the level is set to one more than the max level of the actions in the list
                    self.level = max(ac.level for ac in self.data_sequence) + 1

                    # used to compute the inter-actions features
                    first_edge_data = self.data_sequence[0].get_edge_raw_data()[0]
                    last_edge_data = self.data_sequence[-1].get_edge_raw_data()[1]
                    setattr(self, "first_edge_data", first_edge_data)
                    setattr(self, "last_edge_data", last_edge_data)

                elif self.raw_data_factory is not None:
                    RD = self.raw_data_factory.get_raw_data_class()
                    rd_seq = []
                    for rd in data_sequence:
                        rd_seq.append(RD(rd))
                    setattr(self, "data_sequence", rd_seq)

                    # lower level action. It holds raw data
                    self.level = 1

                    # used to compute the inter-actions features
                    first_edge_data = self.data_sequence[0]
                    last_edge_data = self.data_sequence[-1]
                    setattr(self, "first_edge_data", first_edge_data)
                    setattr(self, "last_edge_data", last_edge_data)
                else:
                    raise TypeError("raw data sequence must contain a list of raw data or a list of actions")

                # check if the action is in compliance with the action pattern
                assert (self.validate(self.data_sequence))["complete_validation"], "Validation over the raw data sequence pattern failed!"

            else:
                raise TypeError("raw data sequence must be a a list")
        except Exception, e:
            raise e

    def set_from_buffer(self, last_point=True):
        if last_point:
            sequence = [s for s in self.tmp_data_buffer]
            self.set_data_sequence(sequence)
        else:
            sequence = [s for s in self.tmp_data_buffer[:-1]]
            self.set_data_sequence(sequence)

        self.reset_buffer()


class ActionExtractor(object):
    def __init__(self, action_classes):
        """

        :param action_classes: List of class objects. Each class must inherit the Action class
        :return:
        """
        super(ActionExtractor, self).__init__()
        self.action_classes = action_classes

    def extract_actions(self, raw_data_stream):
        """
        Extract a list of actions from a list of raw data according to the action pattern.
        :param raw_data_stream: list of raw data
        :return: list of actions
        """
        try:
            if not raw_data_stream:
                raise IOError("raw data stream cannot be empty")

            # Instance each Action class and call the add raw data method
            # the class that return True is the right one
            output_actions = []

            action_instances = self.get_action_instances()
            max_level = max([ac.level for ac in action_instances])

            # firstly extract level 1 actions from raw data stream.
            # than proceed extracting higher level actions from a sequence of lower level ones
            for level in range(max_level):
                raw_data_stream = self._extract_actions_by_level(sequence=raw_data_stream, level=level+1)
                output_actions = raw_data_stream if not not raw_data_stream else output_actions

            return output_actions
        except Exception, e:
            raise e

    def get_action_instances(self, level=None):
        """

        :param level:
        :return:
        """
        try:
            all_action_instances = [ac() for ac in self.action_classes]
            if level is not None:
                action_instances = [a for a in all_action_instances if a.level == level]
                return action_instances

            return all_action_instances

        except Exception, e:
            raise e

    def _extract_actions_by_level(self, sequence, level):
        """
        Extract actions of the specified level from the sequence
        :param sequence: list of raw data or list of actions
        :param level: level of the action to be extracted. If None, the sequence is
        :return:
        """

        try:
            # TODO improve status handling
            STATUS_WAIT = "wait"
            STATUS_READY = "ready"
            STATUS_READY_WL = "ready_wl"
            STATUS_INVALID = "invalid"

            output_actions = []
            if not not sequence:
                action_instances = self.get_action_instances(level=level)

                # used to record the data (lower level actions) in case of level > 2
                high_level_action_buffer = []

                for data in sequence:
                    try:
                        if level > 1:
                            # store if no action is detected, to not modify the output actions (that will contain the
                            # lower level actions given as inumpyut until now)
                            high_level_action_buffer.append(data)

                        buffer_status = []
                        for idx in range(len(action_instances)):
                            buffer_status.append(action_instances[idx].add_data(data))

                        if STATUS_READY in buffer_status:
                            ready_idx = buffer_status.index(STATUS_READY)
                            # initialize the action with the sequence
                            action_instances[ready_idx].set_from_buffer()
                            # add the action to output list
                            output_actions.append(action_instances[ready_idx])
                            if level > 1:
                                # reset higher level action buffer
                                del high_level_action_buffer[:]
                            # create new instances
                            # TODO optimization
                            del action_instances[:]
                            action_instances = self.get_action_instances(level=level)
                        elif STATUS_READY_WL in buffer_status:
                            ready_idx = buffer_status.index(STATUS_READY_WL)
                            # initialize the action with the sequence without the last point
                            action_instances[ready_idx].set_from_buffer(last_point=False)
                            # add the action to output list
                            output_actions.append(action_instances[ready_idx])
                            if level > 1:
                                # reset higher level action buffer
                                del high_level_action_buffer[:]
                            # create new instances
                            # TODO optimization
                            del action_instances[:]
                            action_instances = self.get_action_instances(level=level)
                            # add the last point again to new instances
                            for act in action_instances:
                                act.add_data(data)
                        else:
                            # remove the invalid actions
                            action_instances = [action_instances[i] for i in range(len(action_instances)) if buffer_status[i] != STATUS_INVALID]
                            if not action_instances:
                                # it means that there aren't action instances available, so add the current data
                                # (should be a level 2 action at least, not a raw data) and go further
                                if level > 1:
                                    # do not remove the current low level action
                                    output_actions.extend(high_level_action_buffer)
                                    # reset higher level action buffer
                                    del high_level_action_buffer[:]
                                    # reset action instances
                                    action_instances = self.get_action_instances(level=level)

                    except Exception, e:
                        print e

            return output_actions
        except Exception, e:
            raise e


class FeaturesExtractor(object):
    def __init__(self, actions):
        super(FeaturesExtractor, self).__init__()
        self.actions = actions
        self.datasets = []

    def extract_features(self):
        """
        recursively call the extract_features method of each action in self.actions. A dataset for each action type is
        added to sel.datasets. Each dataset contains a matrix of features values along with a list of features names.
        :return:
        """
        try:
            # call the extract features method of each action
            for ac in self.actions:
                ac.extract_features()

                # string that identifies the action (class name)
                action_label = ac.__class__.__name__
                # index of dataset with this action label, if it exists
                idx = self._get_dataset_index(action_label)
                # check if the dataset for this action already exists
                if idx is None:
                    # dataset doesn't exist, create it.
                    # f_names = [fn for fn in ac.get_feature()]
                    f_names = [key for (key, value) in sorted(ac.get_features().items())]
                    if not not f_names:
                        dsh = DatasetHandler(label=action_label, features_names=f_names)
                        f_values = [value for (key, value) in sorted(ac.get_features().items())]
                        dsh.add_data(f_values)
                        self.datasets.append(dsh)
                else:
                    # dataset already exists, add data to it
                    f_values = [value for (key, value) in sorted(ac.get_features().items())]
                    self.datasets[idx].add_data(f_values)

        except Exception, e:
            raise e

    def _get_dataset_index(self, label):
        """
        :param label: string
        :return: the index of the dataset element in self.dataset whose label is equal to label.
                None if dataset doesn't exist
        """
        try:
            check = None
            for dsi in range(len(self.datasets)):
                if self.datasets[dsi].get_label() == label:
                    check = dsi
                    break

            return check
        except Exception, e:
            raise e


class DatasetHandler(object):
    def __init__(self, label, features_names):
        super(DatasetHandler, self).__init__()
        self.label = label
        self.data = []
        self.features_names = features_names

    def add_data(self, data):
        """

        :param data: list or list of lists. Each list has features values as elements.
        :return:
        """
        try:
            self.data.append(data)
        except Exception, e:
            raise e

    def get_feature_names(self):
        return self.features_names

    def get_data(self, clean=False):
        """

        :param clean: if True, it removes empty lists from data before returning it
        :return:data
        """
        try:
            if not clean:
                return self.data
            else:
                data_cleaned = [z for z in self.data if z != []]
                return data_cleaned
        except Exception, e:
            raise e

    def get_label(self):
        return self.label
