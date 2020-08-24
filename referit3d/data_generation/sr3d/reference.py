import random
import matplotlib
import numpy as np
from collections import defaultdict
from shapely.geometry import MultiPoint

from referit3d.in_out.scannet_scan import ScannetScan
from referit3d.in_out.three_d_object import ThreeDObject


class Reference(object):
    """
    Represents a synthetic spatial reference
    """

    def __init__(self, target: ThreeDObject, anchor: ThreeDObject, reference_type: str, second_anchor=None):
        """
        @param target: the target object
        @param anchor: the anchor object
        @param second_anchor: ThreeDObject the secon anchor (in case of the between references)
        @param reference_type: e.g. 'farthest'
        """
        self.target = target
        self.anchor = anchor
        self.second_anchor = second_anchor
        self.type = reference_type

    def distractors(self) -> list:
        """
        Same 'type' as the self.target in its scan but different object(s)

        @return: list of the distractors ThreeDObjects
        """
        res = []
        scan = self.scan()

        for o in scan.three_d_objects:
            if o.instance_label == self.target.instance_label and \
                    o.object_id != self.target.object_id:
                res.append(o)

        return res

    def anchors(self) -> list:
        """
        Anchors of the reference object
        @return: list of anchors
        """
        ret = [self.anchor]
        if self.second_anchor is not None:
            ret.append(self.second_anchor)
        return ret

    def clutter(self) -> list:
        """
        Different 'type' from self.target or anchor(s) in their scan

        @return: list of the clutter ThreeDObjects
        """
        ret = []
        scan = self.scan()

        related_instances = {self.target.instance_label}
        for anchor in self.anchors():
            related_instances.add(anchor.instance_label)

        for o in scan.three_d_objects:
            if o.instance_label not in related_instances:
                ret.append(o)

        return ret

    def scan(self) -> ScannetScan:
        assert (self.target.scan == self.anchor.scan)
        return self.target.scan

    def context_size(self) -> int:
        """count: target + distractors"""
        return len(self.distractors()) + 1

    def __str__(self, verbose=True):
        res = '{}: {}({}), {}({})'.format(self.type, self.target.instance_label, self.target.object_id,
                                          self.anchor.instance_label, self.anchor.object_id)
        if self.second_anchor:
            res += ' {}({})'.format(self.second_anchor.instance_label, self.second_anchor.object_id)
        if verbose:
            res = '{}: '.format(self.scan().scan_id) + res

        return res

    def plot(self, subsample=None, fig=None, valid_instance_types=None) -> matplotlib.pyplot.figure:
        """
        Plot the target with its anchor(s) and the other objects in the scan
        @param subsample: The number of points to be sampled from the scan
         point cloud
        @param fig: matplotlib.pyplot.fig
        @param valid_instance_types: The instances to be plotted
        @return: matplotlib.pyplot.fig with plotted reference objects
        """
        if fig is None:
            fig = self.scan().plot(subsample, valid_instance_types)

        axis = fig.axes[0]
        self.target.get_bbox().plot(axis=axis, c='g')

        for anchor in self.anchors():
            anchor.get_bbox().plot(axis=axis, c='b')

        for d in self.distractors():
            d.get_bbox().plot(axis=axis, c='r')

        return fig

    @staticmethod
    def valid_anchors(scan: ScannetScan, must_be_unique=True, valid_instances=None, has_front=None,
                      group_by_label=False,
                      too_hard=None, exclude_instances=None):
        """
        Returns a list of three_d object if not group_by_label, otherwise returns dict.

        @param scan: A ScanNet scan
        @param must_be_unique: The anchors should be unique (have multiplicty of 1) in the scan
        @param valid_instances: A list of possible instance classes to choose from (None means no such constraint)
        @param has_front: bool, Choose the anchor objects that have the front property (needed in the allocentric references)
        @param group_by_label: bool, if true return a dict not a list where a key is the anchor instance type and
        the value is a list of objects
        @param too_hard: int, if provided choose the objects that have a multiplicity of at most too_hard
        @param exclude_instances: A list of forbidden instance classes to avoid (None means no such constraint)
        """
        result = defaultdict(list)

        occurrence = scan.instance_occurrences()
        for obj in scan.three_d_objects:
            obj_instance_label = obj.instance_label
            if must_be_unique:
                if occurrence[obj_instance_label] != 1:
                    continue
            if valid_instances is not None:
                if obj_instance_label not in valid_instances:
                    continue

            if too_hard is not None:
                if occurrence[obj_instance_label] > too_hard:
                    continue

            if has_front is not None:
                if obj.has_front_direction != has_front:
                    continue

            if exclude_instances is not None:
                if obj_instance_label in exclude_instances:
                    continue

            result[obj.instance_label].append(obj)

        if group_by_label:
            return result

        ret = []
        for k, v in result.items():
            ret.extend(v)

        return ret

    @staticmethod
    def valid_targets(scan, must_be_multiple=True, valid_instances=None, group_by_label=False, too_hard=None):
        """
        Returns a list of three_d object if not group_by_label, otherwise returns dict.

        @param scan: A ScanNet scan
        @param must_be_multiple: The anchors should not be unique (have multiplicty greater than 1) in the scan
        @param valid_instances: A list of possible instance classes to choose from (None means no such constraint)
        @param group_by_label: bool, if true return a dict not a list where a key is the anchor instance type and
        the value is a list of objects
        @param too_hard: int, if provided choose the objects that have a multiplicity of at most too_hard
        """
        result = defaultdict(list)

        occurrence = scan.instance_occurrences()
        for obj in scan.three_d_objects:
            obj_instance_label = obj.instance_label

            if must_be_multiple:
                if occurrence[obj_instance_label] < 2:
                    continue

            if too_hard is not None:
                if occurrence[obj_instance_label] > too_hard:
                    continue

            if valid_instances is not None:
                if obj_instance_label not in valid_instances:
                    continue
            result[obj.instance_label].append(obj)

        if group_by_label:
            return result

        ret = []
        for k, v in result.items():
            ret.extend(v)

        return ret

    @staticmethod
    def force_uniqueness(all_refs) -> list:
        """
        If the list contains two or more identical references then it discards them.
        @param all_refs: list with Reference objects

        @return: list with all Reference objects that are unique in the list
        """
        counter = defaultdict(list)
        for ref in all_refs:
            if ref.second_anchor is not None:
                key = [ref.target.instance_label, ref.anchor.instance_label, ref.second_anchor.instance_label]
            else:
                key = [ref.target.instance_label, ref.anchor.instance_label]
            if ref.type in ['supported-by', 'supporting', 'above',
                                      'below']:  # To reduce the effect of the bad annotations of the bounding boxes
                key.append('vertical')
            else:
                key.append(ref.type)
            counter[tuple(key)].append(ref)

        res = []
        for key, val in counter.items():
            if len(val) == 1:
                res.extend(counter[key])
        return res

    def satisfies_template_conditions(self, conditions, t_set):
        def get_angle(x, y):
            lx = np.sqrt(x.dot(x))
            ly = np.sqrt(y.dot(y))
            cos_angle = x.dot(y) / (lx * ly)
            angle = np.arccos(cos_angle)
            return np.rad2deg(angle)

        def check_in_center():
            assert self.type == 'between'
            anc_a = self.anchor.get_bbox(axis_aligned=True)
            anc_b = self.second_anchor.get_bbox(axis_aligned=True)
            t = self.target.get_bbox(axis_aligned=True)

            anc_a_center = np.array([anc_a.cx, anc_a.cy])
            anc_b_center = np.array([anc_b.cx, anc_b.cy])
            t_center = np.array([t.cx, t.cy])

            anc_a_anc_b = anc_b_center - anc_a_center
            anc_a_t = t_center - anc_a_center
            angle_a = get_angle(anc_a_anc_b, anc_a_t)

            anc_b_anc_a = anc_a_center - anc_b_center
            anc_b_t = t_center - anc_b_center
            angle_b = get_angle(anc_b_anc_a, anc_b_t)

            return 0 < angle_a < 30 and 0 < angle_b < 30

        def check_next_to():
            if self.type != 'closest':
                return False

            # if not close to each other, return false
            if self.target.distance_from_other_object(self.anchor, optimized=True) > 1.2:
                return False

            # Get the anchor and the target convex hull
            anchor_face = self.anchor.get_bbox().z_faces()[0]  # Top face
            anchor_points = tuple(map(tuple, anchor_face[:, :2]))  # x, y coordinates

            target_z_face = self.target.get_bbox().z_faces()[0]
            target_points = tuple(map(tuple, target_z_face[:, :2]))

            anchor_target_polygon = MultiPoint(anchor_points + target_points).convex_hull

            # Loop over the scan object and check no one intersects this convex hull
            for o in self.scan().three_d_objects:
                if o.instance_label in ['wall', 'floor']:
                    continue

                if o.object_id in [self.anchor.object_id, self.target.object_id]:
                    continue

                o_z_face = o.get_bbox().z_faces()[0]
                o_points = tuple(map(tuple, o_z_face[:, :2]))
                o_polygon = MultiPoint(o_points).convex_hull

                # if it is found in the area between the two objects, return false
                if o_polygon.intersection(anchor_target_polygon).area / o_polygon.area > 0.5:
                    return False

            return True

        res = True
        for cond in conditions:
            if cond == "NEXT_TO":
                res &= check_next_to()
            elif cond == "NOT_NEXT_TO":
                res &= not check_next_to()
            elif cond == "NOT_FRONT_RELATION":
                res &= (self.get_reference_type(coarse=False) != 'front')
            elif cond == "FRONT_RELATION":
                res &= self.get_reference_type(coarse=False) == 'front'
            elif cond == "NOT_IN_CENTER":
                res &= not check_in_center()
            elif cond == "IN_CENTER":
                res &= check_in_center()
            elif cond == "ONLY_INCLUDED_INSTANCES":
                res &= self.anchor.instance_label in t_set["instances"]
            elif cond == "SUPPORTING_RELATION":
                res &= self.get_reference_type(coarse=False) == 'supporting'

        return res

    @staticmethod
    def sample_from_template_set(reference, template_set, template_dict):
        """

        @param reference:
        @param template_set:
        @param template_dict:
        @return:
        """
        assert len(template_set["sentences"]) > 0
        utterances = []

        # Pick one sentence from the set at random
        sentence = np.random.choice(template_set["sentences"], 1, replace=False)[0]

        # Replace the placeholders
        target_verb_index = template_dict['verb_index'][reference.target.instance_label]
        anchor_verb_index = template_dict['verb_index'][reference.anchor.instance_label]
        target_verb = template_dict['verbs'][target_verb_index]
        anchor_object_pronoun = template_dict['object_pronouns'][anchor_verb_index]
        anchor_demonstrative = template_dict['demonstratives'][anchor_verb_index]
        target_verb_to_have = template_dict['verb_to_have'][target_verb_index]
        target_possessive_pronoun = template_dict['possessive_pronouns'][target_verb_index]
        target_object_pronoun = template_dict['object_pronouns'][target_verb_index]

        sentence = sentence.replace('%target%', reference.target.instance_label)
        sentence = sentence.replace('%target_verb%', target_verb)
        sentence = sentence.replace('%anchor%', reference.anchor.instance_label)
        sentence = sentence.replace('%anchor_object_pronoun%', anchor_object_pronoun)
        sentence = sentence.replace('%anchor_demonstrative%', anchor_demonstrative)
        sentence = sentence.replace("%target_verb_to_have%", target_verb_to_have)
        sentence = sentence.replace("%target_possessive_pronoun%", target_possessive_pronoun)
        sentence = sentence.replace("%target_object_pronoun%", target_object_pronoun)

        if reference.get_reference_type(coarse=True) == 'between':
            utterance = sentence.replace('%anchor_2%', reference.second_anchor.instance_label)
            utterances.append(utterance)
        else:
            # Create utterances by replacing the sentence with each the relation
            relations = template_dict["relations"][reference.get_reference_type(coarse=False)].split('/')
            for rel in relations:
                utterances.append(sentence.replace('%relation%', rel))

        return utterances

    @staticmethod
    def to_human_language(reference, template_dict: dict, n_utterances: int) -> list:
        """
        Sample human-like utterances for a synthetic spatial reference
        @param reference: Reference
        @param template_dict: The dict containing the look up template information
        @param n_utterances: The number of utterances to sample for this reference

        @return: a list of human-like utterances (str) describing the spatial reference
        """
        utterances = []
        assert n_utterances >= 1

        # Get all templates sets according to the reference type
        template_sets = template_dict["templates"][reference.get_reference_type(coarse=True)]
        random.shuffle(template_sets)

        # Loop over each set that this reference satisfies its conditions and sample one of its sentences
        for t_set in template_sets:
            # Check whether the conditions are satisfied or not
            conditions = t_set["conditions"]
            if reference.satisfies_template_conditions(conditions, t_set):
                utterances.extend(Reference.sample_from_template_set(reference, t_set, template_dict))

            if len(utterances) == n_utterances:
                return utterances

        return utterances

    def get_reference_type(self, coarse=False) -> str:
        """
        Get the reference type.

        @param coarse: return the coarse type (e.g. horizontal) if true,
        the fine-grained type (e.g. closest) otherwise
        """
        if not coarse:
            return self.type
        if self.type in ['left', 'right', 'front', 'back']:
            return "allocentric"
        if self.type in ['above', 'below']:
            return "vertical"
        if self.type in ['farthest', 'closest']:
            return "horizontal"
        if self.type in ['supported-by', 'supporting']:
            return "support"
        if self.type == 'between':
            return "between"
        else:
            raise ValueError

    def serialize(self):
        """
        Serialize the reference into a dict with the essential information so we can
        save it to disk with minimal size requirements.
        """
        # Get the target, anchors object ids and classes
        target_object_id = self.target.object_id
        anchor_object_id = self.anchor.object_id

        if self.second_anchor is not None:
            second_anchor_object_id = self.second_anchor.object_id
        else:
            second_anchor_object_id = None

        # Get the reference type
        reference_type = self.type

        # Get the corresponding scan
        scan_id = self.scan().scan_id

        return {
            'scan_id': scan_id,
            'reference_type': reference_type,
            'target_object_id': target_object_id,
            'anchor_object_id': anchor_object_id,
            'second_anchor_object_id': second_anchor_object_id
        }

    @staticmethod
    def deserialize(all_scans, serialized_ref):
        """
        Create a reference out of a serialized one.
        @param all_scans: dictionary of all scans, scan_id: scannet scan object
        @param serialized_ref: the serialized reference

        @return: Reference object
        """
        # Get the corresponding scan
        scan_id = serialized_ref['scan_id']
        scan = all_scans[scan_id]

        # Get the reference information
        reference_type = serialized_ref['reference_type']
        target_object_id = serialized_ref['target_object_id']
        anchor_object_id = serialized_ref['anchor_object_id']
        second_anchor_object_id = serialized_ref['second_anchor_object_id']

        # Get the referenced objects
        second_anchor = None
        target = None
        anchor = None
        for o in scan.three_d_objects:
            assert o.object_id >= 0
            if o.object_id == target_object_id:
                target = o
            elif o.object_id == anchor_object_id:
                anchor = o
            elif o.object_id == second_anchor_object_id:
                second_anchor = o

        assert (target is not None and anchor is not None)
        if second_anchor_object_id is not None:
            assert second_anchor is not None

        return Reference(target=target,
                         anchor=anchor,
                         reference_type=reference_type,
                         second_anchor=second_anchor)

    @staticmethod
    def deserialize_all(all_scans, serialized_refs):
        """
        Create a reference out of a serialized one.
        @param all_scans: list carrying ScannetScan
        @param serialized_refs: list with serialized references

        @return: list with Reference objects
        """
        all_scans_dict = {s.scan_id: all_scans[i] for i, s in enumerate(all_scans)}
        res = []
        for ref in serialized_refs:
            res.append(Reference.deserialize(all_scans_dict, ref))
        return res
