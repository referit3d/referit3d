import yaml
import numpy as np
from enum import Enum, unique
from collections import defaultdict
from shapely.geometry import MultiPoint, Point

from .. import ReferenceGenerator, Reference


@unique
class OrientedSections(Enum):
    front = 0
    right = 1
    back = 2
    left = 3
    grey_area = 4


class AllocentricGenerator(ReferenceGenerator):
    def __init__(self, verbose=True):
        super().__init__(verbose)

        self.type = 'allocentric'

        # Read the hyper-parameter
        with open('allocentric/hp.yml') as fin:
            self.hp = yaml.full_load(fin)

    def generate_for_single_scan(self, scan):
        # For convenience
        anchors_must_be_unique = self.hp['anchors_must_be_unique']
        exclude_anchor_instances = self.hp['exclude_anchor_instances']
        max_df = self.hp['max_df']
        max_dl = self.hp['max_dl']
        a = self.hp['angle']
        d2 = self.hp['d2']
        positive_occ_thresh = self.hp['positive_occ_thresh']
        negative_occ_thresh = self.hp['negative_occ_thresh']

        targets = Reference.valid_targets(scan,
                                          must_be_multiple=self.targets_must_be_multiple,
                                          valid_instances=self.valid_target_instances,
                                          group_by_label=True,
                                          too_hard=self.too_hard)

        # Anchors that have oriented bounding boxes and an intrinsic front face
        anchors = Reference.valid_anchors(scan,
                                          must_be_unique=anchors_must_be_unique,
                                          valid_instances=self.valid_anchor_instances,
                                          has_front=True,
                                          too_hard=self.too_hard)

        candidate_refs = []
        # Loop over the anchors list
        for anchor in anchors:
            if anchor.instance_label in exclude_anchor_instances:
                continue

            df = min(2 * anchor.get_bbox().lx, max_df)
            dl = min(2 * anchor.get_bbox().ly, max_dl)

            # Get anchor oriented sections
            [xmin, ymin, _, xmax, ymax, _] = anchor.get_bbox(axis_aligned=False).extrema
            anchor_bbox_extrema = [xmin, xmax, ymin, ymax]
            anchor_sections = self.get_anchor_sections(anchor_bbox_extrema, a, dl, df, d2)

            for target_instance_label in targets.keys():
                bad_ref = False

                candidate_targets = {}
                for target in targets[target_instance_label]:

                    iou_2d, i_ratios, a_ratios = target.iou_2d(anchor)

                    # Ignore references where an anchor intersects with a target object Can be relaxed
                    if np.any(np.array(i_ratios) > 0.2):
                        bad_ref = True
                        break

                    candidate_targets[target.object_id] = target

                if bad_ref:
                    continue

                # get the oriented sections that each target object occupy
                oriented_sections = defaultdict(lambda: defaultdict(int))

                for target in candidate_targets.values():
                    target_points = target.get_pc()
                    for point in target_points:
                        section_id = self.which_section_point_in(anchor.get_bbox(), anchor_sections, point).value
                        oriented_sections[section_id][target.object_id] += 1

                # Loop over each oriented section and if it is only occupied
                # by one target with acceptable occupancy threshold, then this
                # is a good reference
                for sec, in_sec_targets in oriented_sections.items():
                    if OrientedSections(sec).name == 'grey_area':
                        continue

                    target_occupancy_ratios = defaultdict(float)
                    occupying_targets = []
                    n_non_occpying_targets = 0

                    # Calculate the occupancy ratio of this target at each of the oriented section
                    for object_id in in_sec_targets.keys():
                        target = candidate_targets[object_id]
                        assert (target.object_id == object_id)

                        n_points = in_sec_targets[object_id] * 1.0
                        target_occupancy_ratios[object_id] = (n_points / len(target.points))

                        if target_occupancy_ratios[object_id] > positive_occ_thresh:
                            occupying_targets.append(target)
                        elif target_occupancy_ratios[object_id] < negative_occ_thresh:
                            # check if no other targets are occupying the same section with considerable occupying ratio
                            n_non_occpying_targets += 1

                    if len(occupying_targets) == 1 and n_non_occpying_targets == len(in_sec_targets) - 1:
                        candidate_refs.append(Reference(occupying_targets[0], anchor,
                                                        OrientedSections(sec).name))

        self.generated_references = Reference.force_uniqueness(candidate_refs)
        return self.generated_references

    @staticmethod
    def get_anchor_sections(extrema, a, dl, df, d2):
        """

        @param extrema:
        @param a:
        @param dl:
        @param df:
        @param d2:

        @return:
        """
        xmin, xmax, ymin, ymax = extrema
        b = 90 - a
        a = np.deg2rad(a)
        b = np.deg2rad(b)

        section_names = [OrientedSections.front, OrientedSections.back, OrientedSections.right, OrientedSections.left]
        ret = {}
        for section in section_names:
            if section.name == 'front':
                p1 = (xmin, ymin)
                p2 = (xmin, ymax)
                p3 = (xmin - df, ymax + (np.sin(a) * df / np.sin(b)))
                p4 = (xmin - df, ymin - (np.sin(a) * df / np.sin(b)))
                p5 = (xmin - df - d2, ymin - (np.sin(a) * df / np.sin(b)))
                p6 = (xmin - df - d2, ymax + (np.sin(a) * df / np.sin(b)))
                ret[section] = MultiPoint([p1, p2, p3, p4, p5, p6]).convex_hull
            elif section.name == 'back':
                p1 = (xmax, ymin)
                p2 = (xmax, ymax)
                p3 = (xmax + df, ymax + (np.sin(a) * df / np.sin(b)))
                p4 = (xmax + df, ymin - (np.sin(a) * df / np.sin(b)))
                p5 = (xmax + df + d2, ymin - (np.sin(a) * df / np.sin(b)))
                p6 = (xmax + df + d2, ymax + (np.sin(a) * df / np.sin(b)))
                ret[section] = MultiPoint([p1, p2, p3, p4, p5, p6]).convex_hull
            elif section.name == 'left':
                p1 = (xmin, ymax)
                p2 = (xmax, ymax)
                p3 = (xmax + (np.sin(a) * dl / np.sin(b)), ymax + dl)
                p4 = (xmin - (np.sin(a) * dl / np.sin(b)), ymax + dl)
                p6 = (xmin - (np.sin(a) * dl / np.sin(b)), ymax + dl + d2)
                p5 = (xmax + (np.sin(a) * dl / np.sin(b)), ymax + dl + d2)
                ret[section] = MultiPoint([p1, p2, p3, p4, p5, p6]).convex_hull
            elif section.name == 'right':
                p1 = (xmin, ymin)
                p2 = (xmax, ymin)
                p3 = (xmax + (np.sin(a) * dl / np.sin(b)), ymin - dl)
                p4 = (xmin - (np.sin(a) * dl / np.sin(b)), ymin - dl)
                p5 = (xmin - (np.sin(a) * dl / np.sin(b)), ymin - dl - d2)
                p6 = (xmax + (np.sin(a) * dl / np.sin(b)), ymin - dl - d2)
                ret[section] = MultiPoint([p1, p2, p3, p4, p5, p6]).convex_hull
        return ret

    @staticmethod
    def which_section_point_in(anchor_bbox, anchor_sections, target_point):
        # Transform the point in order to be compared with the object's
        # axes aligned bb
        point = target_point - [anchor_bbox.cx, anchor_bbox.cy, anchor_bbox.cz]
        point = np.hstack([point, [1]]).reshape(1, -1)
        rotation = anchor_bbox.inverse_rotation_matrix()
        axis_aligned_point = np.dot(rotation, point.T).T[:, 0:3]
        [px, py, _] = axis_aligned_point.reshape(-1)

        for sec_name, section in anchor_sections.items():
            if section.contains(Point(px, py)):
                return sec_name

        # No section
        return OrientedSections.grey_area
