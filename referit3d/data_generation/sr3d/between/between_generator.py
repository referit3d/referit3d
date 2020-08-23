import yaml
import itertools
from shapely.geometry import MultiPoint

from .. import ReferenceGenerator, Reference
from ....in_out.three_d_object import ThreeDObject


class BetweenGenerator(ReferenceGenerator):
    def __init__(self, verbose=True):
        super().__init__(verbose)

        self.type = 'between'

        # Read the hyper-parameter
        with open('between/hp.yml') as fin:
            self.hp = yaml.full_load(fin)

    def generate_for_single_scan(self, scan):
        # For convience
        must_be_unique_anchors = self.hp['anchors_must_be_unique']
        exclude_anchor_instances = self.hp['exclude_anchor_instances']
        occ_thresh = self.hp['occ_thresh']
        min_forbidden_occ_ratio = self.hp['min_forbidden_occ_ratio']
        target_anchor_intersect_ratio_thresh = self.hp['target_anchor_intersect_ratio_thresh']
        min_target_to_anchor_z_intersection = self.hp['min_target_to_anchor_z_intersection']
        safe_distance = self.hp['safe_distance']

        targets = Reference.valid_targets(scan=scan,
                                          must_be_multiple=self.targets_must_be_multiple,
                                          valid_instances=self.valid_target_instances,
                                          too_hard=self.too_hard,
                                          group_by_label=True)

        anchors = Reference.valid_anchors(scan=scan,
                                          must_be_unique=must_be_unique_anchors,
                                          valid_instances=self.valid_anchor_instances,
                                          too_hard=self.too_hard,
                                          exclude_instances=exclude_anchor_instances)

        all_refs = list()
        bad_combinations = []

        for anc_a, anc_b in itertools.combinations(anchors, 2):
            if not self.valid_between_anchors(anc_a, anc_b):
                continue

            for target_label, target_objects in targets.items():
                if (anc_a.object_id, anc_b.object_id, target_label) in bad_combinations:
                    continue

                if target_label in [anc_a.instance_label, anc_b.instance_label]:
                    continue

                for target in target_objects:
                    if target.object_id in [anc_a.object_id, anc_b.object_id]:
                        continue

                    # Get the top view 2D bounding boxes and check whether it is between or not
                    anchor_a_z_face = anc_a.get_bbox().z_faces()[0]  # Top face
                    anchor_a_points = tuple(map(tuple, anchor_a_z_face[:, :2]))  # x, y coordinates

                    anchor_b_z_face = anc_b.get_bbox().z_faces()[0]
                    anchor_b_points = tuple(map(tuple, anchor_b_z_face[:, :2]))

                    target_z_face = target.get_bbox().z_faces()[0]
                    target_points = tuple(map(tuple, target_z_face[:, :2]))

                    is_between, is_bad_anchor_comb = self.is_between_candidate(
                        anc_a_points=anchor_a_points,
                        anc_b_points=anchor_b_points,
                        target_points=target_points,
                        occ_thresh=occ_thresh,
                        min_forbidden_occ_ratio=min_forbidden_occ_ratio,
                        target_anchor_intersect_ratio_thresh=target_anchor_intersect_ratio_thresh)

                    if is_bad_anchor_comb:
                        bad_combinations.append((anc_a.object_id, anc_b.object_id, target_label))
                        continue

                    # Target should be in the same z range for each of the two anchors
                    _, t_anc_a, _ = target.intersection(anc_a)
                    _, t_anc_b, _ = target.intersection(anc_b)
                    if t_anc_a < min_target_to_anchor_z_intersection or \
                            t_anc_b < min_target_to_anchor_z_intersection:
                        bad_combinations.append((anc_a.object_id, anc_b.object_id, target_label))
                        continue

                    # Target should be away from every other distractor by a certain distance
                    target_away_from_others = True
                    for distractor in target_objects:
                        if distractor.object_id == target.object_id:
                            continue
                        if target.distance_from_other_object(distractor, optimized=True) < safe_distance:
                            target_away_from_others = False
                            bad_combinations.append((anc_a.object_id, anc_b.object_id, target_label))
                            break

                    is_between &= target_away_from_others

                    if is_between:
                        ref = Reference(target, anc_a, reference_type='between', second_anchor=anc_b)
                        all_refs.append(ref)

        all_refs = Reference.force_uniqueness(all_refs)

        ret_refs = []
        for r in all_refs:
            key = (r.anchor.object_id, r.second_anchor.object_id, r.target.instance_label)
            if key in bad_combinations:
                continue
            ret_refs.append(r)

        self.generated_references = Reference.force_uniqueness(ret_refs)

        return self.generated_references

    @staticmethod
    def is_between_candidate(anc_a_points: tuple,
                             anc_b_points: tuple,
                             target_points: tuple,
                             occ_thresh: float,
                             min_forbidden_occ_ratio: float,
                             target_anchor_intersect_ratio_thresh: float) -> (bool, bool):
        """
        Check whether a target object lies in the convex hull of the two anchors.
        @param anc_a_points: The vertices of the first anchor's 2d top face.
        @param anc_b_points: The vertices of the second anchor's 2d top face.
        @param target_points: The vertices of the target's 2d top face.
        @param occ_thresh: By considering the target intersection ratio with the convexhull of the two anchor,
        which is calculated by dividing the target intersection area to the target's area, if the ratio is
        bigger than the occ_thresh, then we consider this target is between the two anchors.
        @param min_forbidden_occ_ratio: used to create a range of intersection area ratios wherever any target
        object occupies the convexhull with a ratio within this range, we consider this case is ambiguous and we
        ignore generating between references with such combination of possible targets and those two anchors
        @param target_anchor_intersect_ratio_thresh: The max allowed target-to-anchor intersection ratio, if the target
        is intersecting with any of the anchors with a ratio above this thresh, we should ignore generating between
        references for such combinations

        @return: (bool, bool) --> (target_lies_in_convex_hull_statisfying_constraints, bad_target_anchor_combination)
        """
        bad_comb = False
        forbidden_occ_range = [min_forbidden_occ_ratio, occ_thresh - 0.001]
        intersect_ratio_thresh = target_anchor_intersect_ratio_thresh

        # Get the convex hull of all points of the two anchors
        convex_hull = MultiPoint(anc_a_points + anc_b_points).convex_hull

        # Get anchor a, b polygons
        polygon_a = MultiPoint(anc_a_points).convex_hull
        polygon_b = MultiPoint(anc_b_points).convex_hull
        polygon_t = MultiPoint(target_points).convex_hull

        # Candidate should fall completely/with a certain ratio in the convex_hull polygon
        occ_ratio = convex_hull.intersection(polygon_t).area / polygon_t.area
        if occ_ratio < occ_thresh:  # The object is not in the convex-hull enough to be considered between
            if forbidden_occ_range[0] < occ_ratio < forbidden_occ_range[1]:
                # but also should not be causing any ambiguities for other candidate targets
                bad_comb = True
            return False, bad_comb

        # Candidate target should never be intersecting any of the anchors
        if polygon_t.intersection(polygon_a).area / polygon_t.area > intersect_ratio_thresh:
            bad_comb = True
            return False, bad_comb

        if polygon_t.intersection(polygon_b).area / polygon_t.area > intersect_ratio_thresh:
            bad_comb = True
            return False, bad_comb

        return True, bad_comb

    @staticmethod
    def valid_between_anchors(anchor_a: ThreeDObject, anchor_b: ThreeDObject) -> bool:
        """
        Check whether two anchor objects can be considered for between reference generation
        @param anchor_a: The first anchor object
        @param anchor_b: The second anchor object
        """
        # Anchors must not be of the same instance label
        same_label = anchor_a.instance_label == anchor_b.instance_label

        # Anchors must not be intersecting each other by any means in the top view
        iou_2d, _, _ = anchor_a.iou_2d(anchor_b)

        return (not same_label) and iou_2d < 0.001
