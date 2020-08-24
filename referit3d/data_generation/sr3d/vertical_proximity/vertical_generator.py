import json
import yaml

from .. import ReferenceGenerator, Reference


class VerticalProximityGenerator(ReferenceGenerator):
    def __init__(self, verbose=True):
        super().__init__(verbose)

        self.type = 'vertical proximity'

        # Read the hyper-parameter
        with open('vertical_proximity/hp.yml') as fin:
            self.hp = yaml.full_load(fin)

        # Read the necessary dictionary
        with open('../../data/language/sr3d/semantics/instances_that_can_support.json') as fin:
            self.instance_can_support = json.load(fin)

        with open('../../data/language/sr3d/semantics/instances_that_cannot_be_supported.json') as fin:
            self.instance_cannot_be_supported = json.load(fin)

    def generate_for_single_scan(self, scan):
        all_targets = Reference.valid_targets(scan,
                                              must_be_multiple=self.targets_must_be_multiple,
                                              valid_instances=self.valid_target_instances,
                                              too_hard=self.too_hard)
        all_anchors = Reference.valid_anchors(scan,
                                              must_be_unique=False,
                                              valid_instances=self.valid_anchor_instances,
                                              too_hard=self.too_hard)

        if len(all_targets) < 1 or len(all_anchors) < 1:
            return []

        # For convenience
        max_to_be_touching_distance = self.hp['max_touch_distance']
        min_above_below_distance = self.hp['min_above_below_distance']
        max_to_be_supporting_area_ratio = self.hp['max_supporting_area_ratio']
        min_to_be_supported_area_ratio = self.hp['min_supported_area_ratio']
        min_to_be_above_below_area_ratio = self.hp['min_to_be_above_below_area_ratio']

        all_refs = list()
        for anchor in all_anchors:
            if anchor.instance_label in self.hp['exclude_anchor_instances']:
                continue

            a_zmin = anchor.z_min()
            a_zmax = anchor.z_max()

            for target in all_targets:
                if target.instance_label == anchor.instance_label:
                    continue

                # Check whether the target object is in the vicinity of the anchor
                iou_2d, i_ratios, a_ratios = target.iou_2d(anchor)
                i_target_ratio, i_anchor_ratio = i_ratios
                target_anchor_area_ratio, anchor_target_area_ratio = a_ratios

                if iou_2d < 0.001:  # No intersection at all (not in the vicinty of each other)
                    continue

                target_bottom_anchor_top_dist = target.z_min() - a_zmax
                target_top_anchor_bottom_dist = a_zmin - target.z_max()

                # Is the target/anchor object can support objects (e.g. like table)?
                target_can_support = self.instance_can_support[target.instance_label].lower() == "true"
                anchor_can_support = self.instance_can_support[anchor.instance_label].lower() == "true"

                # Is the target/anchor cannot be supported by anything other than the walls/floor?
                target_cannot_be_supported = self.instance_cannot_be_supported[target.instance_label].lower() == "true"
                anchor_cannot_be_supported = self.instance_cannot_be_supported[anchor.instance_label].lower() == "true"

                # Target supported-by the anchor
                target_supported_by_anchor = False
                if anchor_can_support and i_target_ratio > min_to_be_supported_area_ratio and abs(
                        target_bottom_anchor_top_dist) <= max_to_be_touching_distance and not target_cannot_be_supported and \
                        target_anchor_area_ratio < max_to_be_supporting_area_ratio:  # target is not quite larger in area
                    # than the anchor
                    target_supported_by_anchor = True

                # Target supporting the anchor
                target_supporting_anchor = False
                if target_can_support and i_anchor_ratio > min_to_be_supported_area_ratio and abs(
                        target_top_anchor_bottom_dist) <= max_to_be_touching_distance and not anchor_cannot_be_supported and \
                        anchor_target_area_ratio < max_to_be_supporting_area_ratio:
                    target_supporting_anchor = True

                target_above_anchor = target_bottom_anchor_top_dist > min_above_below_distance and \
                                      max(i_anchor_ratio, i_target_ratio) > min_to_be_above_below_area_ratio  # above
                target_below_anchor = target_top_anchor_bottom_dist > min_above_below_distance and \
                                      max(i_anchor_ratio, i_target_ratio) > min_to_be_above_below_area_ratio  # below

                # We prefer touching relations than the vertical (above/below) relations
                if target_supported_by_anchor:
                    ref = Reference(target, anchor, reference_type='supported-by')
                    all_refs.append(ref)
                elif target_supporting_anchor:
                    ref = Reference(target, anchor, reference_type='supporting')
                    all_refs.append(ref)
                elif target_above_anchor:
                    ref = Reference(target, anchor, reference_type='above')
                    all_refs.append(ref)
                elif target_below_anchor:
                    ref = Reference(target, anchor, reference_type='below')
                    all_refs.append(ref)

        ret_refs = []
        for r in all_refs:
            # if targets are intersecting with each other --> ignore
            distractors = r.distractors() + [r.target]

            objects_intersect = False
            for i in range(len(distractors)):
                for j in range(i + 1, len(distractors)):
                    i_bbox, j_bbox = distractors[i].get_bbox(axis_aligned=True), distractors[j].get_bbox(
                        axis_aligned=True)
                    inter = i_bbox.intersection_with(j_bbox)

                    i_inter_area_ratio = inter / i_bbox.volume()
                    j_inter_area_ratio = inter / j_bbox.volume()

                    if i_inter_area_ratio > 0.3 or j_inter_area_ratio > 0.3:
                        objects_intersect = True
                        break
                if objects_intersect:
                    break

            if objects_intersect:
                continue
            ret_refs.append(r)

        return Reference.force_uniqueness(ret_refs)
