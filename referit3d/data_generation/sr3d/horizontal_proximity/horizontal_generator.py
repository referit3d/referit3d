import yaml
import numpy as np
from collections import defaultdict

from .. import ReferenceGenerator, Reference


class HorizontalProximityGenerator(ReferenceGenerator):
    def __init__(self, verbose=True):
        super().__init__(verbose)

        self.type = 'horizontal proximity'

        # Read the hyper-parameter
        with open('horizontal_proximity/hp.yml') as fin:
            self.hp = yaml.full_load(fin)

    def generate_for_single_scan(self, scan):
        # For convenience
        exclude_anchor_instances = self.hp['exclude_anchor_instances']
        epsilon_gap = self.hp['horizontal_gap']

        all_targets = Reference.valid_targets(scan,
                                              must_be_multiple=True,
                                              valid_instances=self.valid_target_instances,
                                              too_hard=self.too_hard)
        all_anchors = Reference.valid_anchors(scan,
                                              must_be_unique=True,
                                              valid_instances=self.valid_anchor_instances,
                                              too_hard=self.too_hard,
                                              exclude_instances=exclude_anchor_instances)

        instance_to_targets = defaultdict(list)  # Group same-instance type, since you care only about distances
        for target in all_targets:  # among them.
            instance_to_targets[target.instance_label].append(target)

        if len(all_targets) < 1 or len(all_anchors) < 1:
            return []

        all_refs = list()
        for anchor in all_anchors:
            for instance_type in instance_to_targets.keys():
                if anchor.instance_label == instance_type:
                    assert False  # because of multiple/unique this must fail

                current_target_group = instance_to_targets[instance_type]

                all_dists = []
                for target in current_target_group:
                    t_distance = anchor.distance_from_other_object(target)
                    all_dists.append(t_distance)

                s_idx = np.argsort(all_dists)
                all_dists = np.array(all_dists)[s_idx]

                if all_dists[0] + epsilon_gap < all_dists[1]:  # "closest" ref-type
                    ref = Reference(current_target_group[s_idx[0]], anchor, reference_type='closest')
                    all_refs.append(ref)
                if all_dists[-1] - epsilon_gap > all_dists[-2]:  # "farthest" ref-type
                    ref = Reference(current_target_group[s_idx[-1]], anchor, reference_type='farthest')
                    all_refs.append(ref)

        dummy = Reference.force_uniqueness(all_refs)
        assert (len(dummy) == len(all_refs))
        self.generated_references = all_refs

        return self.generated_references
