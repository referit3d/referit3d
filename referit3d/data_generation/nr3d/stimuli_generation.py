import numpy as np
from ...in_out.cuboid import iou_3d
from referit3d.data_generation.sr3d.reference import Reference


class SameInstanceStimulus(object):
    """ a stimulus comprised by objects of the same instance class"""

    def __init__(self, scan_id, target_id, distractors_ids,
                 instance_types, target_bbox, distractor_bboxes):
        self.scan_id = scan_id
        self.target_id = target_id
        self.distractors_ids = distractors_ids
        self.instance_types = instance_types
        self.target_bbox = target_bbox
        self.distractor_bboxes = distractor_bboxes

    def __len__(self):
        return 1 + len(self.distractors_ids)

    def __hash__(self):
        return hash(str(self))

    def __str__(self):
        res = '-'.join([str(x) for x in [self.scan_id,
                                         self.instance_types,
                                         len(self),
                                         self.target_id]])
        if len(self) > 1:
            res += '-' + '-'.join([str(x) for x in self.distractors_ids])

        res = res.replace(' ', '_')
        return res

    def __eq__(self, other):
        return str(self) == str(other)

    @staticmethod
    def decode_stimulus_string(s):
        """
        Split into scene_id, instance_label, # objects, target object id,
        distractors object id.

        :param s: the stimulus string
        """
        if len(s.split('-', maxsplit=4)) == 4:
            scene_id, instance_label, n_objects, target_id = \
                s.split('-', maxsplit=4)
            distractors_ids = ""
        else:
            scene_id, instance_label, n_objects, target_id, distractors_ids = \
                s.split('-', maxsplit=4)

        instance_label = instance_label.replace('_', ' ')
        n_objects = int(n_objects)
        target_id = int(target_id)
        distractors_ids = [int(i) for i in distractors_ids.split('-') if i != '']
        assert len(distractors_ids) == n_objects - 1

        return scene_id, instance_label, n_objects, target_id, distractors_ids


class SameInstanceSyntheticStimulus(SameInstanceStimulus):

    def __init__(self, scan_id, ref_type, target_id, distractors_ids,
                 target_instance, anchor_instances, target_bbox, distractor_bboxes):
        super().__init__(scan_id, target_id, distractors_ids,
                         target_instance, target_bbox, distractor_bboxes)
        self.type = ref_type
        self.anchor_instances = anchor_instances
        self.description = ''


def generate_stimuli(scans, target_instance_type, must_be_multiple=True, too_hard=None):
    if len(target_instance_type) > 1:
        raise NotImplementedError()

    hits = dict()
    for scan in scans:
        targets = Reference.valid_targets(scan, must_be_multiple=must_be_multiple, valid_instances=target_instance_type)
        if too_hard is not None and len(targets) > too_hard:
            continue

        bboxs = list()
        target_ids = list()
        for o in targets:
            bboxs.append(o.get_bbox().corners)
            target_ids.append(o.object_id)

        bboxs = np.array(bboxs)
        target_ids = np.array(target_ids)
        n_objects = len(targets)
        all_idx = set(range(n_objects))

        for i in range(n_objects):
            distractor_idx = list(all_idx.difference([i]))
            distractor_ids = target_ids[distractor_idx]
            hit = SameInstanceStimulus(scan.scan_id, target_ids[i],
                                       distractor_ids, target_instance_type[0],
                                       bboxs[i], bboxs[distractor_idx])
            #             print(hit)
            assert hit not in hits
            hits[hit] = hit
    return list(hits.values())


def is_valid_hit(hit, max_coverage_threshold=1, max_iou_threshold=1, manual_black_list=None, all_pairs=False):
    """ if all_pairs==True, then the overlap constraint has to be applicable among all pairs of contrasting
    objects, else, only among the target and each other one.
    ONLY FOR AXIS-ALIGNED BOXES
    """
    if manual_black_list is not None:
        if str(hit) in manual_black_list:
            return False

    # That has no intersecting bboxes
    boxes = [hit.target_bbox]
    boxes.extend(hit.distractor_bboxes)

    # Cuboid.from_corner_points_to_extrema()

    if all_pairs:
        left_bound = len(boxes)
    else:
        left_bound = 1  # only against target

    for i in range(left_bound):
        for j in range(i + 1, len(boxes)):
            res = iou_3d(boxes[i], boxes[j])
            if res == 0:
                continue
            else:
                iou_ij, intersection, vol_i, vol_j = res
                coverage = intersection / min([vol_i, vol_j])
                if coverage > max_coverage_threshold:
                    return False
                if iou_ij > max_iou_threshold:
                    return False
    return True
