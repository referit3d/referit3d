import json
import numpy as np
import os.path as osp
import warnings
from collections import defaultdict

from plyfile import PlyData

from ..utils.point_clouds import uniform_sample
from ..utils import invert_dictionary, read_dict
from ..utils.plotting import plot_pointcloud
from .three_d_object import ThreeDObject


class ScannetDataset(object):
    """
    Holds Scannet mesh and labels data paths and some needed class labels mappings
    Note: data downloaded from: http://www.scan-net.org/changelog#scannet-v2-2018-06-11
    """

    def __init__(self, top_scan_dir, idx_to_semantic_cls_file,
                 instance_cls_to_semantic_cls_file, axis_alignment_info_file):
        self.top_scan_dir = top_scan_dir

        self.idx_to_semantic_cls_dict = read_dict(idx_to_semantic_cls_file)

        self.semantic_cls_to_idx_dict = invert_dictionary(self.idx_to_semantic_cls_dict)

        self.instance_cls_to_semantic_cls_dict = read_dict(instance_cls_to_semantic_cls_file)

        self.semantic_cls_to_instance_cls_dict = defaultdict(list)

        for k, v in self.instance_cls_to_semantic_cls_dict.items():
            self.semantic_cls_to_instance_cls_dict[v].append(k)

        self.scans_axis_alignment_matrices = read_dict(axis_alignment_info_file)

    def idx_to_semantic_cls(self, semantic_idx):
        return self.idx_to_semantic_cls_dict[str(semantic_idx)]

    def semantic_cls_to_idx(self, semantic_cls):
        return self.semantic_cls_to_idx_dict[str(semantic_cls)]

    def instance_cls_to_semantic_cls(self, instance_cls):
        return self.instance_cls_to_semantic_cls_dict[str(instance_cls)]

    def get_axis_alignment_matrix(self, scan_id):
        return self.scans_axis_alignment_matrices[scan_id]


class ScannetScan(object):
    """
    Keep track of the point-cloud associated with the scene of Scannet. Includes meta-information such as the
    object that exist in the scene, their semantic labels and their RGB color.
    """

    def __init__(self, scan_id, scannet_dataset, apply_global_alignment=True):
        """
            :param scan_id: (string) e.g. 'scene0705_00'
            :scannet_dataset: (ScannetDataset) captures the details about the class-names, top-directories etc.
        """
        self.dataset = scannet_dataset
        self.scan_id = scan_id
        self.pc, self.semantic_label, self.color = \
            self.load_point_cloud_with_meta_data(self.scan_id, apply_global_alignment=apply_global_alignment)

        self.three_d_objects = None  # A list with ThreeDObject contained in this Scan

    def __str__(self, verbose=True):
        res = '{}'.format(self.scan_id)
        if verbose:
            res += ' with {} points'.format(self.n_points())
        return res

    def n_points(self):
        return len(self.pc)

    def verify_read_data_correctness(self, scan_aggregation, segment_file, segment_indices):
        c1 = scan_aggregation['sceneId'][len('scannet.'):] == self.scan_id
        scan_segs_suffix = '_vh_clean_2.0.010000.segs.json'
        segment_dummy = self.scan_id + scan_segs_suffix
        c2 = segment_file == segment_dummy
        c3 = len(segment_indices) == self.n_points()
        c = np.array([c1, c2, c3])
        if not np.all(c):
            warnings.warn('{} has some issue'.format(self.scan_id))
        return c

    def load_point_cloud_with_meta_data(self, load_semantic_label=True, load_color=True, apply_global_alignment=True):
        """
        :param load_semantic_label:
        :param load_color:
        :param apply_global_alignment: rotation/translation of scan according to Scannet meta-data.
        :return:
        """
        scan_ply_suffix = '_vh_clean_2.labels.ply'
        mesh_ply_suffix = '_vh_clean_2.ply'

        scan_data_file = osp.join(self.dataset.top_scan_dir, self.scan_id, self.scan_id + scan_ply_suffix)
        data = PlyData.read(scan_data_file)
        x = np.asarray(data.elements[0].data['x'])
        y = np.asarray(data.elements[0].data['y'])
        z = np.asarray(data.elements[0].data['z'])
        pc = np.stack([x, y, z], axis=1)

        label = None
        if load_semantic_label:
            label = np.asarray(data.elements[0].data['label'])

        color = None
        if load_color:
            scan_data_file = osp.join(self.dataset.top_scan_dir, self.scan_id, self.scan_id + mesh_ply_suffix)
            data = PlyData.read(scan_data_file)
            r = np.asarray(data.elements[0].data['red'])
            g = np.asarray(data.elements[0].data['green'])
            b = np.asarray(data.elements[0].data['blue'])
            color = (np.stack([r, g, b], axis=1) / 256.0).astype(np.float32)

        # Global alignment of the scan
        if apply_global_alignment:
            pc = self.align_to_axes(pc)

        return pc, label, color

    def load_point_clouds_of_all_objects(self, exclude_instances=None):
        scan_aggregation_suffix = '.aggregation.json'
        aggregation_file = osp.join(self.dataset.top_scan_dir, self.scan_id, self.scan_id + scan_aggregation_suffix)
        with open(aggregation_file) as fin:
            scan_aggregation = json.load(fin)

        scan_segs_suffix = '_vh_clean_2.0.010000.segs.json'
        segment_file = self.scan_id + scan_segs_suffix

        segments_file = osp.join(self.dataset.top_scan_dir, self.scan_id, segment_file)

        with open(segments_file) as fin:
            segments_info = json.load(fin)
            segment_indices = segments_info['segIndices']

        segment_dummy = scan_aggregation['segmentsFile'][len('scannet.'):]

        check = self.verify_read_data_correctness(scan_aggregation, segment_dummy, segment_indices)

        segment_indices_dict = defaultdict(list)
        for i, s in enumerate(segment_indices):
            segment_indices_dict[s].append(i)  # Add to each segment, its point indices

        # iterate over every object
        all_objects = []
        for object_info in scan_aggregation['segGroups']:
            object_instance_label = object_info['label']
            object_id = object_info['objectId']
            if exclude_instances is not None:
                if object_instance_label in exclude_instances:
                    continue

            segments = object_info['segments']
            pc_loc = []
            # Loop over the object segments and get the all point indices of the object
            for s in segments:
                pc_loc.extend(segment_indices_dict[s])
            object_pc = pc_loc
            all_objects.append(ThreeDObject(self, object_id, object_pc, object_instance_label))
        self.three_d_objects = all_objects
        return check

    def override_instance_labels_by_semantic_labels(self):
        for o in self.three_d_objects:
            o._use_true_instance = False

    def activate_instance_labels(self):
        for o in self.three_d_objects:
            o._use_true_instance = True

    def all_semantic_types(self):
        unique_types = np.unique(self.semantic_label)
        human_types = []
        for t in unique_types:
            human_types.append(self.dataset.idx_to_semantic_cls(t))
        return sorted(human_types)

    def instance_occurrences(self):
        """
        :return: (dict) instance_type (string) -> number of occurrences in the scan (int)
        """
        res = defaultdict(int)
        for o in self.three_d_objects:
            res[o.instance_label] += 1
        return res

    def clone(self):
        raise NotImplementedError('Implement me.')

    def points_of_instance_types(self, valid_instance_types, exclude_instance_types):
        idx = []
        for o in self.three_d_objects:
            o_label_valid = True if (valid_instance_types is None) else (o.instance_label in valid_instance_types)
            o_label_excluded = False if (exclude_instance_types is None) else (
                    o.instance_label in exclude_instance_types)

            if o_label_valid and not o_label_excluded:
                idx.extend(o.points)
        return np.array(idx)

    def sample_indices(self, subsample=None, valid_instance_types=None, seed=None, exclude_instance_types=None):
        """
        Sample ids from the scan point cloud.
        :param exclude_instance_types:
        :param seed: Random seed (default=None)
        :param subsample: The number of ids to be sampled from the scan
         point cloud
        :param valid_instance_types: The instances to be sampled from
        :return: sampled point indices
        """
        if valid_instance_types is not None or exclude_instance_types is not None:
            valid_idx = self.points_of_instance_types(valid_instance_types, exclude_instance_types)
        else:
            valid_idx = np.arange(self.n_points())

        if subsample is None:
            return valid_idx  # return all valid points
        else:
            return uniform_sample(points=valid_idx, n_samples=subsample, random_seed=seed)

    def plot(self, subsample=None, valid_instance_types=None):
        """
        Plot the scan point cloud
        :param subsample: The number of points to be sampled from the scan
         point cloud
        :param valid_instance_types: The instances to be plotted

        :return: matplotlib.pyplot.fig of the scan
        """
        pt = self.sample_indices(subsample, valid_instance_types)

        x, y, z = self.pc[pt, 0], self.pc[pt, 1], self.pc[pt, 2]
        color = self.color[pt]

        return plot_pointcloud(x, y, z, color=color)

    def align_to_axes(self, point_cloud):
        """
        Align the scan to xyz axes using the alignment matrix found in scannet.
        """
        # Get the axis alignment matrix
        alignment_matrix = self.dataset.get_axis_alignment_matrix(self.scan_id)
        alignment_matrix = np.array(alignment_matrix, dtype=np.float32).reshape(4, 4)

        # Transform the points
        pts = np.ones((point_cloud.shape[0], 4), dtype=point_cloud.dtype)
        pts[:, 0:3] = point_cloud
        point_cloud = np.dot(pts, alignment_matrix.transpose())[:, :3]  # Nx4

        # Make sure no nans are introduced after conversion
        assert (np.sum(np.isnan(point_cloud)) == 0)

        return point_cloud


def scan_and_target_id_to_context_info(scan_id, target_id, all_scans_in_dict):
    """ Get context information (e.g., same instance-class objects) of the object specified by the target_id in the
    scene specified by the scene_id.
    :param scan_id:    (string) scene0010_00
    :param target_id:   (int) 36
    :param all_scans_in_dict: dict from strings: scene0010_00 to objects of ScannetScan
    :return: (chair, [35, 37, 38, 39], scene0010_00-chair-5-36-35-37-38-39)
    """
    scene_objects = all_scans_in_dict[scan_id].three_d_objects
    target = scene_objects[target_id]
    instance_label = target.instance_label
    distractors = [x.object_id for x in scene_objects if x.instance_label == instance_label and x != target]
    half_context_info = [scan_id, instance_label, str(len(distractors) + 1), str(target_id)]
    context_string = '-'.join(half_context_info + [str(d) for d in distractors])
    context_string = context_string.replace(' ', '_')
    return instance_label, distractors, context_string
