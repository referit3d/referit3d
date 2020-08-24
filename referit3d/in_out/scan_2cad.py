import pandas as pd

from ..utils import read_dict

shape_net_synthetic_id_to_category = {
    '02691156': 'airplane', '02773838': 'bag', '02801938': 'basket',
    '02808440': 'bathtub', '02818832': 'bed', '02828884': 'bench',
    '02834778': 'bicycle', '02843684': 'birdhouse', '02871439': 'bookshelf',
    '02876657': 'bottle', '02880940': 'bowl', '02924116': 'bus',
    '02933112': 'cabinet', '02747177': 'can', '02942699': 'camera',
    '02954340': 'cap', '02958343': 'car', '03001627': 'chair',
    '03046257': 'clock', '03207941': 'dishwasher', '03211117': 'monitor',
    '04379243': 'table', '04401088': 'telephone', '02946921': 'tin_can',
    '04460130': 'tower', '04468005': 'train', '03085013': 'keyboard',
    '03261776': 'earphone', '03325088': 'faucet', '03337140': 'file',
    '03467517': 'guitar', '03513137': 'helmet', '03593526': 'jar',
    '03624134': 'knife', '03636649': 'lamp', '03642806': 'laptop',
    '03691459': 'speaker', '03710193': 'mailbox', '03759954': 'microphone',
    '03761084': 'microwave', '03790512': 'motorcycle', '03797390': 'mug',
    '03928116': 'piano', '03938244': 'pillow', '03948459': 'pistol',
    '03991062': 'pot', '04004475': 'printer', '04074963': 'remote_control',
    '04090263': 'rifle', '04099429': 'rocket', '04225987': 'skateboard',
    '04256520': 'sofa', '04330267': 'stove', '04530566': 'vessel',
    '04554684': 'washer', '02858304': 'boat', '02992529': 'cellphone'
}


def load_scan2cad_meta_data(scan2CAD_meta_file):
    """
    scan2CAD_meta_file: scan2CAD meta-information about the oo-bbox, and 'front-dir'
    """
    scan2CAD_temp = read_dict(scan2CAD_meta_file)

    scan2CAD = dict()
    for scan_id_object_id, object_info in scan2CAD_temp.items():
        # Split the key to get the scan_id and object_id, eg scene0000_00_1
        scan_id = scan_id_object_id[:12]
        object_id = scan_id_object_id[13:]

        scan2CAD[(scan_id, object_id)] = object_info
    return scan2CAD


def load_has_front_meta_data(has_front_file):
    """
    Load the has front property (0, 1) of each shapenet object.

    :param has_front_file: The path to the has front annotations file
    :return: dictionary mapping object id -> (0 or 1 'has front property')
    """
    df = pd.read_csv(has_front_file, converters={'syn_id': lambda x: str(x)})
    res = dict()
    for i in range(len(df)):
        yes = df.loc[i]['has_front'] == 1
        if yes:
            res[(str(df.loc[i]['syn_id']), df.loc[i]['model_name'])] = yes
    return res

import numpy as np

def register_scan2cad_bboxes(all_scans, scan2CAD, bad_mappings_file=None):
    """
    Add the oriented bounding boxes information from scan2CAD to each scannet
    object in each scene (if found).

    :param bad_mappings_file: a file of hand picked bad mappings in scan2CAD
    :param all_scans: a list of scannet scans
    :param scan2CAD: dictionary mapping (scannet_scene_id, object_id) ->
        (oriented bounding box information, others)
    """
    registered = 0
    ignored = 0

    if bad_mappings_file is not None:
        bad_mappings = read_dict(bad_mappings_file)
    else:
        bad_mappings = None

    for scan in all_scans:
        scan_id = scan.scan_id
        for o in scan.three_d_objects:
            key = (scan_id, str(o.object_id))

            # Ignore bad mapping (when applicable)
            if bad_mappings is not None and key in scan2CAD:
                cat_id = scan2CAD[key]['catid_cad']
                label_scan2cad = (shape_net_synthetic_id_to_category[cat_id])

                if [o.instance_label, label_scan2cad] in bad_mappings:
                    ignored += 1
                    continue

            if key in scan2CAD:
                o_info = scan2CAD[key]
                # Get the rotation matrix
                rot = np.array(o_info['obj_rot'], dtype=np.float32)[:3, :3]
                o.set_object_aligned_bbox(
                    *o_info['obj_bbox'], rot=rot)
                registered += 1
    print(registered, 'bboxes registered,', ignored, ' bboxes ignored.')


def register_front_direction(all_scans, scan2CAD, has_front):
    """
    Add the has front information to each scannet object in each scene
    (if found).

    :param all_scans: a list of scannet scans
    :param scan2CAD: dictionary mapping (scannet_scan_id, object_id) ->
        (oriented bounding box information, front_point, others)
    :param has_front: a dictionary mapping the shapenet object ->
        (0 or 1 'has front property')
    """
    registered = 0
    for scan in all_scans:
        scan_id = scan.scan_id
        for o in scan.three_d_objects:
            key = (scan_id, str(o.object_id))
            if key in scan2CAD:
                o_info = scan2CAD[key]
                if (o_info['catid_cad'], o_info['id_cad']) in has_front:
                    o.front_direction = o_info['front_point']
                    o.has_front_direction = True
                    registered += 1
    print(registered, 'fronts registered.')

