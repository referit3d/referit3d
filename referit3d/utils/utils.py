import numpy as np
import six
import string
import random
import torch
import os
import re
import sys
import json
import argparse
import logging
import os.path as osp
from six.moves import cPickle
from six.moves import range


def invert_dictionary(d):
    inv_map = {v: k for k, v in six.iteritems(d)}
    return inv_map


def read_dict(file_path):
    with open(file_path) as fin:
        return json.load(fin)


def random_alphanumeric(n_chars):
    character_pool = string.ascii_uppercase + string.ascii_lowercase + string.digits
    array_pool = np.array([c for c in character_pool])
    res = ''.join(np.random.choice(array_pool, n_chars, replace=True))
    return res


def seed_training_code(manual_seed, strict=False):
    """Control pseudo-randomness for reproducibility.
    :param manual_seed: (int) random-seed
    :param strict: (boolean) if True, cudnn operates in a deterministic manner
    """
    random.seed(manual_seed)
    np.random.seed(manual_seed)
    torch.manual_seed(manual_seed)
    torch.cuda.manual_seed_all(manual_seed)
    if strict:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def unit_vector(vector):
    """
    written by David Wolever
    Returns the unit vector of the vector.
    """
    return vector / np.linalg.norm(vector)


def angle_between(v1, v2):
    """
    written by David Wolever
    Returns the angle in radians between vectors 'v1' and 'v2':

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


def files_in_subdirs(top_dir, search_pattern):
    join = os.path.join
    regex = re.compile(search_pattern)
    for path, _, files in os.walk(top_dir):
        for name in files:
            full_name = join(path, name)
            if regex.search(full_name):
                yield full_name


def immediate_subdirectories(top_dir, full_path=True):
    dir_names = [name for name in os.listdir(top_dir) if os.path.isdir(os.path.join(top_dir, name))]
    if full_path:
        dir_names = [osp.join(top_dir, name) for name in dir_names]
    return dir_names


def pickle_data(file_name, *args):
    """
    Using (c)Pickle to save multiple python objects in a single file.
    """
    out_file = open(file_name, 'wb')
    cPickle.dump(len(args), out_file, protocol=2)
    for item in args:
        cPickle.dump(item, out_file, protocol=2)
    out_file.close()


def unpickle_data(file_name, python2_to_3=False):
    """
    Restore data previously saved with pickle_data().
    :param file_name: file holding the pickled data.
    :param python2_to_3: (boolean), if True, pickle happened under python2x, unpickling under python3x.
    :return: an generator over the un-pickled items.
    Note, about implementing the python2_to_3 see
        https://stackoverflow.com/questions/28218466/unpickling-a-python-2-object-with-python-3
    """
    in_file = open(file_name, 'rb')
    if python2_to_3:
        size = cPickle.load(in_file, encoding='latin1')
    else:
        size = cPickle.load(in_file)

    for _ in range(size):
        if python2_to_3:
            yield cPickle.load(in_file, encoding='latin1')
        else:
            yield cPickle.load(in_file)
    in_file.close()


def create_dir(dir_path):
    """
    Creates a directory (or nested directories) if they don't exist.
    """
    if not osp.exists(dir_path):
        os.makedirs(dir_path)

    return dir_path


def str2bool(v):
    """
    Boolean values for argparse
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def read_lines(file_name):
    trimmed_lines = []
    with open(file_name) as fin:
        for line in fin:
            trimmed_lines.append(line.rstrip())
    return trimmed_lines


def load_json(file_name):
    with open(file_name) as fin:
        res = json.load(fin)
    return res


def set_gpu_to_zero_position(real_gpu_loc):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"] = str(real_gpu_loc)


def create_logger(log_dir, std_out=True):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(message)s')

    # Add logging to file handler
    file_handler = logging.FileHandler(osp.join(log_dir, 'log.txt'))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Add stdout to also print statements there
    if std_out:
        logger.addHandler(logging.StreamHandler(sys.stdout))
    return logger
