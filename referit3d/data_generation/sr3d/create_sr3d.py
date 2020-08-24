import argparse
import json
import os
import pprint

import numpy as np
import os.path as osp
import pandas as pd

from referit3d.data_generation.sr3d import AllocentricGenerator, BetweenGenerator, HPGenerator, SupportAndVPGenerator
from referit3d.utils import unpickle_data, read_lines, str2bool
from referit3d.in_out.scan_2cad import load_scan2cad_meta_data, load_has_front_meta_data
from referit3d.in_out.scan_2cad import register_scan2cad_bboxes, register_front_direction
from referit3d.in_out.scannet_scan import scan_and_target_id_to_context_info

EXTRA_ANCHORS = ['staircase', 'bathtub', 'tv stand', 'copier', 'ladder',
                 'fireplace', 'piano', 'refrigerator', 'stove', 'bathroom vanity',
                 'washing machine', 'stairs', 'shower', 'blinds', 'water cooler', 'clothes dryer',
                 'tv', 'bulletin board', 'rack', 'counter', 'closet', 'projector screen']
SCAN2CAD_META_FILE = '../../data/scan2cad/object_oriented_bboxes/object_oriented_bboxes_aligned_scans.json'
SCAN2CAD_PATCHED_META_FILE = '../../data/scan2cad/object_oriented_bboxes/patched_object_oriented_bboxes_aligned_scans.json'
BAD_SCAN2CAD_MAPPINGS_FILE = '../../data/scan2cad/bad_mappings.json'
HAS_FRONT_FILE = '../../data/scan2cad/shapenet_has_front.csv'
REFERENCES_TO_HUMAN_LANGUAGE_FILE = '../../data/language/sr3d/references_to_human_language.json'


def parse_args():
    parser = argparse.ArgumentParser('Generating annotations for spatial 3D reference (Sr3D).')

    parser.add_argument('-preprocessed_scannet_file', type=str, help='.pkl (output) of prepare_scannet_data.py',
                        required=True)
    parser.add_argument('-valid_targets_file', type=str,
                        help='.txt file describing (one line per) the target classes for which we will make language.',
                        required=True)
    parser.add_argument('-save_dir', type=str, help='top-dir to save the results.', required=True)

    parser.add_argument('-name', type=str, help='Name of the gerneated sr3d csv file.', required=True)
    parser.add_argument('--verbose', type=str2bool, default=True, help='verbose')
    parser.add_argument('--stimulus_is_too_hard', type=int, default=6,
                        help='ignore cases that include more than `this` contrasting, same-class objects.')

    # Synthetic references to language parameters
    parser.add_argument('--max_samples_per_context', type=int, default=1,
                        help='max number of utterances to produce per context/reference-type '
                             '(by simple sampling linguistic variations, see:  '
                             'SYNTHETIC_TO_HUMAN_LANGUAGE_FILE).')

    parser.add_argument('--random_seed', type=int, default=3,
                        help='seed used in sampling template utterances for each reference.')
    parser.add_argument('--targets-must-be-multiple', type=str2bool, default='false')
    parser.add_argument('--anchor_must_be_unique', type=str2bool, default='true')

    args = parser.parse_args()

    args_string = pprint.pformat(vars(args))
    print(args_string)

    # Save the configs
    if not osp.isdir(args.save_dir):
        os.makedirs(args.save_dir)

    with open(osp.join(args.save_dir, 'sr3d_configs.json.txt'), 'w') as fout:
        json.dump(vars(args), fout, indent=4, sort_keys=True)

    return args


def sample_and_save_synthetic_utterances(args, references, all_scans, synth_to_human_lang_dict):
    np.random.seed(args.random_seed)

    all_scans_in_dict = {scan.scan_id: scan for scan in all_scans}
    synthetic_refs_human_like_format = list()

    for ref in references:
        scan_id = ref.scan().scan_id
        target_id = ref.target.object_id
        target_instance_label = ref.target.instance_label
        dummy_target_label, distractor_ids, stimulus_id = scan_and_target_id_to_context_info(scan_id, target_id,
                                                                                             all_scans_in_dict)
        assert (target_instance_label == dummy_target_label)

        anchors_instance_labels = [i.instance_label for i in ref.anchors()]

        utterances = ref.to_human_language(ref, synth_to_human_lang_dict, n_utterances=args.max_samples_per_context)

        example = [{'scan_id': scan_id,
                    'target_id': target_id,
                    'distractor_ids': distractor_ids,
                    'utterance': utter,
                    'stimulus_id': stimulus_id,
                    'coarse_reference_type': ref.get_reference_type(coarse=True),
                    'reference_type': ref.get_reference_type(coarse=False),
                    'instance_type': target_instance_label,
                    'anchors_types': anchors_instance_labels,
                    'anchor_ids': [i.object_id for i in ref.anchors()]} for utter in utterances]

        synthetic_refs_human_like_format.extend(example)

    out_file = osp.join(args.save_dir, args.name)
    synthetic_refs_human_like_format = pd.DataFrame.from_records(synthetic_refs_human_like_format)
    synthetic_refs_human_like_format.to_csv(out_file, index=False)


def generate_synthetic_references():
    # Common parameters
    params = [all_scans, valid_targets, valid_anchors, args.targets_must_be_multiple, args.stimulus_is_too_hard]

    # Horizontal References
    horizontal_proximity_generator = HPGenerator(verbose=args.verbose)
    horizontal_proximity_refs = horizontal_proximity_generator.generate(*params)

    # Vertical References
    vertical_proximity_support_generator = SupportAndVPGenerator(verbose=args.verbose)
    vertical_proximity_refs = vertical_proximity_support_generator.generate(*params)

    # Between References
    between_generator = BetweenGenerator(verbose=args.verbose)
    between_refs = between_generator.generate(*params)

    # Allocentric References
    allocentric_generator = AllocentricGenerator(verbose=args.verbose)
    allocentric_refs = allocentric_generator.generate(*params)

    return (horizontal_proximity_refs + allocentric_refs + vertical_proximity_refs
            + between_refs)


if __name__ == '__main__':
    #
    # Parse arguments
    #
    args = parse_args()

    #
    # Read the scans
    #
    scannet, all_scans = unpickle_data(args.preprocessed_scannet_file)

    #
    # Augment data with scan2CAD bboxes
    #
    scan2CAD = load_scan2cad_meta_data(SCAN2CAD_META_FILE)
    patched_scan2CAD = load_scan2cad_meta_data(SCAN2CAD_PATCHED_META_FILE)
    register_scan2cad_bboxes(all_scans, scan2CAD, BAD_SCAN2CAD_MAPPINGS_FILE)

    #
    # Augment data with has-front information (for allocentric questions).
    #
    has_front = load_has_front_meta_data(HAS_FRONT_FILE)
    register_front_direction(all_scans, scan2CAD, has_front)

    #
    # Read the target and anchor instance types
    #
    valid_targets = read_lines(args.valid_targets_file)
    print('# Targets:', len(valid_targets))
    valid_anchors = valid_targets + EXTRA_ANCHORS
    print('# Anchors:', len(valid_anchors))

    # No duplicates
    assert len(valid_anchors) == len(set(valid_anchors))
    assert len(valid_targets) == len(set(valid_targets))

    #
    # Generate synthetic references
    #
    references = generate_synthetic_references()

    with open(REFERENCES_TO_HUMAN_LANGUAGE_FILE) as fin:
        synth_to_human_lang_dict = json.load(fin)

    #
    # Sample human-like utterances from the generated synthetic references
    #
    sample_and_save_synthetic_utterances(args, references, all_scans, synth_to_human_lang_dict)
