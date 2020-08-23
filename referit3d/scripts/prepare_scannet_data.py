import argparse
import pprint
import time
import os.path as osp
import multiprocessing as mp

from referit3d.in_out.scannet_scan import ScannetScan, ScannetDataset
from referit3d.utils import immediate_subdirectories, create_dir, pickle_data, str2bool


def parse_args():
    parser = argparse.ArgumentParser(description='ReferIt3D')

    parser.add_argument('-top-scan-dir', required=True, type=str,
                        help='the path to the downloaded ScanNet scans')
    parser.add_argument('-top-save-dir', required=True, type=str,
                        help='the path of the directory to be saved preprocessed scans as a .pkl')

    # Optional arguments.
    parser.add_argument('--n-processes', default=-1, type=int,
                        help='the number of processes, -1 means use the available max')
    parser.add_argument('--process-only-zero-view', default=True, type=str2bool,
                        help='00_view of scans are used')
    parser.add_argument('--verbose', default=True, type=str2bool, help='')
    parser.add_argument('--apply-global-alignment', default=True, type=str2bool,
                        help='rotate/translate entire scan globally to aligned it with other scans')

    ret = parser.parse_args()

    # Print the args
    args_string = pprint.pformat(vars(ret))
    print(args_string)

    return ret


if __name__ == '__main__':
    args = parse_args()

    if args.process_only_zero_view:
        tag = 'keep_all_points_00_view'
    else:
        tag = 'keep_all_points'

    if args.apply_global_alignment:
        tag += '_with_global_scan_alignment'
    else:
        tag += '_no_global_scan_alignment'

    # Read all scan files.
    all_scan_ids = [osp.basename(i) for i in immediate_subdirectories(args.top_scan_dir)]
    print('{} scans found.'.format(len(all_scan_ids)))

    kept_scan_ids = []
    if args.process_only_zero_view:
        for si in all_scan_ids:
            if si.endswith('00'):
                kept_scan_ids.append(si)
        all_scan_ids = kept_scan_ids
    print('Working with {} scans.'.format(len(all_scan_ids)))

    # Prepare ScannetDataset
    idx_to_semantic_class_file = '../data/mappings/scannet_idx_to_semantic_class.json'
    instance_class_to_semantic_class_file = '../data/mappings/scannet_instance_class_to_semantic_class.json'
    axis_alignment_info_file = '../data/scannet/scans_axis_alignment_matrices.json'

    scannet = ScannetDataset(args.top_scan_dir,
                             idx_to_semantic_class_file,
                             instance_class_to_semantic_class_file,
                             axis_alignment_info_file)

    def scannet_loader(scan_id):
        """Helper function to load the scans in memory.
        :param scan_id:
        :return: the loaded scan.
        """
        global scannet, args
        scan_i = ScannetScan(scan_id, scannet, args.apply_global_alignment)
        scan_i.load_point_clouds_of_all_objects()
        return scan_i

    if args.verbose:
        print('Loading scans in memory...')

    start_time = time.time()
    n_items = len(all_scan_ids)
    if args.n_processes == -1:
        n_processes = min(mp.cpu_count(), n_items)

    pool = mp.Pool(n_processes)
    chunks = int(n_items / n_processes)

    all_scans = dict()
    for i, data in enumerate(pool.imap(scannet_loader, all_scan_ids, chunksize=chunks)):
        all_scans[all_scan_ids[i]] = data

    pool.close()
    pool.join()

    if args.verbose:
        print("Loading raw data took {:.4} minutes.".format((time.time() - start_time) / 60.0))

    # Save data
    if args.verbose:
        print('Saving the results.')
    all_scans = list(all_scans.values())
    save_dir = create_dir(osp.join(args.top_save_dir, tag))
    save_file = osp.join(save_dir, tag + '.pkl')
    pickle_data(save_file, scannet, all_scans)

    if args.verbose:
        print('All done.')