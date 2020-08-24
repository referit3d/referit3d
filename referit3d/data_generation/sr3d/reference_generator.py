import yaml
import abc

from ...utils import pickle_data


class ReferenceGenerator:
    def __init__(self, verbose=True):
        self.generated_references = []
        self.verbose = verbose
        self.type = None

        self.valid_target_instances = None
        self.valid_anchor_instances = None
        self.targets_must_be_multiple = None
        self.too_hard = None

    def generate(self, all_scans, valid_target_instances, valid_anchor_instances, targets_must_be_multiple, too_hard):
        self.valid_target_instances = valid_target_instances
        self.valid_anchor_instances = valid_anchor_instances
        self.targets_must_be_multiple = targets_must_be_multiple
        self.too_hard = too_hard

        for scan in all_scans:
            scan_references = self.generate_for_single_scan(scan)
            self.generated_references.extend(scan_references)
            if self.verbose:
                print('{}:'.format(self.type), scan.scan_id, 'resulted in', len(scan_references), '\ttotal till now',
                      len(self.generated_references))

        return self.generated_references

    @abc.abstractmethod
    def generate_for_single_scan(self, scan):
        pass

    def save_references(self, save_path):
        references_dict_list = []
        for reference in self.generated_references:
            references_dict_list.append(reference.serialize())

        with open(save_path, 'w') as fout:
            pickle_data(fout, references_dict_list)
