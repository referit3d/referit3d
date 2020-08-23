"""
TODO: add description

The MIT License (MIT)
Originally created at 7/13/20, for Python 3.x
Copyright (c) 2020 Panos Achlioptas (pachlioptas@gmail.com) & Stanford Geometric Computing Lab
"""

import pandas as pd

from .utterances import is_explicitly_view_dependent
from ..data_generation.nr3d import decode_stimulus_string
from ..in_out.pt_datasets.utils import dataset_to_dataloader
from ..models.referit3d_net_utils import detailed_predictions_on_dataset


def analyze_predictions(model, dataset, class_to_idx, pad_idx, device, args, out_file=None, visualize_output=True):
    """
    :param dataset:
    :param net_stats:
    :param pad_idx:
    :return:
    # TODO Panos Post 17 July : clear
    """

    references = dataset.references

    # # YOU CAN USE Those to VISUALIZE PREDICTIONS OF A SYSTEM.
    # confidences_probs = stats['confidences_probs']  # for every object of a scan it's chance to be predicted.
    # objects = stats['contrasted_objects'] # the object-classes (as ints) of the objects corresponding to the confidences_probs
    # context_size = (objects != pad_idx).sum(1) # TODO-Panos assert same as from batch!
    # target_ids = references.instance_type.apply(lambda x: class_to_idx[x])

    hardness = references.stimulus_id.apply(lambda x: decode_stimulus_string(x)[2])
    view_dep_mask = is_explicitly_view_dependent(references)
    easy_context_mask = hardness <= 2

    test_seeds = [args.random_seed, 1, 10, 20, 100]
    net_stats_all_seed = []
    for seed in test_seeds:
        d_loader = dataset_to_dataloader(dataset, 'test', args.batch_size, n_workers=5, seed=seed)
        assert d_loader.dataset.references is references
        net_stats = detailed_predictions_on_dataset(model, d_loader, args=args, device=device, FOR_VISUALIZATION=True)
        net_stats_all_seed.append(net_stats)

    if visualize_output:
        from referit3d.utils import pickle_data
        pickle_data(out_file[:-4] + 'all_vis.pkl', net_stats_all_seed)
#        out = pd.DataFrame(net_stats_all_seed[0])
#        out.to_csv(out_file[:-4] + '_vis.csv', index=False)


    all_accuracy = []
    view_dep_acc = []
    view_indep_acc = []
    easy_acc = []
    hard_acc = []
    among_true_acc = []

    for stats in net_stats_all_seed:
        got_it_right = stats['guessed_correctly']
        all_accuracy.append(got_it_right.mean() * 100)
        view_dep_acc.append(got_it_right[view_dep_mask].mean() * 100)
        view_indep_acc.append(got_it_right[~view_dep_mask].mean() * 100)
        easy_acc.append(got_it_right[easy_context_mask].mean() * 100)
        hard_acc.append(got_it_right[~easy_context_mask].mean() * 100)

        got_it_right = stats['guessed_correctly_among_true_class']
        among_true_acc.append(got_it_right.mean() * 100)

    acc_df = pd.DataFrame({'hard': hard_acc, 'easy': easy_acc,
                           'v-dep': view_dep_acc, 'v-indep': view_indep_acc,
                           'all': all_accuracy, 'among-true': among_true_acc})

    acc_df.to_csv(out_file[:-4] + '.csv', index=False)

    pd.options.display.float_format = "{:,.1f}".format
    descriptive = acc_df.describe().loc[["mean", "std"]].T

    if out_file is not None:
        with open(out_file, 'w') as f_out:
            f_out.write(descriptive.to_latex())
    return descriptive

    #
    # # utterances = references['tokens'].apply(lambda x: ' '.join(x)) # as seen by the neural net.
    #
    #
    #
    #
    # data_df['n_target_class'] = data_df.stimulus_id.apply(lambda x: decode_stimulus_string(x)[2])
    #
    #

    #
    # data_df = data_df.assign(found = pd.Series(net_stats['guessed_correctly']))
    # data_df = data_df.assign(target_pos = pd.Series(net_stats['target_pos']))
    #
    # data_df['n_target_class_inv'] = 1 / data_df['n_target_class']
    # data_df['context_size'] = (contrasted_objects != pad_idx).sum(1) # TODO-Panos assert same as from batch!
    #
    # print('Among target\'s classes', data_df.n_target_class_inv.mean())
    # print('among all classes', (1.0 / data_df['context_size']).mean())
    #
    #
    # print('10 biggest')
    # print(data_df.groupby('instance_type')['found'].mean().sort_values()[::-1][:10])
    # print('TODO, adjust by relative boost (how much +more) against random-guessing baselines')
    #
    #
    # # data_df.guessed_correctly.groupby('reference_type').mean()
    # # data_df.guessed_correctly.groupby('instance_type').mean()
