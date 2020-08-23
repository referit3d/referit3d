"""
I/O routines sr3d-oriented.

The MIT License (MIT)
Originally created at 7/1/20, for Python 3.x
Copyright (c) 2020 Panos Achlioptas (pachlioptas@gmail.com) & Stanford Geometric Computing Lab
"""

import pathlib
import pandas as pd
import os.path as osp


def load_sr3d_raw_data(sr3d_csv, drop_bad_context=True):
    """
    :param sr3d_csv:
    :param drop_bad_context:
    :return:
    """
    df = pd.read_csv(sr3d_csv)

    if drop_bad_context:
        # drop according to data/conditions discovered in Nr3D
        basedir = osp.split(pathlib.Path(__file__).parent.absolute())[0]
        bad_context = osp.join(basedir, 'data/language/nr3d/manually_inspected_bad_contexts.csv')
        bad_context = pd.read_csv(bad_context)
        bad_context = set(bad_context['stimulus_id'].unique())
        drop_mask = df.instance_type.apply(lambda x: x in ['clothes', 'clothing'])
        drop_mask |= df.stimulus_id.isin(bad_context)

        # drop according to data/conditions discovered in Sr3D
        bad_context = osp.join(basedir, 'data/language/sr3d/manually_inspected_bad_contexts.csv')
        bad_context = pd.read_csv(bad_context)
        bad_context = set(bad_context['stimulus_id'].unique())
        drop_mask |= df.stimulus_id.isin(bad_context)

        print('dropping ', (drop_mask.sum()), 'utterances marked manually as bad/poor context')
        df = df[~drop_mask]
        df.reset_index(inplace=True, drop=True)

    return df
