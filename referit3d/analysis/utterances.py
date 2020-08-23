import numpy as np

from .word_meanings import instance_syn, instance_to_group, \
    group_members, to_singular, to_plural
from ..data_generation.nr3d import decode_stimulus_string

# TODO this is just a note. Logic for mentions_target_class
## This is not 100% bullet-proof -BECAUSE- scannet is not perfect. We try to keep the
# healthiest of utterances while keeping as many as we can in a semi-automatic way.
# AHMED. please keep a note of all this in the .py code. we will use it on Appendix too.

# if he mentions a generalization  + uniqueness-check  =>  then yes.
# if he mentions a specialization  => then yes.

# if he mentions a singular form + uniqueness-check => its yes.
# if he mentions a plural form  => its yes.  (# here we are generous. mauybe Scannet does have the plural in this room,
#                                             # but they have bad annotation. We keep in HOPE that the human gave a very
#                                             # descriptive utterance.)

def mentions_target_class(x, all_scans_dict, uniqueness_check=True):
    """ Does the utterance used to describe an object mention the object's instance type?
    :param x: row of a pandas.dataframe with an 'utterance' and 'instance_type' columns.
    :return: boolean
    """
    if isinstance((x['tokens']), str):
        utterance = ' '.join(eval(x['tokens']))
    else:
        utterance = ' '.join(x['tokens'])

    # Get the instance type
    stimulus_id = x['stimulus_id']
    scene_id, instance_label, _, _, _ = decode_stimulus_string(stimulus_id)

    scan_instances_occurences = all_scans_dict[scene_id].instance_occurrences()

    if instance_label in utterance:
        return True

    if (instance_label in to_plural) and to_plural[instance_label] in utterance:
        return True

    if (instance_label in to_singular) and to_singular[instance_label] in utterance:
        if uniqueness_check:
            if scan_instances_occurences[to_singular[instance_label]] == 0:
                return True
        else:
            return True

    # Synonyms specific to this instance-class
    for syn in instance_syn[instance_label]:
        if syn in utterance:
            assert syn not in ['', ' ']
            return True

    # Get the group this instance label belongs to and its related instances
    if instance_to_group[instance_label] is None:
        return False

    group_name = instance_to_group[instance_label]
    group_syns = instance_syn[group_name]
    related_group_members = group_members[group_name]

    assert group_name not in related_group_members
    assert len(np.unique(related_group_members)) == len(related_group_members)

    related_occurances = 0

    # See if one of the group classes is specifically mentioned instead
    for member in related_group_members:
        if member == instance_label:
            continue

        assert member != group_name

        member_occurance = scan_instances_occurences[member]
        related_occurances += member_occurance

        member_mentioned = member in utterance
        for member_syn in instance_syn[member]:
            assert member_syn not in ['', ' ']
            member_mentioned |= member_syn in utterance

        if member_mentioned:
            return True

    if related_occurances != 0:
        return False

    group_mentioned = group_name in utterance
    for group_syn in group_syns:
        group_mentioned |= group_syn in utterance

    assert not(group_name == instance_label and group_mentioned)

    return group_mentioned


def is_explicitly_view_dependent(df):
    """
    :param df: pandas dataframe with "tokens" columns
    :return: a boolean mask
    """
    target_words = {'front', 'behind', 'back', 'right', 'left', 'facing', 'leftmost', 'rightmost',
                    'looking', 'across'}
    return df.tokens.apply(lambda x: len(set(x).intersection(target_words)) > 0)
