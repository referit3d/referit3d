import IPython
from IPython.core.display import display

from referit3d.utils.plotting import bold_string

server_prefix = 'https://gibsonannotation.com/view.html?sti='
server_prefix_no_boxes = 'http://gibsonannotation.com:3000/view.html?sti='


def visualize_stimulus_on_server(stimulus, context_highlighted=True, width=1280, height=720):
    prefix = server_prefix
    if not context_highlighted:
        prefix = server_prefix_no_boxes
    return IPython.display.IFrame(prefix + stimulus, width=width, height=height)


def show_stimuli_with_collected_language(stimulus, collected_data_df=None, show_lang=True, print_dataset=False, validation=False):
    # example, stimulus = 'scene0556_00-door-2-12-31',
    # collected_data_df, dataframe returned from ``load_nr3d_raw_data``
    if validation:
        server_prefix = "https://gibsonannotation.com/validation.html?sti="
    else:
        server_prefix = 'https://gibsonannotation.com/view.html?sti='
    display(IPython.display.IFrame(server_prefix + stimulus, width=700, height=350))

    if show_lang and collected_data_df is not None:
        stim_data = collected_data_df[collected_data_df.stimulus_id==stimulus]
        utterances = stim_data.utterance
        guesses = stim_data.correct_guess

        if print_dataset:
            dataset = stim_data.dataset

        for i in range(len(stim_data)):
            if print_dataset:
                print(bold_string('Correct guess= ' + str(guesses.iloc[i])), dataset.iloc[i],utterances.iloc[i])
            else:
                print(bold_string('Correct guess= ' + str(guesses.iloc[i])), utterances.iloc[i])
