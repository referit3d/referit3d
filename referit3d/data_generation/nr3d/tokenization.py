from collections import defaultdict

sentence_spelling_dictionary = {
    'thewaytheshapeschangethespace': 'the way the shapes change the space'}

token_spelling_dictionary = {'furthest': 'farthest',
                             'further': 'farther',
                             'gray': 'grey',
                             'boardthat': ['board', 'that'],
                             'dressform': ['dress', 'form'],
                             'fridgerater': 'refrigerator',
                             'inbewteen': ['in', 'between'],
                             'lefthandside': ['left', 'hand', 'side'],
                             'lightsource': ['light', 'source'],
                             'middlemost': ['middle', 'most'],
                             'minifridge': ['mini', 'fridge'],
                             'nighttable': ['night', 'table'],
                             'redtrash': ['red', 'trash'],
                             'rightside': ['right', 'side'],
                             'thecounter': ['the', 'counter'],
                             'thewooden': ['the', 'wooden'],
                             'wallwithout': ['wall', 'without'],
                             'farthermost': ['farther', 'most'],
                             'lovechair': ['love', 'seat'],
                             'kotcjet': 'kitchen',
                             'shelfs': 'shelves',
                             'bookshelfs': 'bookshelves',
                             'isagainst': ['is', 'against'],
                             'andin': ['and', 'in'],
                             'alined': 'aligned',
                             'farest': 'farthest',
                             'seather': 'seater',
                             'bunkbed': ['bunk', 'bed'],
                             'loooong': 'long',
                             'thebed': ['the', 'bed'],
                             'racknext': ['rack', 'next'],
                             'pilow': 'pillow',
                             'itemon': ['item', 'on'],
                             'endtable': ['end', 'table'],
                             'redish': 'reddish',
                             'longways': ['long', 'ways'],
                             'furtherst': 'farthest',
                             'furtherest': 'farthest',
                             'theroom': ['the', 'room'],
                             'divders': 'dividers',
                             'sofabed': ['sofa', 'bed'],
                             'bathmat': ['bath', 'mat'],
                             'fatherst': 'farhest',
                             'pading': 'padding',
                             'theright': ['the', 'right'],
                             'orangeish': 'orangish',
                             'correctone': ['correct', 'one'],
                             'switchplates': ['switch', 'plates'],
                             'farmost': 'farthest',
                             'endcap': ['end', 'cap'],
                             'doorat': ['door', 'at'],
                             'verticle': 'vertical',
                             'besider': 'besides',
                             'beidge': 'beige',
                             'backwall': ['back', 'wall'],
                             'offour': ['of', 'four'],
                             'tablestand': ['table', 'stand'],
                             'suare': 'square',
                             'floormat': ['floor', 'mat'],
                             'iront': ['in', 'front'],
                             'doorwall': ['door', 'wall'],
                             'carboard': 'cardboard',
                             'planst': 'plant',
                             'ovular': 'oval',
                             'staute': 'statue',
                             'oneon': ['one', 'on'],
                             'coner': 'corner',
                             'sinl': 'sink',
                             'towl': 'towel',
                             'opne': 'open',
                             'opned': 'open',
                             'haning': 'hanging',
                             'sode': 'side',
                             'doorthe': ['door', 'the'],
                             'hightop': ['high', 'top'],
                             'papertowel': ['paper', 'towel'],
                             'coffeetables': ['coffee', 'tables'],
                             '1': 'one',
                             '2': 'two',
                             '3': 'three',
                             '4': 'four',
                             '5': 'five',
                             '6': 'six',
                             '7': 'seven',
                             '8': 'eight',
                             '9': 'nine',
                             '10': 'ten',
                             '11': 'eleven',
                             '12': 'twelve',
                             '1st': 'first',
                             '2nd': 'second',
                             '3rd': 'third',
                             'moonchair': ['moon', 'chair']
                             }

"""
A set of functions that are useful for pre-processing textual data: uniformizing the words, spelling, etc.
"""
import re

contractions_dict = {
    "ain't": "am not",
    "aren't": "are not",
    "can't": "cannot",
    "can't've": "cannot have",
    "'cause": "because",
    "could've": "could have",
    "couldn't": "could not",
    "couldn't've": "could not have",
    "didn't": "did not",
    "doesn't": "does not",
    "don't": "do not",
    "hadn't": "had not",
    "hadn't've": "had not have",
    "hasn't": "has not",
    "haven't": "have not",
    "he'd": "he had",
    "he'd've": "he would have",
    "he'll": "he will",
    "he'll've": "he will have",
    "he's": "he is",
    "how'd": "how did",
    "how'd'y": "how do you",
    "how'll": "how will",
    "how's": "how is",
    "i'd": "I had",
    "i'd've": "I would have",
    "i'll": "I will",
    "i'll've": "I will have",
    "i'm": "I am",
    "i've": "I have",
    "isn't": "is not",
    "it'd": "it had",
    "it'd've": "it would have",
    "it'll": "it will",
    "it'll've": "iit will have",
    "it's": "it is",
    "let's": "let us",
    "ma'am": "madam",
    "mayn't": "may not",
    "might've": "might have",
    "mightn't": "might not",
    "mightn't've": "might not have",
    "must've": "must have",
    "mustn't": "must not",
    "mustn't've": "must not have",
    "needn't": "need not",
    "needn't've": "need not have",
    "o'clock": "of the clock",
    "oughtn't": "ought not",
    "oughtn't've": "ought not have",
    "shan't": "shall not",
    "sha'n't": "shall not",
    "shan't've": "shall not have",
    "she'd": "she had",
    "she'd've": "she would have",
    "she'll": "she will",
    "she'll've": "she will have",
    "she's": "she is",
    "should've": "should have",
    "shouldn't": "should not",
    "shouldn't've": "should not have",
    "so've": "so have",
    "so's": "so is",
    "that'd": "that had",
    "that'd've": "that would have",
    "that's": "that is",
    "there'd": "there had",
    "there'd've": "there would have",
    "there's": "there is",
    "they'd": "they had",
    "they'd've": "they would have",
    "they'll": "they will",
    "they'll've": "they will have",
    "they're": "they are",
    "they've": "they have",
    "to've": "to have",
    "wasn't": "was not",
    "we'd": "we had",
    "we'd've": "we would have",
    "we'll": "we will",
    "we'll've": "we will have",
    "we're": "we are",
    "we've": "we have",
    "weren't": "were not",
    "what'll": "what will",
    "what'll've": "what will have",
    "what're": "what are",
    "what's": "what is",
    "what've": "what have",
    "when's": "when is",
    "when've": "when have",
    "where'd": "where did",
    "where's": "where is",
    "where've": "where have",
    "who'll": "who will",
    "who'll've": "who will have",
    "who's": "who is",
    "who've": "who have",
    "why's": "why is",
    "why've": "why have",
    "will've": "will have",
    "won't": "will not",
    "won't've": "will not have",
    "would've": "would have",
    "wouldn't": "would not",
    "wouldn't've": "would not have",
    "y'all": "you all",
    "y'all'd": "you all would",
    "y'all'd've": "you all would have",
    "y'all're": "you all are",
    "y'all've": "you all have",
    "you'd": "you had",
    "you'd've": "you would have",
    "you'll": "you will",
    "you'll've": "you will have",
    "you're": "you are",
    "you've": "you have"
}

CONTRACTION_RE = re.compile('({})'.format('|'.join(contractions_dict.keys())),
                            flags=re.IGNORECASE | re.DOTALL)


def expand_contractions(text, contractions=None, lower_i=True):
    """ Expand the contractions of the text (if any).
    Example: You're a good father. -> you are a good father.
    :param text: (string)
    :param contractions: (dict)
    :param lower_i: boolean, if True (I'm -> 'i am' not 'I am')
    :return: (string)

    Note:
        Side-effect: lower-casing. E.g., You're -> you are.
    """
    if contractions is None:
        contractions = contractions_dict  # Use one define in this .py

    def expand_match(contraction):
        match = contraction.group(0)
        expanded_contraction = contractions.get(match)
        if expanded_contraction is None:
            expanded_contraction = contractions.get(match.lower())
        if lower_i:
            expanded_contraction = expanded_contraction.lower()
        return expanded_contraction

    expanded_text = CONTRACTION_RE.sub(expand_match, text)
    return expanded_text


QUOTES_RE_STR = r"""(?:['|"][\w]+['|"])"""  # Words encapsulated in apostrophes.
QUOTES_RE = re.compile(r"(%s)" % QUOTES_RE_STR, flags=re.VERBOSE | re.IGNORECASE | re.UNICODE)


def unquote_words(s):
    """ 'king' - > king, "queen" -> queen """
    iterator = QUOTES_RE.finditer(s)
    new_sentence = list(s)
    for match in iterator:
        start, end = match.span()
        new_sentence[start] = ' '
        new_sentence[end - 1] = ' '
    new_sentence = "".join(new_sentence)
    return new_sentence


def manual_sentence_spelling(x, spelling_dictionary):
    """
    Applies spelling on an entire string, if x is a key of the spelling_dictionary.
    :param x: (string) sentence to potentially be corrected
    :param spelling_dictionary: correction map
    :return: the sentence corrected
    """
    if x in spelling_dictionary:
        return spelling_dictionary[x]
    else:
        return x


def manual_tokenized_sentence_spelling(tokens, spelling_dictionary):
    """
    :param tokens: (list of tokens) to potentially be corrected
    :param spelling_dictionary: correction map
    :return: a list of corrected tokens
    """
    new_tokens = []
    for token in tokens:
        if token in spelling_dictionary:
            res = spelling_dictionary[token]
            if type(res) == list:
                new_tokens.extend(res)
            else:
                new_tokens.append(res)
        else:
            new_tokens.append(token)
    return new_tokens


def token_spell_check(token, speller, corrected, missed_words, utterance=None, max_edit_distance=2):
    spells = speller.lookup(token, max_edit_distance)

    if len(spells) > 0:  # spell-check worked
        corrected[token].append(spells[0].term)
        return spells[0].term
    else:
        missed_words.add(token)
        return token


def pre_process_text(text, manual_sentence_speller, manual_token_speller,
                     tokenizer, golden_vocabulary, token_speller):
    missed_words = set()
    corrected = defaultdict(list)

    clean_text = text.apply(lambda x: manual_sentence_spelling(x, manual_sentence_speller))  # sentence-to-sentence map
    clean_text = clean_text.apply(lambda x: x.lower())
    clean_text = clean_text.apply(unquote_words)
    clean_text = clean_text.apply(expand_contractions)

    basic_punct = '.?!,:;/\-~*_='
    punct_to_space = str.maketrans(basic_punct, ' ' * len(basic_punct))  # map punctuation to space
    clean_text = clean_text.apply(lambda x: x.translate(punct_to_space))
    tokens = clean_text.apply(tokenizer)

    def spell_if_not_in_golden(token_list):
        new_tokens = []
        for token in token_list:
            if type(token) != str:
                assert False
            if token not in golden_vocabulary:
                new_tokens.append(token_spell_check(token, token_speller, corrected, missed_words, utterance=token_list))
            else:
                new_tokens.append(token)
        return new_tokens, corrected, missed_words

    spelled_tokens = tokens.apply(lambda x: manual_tokenized_sentence_spelling(x,
                                                                               spelling_dictionary=manual_token_speller)
                                  )
    spelled_tokens = spelled_tokens.apply(spell_if_not_in_golden)

    return clean_text, tokens, spelled_tokens
