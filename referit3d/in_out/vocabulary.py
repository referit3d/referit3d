import pickle
from collections import Counter


class Vocabulary(object):
    """Simple vocabulary wrapper."""
    def __init__(self, special_symbols=None):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0
        self.special_symbols = None
        self.intialize_special_symbols(special_symbols)

    def intialize_special_symbols(self, special_symbols):
        # Register special-symbols
        if special_symbols is None:
            self.special_symbols = ['<pad>', '<sos>', '<eos>', '<unk>']
        else:
            self.special_symbols = special_symbols

        # Map special-symbols to ints
        for s in self.special_symbols:
            self.add_word(s)

        # Add them as special symbols
        for s in self.special_symbols:
            name = s.replace('<', '')
            name = name.replace('>', '')
            setattr(self, name, self(s))

    def n_special(self):
        return len(self.special_symbols)

    def add_word(self, word):
        if word not in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def __call__(self, word):
        if word not in self.word2idx:
            return self.word2idx['<unk>']
        return self.word2idx[word]

    def __len__(self):
        return len(self.word2idx)

    def encode(self, text, max_len=None, add_begin_end=True):
        """
        :param text: (list) of tokens ['a', 'nice', 'sunset']
        :param max_len:
        :param add_begin_end:
        :return: (list) of encoded tokens.
        """
        encoded = [self(token) for token in text]
        if max_len is not None:
            encoded = encoded[:max_len]  # crop if too big

        if add_begin_end:
            encoded = [self('<sos>')] + encoded + [self('<eos>')]

        if max_len is not None:  # pad if too small (works because [] * [negative] does nothing)
            encoded += [self('<pad>')] * (max_len - len(text))
        return encoded

    def decode(self, tokens):
        return [self.idx2word[token] for token in tokens]

    def decode_print(self, tokens):
        exclude = set([self.word2idx[s] for s in ['<sos>', '<eos>', '<pad>']])
        words = [self.idx2word[token] for token in tokens if token not in exclude]
        return ' '.join(words)

    def __iter__(self):
        return iter(self.word2idx)

    def save(self, file_name):
        """ Save as a .pkl the current Vocabulary instance.
        :param file_name:  where to save
        :return: None
        """
        with open(file_name, mode="wb") as f:
            pickle.dump(self, f, protocol=2)  # protocol 2 => works both on py2.7  and py3.x

    @staticmethod
    def load(file_name):
        """ Load a previously saved Vocabulary instance.
        :param file_name: where it was saved
        :return: Vocabulary instance.
        """
        with open(file_name, 'rb') as f:
            vocab = pickle.load(f)
        return vocab


def build_vocab(token_list, min_word_freq):
    """Build a simple vocabulary wrapper."""

    counter = Counter()
    for tokens in token_list:
        counter.update(tokens)

    # If the word frequency is less than 'threshold', then the word is discarded.
    words = [word for word, cnt in counter.items() if cnt >= min_word_freq]

    # Create a vocab wrapper and add some special tokens.
    vocab = Vocabulary()

    # Add the words to the vocabulary.
    for i, word in enumerate(words):
        vocab.add_word(word)

    return vocab