"""Pure-Python vocabulary for the translation notebook (not TorchScript)."""
from collections import Counter


class Vocab:
    def __init__(self, tokens):
        self._itos = list(tokens)
        self._stoi = {token: index for index, token in enumerate(self._itos)}
        self._default_index = None

    def __len__(self):
        return len(self._itos)

    def __contains__(self, token):
        return token in self._stoi

    def __getitem__(self, token):
        if token in self._stoi:
            return self._stoi[token]
        if self._default_index is not None:
            return self._default_index
        raise KeyError(token)

    def __call__(self, tokens):
        return self.lookup_indices(tokens)

    def lookup_indices(self, tokens):
        return [self[token] for token in tokens]

    def lookup_token(self, index):
        if index < 0 or index >= len(self):
            raise IndexError(index)
        return self._itos[index]

    def lookup_tokens(self, indices):
        return [self.lookup_token(index) for index in indices]

    def get_stoi(self):
        return dict(self._stoi)

    def get_itos(self):
        return list(self._itos)

    def set_default_index(self, index):
        self._default_index = index

    def get_default_index(self):
        return self._default_index


def build_vocab_from_iterator(iterator, min_freq=1, specials=None,
                              special_first=True, max_tokens=None):
    counter = Counter()
    for tokens in iterator:
        counter.update(tokens)
    specials = list(dict.fromkeys(specials or []))
    special_set = set(specials)
    tokens = [token for token, count in
              sorted(counter.items(), key=lambda item: (-item[1], item[0]))
              if count >= min_freq and token not in special_set]
    if max_tokens is not None:
        if max_tokens < len(specials):
            raise ValueError("max_tokens must accommodate all special tokens")
        tokens = tokens[:max_tokens - len(specials)]
    return Vocab(specials + tokens if special_first else tokens + specials)
