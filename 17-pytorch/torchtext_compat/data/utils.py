"""Small subset of torchtext tokenization used by the notebooks."""
from functools import partial


def _spacy_tokens(text, tokenizer):
    return [token.text for token in tokenizer(text)]


def get_tokenizer(tokenizer, language="en"):
    if callable(tokenizer):
        return tokenizer
    if tokenizer is None:
        return str.split
    if tokenizer == "spacy":
        import spacy
        nlp = spacy.load(language)
        return partial(_spacy_tokens, tokenizer=nlp.tokenizer)
    raise ValueError(f"Unsupported tokenizer: {tokenizer!r}")
