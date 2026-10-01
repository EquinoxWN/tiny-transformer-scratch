"""Tokenizer behaviour: lossless round trips, deterministic training, persistence."""

import random

import pytest

from tiny_transformer_scratch.bpe import END_OF_TEXT, BPETokenizer

CORPUS = (
    (
        "Once upon a time, there was a little girl named Lily. She loved to play outside. "
        "One day, Lily saw a big red ball in the park. She was very happy! "
    )
    * 20
    + END_OF_TEXT
    + "Tom had a small dog. The dog liked to run and play in the sun. " * 20
)


@pytest.fixture(scope="module")
def tok():
    return BPETokenizer.train(CORPUS, vocab_size=300, special_tokens=[END_OF_TEXT])


def test_vocab_size_is_respected(tok):
    assert tok.vocab_size == 300
    assert len(tok.merges) == 300 - 256 - 1


def test_first_merge_is_most_frequent_pair():
    tok = BPETokenizer.train("aaab aaab aaab", vocab_size=257)
    assert tok.merges == [(ord("a"), ord("a"))]


def test_training_is_deterministic():
    a = BPETokenizer.train(CORPUS, vocab_size=280)
    b = BPETokenizer.train(CORPUS, vocab_size=280)
    assert a.merges == b.merges


def test_roundtrip_on_training_text(tok):
    assert tok.decode(tok.encode(CORPUS)) == CORPUS


@pytest.mark.parametrize(
    "text",
    [
        "",
        " ",
        "\n\n  \t",
        "café naïve 日本語 🙂👍🏽",
        "it's we'll they've I'm",
        "1234567890 3.14159 -42",
        "!!!??? ... ---",
    ],
)
def test_roundtrip_on_unseen_text(tok, text):
    assert tok.decode(tok.encode(text)) == text


def test_roundtrip_on_random_unicode(tok):
    gen = random.Random(7)
    alphabet = "abc XYZ\n\t.,!'é日🙂́"
    for _ in range(200):
        text = "".join(gen.choice(alphabet) for _ in range(gen.randint(0, 40)))
        assert tok.decode(tok.encode(text)) == text


def test_merges_compress(tok):
    text = "Once upon a time, Lily played with a big red ball."
    assert len(tok.encode(text)) < len(text.encode("utf-8")) * 0.75


def test_special_token_is_one_id(tok):
    ids = tok.encode("Hi" + END_OF_TEXT + "Bye")
    assert tok.special[END_OF_TEXT] in ids
    assert tok.decode(ids) == "Hi" + END_OF_TEXT + "Bye"


def test_save_and_load(tok, tmp_path):
    path = tmp_path / "tok.json"
    tok.save(path)
    loaded = BPETokenizer.load(path)
    assert loaded.merges == tok.merges
    assert loaded.special == tok.special
    assert loaded.encode(CORPUS) == tok.encode(CORPUS)


def test_rejects_too_small_vocab():
    with pytest.raises(ValueError, match="vocab_size must leave room"):
        BPETokenizer.train("abc", vocab_size=100)


def test_unknown_id_raises(tok):
    with pytest.raises(KeyError):
        tok.decode([10**6])
