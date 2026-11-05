"""Behaviour gradient checks cannot see: values, graph mode, tokenizer rules and bad input."""

import json
import threading

import numpy as np
import pytest

from tiny_transformer_scratch import bpe
from tiny_transformer_scratch.autograd import Tensor, cross_entropy, no_grad
from tiny_transformer_scratch.bpe import BPETokenizer

# ---- autograd ---------------------------------------------------------------


def test_mean_over_an_axis_divides_by_that_axis_only():
    x = Tensor(np.arange(6.0).reshape(2, 3), requires_grad=True)
    m = x.mean(axis=1)
    np.testing.assert_allclose(m.data, [1.0, 4.0])
    m.sum().backward()
    np.testing.assert_allclose(x.grad, np.full((2, 3), 1 / 3))


def test_nested_no_grad_keeps_recording_off_until_the_outer_block_ends():
    x = Tensor([1.0], requires_grad=True)
    with no_grad():
        with no_grad():
            pass
        assert not (x * 2).requires_grad
    assert (x * 2).requires_grad


def test_no_grad_in_one_thread_does_not_stop_another_threads_graph():
    inside, done = threading.Event(), threading.Event()

    def inference():
        with no_grad():
            inside.set()
            done.wait(5)

    worker = threading.Thread(target=inference)
    worker.start()
    try:
        assert inside.wait(5)
        x = Tensor([1.0, 2.0], requires_grad=True)
        (x * x).sum().backward()  # this thread is training while the other runs inference
        np.testing.assert_allclose(x.grad, [2.0, 4.0])
    finally:
        done.set()
        worker.join(5)


@pytest.mark.parametrize(
    ("logits_shape", "targets"),
    [((3, 4), [0, 1]), ((3,), [0, 1, 2]), ((3, 4), [[0], [1], [2]])],
)
def test_cross_entropy_rejects_mismatched_shapes(logits_shape, targets):
    with pytest.raises(ValueError, match="expected logits"):
        cross_entropy(Tensor(np.zeros(logits_shape)), np.array(targets))


# ---- tokenizer rules ----------------------------------------------------------


def test_ties_go_to_the_smallest_pair():
    # (a, b) and (c, d) both occur twice; training must pick (97, 98) first, deterministically.
    tok = BPETokenizer.train("abab cdcd", vocab_size=257)
    assert tok.merges == [(97, 98)]


def test_pairs_seen_once_are_not_merged():
    assert BPETokenizer.train("abc", vocab_size=300).merges == []


def test_encoding_applies_the_earliest_merge_first():
    tok = BPETokenizer(merges=[(97, 98), (98, 99)])  # 256 = "ab", 257 = "bc"
    assert tok.encode("abc") == [256, 99]


# ---- tokenizer files ------------------------------------------------------------


def write_doc(tmp_path, **changes):
    doc = {"version": 1, "pattern": bpe.PATTERN, "special_tokens": {}, "merges": [[97, 98]]}
    doc.update(changes)
    path = tmp_path / "tok.json"
    path.write_text(json.dumps(doc), encoding="utf-8")
    return path


def test_unknown_file_version_is_refused(tmp_path):
    with pytest.raises(ValueError, match="unsupported tokenizer version"):
        BPETokenizer.load(write_doc(tmp_path, version=2))


@pytest.mark.parametrize(
    "merges",
    [[[97, 256]], [[97]], [[97, 98, 99]], [["a", "b"]], [[-1, 98]], [[True, 98]], "not a list"],
)
def test_malformed_merges_are_refused_with_a_clear_error(tmp_path, merges):
    with pytest.raises(ValueError, match="merge"):
        BPETokenizer.load(write_doc(tmp_path, merges=merges))


def test_a_merge_may_use_tokens_from_earlier_merges(tmp_path):
    tok = BPETokenizer.load(write_doc(tmp_path, merges=[[97, 98], [256, 99]]))
    assert tok.encode("abc") == [257]


def test_save_leaves_no_temporary_file(tmp_path):
    BPETokenizer(merges=[(97, 98)]).save(tmp_path / "tok.json")
    assert [p.name for p in tmp_path.iterdir()] == ["tok.json"]


# ---- command line ---------------------------------------------------------------


@pytest.mark.parametrize(
    "args",
    [
        ["train", "--vocab-size", "100"],
        ["train", "--max-bytes", "0"],
        ["stats", "--tokenizer", "t.json", "--max-bytes", "-1"],
        ["stats", "--tokenizer", "t.json", "--skip-bytes", "-5"],
    ],
)
def test_command_line_refuses_out_of_range_numbers(tmp_path, args, capsys):
    text = tmp_path / "in.txt"
    text.write_text("hello world", encoding="utf-8")
    extra = ["--out", str(tmp_path / "o.json")] if args[0] == "train" else []
    with pytest.raises(SystemExit) as info:
        bpe.main([*args, "--input", str(text), *extra])
    assert info.value.code == 2
    assert "must be at least" in capsys.readouterr().err


def test_stats_on_empty_input_is_a_clear_error(tmp_path, capsys):
    tok = tmp_path / "t.json"
    BPETokenizer(merges=[(97, 98)]).save(tok)
    empty = tmp_path / "empty.txt"
    empty.write_bytes(b"")
    with pytest.raises(SystemExit):
        bpe.main(["stats", "--tokenizer", str(tok), "--input", str(empty)])
    assert "no text read" in capsys.readouterr().err
