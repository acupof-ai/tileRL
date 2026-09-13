"""mmlu_thinking_spec question-set hash: ordered-question fingerprint and
--pair refusal across mismatched sets."""

import importlib.util
import json
import os

import pytest

from tilerl.eval import mmlu_indices

# CPU gate has no HF dataset; exercise the shared sampler at the real MMLU test
# size (cais/mmlu "all"), so the historical-overlap figure is computed, not typed.
MMLU_TEST_SIZE = 14042

_spec = importlib.util.spec_from_file_location(
    "mmlu_thinking_spec",
    os.path.join(os.path.dirname(__file__), "..", "scripts", "mmlu_thinking_spec.py"),
)
mts = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mts)


def test_indices_use_the_single_sampler(monkeypatch):
    # Harness delegates to tilerl.eval.mmlu_indices rather than reimplementing
    # the sample (drift guard); monkeypatch avoids the HF dataset in CPU CI.
    sentinel = [5, 9, 27]
    monkeypatch.setattr(mts, "mmlu_indices", lambda n, seed: sentinel)
    assert mts.question_indices(27, 4) is sentinel


def test_n_changes_the_sample_historical_split():
    # The exact confound the hash guards: the guard arm ran --n 400 while the
    # dense comparison took the first 400 of a --n 2000 slice. Sorted samples
    # of different n are NOT nested prefixes — most rows differ. Computed from
    # the sampler, not a typed constant, so it tracks the real dataset size.
    i400 = mmlu_indices(400, 0, MMLU_TEST_SIZE)
    first400_of_2000 = mmlu_indices(2000, 0, MMLU_TEST_SIZE)[:400]
    assert mts.question_set_hash(i400) != mts.question_set_hash(first400_of_2000)
    overlap = len(set(i400) & set(first400_of_2000))
    assert overlap < 400 / 2  # majority differ; 80/400 at the current 14042 rows
    # A true prefix (same n, first-N) keeps full overlap and an ordered hash.
    assert mts.question_set_hash(i400[:128]) != mts.question_set_hash(i400)
    assert set(i400[:128]) <= set(i400)


def _arm(n_done, preds, gold):
    correct = sum(p == g for p, g in zip(preds, gold))
    return {"n_done": n_done, "done_idx": list(range(n_done)),
            "predictions": preds, "correct": correct,
            "accuracy": correct / n_done, "mean_output_tokens": 1,
            "elapsed_s": 1.0, "tok_s": 10.0,
            "spec_accepted": 0, "spec_drafted": 0}


def _doc(qhash, n, seed, arm_name, preds, gold):
    return {"n": n, "seed": seed, "qset_hash": qhash, "gold": gold,
            "arms": {arm_name: _arm(len(preds), preds, gold)}}


def test_hash_ordered_and_stable():
    h = mts.question_set_hash([3, 1, 2])
    assert h == mts.question_set_hash([3, 1, 2])
    assert h != mts.question_set_hash([1, 2, 3])  # order matters
    assert len(h) == 16


def test_pair_refuses_mismatched_sets(tmp_path):
    dense = _doc("aaaaaaaaaaaaaaa1", 400, 0, "dense",
                 ["A", "B"], ["A", "B"])
    sparse = _doc("bbbbbbbbbbbbbbb2", 2000, 0, "sparse",
                  ["A", "C"], ["A", "B"])
    d, s = tmp_path / "d.json", tmp_path / "s.json"
    d.write_text(json.dumps(dense))
    s.write_text(json.dumps(sparse))
    with pytest.raises(SystemExit, match="question-set mismatch"):
        mts.pair_arms(str(d), str(s), str(tmp_path / "out.json"))
    assert not (tmp_path / "out.json").exists()


def test_pair_refuses_hashless_old_jsons(tmp_path):
    dense = _doc(None, 400, 0, "dense", ["A"], ["A"])
    sparse = _doc(None, 400, 0, "sparse", ["A"], ["A"])
    for doc, name in ((dense, "d.json"), (sparse, "s.json")):
        del doc["qset_hash"]
        (tmp_path / name).write_text(json.dumps(doc))
    with pytest.raises(SystemExit, match="qset_hash"):
        mts.pair_arms(str(tmp_path / "d.json"), str(tmp_path / "s.json"),
                      str(tmp_path / "out.json"))


def test_pair_matched_set_reports(tmp_path):
    h = "ccccccccccccccc3"
    dense = _doc(h, 400, 0, "dense", ["A", "A"], ["A", "B"])
    sparse = _doc(h, 400, 0, "sparse", ["A", "B"], ["A", "B"])
    d, s = tmp_path / "d.json", tmp_path / "s.json"
    d.write_text(json.dumps(dense))
    s.write_text(json.dumps(sparse))
    out = tmp_path / "out.json"
    mts.pair_arms(str(d), str(s), str(out))
    paired = json.loads(out.read_text())["paired"]
    assert paired["n_common"] == 2
    assert paired["sparse_only_correct"] == 1
    assert paired["dense_only_correct"] == 0
