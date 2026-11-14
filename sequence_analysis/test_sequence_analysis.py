"""Tests for the Milestone 2 code. Run from the repository root:  python -m pytest -q sequence_analysis

Tests that need the corpus under data/raw are skipped when it is missing (as in CI)."""
import numpy as np
import pytest

import corpus
import ngram
import restore
import run_milestone2 as m2
import stats


def test_selfchecks():
    ngram.selfcheck()
    stats.selfcheck()
    restore.selfcheck()


def test_restoration_bigram_matches_closed_form():
    # for a bigram, the generic scorer must equal P(w | prev) * P(next | w)
    texts = [(1, 2, 3), (1, 2, 4), (2, 3), (1, 3, 4), (4, 2, 3)] * 3
    vocab = [1, 2, 3, 4]
    got = m2.restoration(texts, vocab)
    hits = total = 0
    for train, test in ngram.folds(texts):
        P = ngram.transition_matrix(ngram.WittenBell(train, vocab, 2, backoff=True), vocab)
        for t in test:
            for j, true in enumerate(t):
                prev = 0 if j == 0 else vocab.index(t[j - 1]) + 1
                nxt = len(vocab) if j == len(t) - 1 else vocab.index(t[j + 1])
                hits += vocab[int(np.argmax(P[prev, :len(vocab)] * P[1:, nxt]))] == true
                total += 1
    assert got["trials"] == total and got["top1"] == hits / total


def test_positional_classes_and_order_rule():
    rng = np.random.default_rng(0)
    texts = [(1, *rng.integers(10, 20, 3), 9) for _ in range(60)]  # 1 always opens, 9 always closes
    cls = m2.positional_classes(texts)
    assert cls[1] == "initial" and cls[9] == "final"
    assert m2.class_order_rule(cls, texts) == (60, 60)
    assert m2.class_order_rule(cls, [t[::-1] for t in texts]) == (0, 60)


def test_recurring_segments():
    texts = [(5, 6, k) for k in range(10, 20)] + [(k, 7, 8) for k in range(20, 30)]
    segs = m2.recurring_segments(texts)
    assert (5, 6) in segs and (7, 8) in segs
    assert tuple(m2.coverage_by_segments(segs, [(1, 5, 6, 2)])) == (2, 4)


def test_dependence_orders_rigid_above_random():
    rng = np.random.default_rng(0)
    random_texts = [tuple(rng.integers(0, 50, 5)) for _ in range(400)]
    rigid = [tuple((s + i) % 50 for i in range(5)) for s in rng.integers(0, 50, 400)]
    assert m2.dependence(rigid) > 0.2 > abs(m2.dependence(random_texts))


@pytest.mark.skipif(not corpus.IM77_CSV.exists(), reason="IM77 corpus not downloaded")
def test_ebuds_reproduces_published_counts():
    texts, _ = corpus.ebuds(corpus.load_lines())
    assert (len(texts), sum(map(len, texts)), len({s for t in texts for s in t})) == corpus.EBUDS_PUBLISHED


@pytest.mark.skipif(not corpus.WELLS_SQL.exists(), reason="Wells/ICIT corpus not downloaded")
def test_wells_reading_order_ends_in_jar_sign():
    texts, head = corpus.load_wells()
    assert sum(t[-1] == 740 for t in texts) > 10 * sum(t[0] == 740 for t in texts)
    assert not set(head) & set(head.values())  # variant groups are one level deep


def test_segment_tokenizer_round_trips():
    segs = {(1, 2), (3, 4, 5)}
    for t in [(1, 2, 3, 4, 5, 6), (6, 1, 2), (3, 4), (1,)]:
        assert tuple(s for tok in m2.tokenize(t, segs, 3) for s in tok) == t
    assert m2.tokenize((1, 2, 3, 4, 5, 6), segs, 3) == [(1, 2), (3, 4, 5), (6,)]
