"""Milestone 2: positional bias and Markov-chain modelling of Indus sign sequences (IM77 corpus).

    python sequence_analysis/run_milestone2.py            # everything; a few minutes on a laptop CPU
    python sequence_analysis/run_milestone2.py --quick    # fewer resamples, for a fast check

Writes reports/milestone_2/: results.json, CSV tables, figures, transition matrices, and
milestone_2_report.md. Only aggregate statistics are written, never the corpus texts.
"""
import argparse
import csv
import hashlib
import importlib.metadata
import json
import subprocess
import sys
from collections import Counter, defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import csr_matrix
from scipy.stats import chi2, fisher_exact

import ngram
import stats
from corpus import (EBUDS_PUBLISHED, IM77_CSV, REFERENCE, ROOT, WELLS_SQL, ebuds, ebuds_rows, load_lines, load_wells, physical,
                    reference_systems)
from ngram import BOS, EOS, AddK, KneserNey, Positional, Uniform, WittenBell

OUT = ROOT / "reports/milestone_2"
PUBLISHED = {  # Yadav et al. 2010 (PLoS ONE 5(3): e9506), all on EBUDS
    "perplexity": {1: 68.82, 2: 26.69, 3: 26.09, 4: 25.26, 5: 25.26},
    "unigram_entropy": 6.68, "bigram_mi": 2.24,
    "coverage80": {"all": 69, "beginners": 82, "enders": 23},
    "top_beginners": [267, 391, 293], "top_enders": [342, 176, 211],
    "top_pair": ((267, 99), 168, 792.40), "top_triplet": ((336, 89, 211), 34),
    "zipf_mandelbrot": (15.39, 2.59, 44.47), "restoration_90mass": 0.74,
}
M1_SIGN_ERROR_RATE = 0.237  # Milestone 1, YOLOv8n, SYNTHETIC test seals only


def mean_sd(x):
    return [float(np.mean(x)), float(np.std(x, ddof=1))]


def language_models(texts, vocab):
    """5-fold cross-validated cross-entropy for each model, plus Witten-Bell fitted and scored on
    the whole corpus (Yadav et al. do not state their split, so both protocols are reported)."""
    makers = {"uniform": Uniform, "positional (no sign-to-sign dependence)": Positional}
    for n in range(1, 6):
        makers[f"witten-bell backoff n={n}"] = lambda tr, v, n=n: WittenBell(tr, v, n, backoff=True)
        makers[f"witten-bell interpolated n={n}"] = lambda tr, v, n=n: WittenBell(tr, v, n)
    makers["add-1 n=2"] = AddK
    makers["add-0.01 n=2"] = lambda tr, v: AddK(tr, v, 0.01)
    makers["kneser-ney n=2"] = KneserNey
    out = {}
    for name, make in makers.items():
        with_eos, without = ngram.cross_validate(make, texts, vocab)
        ll, n = ngram.log2_likelihood(make(texts, vocab), texts)
        out[name] = {"cv_bits": mean_sd(with_eos), "cv_perplexity": 2 ** float(np.mean(with_eos)),
                     "cv_perplexity_no_eos": 2 ** float(np.mean(without)), "resubstitution_perplexity": 2 ** (-ll / n)}
    for n in range(1, 6):  # candidate explanation of the gap to the published table
        bits, skipped = ngram.cross_validate_skipping_unseen(lambda tr, v, n=n: WittenBell(tr, v, n, backoff=True), texts)
        out[f"witten-bell backoff n={n}"].update(skip_unseen_perplexity=2 ** float(np.mean(bits)), skip_unseen_share=skipped)
    return out


def restoration(texts, vocab, make=lambda tr, v: WittenBell(tr, v, 2, backoff=True)):
    """Remove one sign and rank every candidate by the model probability of the text with the candidate
    in its place, from both sides: only the terms that change are scored (the candidate's own probability
    and the next order-1 events). For a bigram, P(w | prev, next) is proportional to P(w | prev) * P(next | w).
    5-fold, every position of every test text."""
    idx = {s: i for i, s in enumerate(vocab)}
    top1 = top5 = in90 = strict = total = 0
    set_sizes = []
    for train, test in ngram.folds(texts):
        m = make(train, vocab)
        pairs = [(k, j) for k, t in enumerate(test) for j in range(len(t))]
        for (k, j), score in zip(pairs, ngram.candidate_scores(m, vocab, test, pairs)):
            true = test[k][j]
            order = np.argsort(-score)
            rank = int(np.where(order == idx[true])[0][0])
            k90 = int(np.searchsorted(np.cumsum(score[order]) / score.sum(), 0.9) + 1)
            top1, top5, in90, total = top1 + (rank == 0), top5 + (rank < 5), in90 + (rank < k90), total + 1
            strict += rank < k90 - 1  # the sign that crosses the 90% line not counted
            set_sizes.append(k90)
    return {"trials": total, "top1": top1 / total, "top5": top5 / total, "hit_in_90pct_mass_set": in90 / total,
            "hit_in_90pct_mass_set_strict": strict / total,
            "mean_size_of_90pct_set": float(np.mean(set_sizes))}


def permute_tokens(texts, rng):
    """Every token permuted across the corpus, text lengths kept: memoryless, same sign frequencies."""
    cuts = np.cumsum([len(t) for t in texts])[:-1]
    return [tuple(t) for t in np.split(rng.permutation(np.concatenate(texts)), cuts)]


def structure_test(texts, n_resample, seed=0):
    """Is sign order within a text non-random? Compare adjacent-sign mutual information with the
    same texts after shuffling the signs inside each one (keeps every text's length and contents)."""
    rng = np.random.default_rng(seed)
    observed = stats.mutual_information(texts)
    null = [stats.mutual_information([tuple(rng.permutation(t)) for t in texts]) for _ in range(n_resample)]
    memoryless = [stats.mutual_information(permute_tokens(texts, rng)) for _ in range(20)]
    return {"observed_bits": observed, "shuffled_mean_bits": float(np.mean(null)), "all_tokens_permuted_bits": float(np.mean(memoryless)),
            "shuffled_95": np.percentile(null, [2.5, 97.5]).tolist(),
            "p": float((np.sum(np.array(null) >= observed) + 1) / (n_resample + 1))}


def direction_analysis(lines, texts, n_resample):
    """Run the direction test on PHYSICAL left-to-right layout, recovered from each line's recorded
    direction. Stored order is reading order, so testing it directly would only return the
    transcriber's assumption."""
    d = lines[lines.direction.isin(["right-to-left", "left-to-right"]) & (lines.signs.map(len) >= 2)].copy()
    d["phys"] = [physical(r) for r in d.itertuples()]
    d = d[d.phys.map(lambda p: p[0] != 0 and p[-1] != 0)]  # both end signs must be legible

    def group(mask, dedupe=False):
        phys = list(d[mask].phys)
        if dedupe:
            phys = sorted(set(phys))
        return stats.direction_test(phys, n_resample) if len(phys) >= 30 else {"lines": len(phys), "skipped": "fewer than 30 lines"}

    rtl, ltr, single = d.direction == "right-to-left", d.direction == "left-to-right", d.line == "0"
    out = {"share labelled right-to-left": float(rtl.mean()),
        "primary: labelled right-to-left, single-line": group(rtl & single),
        "same, duplicate texts removed": group(rtl & single, dedupe=True),
        "labelled right-to-left, all lines": group(rtl),
        "labelled left-to-right, all lines": group(ltr),
        "labelled left-to-right, single-line": group(ltr & single),
    }
    for col in ("object_type", "site"):
        for value, count in d[rtl & single][col].value_counts().items():
            if count >= 100:
                out[f"right-to-left single-line, {col} = {value}"] = group(rtl & single & (d[col] == value))
    # The labels were plausibly assigned by looking at signs with strong end preferences (342 above all).
    # Drop every line with such a sign at either physical end and test what remains.
    begin, end = Counter(t[0] for t in texts), Counter(t[-1] for t in texts)
    for k in (1, 3, 5, 10, 20):
        markers = {s for s, _ in begin.most_common(k)} | {s for s, _ in end.most_common(k)}
        clean = d.phys.map(lambda p: p[0] not in markers and p[-1] not in markers).to_numpy()
        out[f"primary, no top-{k} text-beginner or ender at either end"] = group(rtl & single & clean)
    # Do the two labelled groups differ more than a random relabelling would produce?
    rng = np.random.default_rng(0)
    left, right = stats.terminals(list(d.phys))
    is_rtl = rtl.to_numpy()
    gap = lambda m: stats.asymmetry(left[m], right[m])[1] - stats.asymmetry(left[~m], right[~m])[1]
    null = [gap(rng.permutation(is_rtl)) for _ in range(n_resample)]
    out["label permutation"] = {"delta_S_gap_rtl_minus_ltr": float(gap(is_rtl)),
                                "null_95": np.percentile(null, [2.5, 97.5]).tolist()}
    for name, mask in (("right-to-left", rtl), ("left-to-right", ltr)):  # what the labels add
        stored = [r.signs for r in d[mask].itertuples()]
        out[f"stored reading order, lines labelled {name}"] = {"lines": len(stored), "delta_S": float(stats.asymmetry(*stats.terminals(stored))[1])}
    l_, r_ = stats.terminals(list(d[rtl & single].phys))
    out["primary delta_G over signs seen at either end"] = float(stats.asymmetry(l_, r_, alphabet=None)[0])
    jar = d[rtl & single].phys
    out["sign 342 in the primary group"] = {"at_physical_left": int(sum(p[0] == 342 for p in jar)),
                                            "at_physical_right": int(sum(p[-1] == 342 for p in jar)), "lines": len(jar)}
    jl = d[ltr].phys
    out["sign 342 in lines labelled left-to-right"] = {"at_physical_left": int(sum(p[0] == 342 for p in jl)),
                                                       "at_physical_right": int(sum(p[-1] == 342 for p in jl)), "lines": len(jl)}
    return out, list(d[rtl & single].phys)


def corrupt(seq, rate, vocab, rng):
    """Recognition errors: each sign is, with probability `rate`, substituted, deleted, or given a
    spurious neighbour on a random side (equal thirds). The random side keeps both ends equally exposed."""
    out = []
    for s in seq:
        if rng.random() < rate:
            kind = rng.integers(3)
            if kind == 0:
                out.append(int(rng.choice(vocab)))
            elif kind == 2:
                extra = int(rng.choice(vocab))
                out += [s, extra] if rng.random() < 0.5 else [extra, s]
        else:
            out.append(s)
    return tuple(out)


def error_propagation(texts, phys_lines, vocab, clean_pairs, repeats, seed=0):
    """How do recognition errors in the input change this milestone's conclusions?"""
    rng = np.random.default_rng(seed)
    top = {(a, b) for a, b, *_ in clean_pairs[:20]}  # ties at rank 20 make this figure approximate
    rows = []
    for rate in (0.0, 0.05, 0.10, 0.15, 0.20, 0.237, 0.30):
        dg, ds, ppl, kept, ender = [], [], [], [], []
        for _ in range(1 if rate == 0 else repeats):
            noisy_lines = [l for l in (corrupt(p, rate, vocab, rng) for p in phys_lines) if len(l) >= 2 and l[0] and l[-1]]
            g, s_ = stats.asymmetry(*stats.terminals(noisy_lines))
            dg.append(g)
            ds.append(s_)
            noisy = sorted({t for t in (corrupt(t, rate, vocab, rng) for t in texts) if t})
            ppl.append(2 ** float(np.mean(ngram.cross_validate(lambda tr, v: WittenBell(tr, v, 2, backoff=True), noisy, vocab)[0])))
            table, _ = stats.pair_table(noisy)
            kept.append(len(top & {(a, b) for a, b, *_ in table[:20]}) / 20)
            ender.append(Counter(t[-1] for t in noisy).most_common(1)[0][1] / len(noisy))
        rows.append({"error_rate": rate, "delta_G": mean_sd(dg) if len(dg) > 1 else [dg[0], 0.0],
                     "delta_S": mean_sd(ds) if len(ds) > 1 else [ds[0], 0.0],
                     "bigram_perplexity": float(np.mean(ppl)), "top20_pairs_retained": float(np.mean(kept)),
                     "share_of_texts_ending_in_top_ender": float(np.mean(ender))})
    return rows


def cv_stability(texts, vocab, seeds=10):
    """Perplexity over 10 different random 5-fold splits, and how often the bigram beats the positional
    baseline fold by fold."""
    makers = {"positional (no sign-to-sign dependence)": Positional, "kneser-ney n=2": KneserNey,
              **{f"witten-bell backoff n={n}": (lambda tr, v, n=n: WittenBell(tr, v, n, backoff=True)) for n in (1, 2, 3)}}
    folds = {name: [ngram.cross_validate(make, texts, vocab, seed=s)[0] for s in range(seeds)] for name, make in makers.items()}
    ppl = {name: [2 ** float(np.mean(f)) for f in fs] for name, fs in folds.items()}
    out = {name: {"perplexity_mean": float(np.mean(p)), "perplexity_sd_across_splits": float(np.std(p, ddof=1)),
                  "perplexity_range": [min(p), max(p)]} for name, p in ppl.items()}
    pos, big = np.ravel(folds["positional (no sign-to-sign dependence)"]), np.ravel(folds["witten-bell backoff n=2"])
    out["bigram better than positional"] = {"folds": int((big < pos).sum()), "of": len(big),
                                            "smallest_gain_bits": float((pos - big).min())}
    return out


def unseen_signs(texts, vocab):
    """How much of the bigram's held-out cross-entropy comes from signs absent from the training fold."""
    ll, n = Counter(), Counter()
    for train, test in ngram.folds(texts):
        m, known = WittenBell(train, vocab, 2, backoff=True), {s for t in train for s in t}
        for t in test:
            s = [BOS, *t, EOS]
            for a, b in zip(s, s[1:]):
                kind = "seen" if b == EOS or b in known else "unseen"
                ll[kind] -= np.log2(m.prob(b, (a,)))
                n[kind] += 1
    total_bits, total_n = ll["seen"] + ll["unseen"], n["seen"] + n["unseen"]
    return {"unseen_share_of_events": n["unseen"] / total_n, "unseen_share_of_bits": ll["unseen"] / total_bits,
            "bits_per_unseen_event": ll["unseen"] / n["unseen"], "bits_per_seen_event": ll["seen"] / n["seen"],
            "perplexity_all": 2 ** (total_bits / total_n), "perplexity_seen_only": 2 ** (ll["seen"] / n["seen"])}


def position_vs_bigram(texts, vocab):
    """Is the bigram's gain just position? All three models predict signs only, given that the text
    has not ended (end-of-text renormalised away), so none is credited for predicting the ending."""
    bits = {"position from the start": [], "position from both ends (knows the text's length)": [], "bigram (Witten-Bell backoff)": []}
    for train, test in ngram.folds(texts):
        pos, two, big = Positional(train, vocab), ngram.TwoEnded(train, vocab), WittenBell(train, vocab, 2, backoff=True)
        ll, n = np.zeros(3), 0
        for t in test:
            h = (BOS,)
            for i, w in enumerate(t):
                ll -= np.log2([pos.prob(w, h) / (1 - pos.prob(EOS, h)), two.prob(w, i, len(t)),
                               big.prob(w, h[-1:]) / (1 - big.prob(EOS, h[-1:]))])
                h, n = h + (w,), n + 1
        for k, v in zip(bits, ll / n):
            bits[k].append(float(v))
    return {k: {"bits_per_sign": mean_sd(v), "perplexity": 2 ** float(np.mean(v))} for k, v in bits.items()}


def cross_system(texts, refs, samples, seed=0):
    """The same statistics on the Indus corpus and on equally sized samples of systems of known type.
    Each reference stream is cut into 'texts' with the Indus length distribution (one random window per
    text), so sample size and text length, which bias every plug-in estimate, are matched."""
    rng = np.random.default_rng(seed)
    lengths = [len(t) for t in texts]
    window = lambda stream: [tuple(stream[s:s + n]) for s, n in zip(rng.integers(0, len(stream) - max(lengths), len(lengths)), lengths)]
    successor = rng.permutation(377)

    def rigid(n):
        s = [int(rng.integers(377))]
        while len(s) < n:
            s.append(int(successor[s[-1]]))
        return tuple(s)

    systems = {"Indus (IM77, EBUDS)": [texts], **{k: [window(v) for _ in range(samples)] for k, v in refs.items()},
               "Indus signs permuted across the corpus (memoryless)": [permute_tokens(texts, rng) for _ in range(samples)],
               "random, 377 equiprobable signs (Rao type 1)": [[tuple(rng.integers(0, 377, n)) for n in lengths] for _ in range(samples)],
               "rigid, each sign has one fixed successor (Rao type 2)": [[rigid(n) for n in lengths] for _ in range(samples)]}
    out = {}
    for name, corpora in systems.items():
        rows = []
        for c in corpora:
            rows.append([len({s for t in c for s in t}), stats.entropy(Counter(s for t in c for s in t).values()),
                         stats.conditional_entropy(c), dependence(c), dependence(c, top_n=20),
                         *[stats.conditional_entropy(c, n) for n in RAO_N]])
        m, sd = np.mean(rows, 0), np.std(rows, 0, ddof=1) if len(rows) > 1 else np.zeros(len(rows[0]))
        out[name] = {"signs_in_sample": float(m[0]), "unigram_entropy": float(m[1]), "conditional_entropy": float(m[2]),
                     "dependence_index": [float(m[3]), float(sd[3])], "dependence_index_top20": [float(m[4]), float(sd[4])],
                     "conditional_entropy_top_n": dict(zip(map(str, RAO_N), map(float, m[5:])))}
    return out


RAO_N = (2, 4, 10, 20, 50, 100, 200, 400)


def dependence(texts, top_n=None, shuffles=10, seed=0):
    """Share of the next sign's uncertainty removed by knowing the current one, beyond what is left
    after shuffling signs within each text (same length and contents):
    (H_shuffled(next | current) - H(next | current)) / H(sign). 0 = no order dependence.
    The plug-in H(next | current) is biased low when most pairs are seen once; the shuffle shares the
    bias, but a large vocabulary still compresses the index (see the report)."""
    rng = np.random.default_rng(seed)
    counts = Counter(s for t in texts for s in t)
    h = stats.entropy([c for _, c in counts.most_common(top_n)])
    shuffled = np.mean([stats.conditional_entropy([tuple(rng.permutation(t)) for t in texts], top_n) for _ in range(shuffles)])
    return float((shuffled - stats.conditional_entropy(texts, top_n)) / h)


def corpus_summary(texts, cv=True):
    """Headline statistics for one corpus variant, in forms comparable across sign lists."""
    vocab = sorted({s for t in texts for s in t})
    counts, end = Counter(s for t in texts for s in t), Counter(t[-1] for t in texts)
    table, _ = stats.pair_table(texts)
    attracting, avoiding = stats.significant_pairs(table)
    tested = sum(r[2] >= stats.MIN_PAIR_COUNT for r in table)
    out = {"texts": len(texts), "tokens": sum(counts.values()), "signs": len(vocab), "unigram_entropy": stats.entropy(counts.values()),
           "dependence_index": dependence(texts),
           "pairs_tested": tested, "pairs_attracting": attracting, "pairs_avoiding": avoiding,
           "top_ender": end.most_common(1)[0], "top_ender_share": end.most_common(1)[0][1] / len(texts),
           "enders_covering_80pct": stats.coverage(end.values()),
           "top20_pairs": [[a, b] for a, b, *_ in table[:20]]}
    if cv:
        pos = ngram.cross_validate(Positional, texts, vocab)[0]
        big = ngram.cross_validate(lambda tr, v: WittenBell(tr, v, 2, backoff=True), texts, vocab)[0]
        out.update(bigram_perplexity=2 ** float(np.mean(big)), bigram_perplexity_over_uniform=2 ** float(np.mean(big)) / (len(vocab) + 1),
                   bits_saved_over_positional=float(np.mean(pos) - np.mean(big)))
    return out


def corpus_variants(lines, wells, extra):
    base, _ = ebuds(lines)
    variants = {"IM77, EBUDS (main analysis)": corpus_summary(base),
                "IM77, doubtful readings kept": corpus_summary(ebuds(lines, keep_doubtful=True)[0]),
                "IM77, every copy of repeated texts kept": corpus_summary(ebuds(lines, keep_copies=True)[0], cv=False),
                **{name: corpus_summary(t) for name, t in extra.items()}}
    if wells:
        w, head = wells
        variants["Wells/ICIT (yajnadevam), signs as listed"] = corpus_summary(w)
        variants["Wells/ICIT, variant groups merged"] = corpus_summary(sorted({tuple(head.get(s, s) for s in t) for t in w}))
    top = {tuple(p) for p in variants["IM77, EBUDS (main analysis)"]["top20_pairs"]}
    for name, v in variants.items():
        if name.startswith("IM77"):
            v["top20_pairs_shared_with_main"] = len(top & {tuple(p) for p in v["top20_pairs"]})
    return variants


def replication_gaps(texts, lines):
    """Candidate explanations for the two published numbers not reproduced."""
    copies, _ = ebuds(lines, keep_copies=True)
    ln2 = float(np.log(2))
    mi = {"distinct texts": stats.mutual_information(texts),
          "distinct texts, start and end marks counted as signs": stats.mutual_information([(BOS, *t, EOS) for t in texts]),
          "every copy of repeated texts kept": stats.mutual_information(copies)}
    out = {"mutual_information": {k: {"bits": v, "nats": v * ln2} for k, v in mi.items()}, "perplexity_protocols": {}}
    for k in (5, 10):
        for cut in (1, 2):
            make = lambda n: (lambda tr, v: WittenBell(tr, v, n, backoff=True, min_count=cut))
            out["perplexity_protocols"][f"{k}-fold, " + ("no count cutoff" if cut == 1 else "SRILM default cutoff (order 3+ n-grams seen once dropped)")] = \
                [2 ** float(np.mean(ngram.cross_validate_skipping_unseen(make(n), texts, k=k)[0])) for n in range(1, 6)]
    return out


MIN_CLASS_COUNT = 25  # signs rarer than this get no class: expected initial/final counts near 5, where chi-square holds


def positional_classes(texts):
    """Classify each sign with at least MIN_CLASS_COUNT tokens by where it occurs in texts of two or more
    signs (reading order). Under no positional preference a sign in a text of length L is initial with
    probability 1/L, final 1/L, medial (L-2)/L; a sign whose observed counts differ from that (chi-square,
    2 d.o.f., Benjamini-Hochberg 5%) takes the class with the largest observed/expected ratio, else 'free'."""
    obs, exp = defaultdict(lambda: np.zeros(3)), defaultdict(lambda: np.zeros(3))
    for t in texts:
        if len(t) < 2:
            continue
        for i, s in enumerate(t):
            obs[s][0 if i == 0 else 2 if i == len(t) - 1 else 1] += 1
            exp[s] += [1 / len(t), (len(t) - 2) / len(t), 1 / len(t)]
    signs = [s for s in obs if obs[s].sum() >= MIN_CLASS_COUNT]
    p = [chi2.sf(float(((obs[s] - exp[s]) ** 2 / np.maximum(exp[s], 1e-9)).sum()), 2) for s in signs]
    q = stats.benjamini_hochberg(p) if signs else []
    return {s: ("initial", "medial", "final")[int(np.argmax(obs[s] / np.maximum(exp[s], 1e-9)))] if qq < 0.05 else "free"
            for s, qq in zip(signs, q)}


def class_order_rule(cls, texts):
    """Share of texts holding both an initial-class and a final-class sign in which every initial-class sign
    comes before every final-class sign (reading order)."""
    both = ok = 0
    for t in texts:
        ini = [i for i, s in enumerate(t) if cls.get(s) == "initial"]
        fin = [i for i, s in enumerate(t) if cls.get(s) == "final"]
        if ini and fin:
            both += 1
            ok += max(ini) < min(fin)
    return ok, both


def syntax(texts, n_resample, seed=0):
    """Item 1 of the proposal's 'syntax rules': positional sign classes, the order of classes within a text,
    and recurring multi-sign segments. Rules are learnt on training folds and checked on held-out texts
    against the same texts with their signs shuffled."""
    rng = np.random.default_rng(seed)
    cls = positional_classes(texts)
    counts = Counter(s for t in texts for s in t)
    by_class = defaultdict(list)
    for s, c in sorted(cls.items(), key=lambda kv: -counts[kv[0]]):
        by_class[c].append(s)
    # class transitions against within-text shuffles
    def class_pairs(seqs):
        labelled = [[cls.get(s, "rare") for s in t] for t in seqs]
        return Counter(p for l in labelled for p in zip(l, l[1:]))

    # null: medial signs shuffled, first and last kept, so positional preference alone is not counted as order
    inner = lambda t: t if len(t) < 3 else (t[0], *rng.permutation(t[1:-1]), t[-1])
    obs = class_pairs(texts)
    null = [class_pairs([inner(t) for t in texts]) for _ in range(n_resample // 5)]
    trans = []
    for pair in sorted(set(obs) | {p for c in null for p in c}):
        nv = np.array([c[pair] for c in null], float)
        trans.append({"first": pair[0], "second": pair[1], "observed": obs[pair], "shuffled_mean": float(nv.mean()),
                      "z": float((obs[pair] - nv.mean()) / max(nv.std(ddof=1), 1e-9))})
    # held-out checks: classes and segments from training folds only
    rule_obs = rule_shuf = rule_inner = np.zeros(2)
    seg_cov = seg_cov_shuf = np.zeros(2)
    for train, test in ngram.folds(texts):
        c = positional_classes(train)
        shuffled = [tuple(rng.permutation(t)) for t in test]
        rule_obs, rule_shuf = rule_obs + class_order_rule(c, test), rule_shuf + class_order_rule(c, shuffled)
        rule_inner = rule_inner + class_order_rule(c, [inner(t) for t in test])
        segs = recurring_segments(train)
        seg_cov, seg_cov_shuf = seg_cov + coverage_by_segments(segs, test), seg_cov_shuf + coverage_by_segments(segs, shuffled)
    segs = recurring_segments(texts)
    seg_pos = {}
    for seg, n in segs.most_common(15):
        k = len(seg)
        at = [(t, i) for t in texts for i in range(len(t) - k + 1) if t[i:i + k] == seg]
        where = Counter("whole text" if len(t) == k else "initial" if i == 0 else "final" if i == len(t) - k else "medial" for t, i in at)
        seg_pos[" ".join(map(str, seg))] = {"texts": len({t for t, _ in at}), **where}
    return {"classes": {k: {"signs": len(v), "examples": v[:12]} for k, v in by_class.items()},
            "classified_signs": len(cls), "token_share_classified": sum(counts[s] for s in cls) / sum(counts.values()),
            "class_transitions": sorted(trans, key=lambda r: -abs(r["z"])),
            "order_rule_heldout": {"texts_with_both": int(rule_obs[1]), "share_obeying": rule_obs[0] / rule_obs[1],
                                   "share_obeying_after_shuffle": rule_shuf[0] / rule_shuf[1],
                                   "share_obeying_ends_kept": rule_inner[0] / rule_inner[1]},
            "segments": {"distinct_recurring": len(segs), "top": seg_pos,
                         "heldout_token_coverage": seg_cov[0] / seg_cov[1], "heldout_token_coverage_after_shuffle": seg_cov_shuf[0] / seg_cov_shuf[1]}}


SEGMENT_MIN_PAIR, SEGMENT_MIN_PMI, SEGMENT_MIN_TEXTS = 3, 1.0, 3


def recurring_segments(texts):
    """Cut each text between adjacent signs that are not strongly associated (pair seen at least
    SEGMENT_MIN_PAIR times with PMI of at least SEGMENT_MIN_PMI bits); keep the pieces of two or more
    signs that recur in at least SEGMENT_MIN_TEXTS distinct texts. Returns Counter(segment -> texts)."""
    table, _ = stats.pair_table(texts)
    glue = {(a, b) for a, b, k, e, *_ in table if k >= SEGMENT_MIN_PAIR and np.log2(k / e) >= SEGMENT_MIN_PMI}
    found = Counter()
    for t in set(texts):
        piece, pieces = [t[0]], []
        for a, b in zip(t, t[1:]):
            if (a, b) in glue:
                piece.append(b)
            else:
                pieces.append(tuple(piece))
                piece = [b]
        pieces.append(tuple(piece))
        found.update({p for p in pieces if len(p) >= 2})
    return Counter({p: n for p, n in found.items() if n >= SEGMENT_MIN_TEXTS})


def coverage_by_segments(segs, texts):
    """(tokens inside an occurrence of a known segment, all tokens)."""
    covered = total = 0
    for t in texts:
        hit = np.zeros(len(t), bool)
        for seg in segs:
            k = len(seg)
            for i in range(len(t) - k + 1):
                if t[i:i + k] == seg:
                    hit[i:i + k] = True
        covered, total = covered + int(hit.sum()), total + len(t)
    return np.array([covered, total])


def count_matrix(seqs, items):
    """Sparse texts x items count matrix; items(t) lists the items (signs, pairs) of one text."""
    keys, rows, cols = {}, [], []
    for i, t in enumerate(seqs):
        for k in items(t):
            rows.append(i)
            cols.append(keys.setdefault(k, len(keys)))
    return csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(seqs), len(keys)))


def icon_group(desc):
    d = desc.lower()
    return "unicorn" if "unicorn" in d else "no field symbol" if d.startswith("no field") else "other field symbol"


def breakdown(lines, n_perm, seed=0):
    """Do sign use and sign order differ by site, object type or field symbol (the animal or motif on the
    object)? Per group: the headline statistics; per group against the rest: Jensen-Shannon divergence of
    sign and adjacent-pair frequencies, with a permutation p (group labels shuffled across texts).
    IM77 records no dates ('level' is excavation depth with a datum per site), so period cannot be tested."""
    rng = np.random.default_rng(seed)
    x = ebuds_rows(lines).copy()
    x["field symbol"] = x.fs_description.map(icon_group)
    x["site_"] = x.site.where(x.site.isin(["Mohenjodaro", "Harappa"]), "other sites")
    x["object"] = x.object_type.where(x.object_type.isin(["seal", "sealing", "miniature tablet", "copper tablet"]), "other objects")
    out = {}
    for name, col in (("site", "site_"), ("object type", "object"), ("field symbol", "field symbol")):
        rows = x[[col, "signs"]].drop_duplicates()  # one copy of a text per group
        labels, seqs = rows[col].to_numpy(), list(rows.signs)
        U, P = count_matrix(seqs, lambda t: t), count_matrix(seqs, lambda t: zip(t, t[1:]))
        js = lambda M, m: stats.js_divergence(M.T @ m.astype(float), M.T @ (~m).astype(float))
        groups = {}
        for g in sorted(set(labels)):
            m = labels == g
            if m.sum() < 100:
                continue
            obs = (js(U, m), js(P, m))
            null = np.array([(js(U, pm), js(P, pm)) for pm in (rng.permutation(m) for _ in range(n_perm))])
            groups[g] = {**{k: v for k, v in corpus_summary(sorted(set(t for t, k in zip(seqs, m) if k))).items() if k != "top20_pairs"},
                         "js_signs": obs[0], "js_signs_null_mean": float(null[:, 0].mean()), "p_signs": float(((null[:, 0] >= obs[0]).sum() + 1) / (n_perm + 1)),
                         "js_pairs": obs[1], "js_pairs_null_mean": float(null[:, 1].mean()), "p_pairs": float(((null[:, 1] >= obs[1]).sum() + 1) / (n_perm + 1))}
        out[name] = groups
    # which signs go with which field symbol (unicorn against every other group with a field symbol)
    rows = x[x["field symbol"] != "no field symbol"][["field symbol", "signs"]].drop_duplicates()
    uni_m = (rows["field symbol"] == "unicorn").to_numpy()
    a, b = Counter(s for t, k in zip(rows.signs, uni_m) if k for s in set(t)), Counter(s for t, k in zip(rows.signs, uni_m) if not k for s in set(t))
    na, nb = int(uni_m.sum()), int((~uni_m).sum())
    assoc = [(s, a[s], b[s], stats.g2(a[s], na - a[s], b[s], nb - b[s])) for s in set(a) | set(b) if a[s] + b[s] >= 10]
    q = stats.benjamini_hochberg([chi2.sf(g, 1) for *_, g in assoc])
    out["signs by field symbol"] = {"texts_unicorn": na, "texts_other_field_symbol": nb, "tested_signs": len(assoc),
                                    "significant": sorted([{"sign": int(s), "texts_unicorn": int(ka), "texts_other": int(kb),
                                                            "share_unicorn": ka / na, "share_other": kb / nb, "q": float(qq)}
                                                           for (s, ka, kb, _), qq in zip(assoc, q) if qq < 0.05], key=lambda r: r["q"])}
    return out


def multiline(lines, texts):
    """Multi-line texts (excluded from EBUDS): does a line break behave like the end of a text? Compares how
    often the sign before a line break, and the sign after it, is one of the signs that typically end or begin
    texts. Lines are taken in their recorded order; a text's lines are read line 1 first."""
    end, begin = Counter(t[-1] for t in texts), Counter(t[0] for t in texts)
    enders = {s for s, _ in end.most_common(stats.coverage(end.values()))}
    beginners = {s for s, _ in begin.most_common(stats.coverage(begin.values()))}
    groups = [g.sort_values("line") for _, g in lines[lines.line != "0"].groupby(["text_no", "side"])]
    clean = [g for g in groups if not g.damaged.any() and not g.doubtful.any() and len(g) >= 2]
    before_break = [g.signs.iloc[i][-1] for g in clean for i in range(len(g) - 1)]
    after_break = [g.signs.iloc[i][0] for g in clean for i in range(1, len(g))]
    text_end = [g.signs.iloc[-1][-1] for g in clean]
    medial = [s for t in texts for s in t[1:-1]]
    share = lambda xs, ref: {"n": len(xs), "share": float(np.mean([s in ref for s in xs]))}
    k, n = sum(s in enders for s in before_break), len(before_break)
    km, nm = sum(s in enders for s in medial), len(medial)
    out = {"groups": len(groups), "groups_fully_legible": len(clean),
           "alternating_direction": sum({"right-to-left", "left-to-right"} <= set(g.direction) for g in groups),
           "ender_sets_size": len(enders), "beginner_set_size": len(beginners),
           "sign before a line break is a typical ender": share(before_break, enders),
           "last sign of a multi-line text is a typical ender": share(text_end, enders),
           "medial sign of a single-line text is a typical ender": share(medial, enders),
           "sign after a line break is a typical beginner": share(after_break, beginners),
           "share of later lines that are a single sign": float(np.mean([len(g.signs.iloc[i]) == 1 for g in clean for i in range(1, len(g))])),
           "p_before_break_vs_medial": float(fisher_exact([[k, n - k], [km, nm - km]])[1])}
    joined = [sum(g.signs, ()) for g in clean]
    separate = [s for g in clean for s in g.signs]
    variants = {"IM77, multi-line texts added, lines joined": sorted(set(texts) | set(joined)),  # sequences: never written to disk
                "IM77, multi-line texts added, each line a text": sorted(set(texts) | set(separate))}
    return out, variants


def estimators(texts, phys, seed=0):
    """The entropy-based statistics under three estimators, with the all-tokens-permuted corpus as a bias
    check: a perfect estimator would give that corpus zero mutual information."""
    rng = np.random.default_rng(seed)
    perm = permute_tokens(texts, rng)
    est = {"plug-in": stats.entropy, "Miller-Madow": lambda c: stats.entropy(c, True), "Chao-Shen": stats.entropy_chao_shen}
    out = {}
    for name, h in est.items():
        out[name] = {"unigram_entropy": h(Counter(s for t in texts for s in t).values()), "mutual_information": stats.mutual_information(texts, h),
                     "mutual_information_permuted_corpus": stats.mutual_information(perm, h),
                     "conditional_entropy": stats.conditional_entropy(texts, h=h),
                     "direction_delta_S_primary": float(stats.asymmetry(*stats.terminals(phys), h=h)[1])}
    return out


def sample_size_curves(texts, reps, seed=0):
    """Headline statistics on random subsets of the corpus, to see whether they have levelled off."""
    rng = np.random.default_rng(seed)
    out = []
    for frac in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0):
        rows = []
        for _ in range(1 if frac == 1.0 else reps):
            sub = [texts[i] for i in sorted(rng.choice(len(texts), int(round(frac * len(texts))), replace=False))]
            vocab = sorted({s for t in sub for s in t})
            counts = Counter(s for t in sub for s in t)
            table, _ = stats.pair_table(sub)
            att, _ = stats.significant_pairs(table)
            rows.append([stats.entropy(counts.values()), stats.entropy_chao_shen(counts.values()), stats.mutual_information(sub),
                         stats.mutual_information(sub, stats.entropy_chao_shen), dependence(sub),
                         2 ** float(np.mean(ngram.cross_validate(lambda tr, v: WittenBell(tr, v, 2, backoff=True), sub, vocab)[0])),
                         Counter(t[-1] for t in sub).most_common(1)[0][1] / len(sub), att, len(vocab)])
        m = np.mean(rows, 0)
        out.append({"fraction": frac, "texts": int(round(frac * len(texts))), **dict(zip(SIZE_METRICS, map(float, m)))})
    return out


SIZE_METRICS = ("unigram_entropy_plugin", "unigram_entropy_chao_shen", "mi_plugin", "mi_chao_shen", "dependence_index",
                "bigram_perplexity", "top_ender_share", "attracting_pairs", "signs_seen")


def resegmentation(texts, reps, seed=0):
    """No machine-readable list of Mahadevan's compound signs (ligatures) and their components was found, so
    the effect of flattening them differently is simulated. 'split': a share of sign types is treated as
    compounds and each occurrence becomes two components drawn from the 50 commonest signs (order fixed per
    type). 'merged': the same share of types is merged pairwise into one sign."""
    rng = np.random.default_rng(seed)
    types = sorted({s for t in texts for s in t})
    common = [s for s, _ in Counter(s for t in texts for s in t).most_common(50)]
    out = {}
    for share in (0.05, 0.1, 0.2):
        for kind in ("split", "merged"):
            rows = []
            for _ in range(reps):
                chosen = rng.choice(types, int(share * len(types)), replace=False)
                if kind == "split":
                    m = {int(s): tuple(int(c) for c in rng.choice(common, 2)) for s in chosen}
                    new = [tuple(c for s in t for c in m.get(s, (s,))) for t in texts]
                else:
                    m = {int(a): int(b) for a, b in zip(chosen[::2], chosen[1::2])}
                    new = [tuple(m.get(s, s) for s in t) for t in texts]
                v = corpus_summary(sorted(set(new)))
                rows.append([v["dependence_index"], v["pairs_attracting"] / v["pairs_tested"], v["top_ender_share"], v["bits_saved_over_positional"]])
            r = np.array(rows)
            m_, sd = r.mean(0), r.std(0, ddof=1)
            out[f"{kind}, {share:.0%} of sign types"] = {**{k: [float(a), float(b)] for k, a, b in
                                                           zip(("dependence_index", "attracting_share", "top_ender_share", "bits_saved_over_positional"), m_, sd)},
                                                        "same_direction": bool(((r[:, 0] > 0) & (r[:, 1] > 0.5) & (r[:, 3] > 0)).all())}
    return out


def adjust_all(res):
    """Every p-value reported as a test in this report, adjusted together: Benjamini-Hochberg (false
    discovery rate) and Holm (family-wise error). Tests already corrected internally (the pair table,
    signs by field symbol, positional classes) enter as their count of significant results, not here."""
    ps = {"sign order vs within-text shuffle (mutual information)": res["structure_test"]["p"]}
    for name, g in res["direction"].items():
        if isinstance(g, dict) and isinstance(g.get("delta_S"), dict):
            ps[f"direction: {name}"] = g["delta_S"]["p_shuffle"]
    for grouping, groups in res["breakdown"].items():
        if grouping != "signs by field symbol":
            for g, v in groups.items():
                ps[f"{grouping} = {g}: sign frequencies differ from the rest"] = v["p_signs"]
                ps[f"{grouping} = {g}: adjacent pairs differ from the rest"] = v["p_pairs"]
    ps["sign before a line break is ender-like more than a medial sign"] = res["multiline"]["p_before_break_vs_medial"]
    names, p = list(ps), np.array(list(ps.values()))
    return [{"test": n, "p": float(pp), "q_bh": float(qq), "p_holm": float(hh)}
            for n, pp, qq, hh in zip(names, p, stats.benjamini_hochberg(p), stats.holm(p))]


def provenance(refs):
    """What produced these numbers: versions and input checksums."""
    files = [IM77_CSV, WELLS_SQL, *sorted(REFERENCE.glob("*.*"))]
    sha = lambda f: hashlib.sha256(f.read_bytes()).hexdigest()
    return {"python": sys.version.split()[0], "packages": {p: importlib.metadata.version(p) for p in ("numpy", "scipy", "pandas", "matplotlib")},
            "git_commit": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=ROOT).stdout.strip(),
            "uncommitted_changes": bool(subprocess.run(["git", "status", "--porcelain", "sequence_analysis"], capture_output=True, text=True, cwd=ROOT).stdout.strip()),
            "inputs_sha256": {str(f.relative_to(ROOT)): sha(f) for f in files if f.exists()},
            "essen_krn_files_sha256": hashlib.sha256(b"".join(f.read_bytes() for f in sorted((REFERENCE / "essen").rglob("*.krn")))).hexdigest(),
            "python_stdlib_tokens_sha256": hashlib.sha256("\x00".join(refs["Python source, tokens"]).encode()).hexdigest() if refs else None}


def figures(res, texts, vocab, P_mle, null_samples):
    counts = Counter(s for t in texts for s in t)
    f = np.sort(np.array(list(counts.values()), float))[::-1]
    r = np.arange(1, len(f) + 1)
    a, b, c = res["zipf_mandelbrot_fit"]
    plt.figure(figsize=(5, 4))
    plt.loglog(r, f, ".", label="EBUDS sign counts")
    plt.loglog(r, np.exp(a - b * np.log(r + c)), label=f"Zipf-Mandelbrot fit (b={b:.2f}, c={c:.1f})")
    plt.xlabel("rank"); plt.ylabel("count"); plt.legend(); plt.tight_layout(); plt.savefig(OUT / "zipf.png", dpi=130); plt.close()

    top = [s for s, _ in counts.most_common(25)]
    idx = [vocab.index(s) for s in top]
    plt.figure(figsize=(9, 7.5))
    plt.imshow(P_mle[np.ix_([0] + [i + 1 for i in idx], idx + [len(vocab)])], cmap="viridis", aspect="auto")
    plt.xticks(range(26), [*top, "END"], rotation=90, fontsize=7); plt.yticks(range(26), ["START", *top], fontsize=7)
    plt.xlabel("next sign (reading order)"); plt.ylabel("current sign"); plt.colorbar(label="P(next | current), maximum likelihood")
    plt.title("Transition probabilities, 25 most frequent signs"); plt.tight_layout(); plt.savefig(OUT / "transition_heatmap.png", dpi=130); plt.close()

    lm = res["language_models"]
    plt.figure(figsize=(6, 4))
    orders = range(1, 6)
    plt.plot(orders, [lm[f"witten-bell backoff n={n}"]["cv_perplexity"] for n in orders], "o-", label="Witten-Bell backoff, 5-fold CV")
    plt.plot(orders, [lm[f"witten-bell interpolated n={n}"]["cv_perplexity"] for n in orders], "s-", label="Witten-Bell interpolated, 5-fold CV")
    plt.plot(orders, [PUBLISHED["perplexity"][n] for n in orders], "k--", label="Yadav et al. 2010")
    plt.xticks(list(orders)); plt.xlabel("n-gram order"); plt.ylabel("perplexity"); plt.legend(); plt.tight_layout(); plt.savefig(OUT / "perplexity_by_order.png", dpi=130); plt.close()

    primary = res["direction"]["primary: labelled right-to-left, single-line"]["delta_S"]
    plt.figure(figsize=(6, 4))
    plt.hist(null_samples, bins=40, color="0.7", label="signs shuffled within each line")
    plt.axvline(primary["value"], color="C3", label=f"observed = {primary['value']:+.3f}")
    plt.axvspan(primary["value"] - 2 * primary["bootstrap_se"], primary["value"] + 2 * primary["bootstrap_se"], color="C3", alpha=0.2, label="+/- 2 bootstrap SE")
    plt.xlabel("delta S (negative = right-to-left)"); plt.legend(); plt.tight_layout(); plt.savefig(OUT / "direction_null.png", dpi=130); plt.close()

    ep = res["error_propagation"]
    x = [row["error_rate"] for row in ep]
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.8))
    ax[0].errorbar(x, [row["delta_S"][0] for row in ep], yerr=[2 * row["delta_S"][1] for row in ep], marker="o")
    ax[0].axhline(0, color="k", lw=0.5); ax[0].set_ylabel("delta S (direction signal; 0 = none)")
    ax[1].plot(x, [row["bigram_perplexity"] for row in ep], "o-"); ax[1].set_ylabel("bigram perplexity (5-fold CV)")
    ax[2].plot(x, [row["top20_pairs_retained"] for row in ep], "o-"); ax[2].set_ylabel("share of top-20 pairs retained")
    for a_ in ax:
        a_.axvline(M1_SIGN_ERROR_RATE, color="C3", ls=":", label="Milestone 1 error rate (synthetic)"); a_.set_xlabel("injected sign error rate")
    ax[0].legend(fontsize=7); plt.tight_layout(); plt.savefig(OUT / "error_propagation.png", dpi=130); plt.close()

    if res["cross_system"]:
        plt.figure(figsize=(7, 4.5))
        for name, v in res["cross_system"].items():
            ns = [n for n in RAO_N if n <= v["signs_in_sample"]]
            plt.semilogx(ns, [v["conditional_entropy_top_n"][str(n)] for n in ns], "o-", lw=3 if name.startswith("Indus (") else 1, label=name)
        plt.xlabel("number of most frequent signs kept (N)"); plt.ylabel("H(next | current), bits")
        plt.title("Conditional entropy as in Rao et al. 2009, samples matched to the Indus corpus", fontsize=9)
        plt.legend(fontsize=6); plt.tight_layout(); plt.savefig(OUT / "conditional_entropy_systems.png", dpi=130); plt.close()

    ss = res["sample_size"]
    fig, ax = plt.subplots(2, 3, figsize=(13, 7))
    panels = [(("unigram_entropy_plugin", "unigram_entropy_chao_shen"), "sign entropy, bits"), (("mi_plugin", "mi_chao_shen"), "adjacent-sign MI, bits"),
              (("dependence_index",), "dependence index"), (("bigram_perplexity",), "bigram perplexity (5-fold CV)"),
              (("top_ender_share",), "share of texts ending in the top ender"), (("attracting_pairs", "signs_seen"), "count")]
    for a_, (keys, label) in zip(ax.ravel(), panels):
        for k in keys:
            a_.plot([r["texts"] for r in ss], [r[k] for r in ss], "o-", label=k.replace("_", " "))
        a_.set_xlabel("texts in the subsample"); a_.set_ylabel(label); a_.legend(fontsize=7)
    plt.tight_layout(); plt.savefig(OUT / "sample_size.png", dpi=130); plt.close()


def write_report(res):
    lm, d, pub, gaps = res["language_models"], res["direction"], PUBLISHED, res["replication_gaps"]
    L = ["# Milestone 2 report: positional bias and Markov-chain modelling", "",
         "Generated by `sequence_analysis/run_milestone2.py` from the IM77 corpus (Mahadevan 1977). "
         "Signs are Mahadevan numbers; sequences are in reading order unless stated.", "",
         "**What this does and does not show.** The results show that sign order in the inscriptions is structured and "
         "non-random, and that the layout is consistent with right-to-left reading. They are not evidence that the "
         "inscriptions encode a language: published work disputes whether any of these statistics separates writing "
         "from structured non-linguistic sign systems (Sproat 2014; Rao et al. 2015), and an unreviewed 2026 preprint "
         "(Raghavendra, arXiv:2608.02999) reports that an invented non-linguistic system reproduces them.", "",
         "## Corpus", "", "| Filter step | Lines | Distinct texts | Sign tokens | Signs |", "|---|---|---|---|---|"]
    for name, c in res["corpus_steps"]:
        L.append(f"| {name} | {c['rows']} | {c['unique_texts']} | {c['tokens']} | {c['signs']} |")
    e = res["ebuds"]
    L += ["", f"Keeping one copy of each text after the last step gives **{e['texts']} texts, {e['tokens']} sign tokens, {e['signs']} signs**. "
          f"Yadav et al. 2010 report {EBUDS_PUBLISHED[0]}, {EBUDS_PUBLISHED[1]} and {EBUDS_PUBLISHED[2]} for their filtered corpus (EBUDS)"
          + (", so the filter is reproduced exactly." if (e["texts"], e["tokens"], e["signs"]) == EBUDS_PUBLISHED else "; this is an EBUDS-style filter, not an exact reproduction."),
          "All statistics below use this filtered corpus except the direction test, which needs the per-line layout.", "",
          "## Replication of Yadav et al. 2010", "", "| Quantity | Published | This run |", "|---|---|---|"]
    r = res["replication"]
    for n in range(1, 6):
        m = lm[f"witten-bell backoff n={n}"]
        L.append(f"| Perplexity, order {n} (Witten-Bell, backoff) | {pub['perplexity'][n]} | {m['cv_perplexity']:.2f} (every held-out sign scored); {m['skip_unseen_perplexity']:.2f} (unseen signs skipped) |")
    L += [f"| Unigram entropy (bits) | {pub['unigram_entropy']} | {r['unigram_entropy']:.2f} plug-in; {r['unigram_entropy_miller_madow']:.2f} Miller-Madow |",
          f"| Adjacent-sign mutual information | {pub['bigram_mi']} (unit not stated) | {r['bigram_mi']:.2f} bits (plug-in); "
          f"{gaps['mutual_information']['every copy of repeated texts kept']['nats']:.2f} in nats with every copy of repeated texts kept |",
          f"| Distinct adjacent sign pairs observed | 2,178 (Sinha et al. 2009) | {res['pairs']['distinct']} |",
          f"| Signs covering 80% of all tokens / text-beginners / text-enders | 69 / 82 / 23 | {r['coverage80']['all']} / {r['coverage80']['beginners']} / {r['coverage80']['enders']} |",
          f"| Three most frequent text-beginners | {pub['top_beginners']} | {r['top_beginners']} |",
          f"| Three most frequent text-enders | {pub['top_enders']} | {r['top_enders']} |",
          f"| Most frequent pair, count, log-likelihood ratio | 267 then 99, 168, 792.40 | {r['top_pair'][0]} then {r['top_pair'][1]}, {r['top_pair'][2]}, {r['top_pair'][3]:.2f} |",
          f"| Most frequent triplet, count | 336, 89, 211; 34 | {r['top_triplet'][0]}; {r['top_triplet'][1]} |",
          f"| Zipf-Mandelbrot fit (a, b, c) | 15.39, 2.59, 44.47 | {res['zipf_mandelbrot_fit'][0]:.2f}, {res['zipf_mandelbrot_fit'][1]:.2f}, {res['zipf_mandelbrot_fit'][2]:.2f} |",
          f"| Damaged-sign restoration, true sign inside the 90%-probability candidate set | 74% | {100 * res['restoration']['hit_in_90pct_mass_set']:.1f}%; {100 * res['restoration']['hit_in_90pct_mass_set_strict']:.1f}% if the sign that crosses the 90% line is not counted |",
          "", "Notes on comparability:", "",
          "- **Perplexity.** The headline figure scores every held-out sign over a fixed vocabulary, and is higher than published. "
          f"The second figure takes the vocabulary from each training fold and skips held-out signs never seen in training "
          f"({100 * lm['witten-bell backoff n=2']['skip_unseen_share']:.1f}% of events), which is the default behaviour of the SRILM toolkit the paper used. "
          "It lands close to the published values for orders 1 to 3, so it is a candidate explanation of the gap, not a confirmed one: the paper "
          "does not state its split or options. Skipping drops the hardest events, so the headline figure is the more honest measure. "
          "Orders 4 and 5 are not reproduced under any of the four protocols tried (see Replication gaps below), so that gap is closed as unexplained.",
          "- **Mutual information.** The paper does not define its estimator or unit. Computed in nats on the single-line corpus before "
          f"repeated texts are reduced to one copy, the plug-in value is {gaps['mutual_information']['every copy of repeated texts kept']['nats']:.3f}, "
          "which matches the published 2.24 to the stated precision. This is the likely explanation, but it rests on a numerical match only. "
          f"The plug-in value is biased upward: with every token permuted across the corpus it is still {res['structure_test']['all_tokens_permuted_bits']:.2f} bits.",
          "- **Restoration.** The published rule can be read two ways at the 90% boundary, and the two readings differ by several points.", "",
          "## Language models compared (5-fold cross-validation over distinct texts)", "",
          "| Model | Bits per event (mean, SD over folds) | Perplexity | Perplexity without end-of-text |", "|---|---|---|---|"]
    for name, m in lm.items():
        L.append(f"| {name} | {m['cv_bits'][0]:.3f}, {m['cv_bits'][1]:.3f} | {m['cv_perplexity']:.2f} | {m['cv_perplexity_no_eos']:.2f} |")
    s = res["structure_test"]
    pos, big = lm["positional (no sign-to-sign dependence)"], lm["witten-bell backoff n=2"]
    L += ["", f"The positional model, which knows only how far into the text a sign is, reaches {pos['cv_bits'][0]:.3f} bits per event; the bigram model reaches "
          f"{big['cv_bits'][0]:.3f}. This baseline counts position from the start only; a baseline that also knows the distance to the end is under Robustness checks.",
          f"The vocabulary is fixed at the {e['signs']} signs of the filtered corpus plus end-of-text; every model gives unseen signs non-zero probability.", "",
          "## Is sign order structured?", "",
          f"Adjacent-sign mutual information is {s['observed_bits']:.3f} bits. After shuffling the signs inside each text it is "
          f"{s['shuffled_mean_bits']:.3f} bits (95% of shuffles in [{s['shuffled_95'][0]:.3f}, {s['shuffled_95'][1]:.3f}]; p < {max(s['p'], 0.001):.3f}). "
          f"With every token permuted across the whole corpus it is {s['all_tokens_permuted_bits']:.3f} bits, which is the size of the estimator's upward bias on data this sparse; "
          "the comparison is therefore against shuffles and not against zero.", "",
          f"Of the {res['pairs']['distinct']} distinct adjacent pairs, {res['pairs']['tested']} occur at least {stats.MIN_PAIR_COUNT} times and were tested "
          f"(log-likelihood ratio, Benjamini-Hochberg at 5%): {res['pairs']['attracting']} occur more often than chance and {res['pairs']['avoiding']} less often. "
          f"The same procedure on a corpus with all tokens permuted flags {res['pairs']['flagged_under_permutation']:.1f} pairs on average. "
          "Rarer pairs are listed but not tested, because the test is unreliable at expected counts below about 1. "
          "Full table with pointwise mutual information: `sign_pairs.csv`. Forward and backward transition matrices: `transition_matrices.npz`.", "",
          "![Transition heatmap](transition_heatmap.png)", "",
          "## Writing direction", "",
          "**What this test can show.** The corpus stores each text in reading order and labels each line right-to-left or left-to-right. "
          "Each line was put back into its physical left-to-right layout using that label, and the measure of Ashraf & Sinha (2018) was applied "
          "to the signs at the two physical ends. Negative delta S and positive delta G mean the left end is the more predictable one, which in "
          "scripts of known direction goes with right-to-left reading.", "",
          "The result is conditional on Mahadevan's labels and is not an independent verification of direction. In stored reading order the "
          f"last-read end is the more concentrated one in both groups (delta S {d['stored reading order, lines labelled right-to-left']['delta_S']:+.3f} for lines labelled "
          f"right-to-left, {d['stored reading order, lines labelled left-to-right']['delta_S']:+.3f} for left-to-right), so the physical values below are that fact "
          "combined with the label. What the data supports is: given the recorded labels, the concentrated end is on the physical left in the "
          f"lines labelled right-to-left, which are {100 * d['share labelled right-to-left']:.0f}% of lines with a recorded direction. A test that does not depend on the labels needs "
          "layout from another source, such as sign positions measured on photographs (Milestone 1's `bbox_xywh`) or the printed concordance. "
          "The proposal asks for 'verifying right-to-left syntax rules'; this report claims consistency with a right-to-left convention and makes no claim about syntax.", "",
          "| Group | Lines | delta S (bootstrap SE) | delta S under within-line shuffle (95%) | p | delta G (bootstrap SE) |", "|---|---|---|---|---|---|"]
    for name, g in d.items():
        if isinstance(g, dict) and "delta_G" in g:
            L.append(f"| {name} | {g['lines']} | {g['delta_S']['value']:+.3f} ({g['delta_S']['bootstrap_se']:.3f}) | "
                     f"[{g['delta_S']['shuffle_null_95'][0]:+.3f}, {g['delta_S']['shuffle_null_95'][1]:+.3f}] | {g['delta_S']['p_shuffle']:.3f} | "
                     f"{g['delta_G']['value']:+.3f} ({g['delta_G']['bootstrap_se']:.3f}) |")
        elif isinstance(g, dict) and "skipped" in g:
            L.append(f"| {name} | {g['lines']} | skipped: {g['skipped']} | | | |")
    j, j2, lp = d["sign 342 in the primary group"], d["sign 342 in lines labelled left-to-right"], d["label permutation"]
    L += ["", f"In the primary group, sign 342 (the commonest text-ender) is at the physical left end of {j['at_physical_left']} lines and at the right end of {j['at_physical_right']}, out of {j['lines']}. "
          f"In lines labelled left-to-right it is at the left end of {j2['at_physical_left']} and the right end of {j2['at_physical_right']}, out of {j2['lines']}. "
          "No exceptions in either group is what one would expect if the labels were assigned with the position of such signs in view.",
          f"Descriptively, the gap in delta S between the two labelled groups is {lp['delta_S_gap_rtl_minus_ltr']:+.3f}, against [{lp['null_95'][0]:+.3f}, {lp['null_95'][1]:+.3f}] when labels are reassigned at random; this follows from the labels and is not separate evidence.", "",
          "**Without the signs most likely to have guided the labels.** The rows 'no top-k text-beginner or ender at either end' drop every primary line "
          "whose physical left or right sign is among the k commonest text-beginners or text-enders (ranked on EBUDS). "
          + " ".join(f"k = {k}: delta S {d[f'primary, no top-{k} text-beginner or ender at either end']['delta_S']['value']:+.3f} on {d[f'primary, no top-{k} text-beginner or ender at either end']['lines']} lines."
                     for k in (1, 3, 5, 10, 20) if 'delta_S' in d[f'primary, no top-{k} text-beginner or ender at either end'])
          + " The asymmetry is not carried by sign 342 alone, but it shrinks steadily as more end-marking signs are removed and is not detectable once the top 20 are excluded. "
          "Either the direction signal lives in a few dozen formulaic end signs, or those are the signs the labels were assigned from; this test cannot tell the two apart.", "",
          f"p is the share of within-line shuffles with an absolute delta S at least as large as observed; {1 / (res['settings']['n_resample'] + 1):.3f} is the smallest value {res['settings']['n_resample']:,} shuffles can give. "
          "delta S (entropy) is the headline measure because it is stable across group sizes. delta G (Gini) shrinks as a group gets smaller and sparser, "
          "so its size should not be compared between groups. delta G here is computed over the full 418-symbol sign list so that groups share one alphabet; "
          f"over only the signs seen at either end, which is closer to Ashraf & Sinha's usage, the primary group gives {d['primary delta_G over signs seen at either end']:+.3f}. "
          "A bootstrap standard error is given in place of a percentile interval because both measures are biased on sparse counts and resampling repeats the bias.", "",
          "Other limits: the left-to-right group is small and unrepresentative (many are miniature tablets from Harappa). Comparing forward and backward "
          "bigram perplexity is not a direction test: a maximum-likelihood bigram model scores a corpus identically in both directions (asserted in `ngram.py`).", "",
          "![Direction null](direction_null.png)", "",
          "## Error propagation from Milestone 1", "",
          "Recognition errors (substitution, deletion, and insertion on a random side, in equal parts) were injected into the corpus and the analysis repeated.", "",
          "| Injected sign error rate | delta S (mean, SD) | delta G (mean, SD) | Bigram perplexity | Top-20 pairs retained | Texts ending in the top ender |", "|---|---|---|---|---|---|"]
    for row in res["error_propagation"]:
        mark = " (Milestone 1, synthetic)" if row["error_rate"] == M1_SIGN_ERROR_RATE else ""
        L.append(f"| {row['error_rate']:.3f}{mark} | {row['delta_S'][0]:+.3f}, {row['delta_S'][1]:.3f} | {row['delta_G'][0]:+.3f}, {row['delta_G'][1]:.3f} | {row['bigram_perplexity']:.1f} | {100 * row['top20_pairs_retained']:.0f}% | {100 * row['share_of_texts_ending_in_top_ender']:.1f}% |")
    L += ["", f"The conclusions weaken but do not reverse: the direction signal shrinks toward zero, the commonest pairs mostly survive, and perplexity rises steeply. "
          f"Milestone 1's sign error rate ({M1_SIGN_ERROR_RATE}) was measured on synthetic seals and is optimistic for real photographs. "
          "The injected errors are uniform over signs; a real classifier confuses similar-looking signs, which this does not model.", "",
          "![Error propagation](error_propagation.png)", "",
          "## Restoring a missing sign", "",
          f"With one sign removed and predicted from both neighbours by the bigram model ({res['restoration']['trials']} trials, 5-fold): "
          f"top-1 {100 * res['restoration']['top1']:.1f}%, top-5 {100 * res['restoration']['top5']:.1f}%, and the true sign is inside the "
          f"90%-probability candidate set {100 * res['restoration']['hit_in_90pct_mass_set']:.1f}% of the time (that set holds "
          f"{res['restoration']['mean_size_of_90pct_set']:.0f} signs on average). The published 74% uses the last criterion; it is not top-1 accuracy. "
          "This measures how predictable the notation is and says nothing about meaning.", ""]
    rb = {"witten-bell backoff n=2": res["restoration"], **res["restoration_by_model"]}
    L += ["| Model | Top-1 | Top-5 | Inside the 90% set | Same, strict | Mean size of the 90% set |", "|---|---|---|---|---|---|"]
    L += [f"| {k} | {100 * v['top1']:.1f}% | {100 * v['top5']:.1f}% | {100 * v['hit_in_90pct_mass_set']:.1f}% | {100 * v['hit_in_90pct_mass_set_strict']:.1f}% | {v['mean_size_of_90pct_set']:.0f} |" for k, v in rb.items()]
    L += ["", "Each model scores a candidate by the probability of the whole text with the candidate in place, so the trigram also uses the sign two places away. "
          "It gains a few points of top-1 accuracy; the 90%-set figures are not comparable across models because the set sizes differ.", ""]
    L += syntax_section(res["syntax"])
    L += robustness_section(res)
    L += extended_section(res)
    (OUT / "milestone_2_report.md").write_text("\n".join(L))


def syntax_section(sx):
    L = ["## Syntax: positional classes, class order and recurring segments", "",
         f"Each sign with at least {MIN_CLASS_COUNT} tokens is tested for a positional preference against what its texts' lengths imply "
         "(chi-square, Benjamini-Hochberg 5%) and assigned the position it over-uses most; the rest are 'free'. "
         f"{sx['classified_signs']} signs qualify, covering {100 * sx['token_share_classified']:.0f}% of tokens.", "",
         "| Class | Signs | Commonest members |", "|---|---|---|"]
    L += [f"| {k} | {v['signs']} | {', '.join(map(str, v['examples']))} |" for k, v in sx["classes"].items()]
    L += ["", "Class-to-class transitions against shuffles of the medial signs, first and last kept (the ten largest departures; z is in shuffle standard deviations):", "",
          "| First | Then | Observed | Shuffled mean | z |", "|---|---|---|---|---|"]
    L += [f"| {r['first']} | {r['second']} | {r['observed']} | {r['shuffled_mean']:.1f} | {r['z']:+.1f} |" for r in sx["class_transitions"][:10]]
    r, s = sx["order_rule_heldout"], sx["segments"]
    L += ["", f"**Rule checked on held-out texts.** With classes learnt on the training folds only: in the {r['texts_with_both']} held-out texts holding both an "
          f"initial-class and a final-class sign, every initial-class sign precedes every final-class sign in {100 * r['share_obeying']:.1f}% of texts, "
          f"against {100 * r['share_obeying_after_shuffle']:.1f}% after shuffling all signs of the same texts and {100 * r['share_obeying_ends_kept']:.1f}% after shuffling only the medial signs. "
          "Most of the rule restates the classes themselves (initial signs open texts, final signs close them); the order of initial- and final-class signs "
          f"inside a text adds {100 * (r['share_obeying'] - r['share_obeying_ends_kept']):.1f} points beyond that. The class transitions above use the same "
          "ends-kept null, so they measure order beyond position.", "",
          f"**Recurring segments.** Texts are cut between adjacent signs that are not strongly associated (pair seen at least {SEGMENT_MIN_PAIR} times "
          f"with pointwise mutual information of at least {SEGMENT_MIN_PMI:.0f} bit); pieces of two or more signs found in at least {SEGMENT_MIN_TEXTS} distinct texts are kept. "
          f"This yields {s['distinct_recurring']} recurring segments. Segments learnt on training folds cover {100 * s['heldout_token_coverage']:.1f}% of held-out sign tokens, "
          f"against {100 * s['heldout_token_coverage_after_shuffle']:.1f}% of the same texts shuffled: much of the corpus is built from a stock of recurring sign groups.", "",
          "| Segment (reading order) | Texts containing it | Occurrences: initial | medial | final | whole text |", "|---|---|---|---|---|---|"]
    L += [f"| {k} | {v['texts']} | {v.get('initial', 0)} | {v.get('medial', 0)} | {v.get('final', 0)} | {v.get('whole text', 0)} |" for k, v in s["top"].items()]
    L += ["", "Segments keep their own positions: the commonest opening pairs open texts and the pairs ending in 342 close them. These are distributional rules "
          "of the notation in reading order. They do not show that the segments are words, or that the classes are grammatical categories.", ""]
    return L


def extended_section(res):
    L = ["### Differences by site, object type and field symbol", "",
         "Per group, the headline statistics; against the rest of the corpus, the Jensen-Shannon divergence of sign frequencies and of adjacent-pair "
         f"frequencies, with a permutation p ({res['settings']['n_resample']:,} relabellings; the smallest possible p is {1 / (res['settings']['n_resample'] + 1):.3f}). Groups with fewer than 100 distinct texts are pooled or left out. "
         "IM77 records no dates ('level' is excavation depth measured from a different datum at each site), so differences by period cannot be tested.", "",
         "| Grouping | Group | Texts | Dependence index | Top ender (share) | JS signs (shuffled mean), p | JS pairs (shuffled mean), p |", "|---|---|---|---|---|---|---|"]
    for grouping, groups in res["breakdown"].items():
        if grouping == "signs by field symbol":
            continue
        for g, v in groups.items():
            L.append(f"| {grouping} | {g} | {v['texts']} | {v['dependence_index']:.3f} | {v['top_ender'][0]} ({100 * v['top_ender_share']:.0f}%) | "
                     f"{v['js_signs']:.3f} ({v['js_signs_null_mean']:.3f}), {v['p_signs']:.3f} | {v['js_pairs']:.3f} ({v['js_pairs_null_mean']:.3f}), {v['p_pairs']:.3f} |")
    f = res["breakdown"]["signs by field symbol"]
    sig = f["significant"]
    flat = [(gr, g, v) for gr, gs in res["breakdown"].items() if gr != "signs by field symbol" for g, v in gs.items()]
    gr, g, v = max(flat, key=lambda x: x[2]["js_signs"] - x[2]["js_signs_null_mean"])
    sig_groups = sum(x[2]["p_signs"] < 0.05 for x in flat)
    L += ["", f"{sig_groups} of {len(flat)} groups differ from the rest in sign use more than random relabelling produces; the most distinct is {gr} = {g} "
          f"(commonest ender {v['top_ender'][0]}). "
          "Site, object type and field symbol overlap, so these differences are not independent of one another.", "",
          f"**Signs and field symbols.** Comparing the {f['texts_unicorn']} unicorn texts with the {f['texts_other_field_symbol']} texts carrying another field symbol, "
          f"{f['tested_signs']} signs occurring in at least 10 of them were tested (log-likelihood ratio, Benjamini-Hochberg 5%): "
          + (", ".join(f"sign {r['sign']} ({100 * r['share_unicorn']:.0f}% of unicorn texts, {100 * r['share_other']:.0f}% of others, q = {r['q']:.3g})" for r in sig[:10]) + "."
             if sig else "none differs significantly. At this sample size no sign is tied to the unicorn rather than to other animals and motifs.") , ""]
    ml = res["multiline"]
    b = ml["sign before a line break is a typical ender"]
    L += ["### Multi-line and boustrophedon texts", "",
          f"IM77 has {ml['groups']} multi-line inscriptions (one side of an object with two or three lines), {ml['groups_fully_legible']} of them fully legible and "
          f"without doubtful readings; {ml['alternating_direction']} have lines labelled in opposite directions (boustrophedon or mixed). Lines are taken in their recorded order. "
          f"'Typical ender' means one of the {ml['ender_sets_size']} signs that end 80% of EBUDS texts; 'typical beginner', one of the {ml['beginner_set_size']} that begin 80%.", "",
          "| Position | Signs | Share that are typical enders / beginners |", "|---|---|---|"]
    for k in ("sign before a line break is a typical ender", "last sign of a multi-line text is a typical ender",
              "medial sign of a single-line text is a typical ender", "sign after a line break is a typical beginner"):
        L.append(f"| {k} | {ml[k]['n']} | {100 * ml[k]['share']:.0f}% |")
    L += ["", f"The sign before a line break is a typical ender far more often than a medial sign (Fisher p = {ml['p_before_break_vs_medial']:.1g}), and more often than "
          "the last sign of the whole inscription. So a line break behaves like the end of a text, not like a pause inside one: either the lines are separate "
          f"texts, or the recorded line order is often not the reading order ({100 * ml['share of later lines that are a single sign']:.0f}% of second and third lines are a single sign). The sample is small. Adding the multi-line "
          "texts to the corpus, joined or line by line, leaves the headline statistics close to the main corpus (see the corpus variants table).", "",
          "### Estimator bias", "",
          "The entropy-based statistics under three estimators. The last column but one is the same mutual information on the corpus with all tokens "
          "permuted, which has no real dependence: whatever remains there is bias.", "",
          "| Estimator | Sign entropy | Mutual information | MI, permuted corpus | MI in excess of the permuted corpus | H(next given current) | Direction delta S (primary) |", "|---|---|---|---|---|---|---|"]
    for k, v in res["estimators"].items():
        L.append(f"| {k} | {v['unigram_entropy']:.3f} | {v['mutual_information']:.3f} | {v['mutual_information_permuted_corpus']:.3f} | "
                 f"{v['mutual_information'] - v['mutual_information_permuted_corpus']:.3f} | {v['conditional_entropy']:.3f} | {v['direction_delta_S_primary']:+.3f} |")
    cs_, pi = res["estimators"]["Chao-Shen"], res["estimators"]["plug-in"]
    L += ["", f"Chao-Shen cuts the bias on the permuted corpus from {pi['mutual_information_permuted_corpus']:.2f} to {cs_['mutual_information_permuted_corpus']:.2f} bits "
          "but does not remove it, which is why every dependence claim in this report is made against shuffles rather than against zero. "
          "The direction statistic barely moves across estimators.", ""]
    ss = res["sample_size"]
    at = lambda f: next(r for r in ss if abs(r["fraction"] - f) < 1e-9)
    L += ["### Sample size", "",
          f"Headline statistics on random subsets of the corpus ({res['settings']['subsets']} subsets per size, means).", "",
          "| Texts | Signs seen | Entropy, plug-in | Entropy, Chao-Shen | MI, plug-in | MI, Chao-Shen | Dependence index | Bigram perplexity | Top-ender share | Attracting pairs |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    L += [f"| {r['texts']} | {r['signs_seen']:.0f} | {r['unigram_entropy_plugin']:.3f} | {r['unigram_entropy_chao_shen']:.3f} | {r['mi_plugin']:.3f} | {r['mi_chao_shen']:.3f} | "
          f"{r['dependence_index']:.3f} | {r['bigram_perplexity']:.1f} | {r['top_ender_share']:.3f} | {r['attracting_pairs']:.0f} |" for r in ss if round(r["fraction"], 1) in (0.1, 0.2, 0.4, 0.6, 0.8, 1.0)]
    ch = lambda k: (at(1.0)[k] - at(0.8)[k]) / at(0.8)[k]
    L += ["", f"From 80% to 100% of the corpus: Chao-Shen entropy changes {100 * ch('unigram_entropy_chao_shen'):+.1f}% and the top-ender share {100 * ch('top_ender_share'):+.1f}%, "
          f"so those have levelled off; the dependence index changes {100 * ch('dependence_index'):+.1f}%, bigram perplexity {100 * ch('bigram_perplexity'):+.1f}% and the count of "
          f"significant attracting pairs {100 * ch('attracting_pairs'):+.1f}%, and {at(1.0)['signs_seen'] - at(0.8)['signs_seen']:.0f} more signs appear. "
          "Pair-level statistics are still moving. Part of that is real (more data supports more context) and part is the sparsity bias of the plug-in estimates "
          "described under Estimator bias; either way a larger corpus would not reverse them. Their direction is the same at every size.", "",
          "![Sample size](sample_size.png)", "",
          "### Compound signs (ligatures)", "",
          "Mahadevan's list numbers many compound signs separately, and how they are flattened into a sequence is a transcriber's choice. No machine-readable "
          "list of the compounds and their components was found, so the effect is simulated: a share of sign types is either split into two components "
          f"(drawn from the 50 commonest signs, order fixed per type) or merged pairwise. Mean (SD) over {res['settings']['subsets']} random draws.", "",
          "| Change | Dependence index | Share of tested pairs attracting | Top-ender share | Bits gained over positional |", "|---|---|---|---|---|"]
    base = res["corpus_variants"]["IM77, EBUDS (main analysis)"]
    L.append(f"| none (main corpus) | {base['dependence_index']:.3f} | {base['pairs_attracting'] / base['pairs_tested']:.2f} | {base['top_ender_share']:.3f} | {base['bits_saved_over_positional']:.2f} |")
    L += [f"| {k} | {v['dependence_index'][0]:.3f} ({v['dependence_index'][1]:.3f}) | {v['attracting_share'][0]:.2f} ({v['attracting_share'][1]:.2f}) | "
          f"{v['top_ender_share'][0]:.3f} ({v['top_ender_share'][1]:.3f}) | {v['bits_saved_over_positional'][0]:.2f} ({v['bits_saved_over_positional'][1]:.2f}) |"
          for k, v in res["resegmentation"].items()]
    shift = lambda k, b: max(abs(v[k][0] - b) for v in res["resegmentation"].values())
    L += ["", f"Re-segmenting up to a fifth of the sign list moves the dependence index by at most {shift('dependence_index', base['dependence_index']):.3f}, "
          f"the share of attracting pairs by at most {shift('attracting_share', base['pairs_attracting'] / base['pairs_tested']):.2f}, the top-ender share by at most "
          f"{shift('top_ender_share', base['top_ender_share']):.3f} and the gain over position by at most {shift('bits_saved_over_positional', base['bits_saved_over_positional']):.2f} bits; "
          + ("in every draw dependence stays positive, attracting pairs stay the majority and the gain over position stays positive. "
             if all(v["same_direction"] for v in res["resegmentation"].values()) else "some draws change direction (see results.json). ") + "Splitting raises the "
          "dependence index, because each split sign becomes a fixed pair; real compounds would do the same if transcribed as separate signs.", "",
          "### All tests in this report, corrected together", ""]
    mc = res["multiple_comparisons"]
    nb, nh = sum(r["q_bh"] < 0.05 for r in mc), sum(r["p_holm"] < 0.05 for r in mc)
    L += [f"{len(mc)} p-values are reported as tests in this report (the pair table, the field-symbol signs and the positional classes are corrected internally and enter "
          f"only through their counts). Corrected together, {nb} stay below 0.05 under Benjamini-Hochberg (false discovery rate) and {nh} under Holm "
          f"(family-wise error). Permutation p-values cannot go below 1/(resamples + 1) = {1 / (res['settings']['n_resample'] + 1):.3f}, which limits how many can pass Holm. Tests that do not pass:", ""]
    L += [f"- {r['test']}: p = {r['p']:.3g}, q = {r['q_bh']:.3g}, Holm = {r['p_holm']:.3g}" for r in mc if r["p_holm"] >= 0.05] or ["- none"]
    pv = res["provenance"]
    L += ["", "### Provenance", "",
          f"Python {pv['python']}; " + ", ".join(f"{k} {v}" for k, v in pv["packages"].items()) + f"; code at commit {pv['git_commit'][:7]}" + (" plus uncommitted changes" if pv["uncommitted_changes"] else "") + ". "
          "Input checksums (SHA-256) and a checksum of the Python-token reference stream (which depends on the local standard library) are in `results.json` under `provenance`. "
          "Every random step uses a fixed seed.", ""]
    return L


def robustness_section(res):
    pv, cs, us, gaps, cvs = res["position_vs_bigram"], res["cross_system"], res["unseen_signs"], res["replication_gaps"], res["cv_stability"]
    L = ["## Robustness checks", "",
         "### Comparison with sign systems of known type", ""]
    if cs:
        L += [f"Each reference system is cut into samples with exactly the Indus text-length distribution ({res['ebuds']['texts']} texts, {res['ebuds']['tokens']} tokens), "
              f"so sample size and text length, which bias every plug-in estimate, are matched; {res['settings']['samples']} samples each, mean (SD over samples). "
              "The samples are windows of a continuous stream, so a window can cross a line or text boundary of the reference system. "
              "Dependence index = share of a sign's uncertainty removed by knowing the previous sign, beyond what remains after shuffling signs "
              "within each text (0 = order carries nothing). Shuffling a short text keeps much of its pair structure, so even the rigid control stays far below 1. Python tokens are from the first 40 modules of the local Python 3.10 standard library, "
              "string and number literals collapsed to one token each.", "",
              "| System | Signs in sample | H(sign), bits | H(next given current), bits | Dependence index, all signs | Dependence index, top 20 signs |", "|---|---|---|---|---|---|"]
        for name, v in cs.items():
            top20 = f"{v['dependence_index_top20'][0]:.3f} ({v['dependence_index_top20'][1]:.3f})" if v["signs_in_sample"] >= 20 else "n/a (fewer than 20 signs)"
            L.append(f"| {name} | {v['signs_in_sample']:.0f} | {v['unigram_entropy']:.2f} | {v['conditional_entropy']:.2f} | "
                     f"{v['dependence_index'][0]:.3f} ({v['dependence_index'][1]:.3f}) | {top20} |")
        ind = cs["Indus (IM77, EBUDS)"]
        real = {k: v for k, v in cs.items() if not k.startswith(("Indus", "random", "rigid"))}
        above = [k for k, v in real.items() if v["dependence_index"][0] > ind["dependence_index"][0]]
        near = min(real, key=lambda k: abs(real[k]["dependence_index"][0] - ind["dependence_index"][0]) + abs(real[k]["dependence_index_top20"][0] - ind["dependence_index_top20"][0]))
        L += ["", "Read this as a scale, not a classification. "
              f"On the all-signs index, {len(real) - len(above)} of the {len(real)} real reference systems show weaker order dependence than the Indus corpus"
              + (f"; {', '.join(above)} show{'s' if len(above) == 1 else ''} stronger" if above else "") + ". "
              f"The closest system on both indices is {near} ({real[near]['dependence_index'][0]:.3f} and {real[near]['dependence_index_top20'][0]:.3f}, "
              f"against {ind['dependence_index'][0]:.3f} and {ind['dependence_index_top20'][0]:.3f}). "
              f"Cuneiform seal inscriptions, short formulaic texts in a known language, score {real['Cuneiform seal inscriptions (signs)']['dependence_index'][0]:.3f} and "
              f"{real['Cuneiform seal inscriptions (signs)']['dependence_index_top20'][0]:.3f}; long running text (the novel, Sumerian texts as a whole) and the non-linguistic "
              "melodies and DNA score lower. Linear B is indicative only: its file mixes real and augmented sequences, and cutting it into windows rather than "
              "whole lines changes its index markedly. Strong order dependence is what a formulaic notation would show, linguistic or not, so this places the "
              "Indus corpus beside short seal texts without saying which kind of system it is. "
              "Sproat's (2014) non-linguistic corpora (Mesopotamian deity symbols, Pictish stones, heraldry) are not publicly released and could not be included; "
              "the Linear B file mixes 513 real Knossos series D sequences with augmented ones it does not mark. The all-signs index for English words is compressed because, at 7000 tokens and over 1,500 word types, "
              "most word pairs occur once in both the real and the shuffled texts; the top-20 column avoids that.", "",
              "![Conditional entropy of sign systems](conditional_entropy_systems.png)", ""]
    else:
        L += ["Skipped: reference texts not found under data/raw/reference (see data/README.md).", ""]
    L += ["### Position counted from both ends", "",
          "All three models below predict signs only, given that the text has not ended (end-of-text is renormalised away), 5-fold CV.", "",
          "| Model | Bits per sign (mean, SD over folds) | Perplexity |", "|---|---|---|"]
    L += [f"| {k} | {v['bits_per_sign'][0]:.3f}, {v['bits_per_sign'][1]:.3f} | {v['perplexity']:.1f} |" for k, v in pv.items()]
    gain = pv["position from both ends (knows the text's length)"]["bits_per_sign"][0] - pv["bigram (Witten-Bell backoff)"]["bits_per_sign"][0]
    L += ["", f"Knowing the distance to both ends helps the positional model, but the bigram, which is not told the text's length, is still {gain:.2f} bits per sign better. "
          "The bigram's gain is therefore sign-to-sign dependence, not position at either end.", "",
          "### Cross-validation stability and unseen signs", "",
          "| Model | Perplexity, mean over 10 random 5-fold splits | SD across splits | Range |", "|---|---|---|---|"]
    L += [f"| {k} | {v['perplexity_mean']:.2f} | {v['perplexity_sd_across_splits']:.2f} | {v['perplexity_range'][0]:.2f} to {v['perplexity_range'][1]:.2f} |"
          for k, v in cvs.items() if "perplexity_mean" in v]
    b = cvs["bigram better than positional"]
    L += ["", f"The bigram beats the positional baseline in {b['folds']} of {b['of']} folds across all splits, by at least {b['smallest_gain_bits']:.2f} bits per event. "
          "The SD across splits is under 1% of perplexity, so the gaps to the published values are not split noise.", "",
          f"Signs absent from the training fold are {100 * us['unseen_share_of_events']:.1f}% of held-out events but {100 * us['unseen_share_of_bits']:.1f}% of the held-out bits "
          f"({us['bits_per_unseen_event']:.1f} bits each against {us['bits_per_seen_event']:.1f}). Scoring only the other events with the same model gives "
          f"perplexity {us['perplexity_seen_only']:.1f} instead of {us['perplexity_all']:.1f}: rare signs account for about a tenth of the headline figure.", "",
          "### Corpus variants: duplicates, doubtful readings, a second transcription, merged sign groups", "",
          "The main corpus already keeps one copy of each repeated text (EBUDS); the third row shows what counting every copy would change. "
          "The Wells/ICIT rows use an independent transcription with a different sign list (yajnadevam/indus-website, GPL-3.0; single-line texts with every sign readable, one copy each). "
          "That data records no doubtful readings and no usable physical layout, so the direction test cannot be repeated on it. "
          f"'Variant groups merged' maps every sign to the head of its Wells variant group ({res['wells_variant_groups'][0]} groups covering {res['wells_variant_groups'][1]} signs). "
          "Perplexity is not comparable across sign lists, so it is also given relative to a uniform model.", "",
          "| Corpus | Texts | Tokens | Signs | Dependence index | Pairs tested / attracting / avoiding | Top ender (share of texts) | Enders covering 80% | Bigram perplexity (relative to uniform) | Bits gained over positional | Top-20 pairs shared with main |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for name, v in res["corpus_variants"].items():
        ppl = f"{v['bigram_perplexity']:.1f} ({v['bigram_perplexity_over_uniform']:.3f})" if "bigram_perplexity" in v else "n/a: copies leak across folds"
        gain = f"{v['bits_saved_over_positional']:.2f}" if "bits_saved_over_positional" in v else "n/a"
        L.append(f"| {name} | {v['texts']} | {v['tokens']} | {v['signs']} | {v['dependence_index']:.3f} | {v['pairs_tested']} / {v['pairs_attracting']} / {v['pairs_avoiding']} | "
                 f"{v['top_ender'][0]} ({100 * v['top_ender_share']:.0f}%) | {v['enders_covering_80pct']} | {ppl} | {gain} | {v.get('top20_pairs_shared_with_main', 'n/a (other sign list)')} |")
    vs = list(res["corpus_variants"].values())
    span = lambda f, fmt="{:.0f}", rows=vs: f"{fmt.format(min(map(f, rows)))} to {fmt.format(max(map(f, rows)))}"
    L += ["", f"Across the variants: dependence index {span(lambda v: v['dependence_index'], '{:.3f}')}; "
          f"{span(lambda v: 100 * v['pairs_attracting'] / v['pairs_tested'])}% of tested pairs attracting; "
          f"one dominant text-ender closing {span(lambda v: 100 * v['top_ender_share'])}% of texts; "
          f"{span(lambda v: v['enders_covering_80pct'])} signs covering 80% of endings; "
          f"{span(lambda v: v['bits_saved_over_positional'], '{:.2f}', [v for v in vs if 'bits_saved_over_positional' in v])} bits per event gained over position alone where computable. "
          f"The vocabulary ranges over {span(lambda v: v['signs'])} signs. "
          + ("On every variant dependence is positive, attracting pairs outnumber avoiding ones and the gain over position is positive." if all(
              v["dependence_index"] > 0 and v["pairs_attracting"] > v["pairs_avoiding"] and v.get("bits_saved_over_positional", 1) > 0 for v in vs)
             else "At least one variant reverses a conclusion (see results.json)."), "",
          "### Replication gaps", "",
          "Mutual information under the definitions tried:", "", "| Definition | Bits | Nats |", "|---|---|---|"]
    L += [f"| {k} | {v['bits']:.3f} | {v['nats']:.3f} |" for k, v in gaps["mutual_information"].items()]
    L += ["", "Witten-Bell backoff perplexity with unseen signs skipped (SRILM's default handling), under four protocols:", "",
          "| Protocol | n=1 | n=2 | n=3 | n=4 | n=5 |", "|---|---|---|---|---|---|",
          f"| Yadav et al. 2010 | {' | '.join(str(PUBLISHED['perplexity'][n]) for n in range(1, 6))} |"]
    L += [f"| {k} | {' | '.join(f'{x:.2f}' for x in v)} |" for k, v in gaps["perplexity_protocols"].items()]
    L += ["", "10-fold cross-validation with unseen signs skipped reproduces orders 1 to 3 within 1%. No protocol tried reproduces orders 4 and 5, "
          "where the published values fall while every reproduction rises; the paper does not give enough detail to go further, so this gap is closed as unexplained.", ""]
    return L


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    ngram.selfcheck()
    stats.selfcheck()
    n_resample, repeats, samples, subsets = (100, 2, 3, 2) if args.quick else (1000, 10, 10, 5)
    OUT.mkdir(parents=True, exist_ok=True)

    lines = load_lines()
    texts, steps = ebuds(lines)
    vocab = sorted({s for t in texts for s in t})
    res = {"settings": {"n_resample": n_resample, "error_repeats": repeats, "samples": samples, "subsets": subsets}, "corpus_steps": steps, "ebuds": {"texts": len(texts), "tokens": sum(map(len, texts)), "signs": len(vocab)}}
    print("corpus:", res["ebuds"])

    counts = Counter(s for t in texts for s in t)
    begin, end = Counter(t[0] for t in texts), Counter(t[-1] for t in texts)
    medial = Counter(s for t in texts for s in t[1:-1])
    pairs, n_pairs = stats.pair_table(texts)
    triplets = Counter(g for t in texts for g in zip(t, t[1:], t[2:]))
    res["replication"] = {
        "unigram_entropy": stats.entropy(counts.values()), "unigram_entropy_miller_madow": stats.entropy(counts.values(), True),
        "bigram_mi": stats.mutual_information(texts),
        "coverage80": {"all": stats.coverage(counts.values()), "beginners": stats.coverage(begin.values()), "enders": stats.coverage(end.values())},
        "top_beginners": [s for s, _ in begin.most_common(3)], "top_enders": [s for s, _ in end.most_common(3)],
        "top_pair": [pairs[0][0], pairs[0][1], pairs[0][2], pairs[0][4]],
        "top_triplet": [list(triplets.most_common(1)[0][0]), triplets.most_common(1)[0][1]],
    }
    res["zipf_mandelbrot_fit"] = stats.zipf_mandelbrot(counts.values())
    attracting, avoiding = stats.significant_pairs(pairs)
    rng = np.random.default_rng(0)
    flagged = [sum(stats.significant_pairs(stats.pair_table(permute_tokens(texts, rng))[0])) for _ in range(10)]
    res["pairs"] = {"distinct": len(pairs), "adjacent_pair_tokens": n_pairs, "tested": int(sum(p[2] >= stats.MIN_PAIR_COUNT for p in pairs)),
                    "attracting": attracting, "avoiding": avoiding, "flagged_under_permutation": float(np.mean(flagged))}
    print("replication:", res["replication"])

    with open(OUT / "sign_pairs.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["first_sign", "second_sign", "count", "expected_if_independent", "pmi_bits", "g2", "p", "q_benjamini_hochberg", "tested"])
        w.writerows([a, b, k, round(e, 3), round(float(np.log2(k / e)), 3), round(g, 2), f"{p:.3g}", "" if q != q else f"{q:.3g}", int(k >= stats.MIN_PAIR_COUNT)]
                    for a, b, k, e, g, p, q in pairs)
    with open(OUT / "sign_positions.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sign", "count", "as_text_beginner", "as_text_ender", "medial", "share_beginner", "share_ender"])
        w.writerows([s, c, begin[s], end[s], medial[s], round(begin[s] / c, 3), round(end[s] / c, 3)] for s, c in counts.most_common())

    full = WittenBell(texts, vocab, 2, backoff=True)
    P_smooth = ngram.transition_matrix(full, vocab)
    back = WittenBell([t[::-1] for t in texts], vocab, 2, backoff=True)  # P(previous | current): texts reversed, so <s> marks the text's end
    np.savez_compressed(OUT / "transition_matrices.npz", signs=np.array(vocab), forward=P_smooth, backward=ngram.transition_matrix(back, vocab))
    P_mle = np.array([[full.counts[(h,)][w_] / max(full.total.get((h,), 0), 1) for w_ in [*vocab, EOS]] for h in [BOS, *vocab]])
    assert np.allclose(P_smooth.sum(1), 1)

    res["language_models"] = language_models(texts, vocab)
    print("bigram WB backoff CV perplexity:", round(res["language_models"]["witten-bell backoff n=2"]["cv_perplexity"], 2))
    res["restoration"] = restoration(texts, vocab)
    res["structure_test"] = structure_test(texts, n_resample)
    res["direction"], phys = direction_analysis(lines, texts, n_resample)
    rng = np.random.default_rng(1)
    null_samples = []
    for _ in range(n_resample):  # null distribution for the figure
        ends = [rng.choice(len(l), 2, replace=False) for l in phys]
        null_samples.append(stats.asymmetry(np.array([l[i] for l, (i, _) in zip(phys, ends)]), np.array([l[j] for l, (_, j) in zip(phys, ends)]))[1])
    print("direction primary:", res["direction"]["primary: labelled right-to-left, single-line"]["delta_S"])
    res["error_propagation"] = error_propagation(texts, phys, vocab, pairs, repeats)

    res["position_vs_bigram"] = position_vs_bigram(texts, vocab)
    res["cv_stability"] = cv_stability(texts, vocab, seeds=3 if args.quick else 10)
    res["unseen_signs"] = unseen_signs(texts, vocab)
    res["replication_gaps"] = replication_gaps(texts, lines)
    have_refs = all((REFERENCE / f).exists() for f in ("pride_and_prejudice.txt", "lambda_phage.fasta"))
    refs = reference_systems() if have_refs else None
    res["cross_system"] = cross_system(texts, refs, samples) if refs else None
    res["multiline"], multiline_texts = multiline(lines, texts)
    wells = load_wells() if WELLS_SQL.exists() else None
    res["corpus_variants"] = corpus_variants(lines, wells, multiline_texts)
    res["wells_variant_groups"] = [len(set(wells[1].values())), len(set(wells[1]) | set(wells[1].values()))] if wells else None
    for name, present in (("comparison with other sign systems", have_refs), ("Wells/ICIT corpus", WELLS_SQL.exists())):
        if not present:
            print(f"skipped the {name}: input missing, see data/README.md")

    res["restoration_by_model"] = {"witten-bell backoff n=3": restoration(texts, vocab, lambda tr, v: WittenBell(tr, v, 3, backoff=True)),
                                   "kneser-ney n=2": restoration(texts, vocab, KneserNey)}
    res["syntax"] = syntax(texts, n_resample)
    res["breakdown"] = breakdown(lines, n_resample)
    res["estimators"] = estimators(texts, phys)
    res["sample_size"] = sample_size_curves(texts, subsets)
    res["resegmentation"] = resegmentation(texts, subsets)
    res["multiple_comparisons"] = adjust_all(res)
    res["provenance"] = provenance(refs)

    (OUT / "results.json").write_text(json.dumps(res, indent=1))
    figures(res, texts, vocab, P_mle, null_samples)
    write_report(res)
    print(f"report -> {OUT / 'milestone_2_report.md'}")
