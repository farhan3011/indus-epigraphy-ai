"""Milestone 2: positional bias and Markov-chain modelling of Indus sign sequences (IM77 corpus).

    python sequence_analysis/run_milestone2.py            # everything; a few minutes on a laptop CPU
    python sequence_analysis/run_milestone2.py --quick    # fewer resamples, for a fast check

Writes reports/milestone_2/: results.json, CSV tables, figures, transition matrices, and
milestone_2_report.md. Only aggregate statistics are written, never the corpus texts.
"""
import argparse
import csv
import json
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import ngram
import stats
from corpus import EBUDS_PUBLISHED, ROOT, ebuds, load_lines, physical
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


def restoration(texts, vocab):
    """Remove one sign and predict it from both neighbours with the bigram model:
    P(w | prev, next) is proportional to P(w | prev) * P(next | w). 5-fold, every position of every test text."""
    idx = {s: i for i, s in enumerate(vocab)}
    top1 = top5 = in90 = strict = total = 0
    set_sizes = []
    for train, test in ngram.folds(texts):
        P = ngram.transition_matrix(WittenBell(train, vocab, 2, backoff=True), vocab)  # [context: BOS, signs] x [event: signs, EOS]
        for t in test:
            for j, true in enumerate(t):
                prev = 0 if j == 0 else idx[t[j - 1]] + 1
                nxt = len(vocab) if j == len(t) - 1 else idx[t[j + 1]]
                score = P[prev, :len(vocab)] * P[1:, nxt]
                order = np.argsort(-score)
                rank = int(np.where(order == idx[true])[0][0])
                k90 = int(np.searchsorted(np.cumsum(score[order]) / score.sum(), 0.9) + 1)
                top1, top5, in90, total = top1 + (rank == 0), top5 + (rank < 5), in90 + (rank < k90), total + 1
                strict += rank < k90 - 1  # the sign that crosses the 90% line not counted
                set_sizes.append(k90)
    return {"trials": total, "top1": top1 / total, "top5": top5 / total, "hit_in_90pct_mass_set": in90 / total,
            "hit_in_90pct_mass_set_strict": strict / total,
            "mean_size_of_90pct_set": float(np.mean(set_sizes))}


def structure_test(texts, n_resample, seed=0):
    """Is sign order within a text non-random? Compare adjacent-sign mutual information with the
    same texts after shuffling the signs inside each one (keeps every text's length and contents)."""
    rng = np.random.default_rng(seed)
    observed = stats.mutual_information(texts)
    null = [stats.mutual_information([tuple(rng.permutation(t)) for t in texts]) for _ in range(n_resample)]
    tokens, cuts = np.concatenate(texts), np.cumsum([len(t) for t in texts])[:-1]
    memoryless = [stats.mutual_information([tuple(t) for t in np.split(rng.permutation(tokens), cuts)]) for _ in range(20)]
    return {"observed_bits": observed, "shuffled_mean_bits": float(np.mean(null)), "all_tokens_permuted_bits": float(np.mean(memoryless)),
            "shuffled_95": np.percentile(null, [2.5, 97.5]).tolist(),
            "p": float((np.sum(np.array(null) >= observed) + 1) / (n_resample + 1))}


def direction_analysis(lines, n_resample):
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
    out = {
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


def write_report(res):
    lm, d, pub = res["language_models"], res["direction"], PUBLISHED
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
          f"| Adjacent-sign mutual information (bits) | {pub['bigram_mi']} | {r['bigram_mi']:.2f} (plug-in; not reproduced) |",
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
          "Orders 4 and 5 are not reproduced under either protocol.",
          "- **Mutual information.** The paper does not define its estimator. An independent check of seven definitions found none that gives 2.24. "
          f"The plug-in value is biased upward: with every token permuted across the corpus it is still {res['structure_test']['all_tokens_permuted_bits']:.2f} bits.",
          "- **Restoration.** The published rule can be read two ways at the 90% boundary, and the two readings differ by several points.", "",
          "## Language models compared (5-fold cross-validation over distinct texts)", "",
          "| Model | Bits per event (mean, SD over folds) | Perplexity | Perplexity without end-of-text |", "|---|---|---|---|"]
    for name, m in lm.items():
        L.append(f"| {name} | {m['cv_bits'][0]:.3f}, {m['cv_bits'][1]:.3f} | {m['cv_perplexity']:.2f} | {m['cv_perplexity_no_eos']:.2f} |")
    s = res["structure_test"]
    pos, big = lm["positional (no sign-to-sign dependence)"], lm["witten-bell backoff n=2"]
    L += ["", f"The positional model, which knows only how far into the text a sign is, reaches {pos['cv_bits'][0]:.3f} bits per event; the bigram model reaches "
          f"{big['cv_bits'][0]:.3f}. The baseline knows position counted from the start only, so part of the difference may still be end-of-text position.",
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
          "lines labelled right-to-left, which are 92% of lines with a recorded direction. A test that does not depend on the labels needs "
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
          "p is the share of within-line shuffles with an absolute delta S at least as large as observed; 0.001 is the smallest value 1,000 shuffles can give. "
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
    (OUT / "milestone_2_report.md").write_text("\n".join(L))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    ngram.selfcheck()
    stats.selfcheck()
    n_resample, repeats = (100, 2) if args.quick else (1000, 10)
    OUT.mkdir(parents=True, exist_ok=True)

    lines = load_lines()
    texts, steps = ebuds(lines)
    vocab = sorted({s for t in texts for s in t})
    res = {"corpus_steps": steps, "ebuds": {"texts": len(texts), "tokens": sum(map(len, texts)), "signs": len(vocab)}}
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
    tokens, cuts = np.concatenate(texts), np.cumsum([len(t) for t in texts])[:-1]
    flagged = [sum(stats.significant_pairs(stats.pair_table([tuple(t) for t in np.split(rng.permutation(tokens), cuts)])[0])) for _ in range(10)]
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
    back = WittenBell([t[::-1] for t in texts], vocab, 2, backoff=True)  # P(previous | current): texts reversed
    P_mle = np.array([[full.counts[(h,)][w_] / max(full.total.get((h,), 0), 1) for w_ in [*vocab, EOS]] for h in [BOS, *vocab]])
    assert np.allclose(P_smooth.sum(1), 1)

    res["language_models"] = language_models(texts, vocab)
    print("bigram WB backoff CV perplexity:", round(res["language_models"]["witten-bell backoff n=2"]["cv_perplexity"], 2))
    res["restoration"] = restoration(texts, vocab)
    res["structure_test"] = structure_test(texts, n_resample)
    res["direction"], phys = direction_analysis(lines, n_resample)
    rng = np.random.default_rng(1)
    null_samples = []
    for _ in range(n_resample):  # null distribution for the figure
        ends = [rng.choice(len(l), 2, replace=False) for l in phys]
        null_samples.append(stats.asymmetry(np.array([l[i] for l, (i, _) in zip(phys, ends)]), np.array([l[j] for l, (_, j) in zip(phys, ends)]))[1])
    print("direction primary:", res["direction"]["primary: labelled right-to-left, single-line"]["delta_S"])
    res["error_propagation"] = error_propagation(texts, phys, vocab, pairs, repeats)

    (OUT / "results.json").write_text(json.dumps(res, indent=1))
    figures(res, texts, vocab, P_mle, null_samples)
    write_report(res)
    print(f"report -> {OUT / 'milestone_2_report.md'}")
