"""Milestone 4 extension: how high can a table score by fitting alone?

    python decipherment/overfit.py           # about 6 minutes on a laptop CPU
    python decipherment/overfit.py --quick   # a fast check

Exploratory, NOT pre-registered: it was written after the Milestone 4 results were known and it is outside the
corrected family of tests in prereg.json. It asks what the Milestone 4 permutation test cannot: the test compares a
claimed table with random tables, but a decipherer does not pick a table at random, they search for one. So random
tables are searched too, with the same score and the claim's own sound values, and the claim is compared with what
the search reaches. The same search is run on real Sanskrit enciphered with the claim's sign-to-sound relation,
where the true table is known, to show what a correct table looks like under the same treatment.

Writes reports/milestone_4/ (overfitting.json, overfitting.png, overfitting_report.md): aggregates only.
"""
import argparse
import json
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import data
import harness
from data import skeleton
from harness import Corpus, assign, encipher, lm_table, units, value_codes

sys.path.insert(0, str(data.ROOT / "sequence_analysis"))
import corpus  # noqa: E402

OUT = harness.OUT
WIDTH = 12  # padded skeleton length per value, as in Corpus.codes


def scoring_tables(target):
    """Only the primary score is searched: Sanskrit trigram minus Sanskrit letter frequencies. The trailing None
    stands where Corpus.score expects the Naive Bayes tables, which the per-text path never touches."""
    return ((lm_table(target), lm_table(target, order=1)), None)


def split_score(cp, M, tables, fit):
    """Primary score in bits per phoneme on the texts in `fit` and on the rest (nan when there is no rest)."""
    s, n = cp.score(M, tables, per_text=True)
    w = s * n
    rest = w[~fit].sum() / n[~fit].sum() if (~fit).any() else float("nan")
    return float(w[fit].sum() / n[fit].sum()), float(rest)


def climb(cp, us, tables, rng, steps, fit=None, start=None, every=50):
    """Hill-climb a table: swap the values of two units and keep the swap if the score on the `fit` texts does not
    fall. The table always holds the claim's own values, like the shuffled tables of the permutation test.
    Returns (values per unit, [(step, score on fit texts, score on the other texts)]).
    ponytail: plain greedy swaps, one start; simulated annealing would reach higher, so the ceiling reported is a floor."""
    vals = list(rng.permutation(np.array([v for _, v in us], dtype=object))) if start is None else list(start)
    fit = np.ones(cp.n_texts, bool) if fit is None else fit
    row = {s: i + 2 for i, s in enumerate(cp.signs)}
    rows = [np.array([row[s] for s in signs if s in row], dtype=int) for signs, _ in us]
    active = [i for i, (signs, _) in enumerate(us) if any(cp.counts[s] for s in signs)]  # units that occur in the texts

    def vec(v):
        c = value_codes(v)[:WIDTH]
        return np.array(c + [-1] * (WIDTH - len(c)))

    M = cp.codes(assign(us, vals), WIDTH)
    cur, rest = split_score(cp, M, tables, fit)
    traj = [(0, cur, rest)]
    for step in range(1, steps + 1):
        i, j = active[rng.integers(len(active))], int(rng.integers(len(vals)))
        if vals[i] != vals[j]:
            M[rows[i]], M[rows[j]] = vec(vals[j]), vec(vals[i])
            new, new_rest = split_score(cp, M, tables, fit)
            if new >= cur:
                vals[i], vals[j], cur, rest = vals[j], vals[i], new, new_rest
            else:
                M[rows[i]], M[rows[j]] = vec(vals[i]), vec(vals[j])
        if step % every == 0:
            traj.append((step, cur, rest))
    return vals, traj


def agreement(cp, us, a, b):
    """Share of sign tokens that two tables read with the same consonants."""
    n = [sum(cp.counts[s] for s in signs) for signs, _ in us]
    return float(sum(k for k, x, y in zip(n, a, b) if skeleton(x) == skeleton(y)) / sum(n))


def pairwise(cp, us, tables_):
    return [agreement(cp, us, a, b) for i, a in enumerate(tables_) for b in tables_[i + 1:]]


def mean_sd(x):
    return {"mean": float(np.mean(x)), "sd": float(np.std(x)), "n": len(x)}


def curve(trajs, col=1):
    """Median over climbs of one trajectory column, as [[step, value], ...]."""
    return [[int(trajs[0][k][0]), float(np.median([t[k][col] for t in trajs]))] for k in range(len(trajs[0]))]


def study(cp, key, tables, rng, climbs, steps, splits, shuffles=200):
    """Everything measured on one corpus for one table whose values are searched: ceiling, held-out half, stability."""
    us = units(key)
    true = [v for _, v in us]
    everything = np.ones(cp.n_texts, bool)
    claim = split_score(cp, cp.codes(assign(us, true), WIDTH), tables, everything)[0]
    random_tables = [list(rng.permutation(np.array(true, dtype=object))) for _ in range(shuffles)]
    null = [split_score(cp, cp.codes(assign(us, v), WIDTH), tables, everything)[0] for v in random_tables]

    found, trajs = zip(*(climb(cp, us, tables, rng, steps) for _ in range(climbs)))
    finals = [t[-1][1] for t in trajs]
    median = curve(trajs)
    passed = next((s for s, v in median if v >= claim), None)
    polished, polish_traj = climb(cp, us, tables, rng, steps, start=true)

    fit_final, rest_final, claim_fit, claim_rest, rest_start, half_trajs = [], [], [], [], [], []
    for _ in range(splits):
        fit = rng.permutation(cp.n_texts) < cp.n_texts // 2
        _, t = climb(cp, us, tables, rng, steps, fit=fit)
        half_trajs.append(t)
        fit_final.append(t[-1][1]); rest_final.append(t[-1][2]); rest_start.append(t[0][2])
        a, b = split_score(cp, cp.codes(assign(us, true), WIDTH), tables, fit)
        claim_fit.append(a); claim_rest.append(b)
    return {
        "texts": cp.n_texts, "tokens": int(sum(cp.counts.values())), "units_in_texts": sum(any(cp.counts[s] for s in signs) for signs, _ in us),
        "table": claim, "random_tables": mean_sd(null),
        "searched_tables": mean_sd(finals) | {"min": float(min(finals)), "max": float(max(finals))},
        "share_of_searched_tables_above_table": float(np.mean(np.array(finals) > claim)),
        "steps_until_median_search_passes_table": passed,
        "median_trajectory": median,
        "table_after_search": {"score": polish_traj[-1][1], "tokens_still_read_the_same": agreement(cp, us, true, polished)},
        "held_out": {"splits": splits, "searched_fit_half": mean_sd(fit_final), "searched_other_half": mean_sd(rest_final),
                     "random_other_half": mean_sd(rest_start), "table_fit_half": mean_sd(claim_fit), "table_other_half": mean_sd(claim_rest),
                     "share_of_splits_searched_beats_table_on_other_half": float(np.mean(np.array(rest_final) > np.array(claim_rest))),
                     "median_trajectory_fit": curve(half_trajs, 1), "median_trajectory_other": curve(half_trajs, 2)},
        "agreement": {"between_searched_tables": mean_sd(pairwise(cp, us, list(found))),
                      "searched_with_table": mean_sd([agreement(cp, us, true, f) for f in found]),
                      "between_random_tables": mean_sd(pairwise(cp, us, random_tables[:20]))},
    }


def pool(studies):
    """Average the scalar results of several replicate studies (the enciphered-Sanskrit control is re-drawn each time)."""
    def merge(vals):
        if isinstance(vals[0], dict):
            return {k: merge([v[k] for v in vals]) for k in vals[0]}
        if isinstance(vals[0], list):
            return vals[0] if not isinstance(vals[0][0], list) else [[vals[0][i][0], float(np.mean([v[i][1] for v in vals]))] for i in range(len(vals[0]))]
        if vals[0] is None or any(v is None for v in vals):
            return None
        return float(np.mean(vals))
    return merge(studies)


def figure(res, path):
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
    for a, (name, title, label) in zip(ax, (("indus", "Indus inscriptions", "claimed table"), ("cipher", "Sanskrit enciphered with a known table", "true table"))):
        r = res[name]
        a.plot(*zip(*r["median_trajectory"]), color="#4a9bd1", lw=2, label="searched table, all texts")
        a.plot(*zip(*r["held_out"]["median_trajectory_fit"]), color="0.45", lw=1.2, label="searched on half: that half")
        a.plot(*zip(*r["held_out"]["median_trajectory_other"]), color="0.45", lw=1.2, ls="--", label="searched on half: the other half")
        a.axhline(r["table"], color="#c9663c", lw=2, label=label)
        a.axhline(r["random_tables"]["mean"], color="0.75", lw=1, ls=":", label="random tables (mean)")
        a.set_title(title, fontsize=10); a.set_xlabel("swaps tried"); a.legend(fontsize=7, loc="lower right")
    ax[0].set_ylabel("Sanskrit sequence score, bits per phoneme")
    plt.tight_layout(); plt.savefig(path, dpi=130); plt.close()


def fmt(x):
    return f"{x['mean']:+.3f} ± {x['sd']:.3f}"


def write_report(res):
    i, c, s = res["indus"], res["cipher"], res["settings"]
    hi, hc = i["held_out"], c["held_out"]
    ai, ac = i["agreement"], c["agreement"]
    rows = [
        ("The table itself", f"{i['table']:+.3f}", f"{c['table']:+.3f}"),
        ("Random tables (same values, shuffled)", fmt(i["random_tables"]), fmt(c["random_tables"])),
        (f"Random tables after {s['steps']:,} search steps", fmt(i["searched_tables"]), fmt(c["searched_tables"])),
        ("Searched tables that end above the table", f"{i['share_of_searched_tables_above_table']:.0%}", f"{c['share_of_searched_tables_above_table']:.0%}"),
        ("Search steps until the median search passes the table", f"{i['steps_until_median_search_passes_table'] or 'never'}",
         f"{c['steps_until_median_search_passes_table'] or 'never'}" if not isinstance(c['steps_until_median_search_passes_table'], float)
         else f"{c['steps_until_median_search_passes_table']:.0f}"),
        ("Searched on half the texts: score on that half", fmt(hi["searched_fit_half"]), fmt(hc["searched_fit_half"])),
        ("Searched on half the texts: score on the other half", fmt(hi["searched_other_half"]), fmt(hc["searched_other_half"])),
        ("The table on the other half", fmt(hi["table_other_half"]), fmt(hc["table_other_half"])),
        ("Random tables on the other half", fmt(hi["random_other_half"]), fmt(hc["random_other_half"])),
        ("Splits where the searched table beats the table on the other half", f"{hi['share_of_splits_searched_beats_table_on_other_half']:.0%}",
         f"{hc['share_of_splits_searched_beats_table_on_other_half']:.0%}"),
        ("Sign tokens two searched tables read alike", f"{ai['between_searched_tables']['mean']:.0%}", f"{ac['between_searched_tables']['mean']:.0%}"),
        ("Sign tokens a searched table reads like the table", f"{ai['searched_with_table']['mean']:.0%}", f"{ac['searched_with_table']['mean']:.0%}"),
        ("Sign tokens two random tables read alike (chance)", f"{ai['between_random_tables']['mean']:.0%}", f"{ac['between_random_tables']['mean']:.0%}"),
        ("The table after the same search: score", f"{i['table_after_search']['score']:+.3f}", f"{c['table_after_search']['score']:+.3f}"),
        ("The table after the same search: tokens still read the same", f"{i['table_after_search']['tokens_still_read_the_same']:.0%}",
         f"{c['table_after_search']['tokens_still_read_the_same']:.0%}"),
    ]
    gap_i = i["searched_tables"]["mean"] - i["table"]
    gap_c = c["searched_tables"]["mean"] - c["table"]
    L = ["# Milestone 4 extension: how high can a table score by fitting alone?", "",
         "**Exploratory, not pre-registered.** This analysis was written after the Milestone 4 results were known and is outside the "
         "corrected family of tests in `decipherment/prereg.json`. Nothing here changes the Milestone 4 result.", "",
         "The Milestone 4 test compares a claimed table with random tables. A decipherer does not pick a table at random: they search for "
         "one that reads well. So here random tables are searched too. Each starts as a shuffle of the claim's own sound values and is "
         f"improved by swapping the values of two signs whenever that does not lower the Sanskrit sequence score ({s['steps']:,} swaps tried, "
         f"{s['climbs']} searches). The search knows no Sanskrit beyond the score and nothing about the inscriptions' meaning.", "",
         "The same search is run on a control where the right answer is known: held-out Sanskrit text enciphered with the claim's own "
         f"sign-to-sound relation and cut to the lengths of real inscriptions ({s['cipher_replicates']} fresh encipherments, averaged). "
         "The control column shows what a correct table looks like under exactly the same treatment.", "",
         f"Claim: Yajnadevam, version of 15 July 2026 ({i['texts']:,} texts, {i['tokens']:,} sign tokens, {i['units_in_texts']} groups of signs "
         "that occur in them). All scores are bits per phoneme; higher is more Sanskrit-like.", "",
         "| | Indus inscriptions, claimed table | Enciphered Sanskrit, true table |", "|---|---|---|"]
    L += [f"| {a} | {b} | {d} |" for a, b, d in rows]
    L += ["", "![search trajectories](overfitting.png)", "", "## Reading the numbers", ""]
    L.append(f"- **Ceiling.** On the Indus inscriptions a blind search ends {gap_i:+.3f} bits per phoneme "
             f"{'above' if gap_i > 0 else 'below'} the claimed table, and {i['share_of_searched_tables_above_table']:.0%} of searches end above it. "
             f"On enciphered Sanskrit the search ends {gap_c:+.3f} {'above' if gap_c > 0 else 'below'} the true table "
             f"({c['share_of_searched_tables_above_table']:.0%} of searches above it).")
    L.append(f"- **Held-out half.** A table searched on half the Indus texts scores {hi['searched_other_half']['mean']:+.3f} on the half it never saw; "
             f"the claimed table scores {hi['table_other_half']['mean']:+.3f} there and a random table {hi['random_other_half']['mean']:+.3f}. "
             f"In the control the searched table scores {hc['searched_other_half']['mean']:+.3f} on the unseen half and the true table "
             f"{hc['table_other_half']['mean']:+.3f}. The claimed table was fitted by its author to all the texts, so neither half is unseen for it; "
             "its two halves are shown only as the reference level.")
    L.append(f"- **Do searches agree?** Two independent searches on the Indus texts read {ai['between_searched_tables']['mean']:.0%} of sign tokens "
             f"with the same consonants (chance: {ai['between_random_tables']['mean']:.0%}); in the control {ac['between_searched_tables']['mean']:.0%} "
             f"(chance {ac['between_random_tables']['mean']:.0%}), and a searched table agrees with the true table on "
             f"{ac['searched_with_table']['mean']:.0%} of tokens.")
    d_i = hi["searched_other_half"]["mean"] - hi["table_other_half"]["mean"]
    d_c = hc["searched_other_half"]["mean"] - hc["table_other_half"]["mean"]
    L.append(f"- **What separates the two columns.** On text the search has not seen, a searched table is {d_c:+.3f} from the true table in the "
             f"control and {d_i:+.3f} from the claimed table on the Indus inscriptions. One caution: the Indus texts are formulaic, so what a "
             "search learns on one half carries over to the other more easily than in the control, where each sound is written with a "
             "randomly chosen sign among those that carry it. The two gaps are therefore not on the same footing.")
    L += ["", "## What this does and does not show", "",
          "- It shows how much of a Sanskrit-like score is available to any table through search alone, on this corpus and with this score. "
          "A score that a blind search reaches or passes cannot by itself be evidence that a table is right.",
          ("- The search passes the true table in the control as well, with tables that disagree with it. So scoring below the search ceiling "
             "is what a correct table does too and is not evidence against the claim; and the sequence score alone cannot pick out the right "
             "table on a corpus of this size. The Milestone 4 positive control (true table against shuffles) is a different and easier question."
             if c["share_of_searched_tables_above_table"] > 0.5 else
             "- In the control the search does not pass the true table, so on a corpus of this size a correct table sits at or above what "
             "search reaches."),
          "- It does not show the claimed table was produced by such a search, and it does not test the claimant's own reading method "
          "(conjuncts, vowels and per-text choices are outside a sign-by-sign table, as in Milestone 4).",
          "- The control is optimistic for the true table: the enciphered text comes from the same corpus as the scoring model, and it is "
          "enciphered sign by sign with no noise. A real Bronze Age language would sit lower.",
          "- The search is plain greedy swapping from one start with the claim's value inventory fixed. A stronger search would reach higher, "
          "so the ceiling reported here is a lower bound on what fitting can reach.",
          "- If independent searches disagree with each other about what the signs say while scoring alike, the score does not pin down one "
          "table on a corpus this size. That is an empirical version of the unicity-distance argument, for this score only.", "",
          f"Settings: {s['climbs']} searches and {s['splits']} half-splits of {s['steps']:,} swaps each on the Indus texts; {s['cipher_replicates']} "
          f"encipherments with {s['cipher_climbs']} searches and {s['cipher_splits']} half-splits each; seed {s['seed']}. "
          f"Run time {s['runtime_min']:.0f} minutes on a laptop CPU.", ""]
    (OUT / "overfitting_report.md").write_text("\n".join(L))


def selfcheck():
    """On a toy cipher with plenty of text, the search must climb, never fall, and keep the multiset of values."""
    rng = np.random.default_rng(0)
    plain = ["".join(rng.choice(list("ktmn"), 6, p=[0.4, 0.3, 0.2, 0.1])) for _ in range(300)]
    plain = [p.replace("kk", "kt").replace("mm", "mk") for p in plain]
    tables = scoring_tables(plain)
    key = {i + 1: (str(i + 1), c + "a") for i, c in enumerate("ktmn")}
    texts = [tuple("ktmn".index(ch) + 1 for ch in p) for p in plain[:120]]
    cp, us = Corpus(texts, key), units(key)
    vals, traj = climb(cp, us, tables, rng, 200, every=10)
    assert sorted(vals) == sorted(v for _, v in us)
    assert all(b[1] >= a[1] - 1e-12 for a, b in zip(traj, traj[1:])), "the score fell during a climb"
    true = split_score(cp, cp.codes(assign(us, [v for _, v in us]), WIDTH), tables, np.ones(cp.n_texts, bool))[0]
    assert abs(traj[-1][1] - true) < 1e-9 and agreement(cp, us, vals, [v for _, v in us]) == 1.0, "a 4-sign cipher with 720 tokens must be solved"
    half = np.arange(cp.n_texts) % 2 == 0
    a, b = split_score(cp, cp.codes(assign(us, vals), WIDTH), tables, half)
    s, n = cp.score(cp.codes(assign(us, vals), WIDTH), tables, per_text=True)
    assert abs((a * n[half].sum() + b * n[~half].sum()) / n.sum() - (s * n).sum() / n.sum()) < 1e-12


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    start = time.time()
    selfcheck()
    climbs, steps, splits, reps, c_climbs, c_splits = (3, 600, 3, 1, 2, 2) if args.quick else (20, 8000, 20, 5, 6, 6)
    seed = 20261114
    rng = np.random.default_rng(seed)
    tables = scoring_tables([skeleton(l) for l in data.dcs_lines("train")])
    key = data.load_key("xlits_2026-07-15.csv")
    wells = corpus.load_wells()[0]
    texts = [t for t in wells if all(s in key for s in t)]
    print("Indus inscriptions ...", flush=True)
    res = {"indus": study(Corpus(texts, key), key, tables, rng, climbs, steps, splits)}
    plain, lengths = data.dcs_lines("heldout"), [len(t) for t in wells]
    runs = []
    for r in range(reps):
        print(f"enciphered Sanskrit {r + 1}/{reps} ...", flush=True)
        ct = encipher([plain[i] for i in rng.permutation(len(plain))[:400]], key, lengths, rng)
        runs.append(study(Corpus(ct, key), key, tables, rng, c_climbs, steps, c_splits))
    res["cipher"] = pool(runs)
    res["settings"] = {"climbs": climbs, "steps": steps, "splits": splits, "cipher_replicates": reps, "cipher_climbs": c_climbs,
                       "cipher_splits": c_splits, "seed": seed, "quick": args.quick, "pre_registered": False,
                       "runtime_min": (time.time() - start) / 60}
    out = OUT = data.ROOT / "reports/quick" if args.quick else OUT  # reports/quick is gitignored
    out.mkdir(parents=True, exist_ok=True)
    (out / "overfitting.json").write_text(json.dumps(res, indent=1))
    figure(res, out / "overfitting.png")
    write_report(res)
    print(json.dumps({k: {x: v[x] for x in ("table", "random_tables", "searched_tables", "share_of_searched_tables_above_table", "agreement")}
                      for k, v in res.items() if k != "settings"}, indent=1))
    print(f"done in {(time.time() - start) / 60:.1f} min -> {out / 'overfitting_report.md'}")
