"""Milestone 4 extension: the same claimed table scored against four different languages.

    python decipherment/languages.py           # about 5 minutes on a laptop CPU
    python decipherment/languages.py --quick   # a fast check, writes to reports/quick

Exploratory, NOT pre-registered. Is Sanskrit special for the claimed table, or does any language do as well? For Sanskrit,
Old Tamil, English and Sumerian (each reduced to the 16-letter consonant skeleton and cut to the same number of letters) the
claimed table is compared with shuffled versions of itself, and a blind search is run to see how high a table can score.
Only the primary score is used (trigram of the language minus its own letter frequencies), as in overfit.py.

Writes reports/milestone_4/ (languages.json, languages_report.md): aggregates only.
"""
import argparse
import json
import re
import sys
import time
from collections import Counter

import numpy as np

import data
import harness
import overfit
from data import skeleton
from harness import Corpus, assign, freq_bins, p_value, units

sys.path.insert(0, str(data.ROOT / "sequence_analysis"))
import corpus  # noqa: E402
import stats  # noqa: E402

LANGS = ("sanskrit", "tamil", "english", "sumerian")
WIDTH = overfit.WIDTH
# English letters -> harness letters; vowels and anything else are dropped. x is two letters.
ENGLISH = {"b": "b", "c": "k", "d": "d", "f": "p", "g": "g", "h": "h", "j": "j", "k": "k", "l": "l", "m": "m", "n": "n", "p": "p",
           "q": "k", "r": "r", "s": "s", "t": "t", "v": "v", "w": "v", "x": "ks", "y": "y", "z": "s"}
# ATF transliteration letters of Sumerian sign readings: sz is the sibilant written sh, j is the velar nasal, x is the velar fricative.
SUMERIAN = {"b": "b", "d": "d", "g": "g", "h": "h", "x": "h", "j": "n", "k": "k", "l": "l", "m": "m", "n": "n", "p": "p", "q": "k",
            "r": "r", "s": "s", "t": "t", "z": "s"}


def english_letters(line):
    return "".join(ENGLISH.get(c, "") for c in line.lower())


def sumerian_letters(sign):
    return "".join(SUMERIAN.get(c, "") for c in sign.lower().replace("sz", "s"))


def sumerian_lines(path):
    """Same filter as corpus.cuneiform (Sumerian-tagged texts, lines with a broken sign dropped), one letter string per line."""
    out, lang = [], None
    for line in path.read_text(errors="ignore").split("\n"):
        if line.startswith("&"):
            lang = None
        elif line.startswith("#atf: lang"):
            lang = line.split()[-1]
        elif lang == "sux" and re.match(r"^\d+'?\.", line):
            body = re.sub(r"[#?!*\[\]<>_⸢⸣]", "", line.split(".", 1)[1]).lower()
            signs = [s for s in re.split(r"[-\s.{}+:]+", body) if s]
            if signs and not any("x" == s or "..." in s or s == "$" for s in signs):
                out.append("".join(sumerian_letters(s) for s in signs))
    return [l for l in out if l]


def english_lines():
    text = (corpus.REFERENCE / "pride_and_prejudice.txt").read_text(encoding="utf-8-sig")
    text = text[text.index("Chapter I.]"):text.index("*** END OF THE PROJECT GUTENBERG")]
    return [l for l in map(english_letters, text.split("\n")) if l]


def equalize(lines_by_lang, rng):
    """Every language cut to the same number of letters (that of the smallest), lines taken in random order."""
    n = min(sum(map(len, v)) for v in lines_by_lang.values())
    out = {}
    for k, v in lines_by_lang.items():
        got, total = [], 0
        for i in rng.permutation(len(v)):
            if total >= n:
                break
            got.append(v[i])
            total += len(v[i])
        out[k] = got
    return out, n


def whole(cp, vals, us, tables):
    return overfit.split_score(cp, cp.codes(assign(us, vals), WIDTH), tables, np.ones(cp.n_texts, bool))[0]


def permutation(cp, key, tables, draws, rng):
    us = units(key)
    true = np.array([v for _, v in us], dtype=object)
    obs = whole(cp, list(true), us, tables)
    r = {"observed": obs}
    for name, bins in (("value_shuffle", None), ("binned_10", freq_bins(us, cp.counts, 10))):
        null = np.array([whole(cp, list(true[perm(len(us), bins, rng)]), us, tables) for _ in range(draws)])
        r[name] = {"mean": float(null.mean()), "sd": float(null.std()), "p": p_value(obs, null), "percentile": float(100 * (null < obs).mean())}
    return r


def perm(n, bins, rng):
    """Index permutation of n units, within bins when given."""
    if bins is None:
        return rng.permutation(n)
    out = np.arange(n)
    for b in bins:
        out[b] = b[rng.permutation(len(b))]
    return out


def search(cp, key, tables, rng, starts, steps):
    """Blind searches fitted on a random half of the texts: held-out score of each, of the claimed table, and the found tables."""
    us = units(key)
    true = [v for _, v in us]
    rest, claim, found = [], [], []
    for _ in range(starts):
        fit = rng.permutation(cp.n_texts) < cp.n_texts // 2
        vals, traj = overfit.climb(cp, us, tables, rng, steps, fit=fit)
        rest.append(traj[-1][2])
        claim.append(overfit.split_score(cp, cp.codes(assign(us, true), WIDTH), tables, fit)[1])
        found.append((traj[-1][1] + 0, vals))
    best = max(range(starts), key=lambda i: found[i][0])
    return {"ceiling_mean": float(np.mean(rest)), "ceiling_sd": float(np.std(rest)), "claim_mean": float(np.mean(claim)),
            "gap_mean": float(np.mean(np.array(rest) - np.array(claim))), "gap_sd": float(np.std(np.array(rest) - np.array(claim)))}, found[best][1]


def write_report(res, out):
    s = res["settings"]
    L = ["# The same claim against four languages (exploratory, not pre-registered)", "",
         "Is Sanskrit special for the claimed table? Four languages are reduced to the same 16-letter consonant skeleton and cut to the same length "
         f"({res['letters']:,} letters each, so each trigram is equally well estimated): Sanskrit (DCS), Old Tamil (Sangam verse), English (Pride and Prejudice) and Sumerian "
         "(CDLI sign readings of Sumerian texts). For each, the claimed table is compared with shuffled versions of itself, and a blind search is run on half the texts "
         "and scored on the other half. Only the primary score is used: log-probability under the language's character trigram minus under its letter frequencies, in bits per phoneme.", "",
         "Scores in bits are not comparable across languages in absolute terms; what compares is the percentile against each language's own shuffles and the gap to each language's own search ceiling.", "",
         "## Permutation test", "", f"{s['draws']:,} draws per null. Holm correction over {len(LANGS) * len(res['keys'])} tests (4 languages, {len(res['keys'])} keys) on the frequency-binned null.", "",
         "| Key | Language | Observed | Value-shuffle percentile (p) | Binned percentile (p) | Binned Holm p |", "|---|---|---|---|---|---|"]
    for k, per in res["keys"].items():
        for lang in LANGS:
            r = per[lang]
            L.append(f"| {k} | {lang} | {r['observed']:+.3f} | {r['value_shuffle']['percentile']:.1f} ({r['value_shuffle']['p']:.4f}) | {r['binned_10']['percentile']:.1f} ({r['binned_10']['p']:.4f}) | {res['holm'][k][lang]:.3f} |")
    L += ["", "## Blind search", "", f"{s['starts']} searches of {s['steps']:,} swaps per language, key `{res['search_key']}`; scores are on the half of the texts the search did not see.", "",
          "| Language | Search ceiling (held-out) | Claimed table (held-out) | Gap, search minus table |", "|---|---|---|---|"]
    for lang in LANGS:
        r = res["search"][lang]
        L.append(f"| {lang} | {r['ceiling_mean']:+.3f} ± {r['ceiling_sd']:.3f} | {r['claim_mean']:+.3f} | {r['gap_mean']:+.3f} ± {r['gap_sd']:.3f} |")
    L += ["", "## Best searched table of each language, scored under every language", "", "Rows: language the table was searched on. Columns: language of the trigram that scores it (whole corpus, bits per phoneme).", "",
          "| Searched on | " + " | ".join(LANGS) + " |", "|---|" + "---|" * len(LANGS)]
    for a in LANGS:
        L.append(f"| {a} | " + " | ".join(f"{res['cross'][a][b]:+.3f}" for b in LANGS) + " |")
    L += ["", "## Reading", "", res["reading"], "", "## Limits", "",
          "Consonant skeletons erase much of what separates languages. The claimed table's values are Sanskrit consonants, so against another language part of its gap to the ceiling may come from a poor fit of its values to that language rather than from the inscriptions; this was not separated. Four languages, one search method, one key. The English and Sumerian letter maps are mine and are given in the script. "
          f"Settings: seed {s['seed']}, runtime {s['runtime_min']:.1f} min, quick run: {s['quick']}.", ""]
    (out / "languages_report.md").write_text("\n".join(L))


def reading(res):
    k = "yajnadevam-2026-07-15"
    pc = {l: res["keys"][k][l]["binned_10"]["percentile"] for l in LANGS}
    sig = [l for l in LANGS if res["holm"][k][l] < 0.05]
    gaps = {l: res["search"][l]["gap_mean"] for l in LANGS}
    spread = float(np.mean([res["search"][l]["gap_sd"] for l in LANGS]))
    a = (f"Against frequency-matched shuffles the current table sits at percentile " + ", ".join(f"{l} {p:.0f}" for l, p in pc.items()) + ". "
         + ("Significant after Holm correction for: " + ", ".join(sig) + "." if sig else "None is significant after Holm correction."))
    b = (f" The search ceiling lies above the claimed table by " + ", ".join(f"{l} {g:+.2f}" for l, g in gaps.items()) + " bits; the largest difference between languages is "
         f"{max(gaps.values()) - min(gaps.values()):.2f} bits against a typical spread over searches of {spread:.2f}, so the ceiling " +
         ("differs between languages by more than its spread." if max(gaps.values()) - min(gaps.values()) > spread else "does not differ between languages by more than its spread."))
    return a + b


def selfcheck():
    assert english_letters("Knight") == "knght" and english_letters("queen") == "kn" and english_letters("exact") == "kskt"
    assert sumerian_letters("lugal") == "lgl" and sumerian_letters("szu") == "s" and sumerian_letters("e2") == "" and sumerian_letters("dumu") == "dm"
    ok = set(harness.ALPHABET)
    assert set("".join(ENGLISH.values())) <= ok and set("".join(SUMERIAN.values())) <= ok
    got, n = equalize({"a": ["kkkk"] * 10, "b": ["kk"] * 4}, np.random.default_rng(0))
    assert n == 8 and sum(map(len, got["a"])) == 8 and sum(map(len, got["b"])) == 8
    n = 6
    assert sorted(perm(n, [np.array([0, 1, 2]), np.array([3, 4, 5])], np.random.default_rng(0))[:3]) == [0, 1, 2]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    start = time.time()
    selfcheck()
    draws, starts, steps = (200, 2, 300) if args.quick else (2000, 10, 2000)
    seed = 20261116
    rng = np.random.default_rng(seed)
    sux = sumerian_lines(corpus.REFERENCE / "cdliatf_unblocked.atf")
    raw = {"sanskrit": [skeleton(l) for l in data.dcs_lines("train")], "tamil": [skeleton(l) for l in data.control_lines()],
           "english": english_lines(), "sumerian": sux}
    raw = {k: [l for l in v if l] for k, v in raw.items()}
    lines, n_letters = equalize(raw, rng)
    tables = {l: overfit.scoring_tables(lines[l]) for l in LANGS}
    keys = {"yajnadevam-2024-11-30": data.load_key("xlits_2024-11-30.csv"), "yajnadevam-2026-07-15": data.load_key("xlits_2026-07-15.csv")}
    wells = corpus.load_wells()[0]
    res = {"letters": n_letters, "keys": {}, "search_key": "yajnadevam-2026-07-15"}
    for k, key in keys.items():
        cp = Corpus([t for t in wells if all(s in key for s in t)], key)
        res["keys"][k] = {}
        for l in LANGS:
            print(k, l, flush=True)
            res["keys"][k][l] = permutation(cp, key, tables[l], draws, rng)
    family = [(k, l) for k in keys for l in LANGS]
    adj = stats.holm([res["keys"][k][l]["binned_10"]["p"] for k, l in family])
    res["holm"] = {k: {} for k in keys}
    for (k, l), p in zip(family, adj):
        res["holm"][k][l] = float(p)
    key = keys[res["search_key"]]
    cp, us = Corpus([t for t in wells if all(s in key for s in t)], key), units(key)
    res["search"], best = {}, {}
    for l in LANGS:
        print("search", l, flush=True)
        res["search"][l], best[l] = search(cp, key, tables[l], rng, starts, steps)
    res["cross"] = {a: {b: whole(cp, best[a], us, tables[b]) for b in LANGS} for a in LANGS}
    res["reading"] = reading(res)
    res["settings"] = {"draws": draws, "starts": starts, "steps": steps, "seed": seed, "quick": args.quick, "pre_registered": False,
                       "runtime_min": (time.time() - start) / 60}
    out = data.ROOT / "reports/quick" if args.quick else harness.OUT
    out.mkdir(parents=True, exist_ok=True)
    (out / "languages.json").write_text(json.dumps(res, indent=1))
    write_report(res, out)
    print(res["reading"])
    print(f"done in {(time.time() - start) / 60:.1f} min -> {out / 'languages_report.md'}")
