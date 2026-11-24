"""Milestone 4 extension: more controls on real scripts, so the method has a hit rate and a false-alarm rate.

    python decipherment/real_controls.py           # about 15 minutes on a laptop CPU
    python decipherment/real_controls.py --quick   # a fast check, writes to reports/quick

Exploratory, NOT pre-registered. Linear B read as Greek (linear_b.py) is one true decipherment. This adds a second true one,
Ugaritic read as Hebrew, and a known-wrong one, Linear A read with Linear B sound values as Greek. Each is run through the
unchanged harness test: a table against shuffled versions of itself.

Sources, fetched at run time into data/raw/real_controls/ (gitignored), SHA-256 recorded on first fetch:
  uga-heb.no_spe.cog       github.com/j-luo93/NeuroDecipher @ 480bad24 (no licence stated)
  LinearAInscriptions.js   github.com/mwenge/lineara.xyz @ 9c81beba (author disclaims copyright)
Homer for the Greek target comes from linear_b.py's fetch.
"""
import argparse
import hashlib
import json
import re
import time
from collections import Counter

import numpy as np

import data
import harness
import linear_b
from data import skeleton
from harness import Corpus, corpus, test_key  # harness puts sequence_analysis on the path

RAW = data.RAW.parent / "real_controls"
NEURO = "https://raw.githubusercontent.com/j-luo93/NeuroDecipher/480bad2487820e3737fecfdd108214efa769e34b/data/{}"
FILES = {"uga-heb.no_spe.cog": NEURO.format("uga-heb.no_spe.cog"), "uga-heb.small.no_spe.cog": NEURO.format("uga-heb.small.no_spe.cog"),
         "LinearAInscriptions.js": "https://raw.githubusercontent.com/mwenge/lineara.xyz/9c81beba669a4ffec3f5b29d4e7748d02c060eff/LinearAInscriptions.js"}
UGA = "$&*<@HSTZabdghiklmnpqrstuvwxyz"
HEB = "$&<HSTabdghklmnpqrstwyz"
HARNESS = "kgcjtdnpbmyrlvs"  # the 15 most frequent Hebrew symbols take these in order; the rest share "h"
PART = re.compile(r"[A-Z]+[₀-₉]*")


def fetch():
    RAW.mkdir(parents=True, exist_ok=True)
    f = RAW / "sha256.json"
    recorded = json.loads(f.read_text()) if f.exists() else {}
    for name, url in FILES.items():
        path = RAW / name
        if not path.exists():
            path.write_bytes(data.get(url))
        got = data.sha256(path.read_bytes())
        assert got == recorded.setdefault(name, got), f"{name} changed since it was first fetched"
    f.write_text(json.dumps(recorded, indent=1))


# ---------- Ugaritic and Hebrew ----------

def half(word):
    return "A" if int(hashlib.md5(word.encode()).hexdigest(), 16) % 2 == 0 else "B"


def parse_uga(text):
    """[(ugaritic, first hebrew cognate or None)] from the cognate TSV (header skipped, '_' = no cognate)."""
    out = []
    for line in text.splitlines()[1:]:
        if "\t" in line:
            u, h = line.split("\t")[:2]
            first = h.split("|")[0]
            out.append((u, None if first == "_" else first))
    return out


def true_uga_key(rows_a):
    """Ugaritic symbol -> Hebrew symbol, from half A only: shared symbols map to themselves; each Ugaritic-only symbol to the
    Hebrew symbol most often found at the same position in equal-length cognate pairs."""
    pos = {u: Counter() for u in UGA if u not in HEB}
    for u, h in rows_a:
        if h and len(u) == len(h):
            for a, b in zip(u, h):
                if a in pos:
                    pos[a][b] += 1
    return {u: (u if u in HEB else (pos[u].most_common(1)[0][0] if pos[u] else "h")) for u in UGA}


def hebrew_letters(rows_a):
    """The 23 Hebrew symbols onto the 16 harness letters, by frequency in half A (lossy; works against the true key)."""
    freq = Counter(c for _, h in rows_a if h for c in h)
    order = sorted(HEB, key=lambda c: (-freq[c], c))
    return {c: (HARNESS[i] if i < len(HARNESS) else "h") for i, c in enumerate(order)}


def ugaritic(text, rng, sub_draws, subsamples, indus_tokens, draws, tables_for):
    rows = parse_uga(text)
    a = [r for r in rows if half(r[0]) == "A"]
    b = sorted({u for u, _ in rows if half(u) == "B" and len(u) >= 2 and all(c in UGA for c in u)})
    hmap = hebrew_letters(a)
    target = ["".join(hmap[c] for c in h) for _, h in a if h and all(c in hmap for c in h)]
    key_heb = true_uga_key(a)
    sign = {c: i + 1 for i, c in enumerate(UGA)}
    key = {sign[u]: (u, hmap[h]) for u, h in key_heb.items()}
    texts = [tuple(sign[c] for c in w) for w in b]
    tables = tables_for(target)
    full, _ = test_key(Corpus(texts, key), key, tables, draws, rng, bin_counts=(10,))
    power = {"value_shuffle": [], "binned_10": []}
    for _ in range(subsamples):
        pick, n = [], 0
        for i in rng.permutation(len(texts)):
            if n >= indus_tokens:
                break
            pick.append(texts[i]); n += len(texts[i])
        r, _ = test_key(Corpus(pick, key), key, tables, sub_draws, rng, bin_counts=(10,))
        for k in power:
            power[k].append(r["nulls"][k]["sequence"]["p"] < 0.05)
    return {"rows": len(rows), "half_a_rows": len(a), "target_words": len(target), "texts": len(texts), "tokens": sum(map(len, texts)),
            "signs": len({s for t in texts for s in t}), "key": {u: h for u, h in key_heb.items()}, "hebrew_letters": hmap,
            "test": full, "indus_size_power": {k: float(np.mean(v)) for k, v in power.items()}, "subsamples": subsamples}


# ---------- Linear A ----------

def linear_a_words(js):
    """Distinct words made only of syllabograms (each '-' part like 'QE' or 'RA₂', at least 2 parts), and how many were dropped."""
    js = re.sub(r",(\s*\])", r"\1", js)  # JavaScript allows a trailing comma before ], JSON does not
    dec = json.JSONDecoder()  # raw_decode, not a regex: words can contain "]"
    entries = [w for m in re.finditer(r'"transliteratedWords"\s*:\s*\[', js) for w in dec.raw_decode(js, m.end() - 1)[0]]
    entries = [w for w in entries if isinstance(w, str) and w.strip()]
    keep = [w for w in entries if len(w.split("-")) >= 2 and all(PART.fullmatch(p) for p in w.split("-"))]
    return sorted(set(keep)), len(entries) - len(keep), len(entries)


def part_letters(part):
    """'RA₂' -> 'r': lowercase, subscript dropped, the consonant series through linear_b.SERIES."""
    syl = re.sub(r"[₀-₉]", "", part).lower()
    return "".join(linear_b.SERIES[c] for c in syl if c in linear_b.SERIES)


def linear_a(js, rng, draws, tables):
    words, dropped, total = linear_a_words(js)
    parts = sorted({p for w in words for p in w.split("-")})
    sign = {p: i + 1 for i, p in enumerate(parts)}
    key = {sign[p]: (p, part_letters(p)) for p in parts}
    texts = [tuple(sign[p] for p in w.split("-")) for w in words]
    full, _ = test_key(Corpus(texts, key), key, tables, draws, rng, bin_counts=(10,))
    return {"entries": total, "kept_words": len(words), "dropped_entries": dropped, "texts": len(texts),
            "tokens": sum(map(len, texts)), "signs": len(parts), "test": full}


# ---------- report ----------

def pct(x):
    return f"{100 * x:.0f}%"


def summary_rows(res):
    lb = res["linear_b"]
    rows = [("Linear B", "Greek", "true", lb["signs"], lb["tokens"], lb["nulls"], "-")]
    u, a = res["ugaritic"], res["linear_a"]
    rows.append(("Ugaritic", "Hebrew", "true", u["signs"], u["tokens"], u["test"]["nulls"],
                 f"{pct(u['indus_size_power']['binned_10'])} (binned), {pct(u['indus_size_power']['value_shuffle'])} (value shuffle)"))
    rows.append(("Linear A, Linear B values", "Greek", "wrong", a["signs"], a["tokens"], a["test"]["nulls"], "-"))
    return rows


def write_report(res, out):
    s, u, a = res["settings"], res["ugaritic"], res["linear_a"]
    L = ["# More controls on real scripts (exploratory, not pre-registered)", "",
         "One true decipherment is one data point. This adds a second true one (Ugaritic read as Hebrew) and a known-wrong one (Linear A "
         "read with Linear B sound values as Greek), and runs each through the unchanged Milestone 4 test: a table against shuffled "
         f"versions of itself, primary score (order beyond letter frequency), {s['draws']:,} draws.", "",
         "## Summary", "",
         "| Script | Read as | Table is | Signs | Tokens | Binned null: percentile, p | Value shuffle: percentile, p | Power at Indus size |",
         "|---|---|---|---|---|---|---|---|"]
    for name, lang, kind, signs, tokens, nulls, power in summary_rows(res):
        b, v = nulls["binned_10"]["sequence"], nulls["value_shuffle"]["sequence"]
        L.append(f"| {name} | {lang} | {kind} | {signs} | {tokens:,} | {b['percentile']:.1f}, {b['p']:.4f} | {v['percentile']:.1f}, {v['p']:.4f} | {power} |")
    L += ["", "Linear B comes from `linear_b.json`. A true table should be significant and the wrong one should not.", "",
          "## Reading", "", *[f"- {x}" for x in res["reading"]], "",
          "## Ugaritic as Hebrew", "",
          f"Source: `uga-heb.no_spe.cog` from NeuroDecipher (commit 480bad24, no licence stated), {u['rows']:,} rows. Rows are split in two by a hash of the "
          f"Ugaritic word. The Hebrew target is the first cognate of each half-A row ({u['target_words']:,} words); the texts are the {u['texts']:,} distinct "
          f"Ugaritic words of half B ({u['tokens']:,} letters), one sign per letter. The true table is built from half A only: a Ugaritic letter that exists in "
          "Hebrew maps to itself, and each of the 7 Ugaritic-only letters to the Hebrew letter most often found at the same position in equal-length cognate pairs.", "",
          "| Ugaritic | Hebrew | Harness letter |", "|---|---|---|",
          *[f"| `{k}` | `{v}` | `{u['hebrew_letters'][v]}` |" for k, v in u["key"].items()], "",
          "The 23 Hebrew letters are fitted into the harness's 16 by frequency: the 15 commonest get their own letter and the other 8 share `h`. "
          "This merge is lossy and works against the true table.", "",
          f"Indus size: {u['subsamples']} random subsamples of the Ugaritic words with about as many sign tokens as the Wells corpus "
          f"({res['indus_tokens']:,}), {s['sub_draws']:,} draws each; power is the share with p < 0.05.", "",
          "## Linear A with Linear B values as Greek", "",
          f"Source: `LinearAInscriptions.js` from lineara.xyz (commit 9c81beba; the author disclaims copyright). Of {a['entries']:,} transliterated entries, "
          f"{a['dropped_entries']:,} were dropped (numerals, logograms, unidentified signs, dividers, single syllabograms) and {a['kept_words']:,} distinct words of "
          f"two or more syllabograms kept ({a['tokens']:,} tokens, {a['signs']} signs). Values: the Linear B consonant series of each syllabogram, as in "
          "`linear_b.py`. Linear A is not thought to be Greek, so this table is wrong by construction.", "",
          "## Limits", "",
          "- Ugaritic is an alphabet of 30 letters, far easier than a system of about 400 signs.",
          "- All three controls are word lists, not running text.",
          "- Three controls are still few for a hit rate and a false-alarm rate.",
          "- The Linear A control is wrong in its language, not necessarily in its sound values: Linear B took its signs from Linear A, and many "
          "values are thought to carry over. So a significant score may mean the order test responds to roughly right sound values in a language "
          "with similar syllable shapes, which is itself a limit on what the test can say about a language.", "",
          f"Settings: seed {s['seed']}, run time {s['runtime_min']:.1f} minutes, quick run: {s['quick']}.", ""]
    (out / "real_controls_report.md").write_text("\n".join(L))


def reading(res):
    sig = lambda n: n["binned_10"]["sequence"]["p"] < 0.05
    lb, u, a = res["linear_b"]["nulls"], res["ugaritic"]["test"]["nulls"], res["linear_a"]["test"]["nulls"]
    out = [f"Ugaritic as Hebrew (true): {'detected' if sig(u) else 'not detected'} against frequency-matched shuffles "
           f"(p {u['binned_10']['sequence']['p']:.4f}); at Indus size it is detected in {pct(res['ugaritic']['indus_size_power']['binned_10'])} of subsamples.",
           f"Linear A as Greek (wrong): {'significant, which is a false alarm and an open problem for the method' if sig(a) else 'not significant, as expected'} "
           f"(p {a['binned_10']['sequence']['p']:.4f}).",
           f"With Linear B ({'detected' if sig(lb) else 'not detected'}), the method finds {sum(map(sig, (lb, u)))} of 2 true tables and raises "
           f"{int(sig(a))} false alarm on 1 wrong table, against the frequency-matched null."]
    return out


def selfcheck():
    rows = [("abcd", "abcd"), ("aZbc", "aSbc"), ("xyz", "_"), ("iub", "yHb")]
    key = true_uga_key([(u, None if h == "_" else h) for u, h in rows])
    assert key["a"] == "a" and key["Z"] == "S" and key["i"] == "y" and key["u"] == "H"
    hm = hebrew_letters([("x", "aab"), ("y", "abc")])
    assert hm["a"] == "k" and hm["b"] == "g" and len(set(hm.values())) == 16
    for v in hm.values():
        assert skeleton(v) == v, v
    js = 'var x = new Map([["HT1",{"transliteratedWords": ["QE-RA₂-U", "\\n", "GRA", "KU-RO", "*301-NA", "10", "A",]}]]);'
    words, dropped, total = linear_a_words(js)
    assert words == ["KU-RO", "QE-RA₂-U"] and dropped == 4 and total == 6, (words, dropped, total)
    assert part_letters("RA₂") == "r" and part_letters("QE") == "c" and part_letters("U") == ""
    assert half("msgr") in "AB"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    start = time.time()
    selfcheck()
    draws, sub_draws, subsamples = (300, 100, 3) if args.quick else (10000, 1000, 20)
    seed = 20261124
    rng = np.random.default_rng(seed)
    fetch()
    linear_b.fetch()
    sanskrit = [skeleton(l) for l in data.dcs_lines("train")]
    lm_s = harness.lm_table(sanskrit)
    def tables_for(target):
        lm_t = harness.lm_table(target)
        return ((lm_t, harness.lm_table(target, order=1)), (lm_t, lm_s), harness.nb_tables(target, sanskrit))
    indus_tokens = sum(map(len, corpus.load_wells()[0]))
    print("Ugaritic ...", flush=True)
    uga = ugaritic((RAW / ("uga-heb.small.no_spe.cog" if args.quick else "uga-heb.no_spe.cog")).read_text(), rng, sub_draws, subsamples,
                   indus_tokens, draws, tables_for)
    print("Linear A ...", flush=True)
    greek = linear_b.verse_words((linear_b.RAW / "iliad.xml").read_text()) + linear_b.verse_words((linear_b.RAW / "odyssey.xml").read_text())
    la = linear_a((RAW / "LinearAInscriptions.js").read_text(), rng, draws, tables_for(greek))
    lbj = json.loads((harness.OUT / "linear_b.json").read_text())
    res = {"ugaritic": uga, "linear_a": la, "indus_tokens": indus_tokens,
           "linear_b": {"signs": lbj["corpus"]["signs"], "tokens": lbj["corpus"]["tokens"], "nulls": lbj["true_key"]["nulls"]}}
    res["reading"] = reading(res)
    res["settings"] = {"draws": draws, "sub_draws": sub_draws, "seed": seed, "quick": args.quick, "pre_registered": False,
                       "runtime_min": (time.time() - start) / 60}
    out = data.ROOT / "reports/quick" if args.quick else harness.OUT
    out.mkdir(parents=True, exist_ok=True)
    (out / "real_controls.json").write_text(json.dumps(res, indent=1))
    write_report(res, out)
    print("\n".join(res["reading"]))
    print(f"done in {(time.time() - start) / 60:.1f} min -> {out / 'real_controls_report.md'}")
