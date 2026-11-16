"""Milestone 4 extension: a positive control on a script that really was deciphered, Linear B read as Greek.

    python decipherment/linear_b.py           # about 5 minutes on a laptop CPU
    python decipherment/linear_b.py --quick   # a fast check, writes to reports/quick

Exploratory, NOT pre-registered. The Milestone 4 positive control enciphers Sanskrit with the claim's own table, so its
"true" table is one this project made. Here the table is the accepted Mycenaean reading of Linear B, and the target is
Homeric Greek. The harness is used unchanged: Linear B consonant series and Greek letters are both written in the harness's
16-letter alphabet (table in the report). Nothing fetched is committed.

Sources, fetched at run time into data/raw/linear_b/ (gitignored) and checked against a SHA-256 recorded on first fetch:
  linear_b-greek.cog   github.com/j-luo93/NeuroDecipher @ 480bad24 (no licence stated); only the Linear B column is used
  Iliad, Odyssey      github.com/PerseusDL/canonical-greekLit @ dacb0490 (CC BY-SA 4.0), Greek text of the verse lines
"""
import argparse
import json
import re
import sys
import time
import unicodedata

import numpy as np

import data
import harness
from data import skeleton
from harness import Corpus, corrupt, test_key

RAW = data.RAW.parent / "linear_b"
NEURO = "https://raw.githubusercontent.com/j-luo93/NeuroDecipher/480bad2487820e3737fecfdd108214efa769e34b/data/linear_b-greek.cog"
PERSEUS = "https://raw.githubusercontent.com/PerseusDL/canonical-greekLit/dacb04909e43dc47738828d33cc3d2b0635bc54c/data/tlg0012/{0}/tlg0012.{0}.perseus-grc2.xml"
FILES = {"linear_b-greek.cog": NEURO, "iliad.xml": PERSEUS.format("tlg001"), "odyssey.xml": PERSEUS.format("tlg002")}
SERIES = {"d": "d", "j": "y", "k": "k", "m": "m", "n": "n", "p": "p", "q": "c", "r": "r", "s": "s", "t": "t", "w": "v", "z": "j"}
GREEK = {"δ": "d", "κ": "k", "γ": "k", "χ": "k", "μ": "m", "ν": "n", "π": "p", "β": "p", "φ": "p", "ρ": "r", "λ": "r",
         "σ": "s", "ς": "s", "τ": "t", "θ": "t", "ζ": "j", "ξ": "ks", "ψ": "ps"}
IN_REPORT = [("d", "d", "δ"), ("j", "y", "-"), ("k", "k", "κ γ χ"), ("m", "m", "μ"), ("n", "n", "ν"), ("p", "p", "π β φ"),
             ("q", "c", "-"), ("r", "r", "ρ λ"), ("s", "s", "σ ς"), ("t", "t", "τ θ"), ("w", "v", "-"), ("z", "j", "ζ")]


# ---------- fetching and parsing ----------

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
    return recorded


def greek_letters(word):
    """Greek word -> harness letters: accents and breathings removed, vowels dropped, consonants mapped by GREEK."""
    d = unicodedata.normalize("NFD", word.lower())
    return "".join(GREEK.get(c, "") for c in d if not unicodedata.combining(c))


def syllable_letters(ch):
    """Linear B character -> (syllable such as 'da', harness letters): the consonant letters of the syllable via SERIES."""
    name = unicodedata.name(ch, "")
    if not name.startswith("LINEAR B SYLLABLE"):
        return None
    syl = name.split()[-1].lower()
    return syl, "".join(SERIES[c] for c in syl if c.isalpha() and c in SERIES)


def parse_cog(text):
    """Distinct Linear B words (column 1 of the cognate file; the Greek column is never read) made only of syllabograms."""
    words = {l.split("\t")[0] for l in text.splitlines()[1:] if "\t" in l}
    return sorted(w for w in words if len(w) >= 2 and all(syllable_letters(c) for c in w))


def verse_words(xml):
    """Greek words of every verse line <l> of a TEI file, tags removed, as harness-letter strings (empty ones dropped)."""
    out = []
    for body in re.findall(r"<l\b[^>]*>(.*?)</l>", xml, flags=re.S):
        out += [greek_letters(w) for w in re.sub(r"<[^>]+>", " ", body).split()]
    return [w for w in out if w]


# ---------- report ----------

def pct(x):
    return f"{100 * x:.0f}%"


def write_report(res, out):
    s, c, cs = res["settings"], res["corpus"], res["settings"]
    t = res["true_key"]
    L = ["# Positive control on a deciphered script: Linear B as Greek (exploratory, not pre-registered)", "",
         "The Milestone 4 positive control enciphers Sanskrit with the claim's own table, so the true table is one this project made. "
         "Linear B was deciphered independently (Ventris and Chadwick, 1952) and the Mycenaean reading is accepted, so it is a table "
         "nobody here chose. The same test is run on it, unchanged: a table is scored for how much its output looks like the target "
         "language, against 10,000 shuffled versions of itself. If the test is any good, the true table should stand out.", "",
         "## Data", "",
         f"- Linear B words: `linear_b-greek.cog` from NeuroDecipher (github.com/j-luo93/NeuroDecipher, commit 480bad24; no licence stated; only column 1 is read, not the Greek cognates, which hold the answers). "
         f"{c['rows']} rows, {c['texts']} distinct words of two or more syllabograms kept, {c['tokens']:,} sign tokens, {c['signs']} distinct signs. One text is one word.",
         f"- Greek target: the Iliad and Odyssey from PerseusDL canonical-greekLit (commit dacb0490, CC BY-SA 4.0), {c['greek_words']:,} words, one word per target line.",
         f"- Control language for the scores: Sanskrit (DCS training split), as in the Indus test.",
         "- Fetched at run time, never committed; this report holds aggregates only. The better source, DAMOS tablet lines (CC BY-NC-SA 4.0), has no bulk export.", "",
         "## How Linear B and Greek are put in one alphabet", "",
         "The harness scores strings over 16 consonant letters. A syllabogram's value is the consonant series of its transliteration (vowels dropped, so `pte` is `pt`, "
         "`a` is empty); Greek is reduced the same way (accents, breathings and vowels dropped; ξ is ks, ψ is ps).", "",
         "| Series | Harness letter | Greek letters |", "|---|---|---|"] + [f"| {a} | {b} | {g} |" for a, b, g in IN_REPORT] + [
         "", "Losses that work against the true table: the j, q and w series have no Greek counterpart (labiovelars and digamma were lost by Homer's time), "
         "Linear B writes almost no syllable-final consonants, Homeric Greek is centuries later than the tablets, and vowels are discarded.", "",
         "## Results", "",
         f"**1. The true table against shuffles** ({s['draws']:,} draws; the primary score is order beyond letter frequency, in bits per phoneme). "
         f"Observed {t['observed']['sequence']:+.3f}.", "", "| Null | Null mean (sd) | Percentile | p |", "|---|---|---|---|"]
    for n, d in t["nulls"].items():
        x = d["sequence"]
        L.append(f"| {n.replace('_', ' ')} | {x['mean']:+.3f} ({x['sd']:.3f}) | {x['percentile']:.1f} | {x['p']:.4f} |")
    L += ["", "Other scorers, value-shuffle p: " + ", ".join(f"{sc.replace('_', ' ')} {t['nulls']['value_shuffle'][sc]['p']:.4f}" for sc in harness.SCORERS) + ".", "",
          f"**2. How wrong can a table be and still pass.** A share of the true table's values replaced by other values; {s['corrupt_keys']} corrupted tables per row, {s['corrupt_draws']:,} draws. Share with p < 0.05 on the primary score:", "",
          "| Values replaced | Value-shuffle null | Binned null |", "|---|---|---|"]
    for share, r in res["corruption"].items():
        L.append(f"| {pct(float(share))} | {pct(r['value_shuffle'])} | {pct(r['binned_10'])} |")
    ind = res["indus_size"]
    L += ["", f"**3. Size.** This corpus has {c['tokens']:,} sign tokens in {c['texts']} texts. The Indus test scored {ind['tokens']:,} tokens in {ind['texts']:,} texts. "
          "This corpus is the smaller one, so a pass here is the stronger result, and a fail could reflect size alone.", "",
          f"**4. Wrong language.** The same table scored with Sanskrit as the target and Greek as the control: observed {res['wrong_language']['observed']['sequence']:+.3f}, "
          f"percentile {res['wrong_language']['nulls']['binned_10']['sequence']['percentile']:.1f} against the binned null, p {res['wrong_language']['nulls']['binned_10']['sequence']['p']:.4f}.", "",
          "## Reading", "", res["reading"], "",
          "## Limits", "",
          "A pass shows the method detects a real decipherment despite the losses above; a fail does not show the method is useless, only that it cannot see this table "
          "on a word list of this size under this reduction. Words are not running text, the target is later Greek, and consonant-only scoring discards what a syllabary records. "
          "Not pre-registered and outside the corrected family of Milestone 4 tests.", "",
          f"Settings: seed {s['seed']}, runtime {s['runtime_min']:.1f} min, quick run: {s['quick']}.", ""]
    (out / "linear_b_report.md").write_text("\n".join(L))


def reading(res):
    t = res["true_key"]["nulls"]
    ok = {n: t[n]["sequence"]["p"] < 0.05 for n in t}
    cor = res["corruption"]
    w = res["wrong_language"]["nulls"]["binned_10"]["sequence"]["p"]
    first = (f"The true Linear B table scores at percentile {t['binned_10']['sequence']['percentile']:.1f} of frequency-matched shuffles "
             f"(p {t['binned_10']['sequence']['p']:.4f}) and percentile {t['value_shuffle']['sequence']['percentile']:.1f} of fully shuffled tables "
             f"(p {t['value_shuffle']['sequence']['p']:.4f}): ")
    first += "the test detects it under both nulls." if all(ok.values()) else ("the test detects it under " + " and ".join(n.replace("_", " ") for n, v in ok.items() if v) + " only." if any(ok.values()) else
             "the test does not detect it under either null. On this evidence the method cannot see a correct table in a word list of this size.")
    second = " With 10% of values replaced the detection rate is " + pct(cor["0.1"]["binned_10"]) + " (binned null), and with 30% replaced " + pct(cor["0.3"]["binned_10"]) + "."
    third = " Scored against the wrong language the table " + ("is not significant, as it should be." if w >= 0.05 else "is still significant, which is an open problem for the method.")
    close = "" if w >= 0.05 and all(ok.values()) else (" Taken together this is at best a partial pass: the true table is detected only against frequency-matched shuffles" if ok["binned_10"] and not ok["value_shuffle"] else " Taken together this control does not pass cleanly") + (", and a table that is right for Greek also scores significantly for Sanskrit, so a significant result from this test is not evidence that a table is right for the language it was scored against." if w < 0.05 else ".")
    return first + second + third + close


def selfcheck():
    assert greek_letters("ξένος") == "ksns" and greek_letters("ἄνθρωπος") == "ntrps" and greek_letters("βασιλεύς") == "psrs"
    ch = lambda n: chr(next(c for c in range(0x10000, 0x10100) if unicodedata.name(chr(c), "").endswith(f" {n}")))
    got = {n: syllable_letters(ch(n))[1] for n in ("DA", "QE", "WO", "PTE", "A", "RO2", "JA", "ZO")}
    assert got == {"DA": "d", "QE": "c", "WO": "v", "PTE": "pt", "A": "", "RO2": "r", "JA": "y", "ZO": "j"}, got
    for v in set(got.values()) | set(GREEK.values()):
        assert skeleton(v) == v, v
    text = "linear_b\tgreek\n" + ch("DA") + ch("RO2") + "\tx\n" + ch("A") + "\ty\n" + ch("DA") + ch("RO2") + "\tz\n" + ch("PA") + ch("TE") + "\tw\n"
    assert parse_cog(text) == sorted([ch("DA") + ch("RO2"), ch("PA") + ch("TE")])  # distinct, one-sign word dropped
    assert verse_words('<l n="1">ἄνθρωπος <q>ξένος</q></l>') == ["ntrps", "ksns"]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    start = time.time()
    selfcheck()
    draws, c_keys, c_draws = (300, 3, 100) if args.quick else (10000, 20, 1000)
    seed = 20261116
    rng = np.random.default_rng(seed)
    fetch()
    rows = (RAW / "linear_b-greek.cog").read_text()
    words = parse_cog(rows)
    sign_id = {c: i + 1 for i, c in enumerate(sorted({c for w in words for c in w}))}
    key = {i: syllable_letters(c) for c, i in sign_id.items()}  # {sign: (syllable, letters)}: each sign its own shuffle unit
    texts = [tuple(sign_id[c] for c in w) for w in words]
    greek = verse_words((RAW / "iliad.xml").read_text()) + verse_words((RAW / "odyssey.xml").read_text())
    sanskrit = [skeleton(l) for l in data.dcs_lines("train")]
    lm_g, lm_s = harness.lm_table(greek), harness.lm_table(sanskrit)
    tables = ((lm_g, harness.lm_table(greek, order=1)), (lm_g, lm_s), harness.nb_tables(greek, sanskrit))
    swapped = ((lm_s, harness.lm_table(sanskrit, order=1)), (lm_s, lm_g), harness.nb_tables(sanskrit, greek))
    cp = Corpus(texts, key)
    res = {"corpus": {"rows": len(rows.splitlines()) - 1, "texts": len(texts), "tokens": sum(map(len, texts)), "signs": len(sign_id), "greek_words": len(greek)}}
    print("true table ...", flush=True)
    res["true_key"], _ = test_key(cp, key, tables, draws, rng, bin_counts=(10,))
    res["corruption"] = {}
    for share in (0.1, 0.2, 0.3, 0.5):
        print(f"corrupted {share} ...", flush=True)
        ps = {"value_shuffle": [], "binned_10": []}
        for _ in range(c_keys):
            r, _ = test_key(cp, corrupt(key, share, rng), tables, c_draws, rng, bin_counts=(10,))
            for n in ps:
                ps[n].append(r["nulls"][n]["sequence"]["p"])
        res["corruption"][str(share)] = {n: float(np.mean(np.array(v) < 0.05)) for n, v in ps.items()}
    print("wrong language ...", flush=True)
    res["wrong_language"], _ = test_key(cp, key, swapped, draws, rng, bin_counts=(10,))
    ind = json.loads((harness.OUT / "results.json").read_text())["claims"]["yajnadevam-2026-07-15"]
    res["indus_size"] = {"texts": ind["texts_scored"], "tokens": ind["tokens_scored"]}
    res["reading"] = reading(res)
    res["settings"] = {"draws": draws, "corrupt_keys": c_keys, "corrupt_draws": c_draws, "seed": seed, "quick": args.quick,
                       "pre_registered": False, "runtime_min": (time.time() - start) / 60}
    out = data.ROOT / "reports/quick" if args.quick else harness.OUT
    out.mkdir(parents=True, exist_ok=True)
    (out / "linear_b.json").write_text(json.dumps(res, indent=1))
    write_report(res, out)
    print(res["reading"])
    print(f"done in {(time.time() - start) / 60:.1f} min -> {out / 'linear_b_report.md'}")
