"""IM77 corpus loading and the EBUDS filter (Yadav et al. 2010).

Signs are Mahadevan numbers 1-417; 0 marks an illegible or lost sign. Stored order is reading
order (first-read sign first). Never write filtered sequences into the repo: the data has no licence.

    python sequence_analysis/corpus.py     # prints the filter counts step by step
"""
import hashlib
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
IM77_CSV = ROOT / "data/raw/im77/im77_corpus_lines.csv"
SHA256 = "385b82bd6a9629e0ab215af46db84ca0c9ec4f6044a2a2ea58a3c64121f26211"
EBUDS_PUBLISHED = (1548, 7000, 377)  # texts, sign tokens, distinct signs


def load_lines():
    """One row per inscribed line: metadata plus `signs` (tuple of ints), `damaged`, `doubtful`."""
    if hashlib.sha256(IM77_CSV.read_bytes()).hexdigest() != SHA256:
        print(f"WARNING: {IM77_CSV.name} differs from the version this code was checked against")
    d = pd.read_csv(IM77_CSV, dtype=str, keep_default_na=False)
    d = d[d.n_signs.astype(int) > 0].reset_index(drop=True)  # 343 'unrecorded' rows are empty placeholders
    d["signs"] = d.signs_clean.map(lambda s: tuple(int(t) for t in s.split()))
    d["damaged"] = d.signs.map(lambda s: 0 in s)
    d["doubtful"] = d.signs_reading_order.str.contains("*", regex=False)  # '*' is stripped in signs_clean
    return d


def summarise(rows):
    texts = set(rows.signs)
    tokens = [s for t in texts for s in t]
    return {"rows": len(rows), "unique_texts": len(texts), "tokens": len(tokens), "signs": len(set(tokens) - {0})}


def ebuds(lines):
    """Yadav et al.'s filter: drop damaged or doubtful texts, drop lines of multi-line texts, keep one
    copy of each repeated text. Returns (sorted list of sign tuples, [(step, counts), ...])."""
    steps = [("all inscribed lines", summarise(lines))]
    x = lines[~lines.damaged]
    steps.append(("no illegible or lost sign", summarise(x)))
    x = x[~x.doubtful]
    steps.append(("no doubtful reading", summarise(x)))
    x = x[x.line == "0"]  # line 0 = the only line on its side; 1/2/3 = a line of a multi-line text
    steps.append(("single-line only", summarise(x)))
    texts = sorted(set(x.signs))
    return texts, steps


def physical(row):
    """Signs in physical left-to-right order on the impression, or None if the layout is not recorded."""
    if row.direction == "right-to-left":
        return row.signs[::-1]
    if row.direction == "left-to-right":
        return row.signs
    return None


if __name__ == "__main__":
    lines = load_lines()
    texts, steps = ebuds(lines)
    for name, c in steps:
        print(f"{name:32s} {c}")
    got = (len(texts), sum(map(len, texts)), len({s for t in texts for s in t}))
    print("EBUDS reproduced exactly" if got == EBUDS_PUBLISHED else f"EBUDS-style filter: {got} vs published {EBUDS_PUBLISHED}")
