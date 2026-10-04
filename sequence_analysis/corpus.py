"""IM77 corpus loading and the EBUDS filter (Yadav et al. 2010).

Signs are Mahadevan numbers 1-417; 0 marks an illegible or lost sign. Stored order is reading
order (first-read sign first). Never write filtered sequences into the repo: the data has no licence.

    python sequence_analysis/corpus.py     # prints the filter counts step by step
"""
import hashlib
import re
import sysconfig
import tokenize
from collections import defaultdict
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


def ebuds_rows(lines):
    """The EBUDS line filter without de-duplication, keeping each line's metadata (site, object, field symbol)."""
    return lines[~lines.damaged & ~lines.doubtful & (lines.line == "0")]


def ebuds(lines, keep_doubtful=False, keep_copies=False):
    """Yadav et al.'s filter: drop damaged or doubtful texts, drop lines of multi-line texts, keep one
    copy of each repeated text. Returns (sorted list of sign tuples, [(step, counts), ...]).
    The two flags undo one step each, for sensitivity checks; copies leak across CV folds."""
    steps = [("all inscribed lines", summarise(lines))]
    x = lines[~lines.damaged]
    steps.append(("no illegible or lost sign", summarise(x)))
    if not keep_doubtful:
        x = x[~x.doubtful]
        steps.append(("no doubtful reading", summarise(x)))
    x = x[x.line == "0"]  # line 0 = the only line on its side; 1/2/3 = a line of a multi-line text
    steps.append(("single-line only", summarise(x)))
    texts = sorted(x.signs) if keep_copies else sorted(set(x.signs))
    return texts, steps


WELLS_SQL = ROOT / "data/raw/yajnadevam/population-script.sql"
WELLS_UNREADABLE = {0, 515, 516, 546, 547, 606, 999}  # 0 = line break, the rest render as '??' or blank


def load_wells():
    """Second, independent transcription: yajnadevam/indus-website (GPL-3.0), Wells / ICIT sign numbers.

    Returns (texts, variant) with the same filter as EBUDS as far as the data allows: single-line,
    every sign readable, one copy per text. The site records no doubtful readings. GLYPHSEQUENCE is
    stored with the jar sign (740, the commonest text-ender in every corpus) first in 816 texts and last in 2,
    regardless of the recorded direction, so it is reversed here to give reading order; physical layout
    cannot be recovered from it. variant maps a sign to the head of its Wells variant group (GLYPHSIMILARITY)."""
    sql = WELLS_SQL.read_text()

    def rows(table):
        i = sql.index(f"INSERT INTO {table}")
        return re.findall(r"\(([^()]*)\)", sql[sql.index("VALUES", i):sql.index(";", i)])

    seq = defaultdict(dict)
    for r in rows("GLYPHSEQUENCE"):
        seal, glyph, idx = map(int, r.split(","))
        seq[seal][idx] = glyph
    texts = [tuple(v[i] for i in sorted(v, reverse=True)) for v in seq.values()]
    texts = sorted({t for t in texts if not WELLS_UNREADABLE & set(t)})
    head = {}
    for r in rows("GLYPHSIMILARITY"):
        a, b = map(int, r.split(","))
        head[b] = head.get(a, a)  # groups are stars: (head, member); ponytail: no union-find needed for this data
    return texts, head


REFERENCE = ROOT / "data/raw/reference"


def reference_systems():
    """Token streams from systems of known type, for comparison: {name: list of tokens}."""
    eng = (REFERENCE / "pride_and_prejudice.txt").read_text(encoding="utf-8-sig")
    eng = eng[eng.index("Chapter I.]"):eng.index("*** END OF THE PROJECT GUTENBERG")].lower()
    dna = "".join((REFERENCE / "lambda_phage.fasta").read_text().split("\n")[1:])
    code = []
    for f in sorted(Path(sysconfig.get_paths()["stdlib"]).glob("*.py"))[:40]:
        with open(f, "rb") as fh:
            code += [t.string if t.type in (tokenize.NAME, tokenize.OP) else tokenize.tok_name[t.type]
                     for t in tokenize.tokenize(fh.readline) if t.type in (tokenize.NAME, tokenize.OP, tokenize.NUMBER, tokenize.STRING)]
    out = {"English, words": re.findall(r"[a-z]+", eng), "English, letters": list(re.sub(r"[^a-z]", "", eng)),
           "DNA (phage lambda)": list(dna), "Python source, tokens": code}
    linear_b = REFERENCE / "linear_b_samples.txt"
    if linear_b.exists():  # syllabograms and logograms; the file mixes 513 real sequences with unmarked augmented copies
        out["Linear B, Knossos series D (signs)"] = [s for line in sorted(set(linear_b.read_text().split("\n"))) for s in re.split(r"[-\s]+", line) if s]
    cdli = REFERENCE / "cdliatf_unblocked.atf"
    if cdli.exists():
        sux, seals = cuneiform(cdli)
        out["Sumerian cuneiform, all texts (signs)"], out["Cuneiform seal inscriptions (signs)"] = sux, seals
    essen = sorted((REFERENCE / "essen").rglob("*.krn"))
    if essen:  # non-linguistic: pitches of folk melodies, octave and accidental kept, rhythm dropped
        notes = (re.search(r"([a-gA-G])\1*[#-]*", line.split("\t")[0]) for f in essen for line in f.read_text(errors="ignore").split("\n")
                 if line and line[0] not in "!*=" and "r" not in line.split("\t")[0])
        out["Folk melodies, Essen collection (notes)"] = [m.group() for m in notes if m]
    return out


def cuneiform(path):
    """Sign readings from CDLI ATF texts tagged Sumerian: (stream of all such texts, stream of seal texts only).
    Lines with a broken or unreadable sign are dropped; determinatives count as signs; damage marks are removed."""
    sux, seals, lang, obj = [], [], None, None
    for line in path.read_text(errors="ignore").split("\n"):
        if line.startswith("&"):
            lang = obj = None
        elif line.startswith("#atf: lang"):
            lang = line.split()[-1]
        elif line.startswith("@object"):
            obj = line.split()[-1]
        elif lang == "sux" and re.match(r"^\d+'?\.", line):
            body = re.sub(r"[#?!*\[\]<>_⸢⸣]", "", line.split(".", 1)[1]).lower()
            signs = [s for s in re.split(r"[-\s.{}+:]+", body) if s]
            if signs and not any("x" == s or "..." in s or s == "$" for s in signs):
                sux += signs
                if obj == "seal":
                    seals += signs
    return sux, seals


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
