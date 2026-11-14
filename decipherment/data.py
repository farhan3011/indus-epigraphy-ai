"""Milestone 4 inputs: claimed sign keys, the Sanskrit target corpus and the Old Tamil control corpus.

    python decipherment/data.py      # fetch everything into data/raw/decipherment/ (gitignored), verify, self-check

Nothing fetched here is committed: the Yajnadevam files have no licence and inscriptions.csv contains corpus texts.
Single files are pinned by commit and SHA-256. DCS is pinned by commit (SHA-256 recorded over the extracted text);
Project Madurai pages are not versioned, so their SHA-256 is recorded on first fetch and checked afterwards.
"""
import csv
import difflib
import hashlib
import json
import re
import subprocess
import sys
import tempfile
import urllib.parse
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data/raw/decipherment"
LIPI = "https://raw.githubusercontent.com/yajnadevam/lipi/{}/src/assets/data/{}"
PARAGORTH = "https://raw.githubusercontent.com/paragorth/Indus-/{}/data/derived/{}"
FILES = {  # name -> (url, sha256)
    "xlits_2024-11-30.csv": (LIPI.format("eeeea5915ea3ed59fd37fc3a0f4e35ecc55aa6c5", "xlits.csv"),
                             "792740502e5ba85edde41d1ce4887216046a9b2870d1dc5c7cc32b5d47e6ddad"),
    "xlits_2026-07-15.csv": (LIPI.format("74f4ca39b65043ee7574da22e9433489fe7806f2", "xlits.csv"),
                             "9beba97345ded4ccc17aa5a184f29473f24bcac3f8db713b7e6696ad5ce03825"),
    "inscriptions.csv": (LIPI.format("6360880397b321699aed07cf75164ba0ac01ae06", "inscriptions.csv"),
                         "b8dcb325a2e2cd3cde12ac12e7a418b0348e45713c9ee28aad81b645d2ac5ee4"),
    "fuls2024.json": (PARAGORTH.format("b2df930e87d2c5c7eaf1e1744d56d00dd2b7bff9", "fuls2024-le-indus-key.json"), None),
    "shaw2026.json": (PARAGORTH.format("e957fc6401f6b48e6c4ab66253e8cad3db38ef01", "shaw2026-phonetic-shadows.json"), None),
}
DCS_COMMIT = "8aeed5a1343d0e48b64eb32af8c00e8c6eb29359"
DCS_TEXTS = ["Arthaśāstra", "Manusmṛti", "Amarakośa", "Hitopadeśa", "Kumārasaṃbhava"]  # post-Vedic, as the claim says
MADURAI = {"ainkurunuru": "pmuni0028", "purananuru": "pmuni0057", "kuruntokai": "pmuni0110", "akananuru": "pmuni0229",
           "natrinai": "pmuni0296"}  # Sangam anthologies (Old Tamil); redistribution allowed with the header kept
VOWELS = set("aAiIuUeEoOfFxX")  # SLP1 vowels (Tamil short e/o mapped onto e/o), removed for the consonant skeleton


# ---------- fetching ----------

def sha256(b):
    return hashlib.sha256(b).hexdigest()


def get(url):
    # curl rather than urllib: urllib timed out on projectmadurai.org from this network while curl did not
    curl = lambda u: subprocess.run(["curl", "-sSLf", "-m", "300", "--retry", "3", "-A", "Mozilla/5.0 (compatible; indus-epigraphy-student-project)", u],
                                    check=True, capture_output=True).stdout
    try:
        return curl(url)
    except subprocess.CalledProcessError:
        # some networks reset raw.githubusercontent.com; jsDelivr serves the same commit, and every file is checked against its SHA-256
        m = re.fullmatch(r"https://raw\.githubusercontent\.com/([^/]+)/([^/]+)/([^/]+)/(.+)", url)
        if not m:
            raise
        return curl("https://cdn.jsdelivr.net/gh/{}/{}@{}/{}".format(*m.groups()))


def dcs_from_git():
    """The same lines as the per-file download, from a sparse checkout of the pinned commit (for networks that
    reset raw.githubusercontent.com; the repository is too large for the jsDelivr mirror)."""
    with tempfile.TemporaryDirectory() as tmp:
        git = lambda *a: subprocess.run(["git", "-C", tmp, "-c", "core.quotepath=off", *a], check=True, capture_output=True)
        git("init", "-q")
        git("remote", "add", "origin", "https://github.com/OliverHellwig/sanskrit")
        git("sparse-checkout", "set", *(f"dcs/data/conllu/files/{t}" for t in DCS_TEXTS))
        git("fetch", "-q", "--depth", "1", "--filter=blob:none", "origin", DCS_COMMIT)
        git("checkout", "-q", "FETCH_HEAD")
        return [f"{text}\t{f.name}\t{l[9:].strip()}" for text in DCS_TEXTS
                for f in sorted(Path(tmp, "dcs/data/conllu/files", text).iterdir(), key=lambda f: f.name)
                for l in f.read_text().splitlines() if l.startswith("# text = ")]


def fetch():
    RAW.mkdir(parents=True, exist_ok=True)
    recorded = json.loads((RAW / "sha256.json").read_text()) if (RAW / "sha256.json").exists() else {}
    for name, (url, digest) in FILES.items():
        path = RAW / name
        if not path.exists():
            path.write_bytes(get(url))
        got = sha256(path.read_bytes())
        assert got == (digest or recorded.setdefault(name, got)), f"{name}: SHA-256 {got} differs from the pinned file"
    dcs = RAW / "dcs.txt"
    if not dcs.exists():
        try:
            lines = []
            for text in DCS_TEXTS:
                api = f"https://api.github.com/repos/OliverHellwig/sanskrit/contents/dcs/data/conllu/files/{urllib.parse.quote(text)}?ref={DCS_COMMIT}"
                for f in sorted(json.loads(get(api)), key=lambda f: f["name"]):
                    raw = get(f"https://raw.githubusercontent.com/OliverHellwig/sanskrit/{DCS_COMMIT}/{urllib.parse.quote(f['path'])}").decode()
                    lines += [f"{text}\t{f['name']}\t{l[9:].strip()}" for l in raw.splitlines() if l.startswith("# text = ")]
        except subprocess.CalledProcessError:
            lines = dcs_from_git()
        dcs.write_text("\n".join(lines))
    assert sha256(dcs.read_bytes()) == recorded.setdefault("dcs.txt", sha256(dcs.read_bytes())), "dcs.txt changed"
    for name, page in MADURAI.items():
        path = RAW / f"madurai_{name}.html"
        if not path.exists():
            path.write_bytes(get(f"https://www.projectmadurai.org/pm_etexts/utf8/{page}.html"))
        got = sha256(path.read_bytes())
        assert got == recorded.setdefault(path.name, got), f"{path.name} changed since it was first fetched"
    (RAW / "sha256.json").write_text(json.dumps(recorded, indent=1))
    return recorded


# ---------- transliteration ----------

IAST = [("ṭh", "W"), ("ḍh", "Q"), ("kh", "K"), ("gh", "G"), ("ch", "C"), ("jh", "J"), ("th", "T"), ("dh", "D"), ("ph", "P"),
        ("bh", "B"), ("ai", "E"), ("au", "O"), ("ā", "A"), ("ī", "I"), ("ū", "U"), ("ṝ", "F"), ("ṛ", "f"), ("ḹ", "X"), ("ḷ", "x"),
        ("ṃ", "M"), ("ṁ", "M"), ("ḥ", "H"), ("ṅ", "N"), ("ñ", "Y"), ("ṭ", "w"), ("ḍ", "q"), ("ṇ", "R"), ("ś", "S"), ("ṣ", "z")]
SLP1 = set("aAiIuUfFxXeEoOMHkKgGNcCjJYwWqQRtTdDnpPbBmyrlvSzsh")


def iast_to_slp1(s):
    """IAST (as in DCS) -> SLP1, one character per sound; anything else becomes a space."""
    s = s.lower()
    for a, b in IAST:
        s = s.replace(a, b)
    return re.sub(r"\s+", " ", "".join(c if c in SLP1 else " " for c in s)).strip()


TA_VOWELS = dict(zip("அஆஇஈஉஊஎஏஐஒஓஔ", ["a", "A", "i", "I", "u", "U", "e", "E", "E", "o", "O", "O"]))
TA_SIGNS = dict(zip("ாிீுூெேைொோௌ", ["A", "i", "I", "u", "U", "e", "E", "E", "o", "O", "O"]))
# Tamil-only consonants get pre-registered extra symbols: ழ -> 1, ற -> 2, ன -> 3 (outside SLP1, never in a key's output)
TA_CONS = dict(zip("கஙசஞடணதநபமயரலவழளறனஜஷஸஹஶ", "kNcYwRtnpmyrlv1L23jzshS"))
VIRAMA = "்"


def tamil_to_slp1(s):
    """Tamil script -> SLP1-like symbols: consonant + virama is a bare consonant, a bare consonant carries a."""
    out, chars = [], list(s)
    for i, c in enumerate(chars):
        nxt = chars[i + 1] if i + 1 < len(chars) else ""
        if c in TA_CONS:
            out.append(TA_CONS[c] + ("" if nxt == VIRAMA or nxt in TA_SIGNS else "a"))
        elif c in TA_SIGNS:
            out.append(TA_SIGNS[c])
        elif c in TA_VOWELS:
            out.append(TA_VOWELS[c])
        elif c == "ஃ":
            out.append("H")
        elif c != VIRAMA:
            out.append(" ")
    return re.sub(r"\s+", " ", "".join(out)).strip()


# The Yajnadevam key writes only 16 plain consonants (no aspirates, retroflexes, sibilant or nasal distinctions);
# the claimant picks those per text. So every string is reduced to the key's distinctions before scoring.
COLLAPSE = str.maketrans({"K": "k", "G": "g", "C": "c", "J": "j", "w": "t", "W": "t", "T": "t", "q": "d", "Q": "d", "D": "d",
                          "P": "p", "B": "b", "N": "n", "Y": "n", "R": "n", "S": "s", "z": "s", "L": "l",
                          "1": "l", "2": "r", "3": "n", "M": None, "H": None})


def skeleton(s):
    """Collapsed consonant skeleton: vowels, spaces, punctuation, anusvara and visarga removed, consonants reduced to the
    key's 16 distinctions. Every corpus, key output and control is compared in this form."""
    return "".join(c for c in s.translate(COLLAPSE) if c in CONSONANTS)


CONSONANTS = set("kgcjtdnpbmyrlvsh")


# ---------- loaders ----------

def norm_id(s):
    return int(str(s).strip())


def load_key(name):
    """Yajnadevam key -> {sign: (canonical head, value)}; break (000) and space (999) rows and blank values left out."""
    out = {}
    for r in csv.DictReader(open(RAW / name)):
        if r["sign"] not in ("000", "999") and r["xlit"].strip():
            out[norm_id(r["sign"])] = (r["canonical"].strip(), r["xlit"].strip())  # a canonical head can be a compound like 031-001-031
    return out


def load_paragorth(name):
    """Fuls / Shaw key -> {sign: value}; for 'ka/kan' style alternatives the first is kept."""
    d = json.loads((RAW / name).read_text())
    d = d.get("key", d)
    return {norm_id(k): v.split("/")[0].strip() for k, v in d.items() if isinstance(v, str) and k.strip().isdigit()}


def load_inscriptions():
    """Rows of the readings file: sign tuple in reading order (the stored order reversed), reading or None, is_ref."""
    out = []
    for r in csv.DictReader(open(RAW / "inscriptions.csv")):
        signs = tuple(norm_id(x) for x in re.findall(r"\d+", r["text"]))
        reading = (r["sanskrit"] or "").strip()
        out.append({"id": r["id"], "signs": signs[::-1], "reading": None if not reading or reading.startswith("ref:") else reading,
                    "is_ref": reading.startswith("ref:")})
    return out


def dcs_lines(split):
    """SLP1 sentences from DCS; about 10% of chapters (by a stable hash of the file name) are held out."""
    out = []
    for line in (RAW / "dcs.txt").read_text().splitlines():
        _, chapter, text = line.split("\t")
        held = int(hashlib.md5(chapter.encode()).hexdigest(), 16) % 10 == 0
        if held == (split == "heldout"):
            out.append(iast_to_slp1(text))
    return [l for l in out if l]


def control_lines():
    """Old Tamil Sangam verse lines from Project Madurai, transliterated."""
    out = []
    for name in MADURAI:
        page = (RAW / f"madurai_{name}.html").read_text(errors="ignore")
        text = re.sub(r"<[^>]+>", "\n", page)
        out += [tamil_to_slp1(l) for l in text.splitlines() if len(re.findall(r"[஀-௿]", l)) >= 8]
    return [l for l in out if l]


def selfcheck():
    assert iast_to_slp1("kṛṣṇa") == "kfzRa" and iast_to_slp1("āditya") == "Aditya" and iast_to_slp1("bhūḥ") == "BUH"
    assert tamil_to_slp1("தமிழ்") == "tami1" and tamil_to_slp1("கா") == "kA" and tamil_to_slp1("அறம்") == "a2am"
    assert skeleton("aYjAna ama") == "njnm" and skeleton("jaqa-arRaH") == "jdrn" and norm_id("032") == 32


def gate_checks():
    """Facts measured during the research; failing one means the pins or the reading direction are wrong."""
    sys.path.insert(0, str(ROOT / "sequence_analysis"))
    import corpus
    old, new = load_key("xlits_2024-11-30.csv"), load_key("xlits_2026-07-15.csv")
    raw = [{r["sign"]: r["xlit"] for r in csv.DictReader(open(RAW / n))} for n in ("xlits_2024-11-30.csv", "xlits_2026-07-15.csv")]
    assert sum(raw[0][s] != raw[1].get(s) for s in raw[0]) == 50
    if corpus.WELLS_SQL.exists():
        used = {s for t in corpus.load_wells()[0] for s in t}
        assert used <= {norm_id(s) for s in raw[1]}, "the key does not cover every Wells sign"
    rev, fwd, sims = [], [], []
    for r in load_inscriptions():
        if r["reading"]:
            want = skeleton(r["reading"])
            got = skeleton("".join(new.get(s, ("", ""))[1] for s in r["signs"]))
            rev.append(got == want)
            fwd.append(skeleton("".join(new.get(s, ("", ""))[1] for s in r["signs"][::-1])) == want)
            sims.append(difflib.SequenceMatcher(None, got, want).ratio())
    share, other = sum(rev) / len(rev), sum(fwd) / len(fwd)
    # measured 2 Nov 2026: 43.5% of his readings reproduced in reading order, 7.9% in stored order
    assert 0.38 <= share <= 0.48 and share > 4 * other, (share, other)
    return {"read_texts": len(rev), "skeleton_match_reading_order": share, "skeleton_match_stored_order": other,
            "mean_similarity": sum(sims) / len(sims), "key_signs": (len(old), len(new))}

if __name__ == "__main__":
    selfcheck()
    print("sha256:", json.dumps(fetch(), indent=1))
    print("gate checks:", gate_checks())
    print(f"DCS train {len(dcs_lines('train')):,} lines, held out {len(dcs_lines('heldout')):,}; control {len(control_lines()):,} lines")
