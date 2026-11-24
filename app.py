"""Demo app for Milestones 1 to 4 and their extensions.

    streamlit run app.py
    python app.py            # self-check: asserts, then runs the whole app once headless

Runs locally only: it reads the IM77 corpus and the trained weights from this machine.
Colours live in two places that must agree: the constants below and .streamlit/config.toml.
"""
import base64
import itertools
import json
import os
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import altair as alt
import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT / "grapheme_extraction"), str(ROOT / "sequence_analysis"), str(ROOT / "decipherment")]

import corpus  # noqa: E402
import masked_model  # noqa: E402
import ngram  # noqa: E402
import restore  # noqa: E402
from classifier import crop_sign, load_classifier  # noqa: E402
from pipeline import extract, iou, levenshtein, load_detector  # noqa: E402
from synthetic_generator import GLYPH_DIR, degradation, load_glyphs, load_sequences, make_seal  # noqa: E402

M2 = ROOT / "reports/milestone_2"
M3 = ROOT / "reports/milestone_3"
M4 = ROOT / "reports/milestone_4"
PUBLISHED_PERPLEXITY = {1: 68.82, 2: 26.69, 3: 26.09, 4: 25.26, 5: 25.26}  # Yadav et al. 2010
M1_SIGN_ERROR_RATE = 0.237  # Milestone 1, YOLOv8n, synthetic test seals
DETECTORS = {"YOLOv8n": ROOT / "runs/yolov8n/weights/best.pt", "RT-DETR-L": ROOT / "runs/rtdetr-l/weights/best.pt",
             "OpenCV baseline (no learning)": "opencv"}
# Seeds chosen to show the real spread of results (exact match on synthetic seals is about 43%), not only successes.
PRESETS = {"A · clean read, 7 signs": 38, "B · clean read, 5 signs": 33, "C · one wrong sign of 8": 6,
           "D · three wrong signs of 5": 19, "E · spurious box, sequence fails": 8}
REAL = {"Impression": ("met_impression.jpg", False), "Carved seal (mirror image)": ("met_seal.jpg", True),
        "Both, as photographed": ("met_cropped.jpg", False)}

INK, MUTED, LINE, ACCENT, BLUE, GOOD, BAD, AMBER = "#ece3d3", "#a39886", "#342e28", "#c9663c", "#4a9bd1", "#7fc48a", "#ec6a5a", "#e0b060"
CSS = """<style>
:root { --ink:#ece3d3; --muted:#a39886; --bg:#12100e; --card:#1b1815; --raised:#24201c; --line:#342e28;
        --accent:#c9663c; --good:#7fc48a; --bad:#ec6a5a; --amber:#e0b060;
        --serif:"Iowan Old Style","Palatino Linotype",Palatino,Georgia,serif; --mono:"SF Mono",ui-monospace,Menlo,monospace; }
.block-container { padding-top: 2.2rem; max-width: 1180px; }
header[data-testid="stHeader"] { background: transparent; }
.masthead { border-bottom: 1px solid var(--line); padding-bottom: 1.1rem; margin-bottom: .6rem; }
.masthead .eyebrow { font: 600 .72rem/1 var(--mono); letter-spacing: .16em; text-transform: uppercase; color: var(--accent); }
.masthead h1 { font: 500 2.5rem/1.15 var(--serif); margin: .5rem 0 .35rem; padding: 0; color: var(--ink); letter-spacing: -.01em; }
.masthead p { color: var(--muted); margin: 0; max-width: 62ch; }
h3.sec { font: 500 1.35rem/1.2 var(--serif); margin: 1.6rem 0 .7rem; padding: 0; color: var(--ink); }
p.lede { color: var(--muted); max-width: 70ch; margin: .2rem 0 1rem; }
.label { font: 600 .7rem/1 var(--mono); letter-spacing: .12em; text-transform: uppercase; color: var(--muted); margin: 1.1rem 0 .5rem; }
button[data-baseweb="tab"] p { font-size: .98rem; }
[data-testid="stImage"] img { border-radius: 12px; }

.strip { display: flex; flex-wrap: wrap; align-items: center; gap: 6px; }
.tile { width: 70px; background: var(--card); border: 1px solid var(--line); border-radius: 10px; padding: 8px 6px 6px; text-align: center; }
.tile img { width: 44px; height: 44px; object-fit: contain; display: block; margin: 0 auto 4px; }
.tile span { font: 500 .74rem/1 var(--mono); color: var(--muted); }
.tile.ok { border-color: var(--good); } .tile.ok span { color: var(--good); }
.tile.bad { border-color: var(--bad); } .tile.bad span { color: var(--bad); }
.tile.new { border-color: var(--accent); } .tile.new span { color: var(--accent); }
.tile small { display: block; font: 500 .64rem/1.3 var(--mono); color: var(--muted); }
.tile.gap { border-style: dashed; border-color: var(--accent); color: var(--accent); font: 500 1.9rem/66px var(--serif); padding: 0; height: 76px; }
.strip .dir { font: 500 .72rem/1.3 var(--mono); color: var(--muted); margin-left: 8px; }

.stats { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 10px; margin: .9rem 0 .4rem; }
.stat { background: var(--card); border: 1px solid var(--line); border-radius: 12px; padding: 14px 16px; }
.stat .k { font-size: .8rem; color: var(--muted); }
.stat .v { font: 500 1.9rem/1.2 var(--serif); color: var(--ink); margin-top: 4px; }
.stat .v small { font: 400 .85rem/1 system-ui, sans-serif; color: var(--muted); margin-left: 4px; }

.grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(250px, 1fr)); gap: 12px; }
.grid.five { grid-template-columns: repeat(5, minmax(0, 1fr)); }
.card { background: var(--card); border: 1px solid var(--line); border-radius: 12px; padding: 14px; }
.card.ok { border-color: var(--good); }
.card .head { display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px; }
.card .n { font: 500 .78rem/1 var(--mono); color: var(--muted); }
.chip { font-size: .72rem; padding: 3px 9px; white-space: nowrap; border-radius: 99px; border: 1px solid var(--line); color: var(--muted); }
.chip.ok { color: var(--good); border-color: var(--good); } .chip.bad { color: var(--bad); border-color: var(--bad); }
.chip.near { color: var(--amber); border-color: var(--amber); }
.pair { display: flex; gap: 10px; margin-bottom: 10px; }
.pair figure { margin: 0; flex: 1; text-align: center; }
.pair img { width: 76px; height: 76px; object-fit: contain; border-radius: 8px; background: var(--raised); padding: 4px; }
.pair figcaption { font-size: .7rem; color: var(--muted); margin-top: 4px; }
.guess { text-align: center; }
.guess img { width: 64px; height: 64px; object-fit: contain; margin: 4px auto 8px; display: block; }
.guess b { font: 500 .85rem/1 var(--mono); color: var(--ink); display: block; margin-bottom: 8px; }
.guess small { color: var(--muted); font-size: .76rem; display: block; margin-top: 6px; }
.bars { display: grid; grid-template-columns: max-content 1fr max-content; gap: 6px 10px; align-items: center; font-size: .8rem; }
.bars .name { font-family: var(--mono); color: var(--muted); } .bars .val { color: var(--muted); text-align: right; font-variant-numeric: tabular-nums; }
.bars .track { height: 8px; background: var(--raised); border-radius: 4px; overflow: hidden; }
.bars .fill { height: 100%; background: var(--accent); border-radius: 4px; }
.bars .true { color: var(--ink); font-weight: 600; }
.bar1 { height: 6px; background: var(--raised); border-radius: 3px; overflow: hidden; } .bar1 div { height: 100%; background: var(--accent); }
.note { border-left: 3px solid var(--amber); background: var(--card); border-radius: 0 10px 10px 0; padding: 12px 16px; color: var(--ink); margin-bottom: 12px; font-size: .92rem; }
.note.good { border-color: var(--good); }
.hero { display: flex; gap: 18px; align-items: center; }
.hero img { width: 110px; height: 110px; object-fit: contain; background: var(--card); border: 1px solid var(--line); border-radius: 14px; padding: 12px; }
.hero .v { font: 500 2.2rem/1.1 var(--serif); } .hero .k { color: var(--muted); font-size: .85rem; }
table.rep { width: 100%; border-collapse: collapse; font-size: .9rem; }
table.rep th { text-align: left; font: 600 .7rem/1 var(--mono); letter-spacing: .1em; text-transform: uppercase; color: var(--muted); padding: 8px 10px; border-bottom: 1px solid var(--line); }
table.rep td { padding: 9px 10px; border-bottom: 1px solid var(--line); } table.rep td.num { font-family: var(--mono); font-size: .84rem; }
</style>"""


# ---------- shared ----------

def uri(img):
    return "data:image/png;base64," + base64.b64encode(cv2.imencode(".png", img)[1]).decode()


@st.cache_data
def glyph_uri(sign):
    """Reference drawing of a Mahadevan sign in the ink colour on a transparent background."""
    mask = cv2.imread(str(GLYPH_DIR / f"{sign}.png"), cv2.IMREAD_GRAYSCALE)
    x, y, w, h = cv2.boundingRect(cv2.findNonZero((mask > 127).astype(np.uint8)))  # the drawings sit small in a 150 px canvas
    side = max(w, h, 60) + 12  # floor keeps relative size: short strokes stay shorter than long ones
    mask = cv2.copyMakeBorder(mask[y:y + h, x:x + w], (side - h) // 2, (side - h + 1) // 2, (side - w) // 2, (side - w + 1) // 2, cv2.BORDER_CONSTANT)
    return uri(np.dstack([np.full_like(mask, c) for c in (0xD3, 0xE3, 0xEC)] + [mask]))  # BGR of INK, mask as alpha


def tiles(signs, hidden=None, marks=None, notes=None):
    """Signs given in reading order, laid out as on an impression: first-read sign on the right.
    hidden: one position or a set of them shown as gaps; marks: a CSS class per tile; notes: a small line under each sign number."""
    hidden = {hidden} if isinstance(hidden, int) else set(hidden or ())
    cells = ['<div class="tile gap">?</div>' if i in hidden else
             f'<div class="tile {marks[i] if marks else ""}"><img src="{glyph_uri(s)}" alt=""><span>{s}</span>'
             f'{f"<small>{notes[i]}</small>" if notes and notes[i] else ""}</div>'
             for i, s in enumerate(signs)]
    return '<div class="strip">' + "".join(cells[::-1]) + '<div class="dir">&larr; read from<br>this end</div></div>'


def stats(*items):
    """items: (label, value) or (label, value, small suffix)."""
    return '<div class="stats">' + "".join(
        f'<div class="stat"><div class="k">{k}</div><div class="v">{v}<small>{rest[0] if rest else ""}</small></div></div>'
        for k, v, *rest in items) + "</div>"


def bars(rows):
    """rows: (name, fraction 0-1, value text, emphasised)."""
    return '<div class="bars">' + "".join(
        f'<div class="name {"true" if em else ""}">{name}</div><div class="track"><div class="fill" style="width:{100 * frac:.1f}%"></div></div>'
        f'<div class="val {"true" if em else ""}">{text}</div>' for name, frac, text, em in rows) + "</div>"


def themed(chart, height=260):
    return (chart.properties(height=height, background="transparent").configure_view(stroke=None)
            .configure_axis(labelColor=MUTED, titleColor=MUTED, gridColor=LINE, domainColor=LINE, tickColor=LINE,
                            labelFontSize=12, titleFontSize=12, titleFontWeight="normal", titlePadding=10)
            .configure_legend(labelColor=INK, title=None, orient="top", labelFontSize=12, symbolType="circle"))


def section(title, lede=None):
    st.html(f'<h3 class="sec">{title}</h3>' + (f'<p class="lede">{lede}</p>' if lede else ""))


# ---------- Milestone 1 ----------

@st.cache_resource
def detector(name, conf):
    return load_detector(DETECTORS[name], conf)


@st.cache_resource
def classifier():
    return load_classifier()


@st.cache_resource
def generator_inputs():
    return load_glyphs(), load_sequences()


@st.cache_resource
def label_font(size=20):
    for path, index in (("/System/Library/Fonts/Helvetica.ttc", 1), ("C:/Windows/Fonts/arialbd.ttf", 0),
                        ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 0)):
        try:
            return ImageFont.truetype(path, size, index=index)
        except OSError:
            pass
    return ImageFont.load_default(size)


def synthetic(seed):
    """Deterministic synthetic seal: (BGR image, truth [(sign, x0, y0, x1, y1)])."""
    random.seed(seed)
    np.random.seed(seed)
    degrade = degradation()
    degrade.set_random_seed(seed)
    return make_seal(*generator_inputs(), degrade)


def match(truth, signs):
    """Per predicted sign: the true sign ID it overlaps (IoU >= 0.5, one-to-one), or None."""
    pred = [(x, y, x + w, y + h) for x, y, w, h in (s["bbox_xywh"] for s in signs)]
    hit = [None] * len(pred)
    for g in truth:
        free = [p for p in range(len(pred)) if hit[p] is None]
        best = max(free, key=lambda p: iou(g[1:], pred[p]), default=None)
        if best is not None and iou(g[1:], pred[best]) >= 0.5:
            hit[best] = g[0]
    return hit


def draw(img, signs, hit=None):
    """Boxes and top-1 sign numbers. With ground truth: green = right sign, red = wrong sign or not a sign."""
    k = max(1.0, 900 / img.shape[1])
    im = Image.fromarray(cv2.cvtColor(cv2.resize(img, None, fx=k, fy=k, interpolation=cv2.INTER_CUBIC), cv2.COLOR_BGR2RGB))
    d, font = ImageDraw.Draw(im), label_font()
    for i, s in enumerate(signs):
        x, y, w, h = (round(v * k) for v in s["bbox_xywh"])
        top1 = s["top_k"][0][0]
        colour = AMBER if hit is None else GOOD if hit[i] == top1 else BAD
        d.rounded_rectangle((x, y, x + w, y + h), radius=6, outline=colour, width=3)
        ty = max(0, y - 27)
        d.rounded_rectangle((x, ty, x + d.textlength(str(top1), font=font) + 12, ty + 25), radius=6, fill=colour)
        d.text((x + 6, ty + 2), str(top1), font=font, fill="#12100e")
    return im


def sign_cards(view, signs, hit):
    gray, cards = cv2.cvtColor(view, cv2.COLOR_BGR2GRAY), []
    for i, s in enumerate(signs[:24]):
        x, y, w, h = s["bbox_xywh"]
        top5, chip = [sid for sid, _ in s["top_k"]], ""
        if hit:
            kind, text = (("bad", "not a sign") if hit[i] is None else ("ok", "correct") if hit[i] == top5[0] else
                          ("near", f"true sign {hit[i]} is in the top 5") if hit[i] in top5 else ("bad", f"missed · true sign {hit[i]}"))
            chip = f'<span class="chip {kind}">{text}</span>'
        cards.append(
            f'<div class="card"><div class="head"><span class="n">SIGN {i + 1}</span>{chip}</div><div class="pair">'
            f'<figure><img src="{uri(crop_sign(gray, (x, y, x + w, y + h)))}" alt=""><figcaption>from the image</figcaption></figure>'
            f'<figure><img src="{glyph_uri(top5[0])}" alt=""><figcaption>best match · {top5[0]}</figcaption></figure></div>'
            + bars([(sid, p, f"{p:.0%}", bool(hit) and sid == hit[i]) for sid, p in s["top_k"]]) + "</div>")
    return '<div class="grid">' + "".join(cards) + "</div>"


def milestone1():
    left, right = st.columns([5, 9], gap="large")
    with left:
        st.html('<p class="lede">A photograph goes in. A detector boxes each sign, a classifier names it with its '
                'Mahadevan number (1 to 417), and the signs are listed in reading order, right to left.</p>')
        source = st.segmented_control("Image", ["Synthetic", "Real seal", "Upload"], default="Synthetic") or "Synthetic"
        truth, mirror = None, False
        if source == "Synthetic":
            choice = st.selectbox("Example", [*PRESETS, "Random"])
            if choice == "Random":
                if st.button("New random seal") or "seed" not in st.session_state:
                    st.session_state.seed = random.SystemRandom().randrange(10 ** 6)
                seed = st.session_state.seed
            else:
                seed = PRESETS[choice]
            img, truth = synthetic(seed)
            st.caption("Sign drawings engraved on generated stone, then blurred and distorted. The answer is known, so every box can be checked.")
        elif source == "Real seal":
            name, mirror = REAL[st.selectbox("Photo", list(REAL))]
            img = cv2.imread(str(ROOT / "data/samples" / name))
            st.caption("Stamp seal with unicorn, Indus, about 2600 to 1900 BC. The Metropolitan Museum of Art, 49.40.1, public domain.")
        else:
            up = st.file_uploader("JPEG or PNG", type=["jpg", "jpeg", "png"])
            img = cv2.imdecode(np.frombuffer(up.getvalue(), np.uint8), cv2.IMREAD_COLOR) if up else None
            if up and img is None:
                st.error("That file could not be read as an image.")
        if truth is None:
            mirror = st.toggle("Photo shows the carved seal, so mirror it first", value=mirror)
        with st.expander("Detector settings"):
            name = st.selectbox("Detector", [n for n, w in DETECTORS.items() if w == "opencv" or (w.exists() and w.stat().st_size)])  # an interrupted training run leaves an empty file
            conf = st.slider("Confidence threshold", 0.05, 0.9, 0.25, 0.05)
    if img is None:
        right.html('<div class="note">Upload a photograph of a seal or an impression to run the pipeline on it.</div>')
        return
    out = extract(img, detector(name, conf), classifier(), is_seal_matrix=mirror)
    view = cv2.flip(img, 1) if mirror else img  # boxes are in impression view
    hit = match(truth, out["signs"]) if truth else None
    with right:
        st.image(draw(view, out["signs"], hit), width="stretch")
        if truth:
            gold = [g[0] for g in sorted(truth, key=lambda g: -(g[1] + g[3]) / 2)]
            right_signs = [h == s["top_k"][0][0] for h, s in zip(hit, out["signs"])]
            st.html(stats(("Signs found", sum(h is not None for h in hit), f"of {len(gold)}"),
                          ("Named correctly", sum(right_signs), f"of {len(gold)}"),
                          ("Edits to fix the sequence", levenshtein(gold, out["sequence"])))
                    + '<div class="label">True sequence</div>' + tiles(gold)
                    + '<div class="label">Pipeline output</div>' + tiles(out["sequence"], marks=["ok" if r else "bad" for r in right_signs]))
        elif out["sequence"]:
            st.html('<div class="label">Pipeline output</div>' + tiles(out["sequence"]))

    if out["signs"]:
        section("Sign by sign", "Each crop next to the reference drawing it matched best, with the classifier's five closest signs.")
        st.html(sign_cards(view, out["signs"], hit))
        if len(out["signs"]) > 24:
            st.caption(f"{len(out['signs']) - 24} more boxes not shown.")
        with st.expander("Raw JSON output"):
            st.json(out, expanded=False)


# ---------- Milestone 2 ----------

@st.cache_resource
def sequence_model():
    """Bigram model trained on four fifths of the corpus; the demo texts are the held-out fifth,
    so the model has never seen the inscription it is asked about."""
    texts, _ = corpus.ebuds(corpus.load_lines())
    vocab = sorted({s for t in texts for s in t})
    train, test = next(ngram.folds(texts))
    return ngram.WittenBell(train, vocab, 2, backoff=True), vocab, [t for t in test if len(t) >= 4]


def bits_per_sign(model, text):
    ll, n = ngram.log2_likelihood(model, [text])
    return -ll / n


def candidates(model, vocab, text, j):
    """Probability of every sign at position j given its two neighbours: P(w | prev) * P(next | w), normalised."""
    s = [ngram.BOS, *text, ngram.EOS]
    score = np.array([model.prob(w, (s[j],)) * model.prob(s[j + 2], (w,)) for w in vocab])
    return score / score.sum()


def reorderings(text, n=500, seed=0):
    """Distinct reorderings of a text other than itself: all of them if short, else a random sample."""
    if len(text) <= 6:
        return sorted(set(itertools.permutations(text)) - {tuple(text)})
    rng, out = random.Random(seed), set()
    while len(out) < n:
        out.add(tuple(rng.sample(text, len(text))))
    return sorted(out - {tuple(text)})


def new_text(n):
    st.session_state.text = random.SystemRandom().randrange(n)


def guess_tab(model, vocab, text, res):
    section("Guess the missing sign", f"One sign is hidden from an inscription the model was not trained on. Using only the two "
            f"neighbours, it ranks all {len(vocab)} signs for the gap.")
    a, b = st.columns([3, 1], vertical_alignment="bottom")
    j = a.slider("Hide sign number (1 is read first, on the right)", 1, len(text), min(2, len(text))) - 1
    b.button("Another inscription", key="next_guess", on_click=new_text, args=(st.session_state.n_texts,), width="stretch")
    reveal = st.toggle("Reveal the answer", key=f"reveal_{st.session_state.text}_{j}")
    st.html(tiles(text, hidden=None if reveal else j, marks=["ok" if reveal and i == j else "" for i in range(len(text))]))
    p = candidates(model, vocab, text, j)
    order = np.argsort(-p)
    st.html('<div class="label">The model\'s five best guesses</div><div class="grid five">' + "".join(
        f'<div class="card guess {"ok" if reveal and vocab[k] == text[j] else ""}"><img src="{glyph_uri(vocab[k])}" alt=""><b>{vocab[k]}</b>'
        f'<div class="bar1"><div style="width:{100 * p[k]:.1f}%"></div></div><small>{p[k]:.0%}</small></div>' for k in order[:5]) + "</div>")
    if reveal:
        rank = int(np.where(order == vocab.index(text[j]))[0][0]) + 1
        st.html(f'<div class="note {"good" if rank <= 5 else ""}">The hidden sign is <b>{text[j]}</b>. '
                f'The model ranked it {rank} of {len(vocab)}.</div>')
    r = res["restoration"]
    st.html(stats(("First guess right", f"{r['top1']:.1%}"), ("Answer in the top five", f"{r['top5']:.1%}"),
                  ("Random guessing", f"{1 / len(vocab):.1%}")))
    st.caption(f"Measured over every position of every text: {r['trials']:,} trials, 5-fold cross-validation.")


def shuffle_tab(model, text, res):
    section("Real order against every other order", "If sign order carried no structure, a real inscription would be no easier to "
            "predict than the same signs rearranged. Fewer bits per sign means more predictable.")
    a, b = st.columns([3, 1], vertical_alignment="center")
    a.html(tiles(text))
    b.button("Another inscription", key="next_shuffle", on_click=new_text, args=(st.session_state.n_texts,), width="stretch")
    real = bits_per_sign(model, text)
    others = np.array([bits_per_sign(model, t) for t in reorderings(text)])
    if not len(others):
        st.html('<div class="note">Every sign in this text is the same, so it has no other ordering.</div>')
        return
    st.html(stats(("Real order", f"{real:.2f}", "bits per sign"), ("Typical reordering", f"{np.median(others):.2f}", "bits per sign"),
                  ("Reorderings it beats", f"{(others > real).mean():.0%}", f"of {len(others)}")))
    hist = alt.Chart(pd.DataFrame({"bits": others})).mark_bar(color=MUTED, cornerRadiusEnd=3).encode(
        alt.X("bits:Q", bin=alt.Bin(maxbins=28), title="Bits per sign"), alt.Y("count():Q", title="Reorderings"),
        tooltip=[alt.Tooltip("bits:Q", bin=alt.Bin(maxbins=28), title="Bits per sign"), alt.Tooltip("count():Q", title="Reorderings")])
    mark = pd.DataFrame({"bits": [real], "label": [f"real order · {real:.2f}"]})
    rule = alt.Chart(mark).mark_rule(color=ACCENT, strokeWidth=3).encode(x="bits:Q", tooltip=[alt.Tooltip("bits:Q", format=".2f", title="Real order")])
    label = alt.Chart(mark).mark_text(align="left", dx=8, color=INK, fontSize=12).encode(x="bits:Q", y=alt.value(10), text="label:N")
    st.altair_chart(themed(hist + rule + label), theme=None, width="stretch")
    s = res["structure_test"]
    st.caption(f"One short text can lose to some of its reorderings; the claim is about the corpus. Across all texts, neighbouring signs "
               f"share {s['observed_bits']:.2f} bits of information, against {s['shuffled_mean_bits']:.2f} when signs are shuffled within "
               f"each text (p = {s['p']:.3f}).")


def explore_tab():
    pos = pd.read_csv(M2 / "sign_positions.csv").set_index("sign")
    pairs = pd.read_csv(M2 / "sign_pairs.csv")
    section("Sign explorer", "Where a sign sits inside inscriptions, and which signs keep it company.")
    sign = st.selectbox("Sign, most frequent first", list(pos.index), index=list(pos.index).index(342),
                        format_func=lambda s: f"{s}  ·  {pos.loc[s, 'count']} occurrences")
    row = pos.loc[sign]
    a, b = st.columns([1, 2], gap="large", vertical_alignment="center")
    a.html(f'<div class="hero"><img src="{glyph_uri(sign)}" alt=""><div><div class="k">Mahadevan sign</div><div class="v">{sign}</div>'
           f'<div class="k">{int(row["count"])} occurrences</div></div></div>')
    b.html('<div class="label">Position in the text</div>' + bars(
        [(name, n / row["count"], f"{int(n)} · {n / row['count']:.0%}", n == max(row.as_text_beginner, row.medial, row.as_text_ender))
         for name, n in (("read first", row.as_text_beginner), ("middle", row.medial), ("read last", row.as_text_ender))]))
    for title, mine, other in (("Most often followed by", "first_sign", "second_sign"), ("Most often preceded by", "second_sign", "first_sign")):
        top = pairs[pairs[mine] == sign].nlargest(5, "count")
        st.html(f'<div class="label">{title}</div>' + ('<div class="grid five">' + "".join(
            f'<div class="card guess"><img src="{glyph_uri(getattr(r, other))}" alt=""><b>{getattr(r, other)}</b>'
            f'<small>{r.count} times<br>{r.count / r.expected_if_independent:.1f}&times; chance</small></div>' for r in top.itertuples())
            + "</div>" if len(top) else '<p class="lede">No recorded neighbour.</p>'))
    st.caption("\"× chance\" compares the count with what the two signs' frequencies alone would give.")


def results_tab(res):
    rep, lm, r = res["replication"], res["language_models"], res["restoration"]
    bigram = lm["witten-bell backoff n=2"]
    section("Headline results", "The corpus and most published statistics from Yadav et al. 2010 are reproduced. Text endings use far "
            "fewer signs than text beginnings, which is consistent with writing in one direction, right to left.")
    st.html(stats(("Texts", "1,548"), ("Sign tokens", "7,000"), ("Distinct signs", "377"),
                  ("Signs covering 80% of endings", rep["coverage80"]["enders"], f"vs {rep['coverage80']['beginners']} for beginnings")))
    a, b = st.columns(2, gap="large")
    with a:
        st.html('<div class="label">Perplexity by model order</div>')
        ours, paper = "This project", "Yadav et al. 2010"
        d = pd.DataFrame([{"order": n, "series": ours, "perplexity": lm[f"witten-bell backoff n={n}"]["skip_unseen_perplexity"]} for n in range(1, 6)]
                         + [{"order": n, "series": paper, "perplexity": v} for n, v in PUBLISHED_PERPLEXITY.items()])
        line = alt.Chart(d).mark_line(strokeWidth=2, point=alt.OverlayMarkDef(size=70, filled=True)).encode(
            alt.X("order:O", title="Signs of context + 1", axis=alt.Axis(labelAngle=0)), alt.Y("perplexity:Q", title="Perplexity"),
            alt.Color("series:N", scale=alt.Scale(domain=[ours, paper], range=[ACCENT, BLUE])),
            tooltip=["series", "order", alt.Tooltip("perplexity:Q", format=".2f")])
        st.altair_chart(themed(line, 280), theme=None, width="stretch")
        st.caption("Scored the way the paper scores it, skipping signs unseen in training. One sign of context does most of the work.")
    with b:
        st.html('<div class="label">What recognition errors would do</div>')
        d = pd.DataFrame([{"error": e["error_rate"], "perplexity": e["bigram_perplexity"]} for e in res["error_propagation"]])
        line = alt.Chart(d).mark_line(color=ACCENT, strokeWidth=2, point=alt.OverlayMarkDef(size=70, filled=True, color=ACCENT)).encode(
            alt.X("error:Q", title="Share of signs misread", axis=alt.Axis(format="%")), alt.Y("perplexity:Q", title="Bigram perplexity"),
            tooltip=[alt.Tooltip("error:Q", format=".0%", title="Signs misread"), alt.Tooltip("perplexity:Q", format=".1f")])
        mark = pd.DataFrame({"error": [M1_SIGN_ERROR_RATE], "label": ["Milestone 1 today"]})
        rule = alt.Chart(mark).mark_rule(color=MUTED, strokeDash=[4, 4]).encode(x="error:Q")
        label = alt.Chart(mark).mark_text(align="right", dx=-8, color=INK, fontSize=12).encode(x="error:Q", y=alt.value(10), text="label:N")
        st.altair_chart(themed(line + rule + label, 280), theme=None, width="stretch")
        st.caption("Corpus signs deliberately corrupted at each rate. Milestone 1's error rate is from synthetic seals.")
    st.html('<div class="label">Replication of Yadav et al. 2010</div><table class="rep"><tr><th>Measure</th><th>Published</th><th>This project</th></tr>' + "".join(
        f'<tr><td>{m}</td><td class="num">{p}</td><td class="num">{o}</td></tr>' for m, p, o in (
            ("Texts / tokens / distinct signs", "1,548 / 7,000 / 377", "1,548 / 7,000 / 377"),
            ("Sign entropy, bits", "6.68", f"{rep['unigram_entropy']:.2f}"),
            ("Commonest pair", "267 then 99, 168 times", f"{rep['top_pair'][0]} then {rep['top_pair'][1]}, {rep['top_pair'][2]} times"),
            ("Signs covering 80% of beginnings / endings", "82 / 23", f"{rep['coverage80']['beginners']} / {rep['coverage80']['enders']}"),
            ("Bigram perplexity", "26.69", f"{bigram['skip_unseen_perplexity']:.2f} same scoring · {bigram['cv_perplexity']:.2f} scoring every sign"),
            ("Information shared by neighbours, bits", "2.24", f"{rep['bigram_mi']:.2f} · not reproduced"),
            ("Hidden sign inside the 90% candidate set", "74%", f"{r['hit_in_90pct_mass_set']:.1%}"))) + "</table>")
    with st.expander("Figures from the written report"):
        for name in ("zipf.png", "direction_null.png", "transition_heatmap.png"):
            if (M2 / name).exists():
                st.image(str(M2 / name))
    st.caption("Method and every number: reports/milestone_2/milestone_2_report.md")


def milestone2():
    model, vocab, held_out = sequence_model()
    res = json.loads((M2 / "results.json").read_text())
    st.session_state.n_texts = len(held_out)
    if "text" not in st.session_state:
        new_text(len(held_out))
    text = held_out[st.session_state.text]
    st.html('<p class="lede">Statistics on the published sign sequences of the Mahadevan 1977 concordance. They show that sign order '
            'is structured and not random. They say nothing about what the signs mean.</p>')
    guess, shuffle, explore, results = st.tabs(["Guess the missing sign", "Real order vs shuffled", "Sign explorer", "Headline results"])
    with guess:
        guess_tab(model, vocab, text, res)
    with shuffle:
        shuffle_tab(model, text, res)
    with explore:
        explore_tab()
    with results:
        results_tab(res)


# ---------- Milestone 3 ----------

@st.cache_resource
def gap_models():
    """The neural masked-sign model and the best n-gram (Kneser-Ney trigram), both trained on every EBUDS text, the
    corpus texts with where each was found, and the damaged corpus texts."""
    model, tokens, temperature = masked_model.load()
    lines = corpus.load_lines()
    texts, _ = corpus.ebuds(lines)
    vocab = sorted({s for t in texts for s in t})
    trigram = masked_model.best_ngram(texts, vocab)
    sites = defaultdict(Counter)
    for row in corpus.ebuds_rows(lines).itertuples():
        sites[row.signs][row.site] += 1
    return {"model": model, "tokens": tokens, "temperature": temperature, "trigram": trigram, "scorer": restore.Scorer(trigram, vocab),
            "vocab": vocab, "texts": texts, "sites": sites, "damaged": masked_model.damaged_texts(lines, vocab)}


def guess_cards(top):
    """top: [(sign, probability)] best first."""
    return '<div class="grid five">' + "".join(
        f'<div class="card guess"><img src="{glyph_uri(s)}" alt=""><b>{s}</b><div class="bar1"><div style="width:{100 * p:.1f}%"></div></div>'
        f'<small>{p:.0%}</small></div>' for s, p in top) + "</div>"


def set90(probs):
    p = np.sort(probs)[::-1]
    return int(np.argmax(np.cumsum(p) >= 0.9) + 1)


def new_damaged(n):
    st.session_state.damaged = random.SystemRandom().randrange(n)


def parse_signs(typed, vocab, gaps=True):
    """Typed sign numbers -> (list with None for ? and restore.STAR for *, tokens not recognised)."""
    out, bad = [], []
    for t in typed.split():
        if gaps and t == "?":
            out.append(None)
        elif gaps and t == restore.STAR:
            out.append(restore.STAR)
        elif t.isdigit() and int(t) in vocab:
            out.append(int(t))
        else:
            bad.append(t)
    return out, bad


def where_found(sites, text):
    c = sites.get(tuple(text))
    return ", ".join(f"{site} ×{n}" if n > 1 else site for site, n in c.most_common(3)) if c else ""


def damaged_tab(g):
    damaged, vocab = g["damaged"], g["vocab"]
    if "damaged" not in st.session_state:
        new_damaged(len(damaged))
    text = damaged[st.session_state.damaged]
    gap = text.index(0)
    gapped = [None if s == 0 else s for s in text]
    section("A damaged inscription", f"One of the {len(damaged)} single-line corpus texts with exactly one illegible sign.")
    a, b = st.columns([3, 1], vertical_alignment="center")
    a.html(tiles(text, hidden=gap))
    b.button("Another inscription", key="next_damaged", on_click=new_damaged, args=(len(damaged),), width="stretch")
    net = masked_model.fill(g["model"], g["tokens"], g["temperature"], gapped, k=len(vocab))[gap]
    bi = ngram.candidate_scores(g["trigram"], vocab, [text], [(0, gap)])[0]
    bi = bi / bi.sum()
    order = np.argsort(-bi)
    left, right = st.columns(2, gap="large")
    left.html('<div class="label">Neural model: reads the whole inscription</div>' + guess_cards(net[:5])
              + f'<p class="lede">Its 90% probability set holds {set90(np.array([p for _, p in net]))} signs.</p>')
    right.html('<div class="label">Best n-gram: reads the neighbouring signs</div>' + guess_cards([(vocab[k], bi[k]) for k in order[:5]])
               + f'<p class="lede">Its 90% probability set holds {set90(bi)} signs.</p>')

    top, bits = masked_model.influence(g["model"], g["tokens"], g["temperature"], gapped, gap)
    section("Which signs the neural model leaned on", f"Each legible sign is blanked in turn. The number is how many bits the model's first "
            f"guess ({top}) loses without that sign; the sign it leans on most is outlined.")
    lean = max(bits, key=bits.get)
    st.html(tiles(text, hidden=gap, marks=["new" if i == lean and bits[i] > 0 else "" for i in range(len(text))],
                  notes=[f"{bits[i]:+.1f} bits" if i in bits else "" for i in range(len(text))]))

    hits, fillers = restore.parallels(g["texts"], gapped, gap, k=4)
    section("Parallel texts", "Complete corpus texts that match the signs around the gap, the start and end of a text included. "
            "The outlined sign stands where this inscription is damaged.")
    if fillers:
        n = sum(fillers.values())
        st.html(f'<div class="label">Between the same neighbours, in {n} corpus text{"s" if n > 1 else ""}</div>'
                + bars([(f"sign {s}", c / n, f"{c} of {n}", k == 0) for k, (s, c) in enumerate(fillers.most_common(5))]))
    if not hits:
        st.html('<div class="note">No corpus text shares a neighbouring sign with this gap.</div>')
    for left_n, right_n, t, j in hits:
        st.html(tiles(t, marks=["new" if i == j else "" for i in range(len(t))])
                + f'<p class="lede">{left_n + right_n} matching signs or text ends around the gap'
                + (f' · found at {where_found(g["sites"], t)}' if where_found(g["sites"], t) else "") + "</p>")


def restore_tab(g):
    vocab = set(g["vocab"])
    section("Restore your own", "Type signs in reading order (first-read sign first). Use ? for one missing sign and * for a damaged "
            f"stretch of unknown length (0 to {restore.MAX_RUN} signs).")
    typed = st.text_input("Inscription", "267 99 ? ? 342", label_visibility="collapsed", key="restore_text")
    seq, bad = parse_signs(typed, vocab)
    singles, stars = seq.count(None), seq.count(restore.STAR)
    longest = masked_model.MAX_LEN - 2 - restore.MAX_RUN * stars
    if not seq or bad or not singles + stars or stars > 1 or singles > 4 or len(seq) > longest:
        st.html('<div class="note">Use sign numbers that occur in the corpus, at least one ? or *, at most four ? and one *, and at most '
                f'{longest} entries.' + (f' Not recognised: {", ".join(bad)}.' if bad else "") + "</div>")
        return
    r = restore.restore(g["scorer"], seq)
    st.html('<div class="label">Best n-gram, all gaps searched together</div>')
    for fill, p, filled in r["fills"]:
        a, b = st.columns([5, 1], vertical_alignment="center")
        a.html(tiles(fill, marks=["new" if i in filled else "" for i in range(len(fill))]) if fill else '<p class="lede">nothing legible</p>')
        b.html(f'<div class="stat"><div class="k">share</div><div class="v">{p:.0%}</div></div>')
    st.caption("Outlined signs are restored. Shares are of the candidates the search kept, so they run slightly high.")
    if "lengths" in r:
        st.html('<div class="label">How many signs are missing in the damaged stretch</div>'
                + bars([(f"{n} sign{'s' if n != 1 else ''}", p, f"{p:.0%}", p == max(r["lengths"].values())) for n, p in r["lengths"].items()]))
    else:
        plain = [None if x is None else x for x in seq]
        for gap, top in masked_model.fill(g["model"], g["tokens"], g["temperature"], plain).items():
            st.html(f'<div class="label">Gap {gap + 1}: neural model, each gap on its own</div>' + guess_cards(top))


def new_miscopy(g):
    rng = random.SystemRandom()
    text = list(rng.choice([t for t in g["texts"] if len(t) >= 4]))
    i = rng.randrange(len(text))
    was = text[i]
    while text[i] == was:
        text[i] = rng.choice(rng.choice(g["texts"]))  # a sign drawn by how often it occurs
    st.session_state.miscopy_text = " ".join(map(str, text))
    st.session_state.miscopy_truth = (st.session_state.miscopy_text, i, was)


def miscopy_tab(g, ext):
    section("Spot a sign that does not fit", "Every sign is scored by how expected it is given all the others. A sign far less expected "
            "than any sign of a normal text is flagged. A rare but genuine sign looks the same to the model, so a flag is a reason to "
            "look at the object again, not a correction.")
    if "miscopy_text" not in st.session_state:
        new_miscopy(g)
    a, b = st.columns([3, 1], vertical_alignment="bottom")
    typed = a.text_input("Inscription in reading order", key="miscopy_text")
    b.button("A corpus text with one sign swapped", on_click=new_miscopy, args=(g,), width="stretch")
    seq, bad = parse_signs(typed, set(g["vocab"]), gaps=False)
    if bad or not 2 <= len(seq) <= masked_model.MAX_LEN - 2:
        st.html(f'<div class="note">Use 2 to {masked_model.MAX_LEN - 2} sign numbers that occur in the corpus.'
                + (f' Not recognised: {", ".join(bad)}.' if bad else "") + "</div>")
        return
    rows = restore.surprise(g["trigram"], g["vocab"], seq)
    cut = ext["miscopy"]["threshold_at_5pct_false_alarms"] if ext else 5e-5
    worst = min(range(len(seq)), key=lambda j: rows[j]["p"])
    flagged = rows[worst]["p"] <= cut
    st.html(tiles(seq, marks=["bad" if flagged and i == worst else "" for i in range(len(seq))],
                  notes=[f"{r['p']:.1%}" if r["p"] >= 0.001 else "&lt;0.1%" for r in rows]))
    if flagged:
        st.html(f'<div class="note">Sign {seq[worst]} is flagged: the model ranks it {rows[worst]["rank"]} of {len(g["vocab"])} candidates for '
                f'this place and would expect {rows[worst]["expected"]} ({rows[worst]["expected_p"]:.0%}).</div>')
    else:
        st.html('<div class="note good">No sign is below the flag threshold (set so that 5% of unaltered held-out texts are flagged).</div>')
    truth = st.session_state.get("miscopy_truth")
    if truth and truth[0] == typed:
        st.caption(f"In this example sign {truth[1] + 1} in reading order was swapped; the corpus has {truth[2]} there. "
                   "The model was trained on the original text, so this example is easier than a new find would be.")
    st.html('<table class="rep"><tr><th>Sign</th><th>Probability given the others</th><th>Rank</th><th>Model would expect</th></tr>' + "".join(
        f'<tr><td class="num">{r["sign"]}</td><td class="num">{r["p"]:.2%}</td><td class="num">{r["rank"]}</td>'
        f'<td class="num">{r["expected"]} ({r["expected_p"]:.0%})</td></tr>' for r in rows) + "</table>")


def m3_results_tab(res, ext):
    r = res["models"]
    section("One hidden sign", "Measured the same way as Milestone 2: every sign of every held-out text hidden in turn, 5-fold.")
    st.html(stats(("Best n-gram, first guess right", f"{r['kn_trigram']['top1']:.1%}"),
                  ("Neural model alone", f"{r['net_calibrated']['top1']:.1%}", f"± {100 * r['net_calibrated']['top1_sd']:.1f}"),
                  ("Neural + n-gram mixture", f"{r['mixture']['top1']:.1%}", f"± {100 * r['mixture']['top1_sd']:.1f}"),
                  ("Mixture calibration error", f"{r['mixture']['ece_top1']:.3f}", "0 is perfect")))
    st.caption(res["verdict"] + " Full report: reports/milestone_3/milestone_3_report.md")
    if not ext:
        st.html('<div class="note">No extension results yet. Run <code>python sequence_analysis/restore.py</code>.</div>')
        return
    two, ul, mc = ext["two_gaps"]["all"], ext["unknown_length"], ext["miscopy"]
    section("Two hidden signs", f"Both signs right, on {two['trials']:,} held-out trials.")
    items = [("Joint search (n-gram)", f"{two['joint']['both_right']:.1%}"), ("Greedy search (n-gram)", f"{two['greedy']['both_right']:.1%}")]
    if "neural" in two:
        items.append(("Neural, each gap on its own", f"{two['neural']['both_right']:.1%}"))
    items.append(("True pair among 5 best (joint)", f"{two['joint']['pair_in_top5']:.1%}"))
    st.html(stats(*items))
    section("Damage of unknown length and miscopied signs")
    st.html(stats(("Length of the damage right", f"{ul['length_right']:.1%}", f"baseline {ul['always_most_common_length']:.0%}"),
                  ("Stretch among 5 best fills", f"{ul['fill_in_top5']:.1%}"),
                  ("Swapped sign located", f"{mc['position_found']:.1%}", f"chance {mc['position_by_chance']:.0%}"),
                  ("Corrupted vs clean text", f"AUC {mc['text_auc']:.2f}", "0.5 is chance")))
    st.caption("Full report: reports/milestone_3/restoration_extensions_report.md")


def milestone3():
    if not masked_model.MODEL_FILE.exists():
        st.html('<div class="note">No trained model yet. Run <code>python sequence_analysis/masked_model.py</code> first.</div>')
        return
    g = gap_models()
    res = json.loads((M3 / "results.json").read_text())
    ext = json.loads((M3 / "restoration_extensions.json").read_text()) if (M3 / "restoration_extensions.json").exists() else None
    st.html('<p class="lede">Some corpus inscriptions have a sign that can no longer be read. Two models suggest what it was: the '
            'best Milestone 2 n-gram, which looks at the neighbouring signs, and a small neural model that reads the whole inscription. '
            'Nobody knows the right answers for these texts, so the suggestions are candidates with a confidence, not readings.</p>')
    damaged, own, miscopy, results = st.tabs(["A damaged inscription", "Restore your own", "Spot a miscopied sign", "How good is it"])
    with damaged:
        damaged_tab(g)
    with own:
        restore_tab(g)
    with miscopy:
        miscopy_tab(g, ext)
    with results:
        m3_results_tab(res, ext)


# ---------- Milestone 4 ----------

@st.cache_resource
def decipher_tools():
    """The current Yajnadevam key, the scoring tables and the Wells texts the key can read (local data only)."""
    import data as dec_data
    import harness
    key = dec_data.load_key("xlits_2026-07-15.csv")
    target = [dec_data.skeleton(l) for l in dec_data.dcs_lines("train")]
    control = [dec_data.skeleton(l) for l in dec_data.control_lines()]
    lm = harness.lm_table(target)
    tables = ((lm, harness.lm_table(target, order=1)), (lm, harness.lm_table(control)), harness.nb_tables(target, control))
    wells = corpus.load_wells()[0]
    readable = [t for t in wells if all(s in key for s in t)]
    texts = [t for t in readable if len(t) >= 3]
    return harness, dec_data, key, tables, texts, wells, harness.Corpus(readable, key)


@st.cache_resource
def lexicon_tools():
    """The dictionary match-rate module and its Sanskrit lexicon (exploratory extension), or None before lexicon.py has run."""
    if not (M4 / "lexicon.json").exists():
        return None
    import data as dec_data
    import lexicon
    return lexicon, lexicon.build_lexicon(dec_data.dcs_lines("train"))


def test_table(harness, dec_data, tables, wells, table, draws=300, seed=0, lex=None):
    """The Milestone 4 test on any table {sign: value}: None if it reads too little of the corpus, else the test's results.
    lex: (lexicon module, lexicon) to add the dictionary match-rate scores."""
    key = {s: (str(s), v) for s, v in table.items()}  # a value with no consonant (a bare vowel) is read as nothing, as in the harness
    texts = [t for t in wells if all(s in key for s in t)]
    if sum(bool(dec_data.skeleton(v)) for v in table.values()) < 20 or len(texts) < 30:
        return None
    r, _ = harness.test_key(harness.Corpus(texts, key), key, tables, draws, np.random.default_rng(seed))
    if lex:
        r["lexicon"] = lex[0].test(texts, key, draws, np.random.default_rng(seed), lex[1])
    return r | {"texts": len(texts), "signs": len(key), "draws": draws}


def search_chart(r, label):
    """Median search trajectories of one column of the overfitting study, with the table's own score as a rule."""
    h = r["held_out"]
    rows = [{"swaps": a, "score": b, "series": name} for name, traj in (
        ("Searched on all texts", r["median_trajectory"]), ("Searched on half: that half", h["median_trajectory_fit"]),
        ("Searched on half: the unseen half", h["median_trajectory_other"])) for a, b in traj]
    lines = alt.Chart(pd.DataFrame(rows)).mark_line(strokeWidth=2).encode(
        alt.X("swaps:Q", title="Swaps tried"), alt.Y("score:Q", title="Sanskrit sequence score, bits per phoneme", scale=alt.Scale(zero=False)),
        alt.Color("series:N", scale=alt.Scale(range=[BLUE, MUTED, AMBER])), tooltip=["series", "swaps", alt.Tooltip("score:Q", format="+.3f")])
    mark = pd.DataFrame({"score": [r["table"]], "label": [label]})
    rule = alt.Chart(mark).mark_rule(color=ACCENT, strokeWidth=3).encode(y="score:Q")
    text = alt.Chart(mark).mark_text(align="right", dy=-8, color=INK, fontSize=12).encode(x=alt.value(420), y="score:Q", text="label:N")
    return themed(lines + rule + text, 300)


def overfitting_section():
    if not (M4 / "overfitting.json").exists():
        st.html('<div class="note">No search results yet. Run <code>python decipherment/overfit.py</code>.</div>')
        return
    o = json.loads((M4 / "overfitting.json").read_text())
    i, c = o["indus"], o["cipher"]
    section("What a blind search can reach", "Exploratory, not part of the pre-registered test. A decipherer searches for a table that "
            "reads well, so random tables are searched too: two signs swap values whenever the score does not fall. The search knows "
            "nothing but the score. The control on the right is real Sanskrit enciphered with a known table.")
    st.html(stats(("Claimed table", f"{i['table']:+.2f}"), ("Random tables after search", f"{i['searched_tables']['mean']:+.2f}"),
                  ("Swaps until search passes the claim", f"{i['steps_until_median_search_passes_table']}"),
                  ("Two searches read alike", f"{i['agreement']['between_searched_tables']['mean']:.0%}", "of sign tokens")))
    a, b = st.columns(2, gap="large")
    a.html('<div class="label">Indus inscriptions</div>')
    a.altair_chart(search_chart(i, "claimed table"), theme=None, width="stretch")
    b.html('<div class="label">Control: enciphered Sanskrit</div>')
    b.altair_chart(search_chart(c, "true table"), theme=None, width="stretch")
    st.caption(f"The search passes the true table in the control too ({c['searched_tables']['mean']:+.2f} against {c['table']:+.2f}), with tables "
               f"that agree with it on only {c['agreement']['searched_with_table']['mean']:.0%} of sign tokens. So this score cannot pick out "
               "the right table on a corpus of this size, and scoring below the search ceiling is not evidence against a table. What the "
               "search shows is how much Sanskrit-like order any table can be given by fitting alone. "
               "Full report: reports/milestone_4/overfitting_report.md")


def new_wells(n):
    st.session_state.wells = random.SystemRandom().randrange(n)


def milestone4():
    if not (M4 / "results.json").exists():
        st.html('<div class="note">No results yet. Run <code>python decipherment/harness.py</code> first.</div>')
        return
    res = json.loads((M4 / "results.json").read_text())
    claim = res["claims"]["yajnadevam-2026-07-15"]
    st.html('<p class="lede">A claimed decipherment is a table from signs to sounds. The test applies the table to the '
            'inscriptions, measures how Sanskrit-like the order of the result is, and compares that with 10,000 tables made by '
            'shuffling the claim\'s own sounds among its signs. The rules were committed before any claim was scored. It can show '
            'that a table carries no signal beyond chance; it cannot show whether the script records a language.</p>')
    pc, nc = res["positive_control"], res["negative_control"]
    p = claim["nulls"]["binned_10"]["sequence"]
    st.html(stats(("Known tables detected (positive control)", f"{min(v['power'] for v in pc['0'].values()):.0%}"),
                  ("Random tables flagged (5% expected)", f"{nc['share_below_0.05']:.1%}"),
                  ("Claimed table beats", f"{p['percentile']:.0f}%", "of frequency-matched shuffles"),
                  ("After correction for 12 tests", f"p = {claim['holm']['sequence']['binned_10']:.2f}", "not significant" if claim['holm']['sequence']['binned_10'] >= 0.05 else "significant")))

    section("The claimed table against shuffled versions of itself", "Yajnadevam's Sanskrit key (version of 15 July 2026), "
            "scored on the inscriptions it can read. Higher means more Sanskrit-like sign order.")
    h = claim["primary_histogram"]
    centres = [(a + b) / 2 for a, b in zip(h["edges"][:-1], h["edges"][1:])]
    rows = [{"score": c, "keys": n, "null": label} for label, k in (("Any shuffle", "value_shuffle"), ("Frequency-matched shuffle", "binned_10"))
            for c, n in zip(centres, h[k])]
    bars = alt.Chart(pd.DataFrame(rows)).mark_bar(opacity=0.75, binSpacing=0).encode(
        alt.X("score:Q", title="Sanskrit sequence score, bits per phoneme", bin=alt.Bin(binned=True, step=centres[1] - centres[0])),
        alt.Y("keys:Q", title="Shuffled tables", stack=None),
        alt.Color("null:N", scale=alt.Scale(domain=["Any shuffle", "Frequency-matched shuffle"], range=[MUTED, BLUE])),
        tooltip=["null", alt.Tooltip("score:Q", format=".2f"), "keys"])
    mark = pd.DataFrame({"score": [claim["observed"]["sequence"]], "label": ["claimed table"]})
    rule = alt.Chart(mark).mark_rule(color=ACCENT, strokeWidth=3).encode(x="score:Q")
    label = alt.Chart(mark).mark_text(align="left", dx=6, color=INK, fontSize=12).encode(x="score:Q", y=alt.value(10), text="label:N")
    st.altair_chart(themed(bars + rule + label, 280), theme=None, width="stretch")
    st.caption(claim["sentence"][0].upper() + claim["sentence"][1:])

    oe, rv, sh = res["order_effect"], res["read_vs_unread"], res["shaw"]
    st.html(stats(("Real order vs shuffled order", f"z = {oe['claim_z']:.1f}", f"random tables {oe['random_keys_z_mean']:+.1f} ± {oe['random_keys_z_sd']:.1f}"),
                  ("Texts he read vs left unread", f"{rv['difference_bits_per_phoneme']:+.2f} bits", f"p = {rv['p']:.3f}"),
                  ("Shaw 2026: values follow sign numbering", f"{sh['observed']} of {sh['pairs']}", f"{sh['expected']:.0f} expected")))
    st.caption("Any table carries some of the inscriptions' own sign order into its output, so the order effect is not specific to "
               "the claim. The table fits the texts its author read better than the ones he left unread, which is a sign of fitting.")

    overfitting_section()

    section("Which claims can be tested")
    chip = {"TESTED": "ok", "NO_TARGET_LANGUAGE": "near", "NOT_RUN": "near", "NOT_TESTABLE_NO_TABLE": "bad"}
    st.html('<table class="rep"><tr><th>Claim</th><th>Status</th><th>Note</th></tr>' + "".join(
        f'<tr><td>{c}</td><td><span class="chip {chip.get(v["status"], "")}">{v["status"].replace("_", " ").lower()}</span></td>'
        f'<td>{v.get("done", v.get("reason", "no fixed sign-to-value table is published"))}</td></tr>' for c, v in res["claims_table"].items()) + "</table>")

    section("Try it", "A real inscription read with the claimed table and with three shuffled tables. Each gets the same score.")
    try:
        harness, dec_data, key, tables, texts, wells, cp_all = decipher_tools()
    except (FileNotFoundError, OSError):
        st.html('<div class="note">Needs the local data: run <code>python decipherment/data.py</code>.</div>')
        return
    if "wells" not in st.session_state:
        new_wells(len(texts))
    text = texts[st.session_state.wells]
    a, b = st.columns([3, 1], vertical_alignment="center")
    a.html('<div class="label">Signs in reading order (Wells numbers)</div><p class="lede" style="font-family:var(--mono)">'
           + " · ".join(map(str, text)) + "</p>")
    b.button("Another inscription", key="next_wells", on_click=new_wells, args=(len(texts),), width="stretch")
    rng = np.random.default_rng(st.session_state.wells)
    us = harness.units(key)
    tables_rows = [("Claimed table", {s: cv[1] for s, cv in key.items()})] + [(f"Shuffled table {i}", harness.shuffled(us, rng)) for i in (1, 2, 3)]
    cp = harness.Corpus([text], key)
    out = []
    for name, values in tables_rows:
        reading = " ".join(values[s] for s in text)
        score = cp.score(cp.codes(values), tables, per_text=True)[0][0]
        out.append(f'<tr><td>{name}</td><td class="num">{reading}</td><td class="num">{dec_data.skeleton(reading)}</td><td class="num">{score:+.2f}</td></tr>')
    st.html('<table class="rep"><tr><th>Table</th><th>Reading</th><th>Consonants scored</th><th>Score</th></tr>' + "".join(out) + "</table>")

    import overfit
    section("Run a search yourself", "Start from a random shuffle of the claimed table's values and let the search swap for 2,000 tries "
            "on all the inscriptions the table can read. It takes a second or two.")
    if st.button("Search a random table", key="run_search"):
        st.session_state.search_seed = st.session_state.get("search_seed", 0) + 1
    if st.session_state.get("search_seed"):
        vals, traj = overfit.climb(cp_all, us, (tables[0], None), np.random.default_rng(st.session_state.search_seed), 2000, every=25)
        claimed = cp_all.score(cp_all.codes({s: cv[1] for s, cv in key.items()}), tables)[0]
        df = pd.DataFrame([{"swaps": a, "score": b} for a, b, _ in traj])
        line = alt.Chart(df).mark_line(color=BLUE, strokeWidth=2).encode(
            alt.X("swaps:Q", title="Swaps tried"), alt.Y("score:Q", title="Sanskrit sequence score, bits per phoneme", scale=alt.Scale(zero=False)))
        rule = alt.Chart(pd.DataFrame({"score": [claimed]})).mark_rule(color=ACCENT, strokeWidth=3).encode(y="score:Q")
        st.altair_chart(themed(line + rule, 260), theme=None, width="stretch")
        passed = next((a for a, b, _ in traj if b >= claimed), None)
        found = harness.assign(us, vals)
        one = harness.Corpus([text], key)
        mine = " ".join(found[s] for s in text)
        st.html(stats(("Started at", f"{traj[0][1]:+.2f}"), ("Reached", f"{traj[-1][1]:+.2f}"), ("Claimed table (orange line)", f"{claimed:+.2f}"),
                      ("Passed the claim after", f"{passed} swaps" if passed is not None else "not yet")))
        st.html('<table class="rep"><tr><th>Table</th><th>Reading of the inscription above</th><th>Consonants scored</th><th>Score</th></tr>'
                f'<tr><td>Searched table</td><td class="num">{mine}</td><td class="num">{dec_data.skeleton(mine)}</td>'
                f'<td class="num">{one.score(one.codes(found), tables, per_text=True)[0][0]:+.2f}</td></tr></table>')
        st.caption("The searched table scores higher than the claim and reads the inscriptions as something else entirely.")

    section("Test your own table", "Upload a CSV with a column <code>sign</code> (Wells numbers) and a column <code>value</code> (sounds in "
            "SLP1 or plain letters; only consonants are scored). It gets the same test as the claim, with 300 shuffles instead of 10,000.")
    up = st.file_uploader("Sign-to-sound table", type="csv", label_visibility="collapsed")
    if up is not None:
        try:
            d = pd.read_csv(up, dtype=str, keep_default_na=False)
            d.columns = [c.strip().lower() for c in d.columns]
            col = next(c for c in ("value", "xlit") if c in d.columns)
            table = {int(a): v.strip() for a, v in zip(d["sign"], d[col]) if a.strip().isdigit() and v.strip()}
        except (KeyError, StopIteration, ValueError, pd.errors.ParserError):
            table = None
        r = test_table(harness, dec_data, tables, wells, table, lex=lexicon_tools()) if table else None
        if r is None:
            st.html('<div class="note">Could not test this table. It needs columns <code>sign</code> and <code>value</code>, at least 20 signs '
                    'with a consonant in their value, and it must read at least 30 whole inscriptions.</div>')
        else:
            n = r["nulls"]
            st.html(stats(("Inscriptions read", f"{r['texts']:,}", f"{r['signs']} signs"), ("Sequence score", f"{r['observed']['sequence']:+.3f}"),
                          ("Against any shuffle", f"p = {n['value_shuffle']['sequence']['p']:.3f}"),
                          ("Against frequency-matched shuffles", f"p = {n['binned_10']['sequence']['p']:.3f}")))
            if "lexicon" in r:
                lx = r["lexicon"]
                st.html(stats(("Outputs that are Sanskrit words", f"{lx['observed']['whole_text_match']:.1%}",
                               f"p = {lx['nulls']['binned_10']['whole_text_match']['p']:.3f} vs matched shuffles"),
                              ("Letters covered by Sanskrit words", f"{lx['observed']['coverage']:.1%}",
                               f"p = {lx['nulls']['binned_10']['coverage']['p']:.3f} vs matched shuffles")))
            st.caption(f"{r['draws']} shuffles, so the smallest possible p is {1 / (r['draws'] + 1):.3f}; one uncorrected test, a screening "
                       "result only. A small p means the table carries structure beyond chance; it does not mean the readings are right.")
    st.caption("Full report: reports/milestone_4/milestone_4_report.md")


def selftest():
    signs = [{"bbox_xywh": [100, 10, 30, 50], "top_k": [[7, 1.0]]}, {"bbox_xywh": [300, 300, 30, 50], "top_k": [[9, 1.0]]}]
    assert match([(7, 98, 9, 131, 61), (5, 10, 10, 40, 60)], signs) == [7, None]  # second box overlaps no true sign
    model = ngram.WittenBell([(1, 2, 3), (1, 2, 3), (3, 2, 1)], [1, 2, 3], 2, backoff=True)
    p = candidates(model, [1, 2, 3], (1, 2, 3), 1)
    assert abs(p.sum() - 1) < 1e-9 and p.argmax() == 1  # the hidden middle sign, 2, is the best guess
    assert len(reorderings((1, 2, 3))) == 5 and (1, 2, 3) not in reorderings(tuple(range(9)), n=50)
    html = tiles([1, 2, 3], hidden=1)
    assert html.index("<span>3</span>") < html.index("?</div>") < html.index("<span>1</span>")  # first-read sign is rightmost
    assert draw(np.zeros((100, 200, 3), np.uint8), signs[:1]).size == (900, 450)
    assert parse_signs("1 ? * x", {1}) == ([1, None, restore.STAR], ["x"]) and parse_signs("1 ?", {1}, gaps=False) == ([1], ["?"])
    assert tiles([1, 2], hidden={0}, notes=["", "n"]).count("tile gap") == 1
    try:
        harness, dec_data, key, tables, _, wells, _ = decipher_tools.__wrapped__()
    except (FileNotFoundError, OSError):
        harness = None  # the decipherment data is not on this machine
    if harness:
        lex = lexicon_tools.__wrapped__()
        r = test_table(harness, dec_data, tables, wells, {s: cv[1] for s, cv in key.items()}, draws=20, lex=lex)
        assert r["texts"] > 1000 and 0 < r["nulls"]["binned_10"]["sequence"]["p"] <= 1
        assert lex is None or 0 <= r["lexicon"]["observed"]["coverage"] <= 1
        assert test_table(harness, dec_data, tables, wells, {1: "ka"}) is None
    from streamlit.testing.v1 import AppTest
    os.environ["INDUS_APP_UNDER_TEST"] = "1"
    for tab in ("m1", "m2", "m3", "m4"):
        at = AppTest.from_file(__file__, default_timeout=300)
        at.session_state["_only"] = tab
        at.run()
        assert not at.exception, at.exception
    print("ok")


if __name__ == "__main__":
    if not st.runtime.exists() and not os.environ.get("INDUS_APP_UNDER_TEST"):
        selftest()
        raise SystemExit
    st.set_page_config(page_title="Indus script · computational epigraphy", layout="wide")
    st.html(CSS + '<div class="masthead"><div class="eyebrow">Computational epigraphy</div><h1>Reading the Indus script by machine</h1></div>')
    only = st.session_state.get("_only")  # set by the self-check, which cannot click tabs
    m1, m2, m3, m4 = st.tabs(["Seal to sign numbers", "Sign-sequence statistics", "Filling damaged inscriptions", "Testing decipherment claims"])
    if only in (None, "m1"):
        with m1:
            milestone1()
    if only in (None, "m2"):
        with m2:
            milestone2()
    if only in (None, "m3"):
        with m3:
            milestone3()
    if only in (None, "m4"):
        with m4:
            milestone4()
