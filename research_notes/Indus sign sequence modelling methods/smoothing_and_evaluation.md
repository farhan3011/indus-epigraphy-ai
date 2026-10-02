# Smoothed n-gram / Markov models on a tiny symbol corpus: Witten-Bell smoothing and honest evaluation (as of October 2026)

Corpus assumed throughout: about 2,900 texts, 13,372 sign tokens, closed vocabulary of 417 signs, 112 signs occurring exactly once, mean text length about 4.6 signs.

Provenance tags used below:
- **[fetched]** = read this session from the linked page.
- **[local test]** = run by me on 2026-10-02 against NLTK 3.10.3 installed from PyPI into a clean venv (Python 3.14). Scripts were throwaway; the numbers are reproduced here.
- **[memory]** = standard result cited to its canonical source, but the source was not re-opened this session. Verify before quoting verbatim.

---

## 1. Witten-Bell smoothing: original formulation, interpolated form, exact formulas, base case, worked example

### Takeaway
Interpolated Witten-Bell is `p(w|h) = (c(h,w) + T(h)·p_lower(w)) / (c(h) + T(h))`, where `T(h)` is the number of distinct symbols seen after `h`; Chen & Goodman end the recursion in the **uniform distribution**, which is what guarantees no zero probabilities. An implementation that ends the recursion in the maximum-likelihood unigram (as NLTK does) still assigns zero to any sign absent from the training data.

### Cited Findings

**Formulas (Chen & Goodman 1999, section 2.5)**
- Witten-Bell "was developed for the task of text compression, and can be considered to be an instance of Jelinek–Mercer smoothing"; the nth-order smoothed model is "a linear interpolation between the nth-order maximum likelihood model and the (n−1)th-order smoothed model". [fetched] — [Chen & Goodman 1999, Computer Speech & Language 13:359–394](https://u.cs.biu.ac.il/~yogo/courses/mt2013/papers/chen-goodman-99.pdf)
- Definition (their eq. 6): `N1+(h·) = |{w : c(h,w) > 0}|`, the number of unique symbols that follow history `h`. [fetched] — [Chen & Goodman 1999](https://u.cs.biu.ac.il/~yogo/courses/mt2013/papers/chen-goodman-99.pdf)
- Interpolation weight (eq. 7): `1 − λ_h = N1+(h·) / ( N1+(h·) + Σ_w c(h,w) )`. [fetched] — [Chen & Goodman 1999](https://u.cs.biu.ac.il/~yogo/courses/mt2013/papers/chen-goodman-99.pdf)
- Resulting estimator (eq. 8): `p_WB(w|h) = ( c(h,w) + N1+(h·)·p_WB(w|h') ) / ( Σ_w c(h,w) + N1+(h·) )`, where `h'` is `h` with its oldest symbol dropped. [fetched] — [Chen & Goodman 1999](https://u.cs.biu.ac.il/~yogo/courses/mt2013/papers/chen-goodman-99.pdf)
- Interpretation: `1 − λ_h` is "the frequency with which we should use the lower-order model"; `N1+(h·)` is "the number of novel words we have seen after the history" and "this is in fact the count assigned by Witten–Bell smoothing to the lower-order distribution". [fetched] — [Chen & Goodman 1999](https://u.cs.biu.ac.il/~yogo/courses/mt2013/papers/chen-goodman-99.pdf)
- Footnote 3: "Witten–Bell smoothing refers to method C in these references. Different notation is used in the original text." The references are Bell, Cleary & Witten 1990 (*Text Compression*) and Witten & Bell 1991. [fetched] — [Chen & Goodman 1999](https://u.cs.biu.ac.il/~yogo/courses/mt2013/papers/chen-goodman-99.pdf)
- Base case: "The implementation witten-bell-interp is a faithful implementation of the original algorithm, where we end the model recursion by taking the zeroth-order distribution to be the uniform distribution." Elsewhere: `p_unif(w) = 1/|V|`. [fetched] — [Chen & Goodman 1999](https://u.cs.biu.ac.il/~yogo/courses/mt2013/papers/chen-goodman-99.pdf)

**Explicit bigram and trigram cases (direct instantiation of eq. 8 with a uniform zeroth order)**

Let `N` = total predicted tokens, `T` = number of distinct symbol types observed, `|V'|` = number of predictable events (417 signs + end-of-text = 418).

```
unigram:  p1(w)      = ( c(w)     + T      · 1/|V'| ) / ( N      + T      )
bigram:   p2(w|v)    = ( c(v,w)   + T(v)   · p1(w)   ) / ( c(v)   + T(v)   )
trigram:  p3(w|u,v)  = ( c(u,v,w) + T(u,v) · p2(w|v) ) / ( c(u,v) + T(u,v) )
unseen context (c(h) = 0):  p(w|h) = p_lower(w)
```
Here `c(v) = Σ_w c(v,w)` (count of `v` as a context, not its raw unigram count) and `T(v) = N1+(v·)`. — [Chen & Goodman 1999, eq. 6–8](https://u.cs.biu.ac.il/~yogo/courses/mt2013/papers/chen-goodman-99.pdf)

**SRILM's statement of the same thing**
- SRILM `-wbdiscount`: `bow(a_) = n(a_*) / (n(a_*) + c(a_))`; interpolated `g(a_z) = c(a_z) / (n(a_*) + c(a_))`; backoff `f(a_z) = c(a_z) / (n(a_*) + c(a_))`. This matches Chen & Goodman eq. 7–8. [fetched, via Deniz Yuret's transcription of the SRILM `ngram-discount(7)` man page; the SRI page itself returned a connection reset] — [SRILM ngram smoothing notes](http://www.denizyuret.com/2007/12/srilm-ngram-smoothing-notes.html); primary: [ngram-discount(7)](http://www.speech.sri.com/projects/srilm/manpages/ngram-discount.7.html)
- Caution: a search-engine summary attributed `D = n1/(n1+2·n2)` to Witten-Bell. That is wrong; it is SRILM's suggested constant for **absolute discounting** (`-cdiscount`). — [SRILM ngram smoothing notes](http://www.denizyuret.com/2007/12/srilm-ngram-smoothing-notes.html)

**Original formulation (Witten & Bell 1991, method C)** [memory]
- Method C estimates the probability of a novel (never-before-seen) event in a context as `t/(n+t)` and each seen event as `c_i/(n+t)`, with `n` tokens and `t` distinct types observed in that context; the paper also defines methods A, B, P and X as alternative escape estimators. — [Witten & Bell 1991, IEEE Trans. Inf. Theory 37(4):1085–1094](https://doi.org/10.1109/18.87000)
- A second, non-recursive textbook variant (Jurafsky & Martin, 2nd ed.) spreads the escape mass `T(h)/(c(h)+T(h))` **equally** over the `Z(h) = |V| − T(h)` unseen continuations: `p(w|h) = T(h) / (Z(h)·(c(h)+T(h)))` for unseen `w`. It is a different estimator from the interpolated form above; do not mix the two. The current 3rd-edition draft chapter no longer presents Witten-Bell. — [Jurafsky & Martin, SLP3 n-gram chapter](https://web.stanford.edu/~jurafsky/slp3/3.pdf); see also [MacCartney 2005 smoothing tutorial](https://nlp.stanford.edu/~wcmac/papers/20050421-smoothing-tutorial.pdf) (not opened this session)

**Worked example** [local test; hand-checked]

Training texts: `abca`, `abd`, `cab`, `e`. Closed vocabulary `{a,b,c,d,e,f,g}` (`f`, `g` never observed). Predictable events = 7 signs + EOS = 8. BOS is a context only.

- Unigram counts (BOS excluded): a=4, b=3, c=2, d=1, e=1, EOS=4. So `N = 15`, `T = 6`.
- `p1(a) = (4 + 6/8)/21 = 0.2262`; `p1(b) = 3.75/21 = 0.1786`; `p1(c) = 2.75/21 = 0.1310`; `p1(f) = 0.75/21 = 0.0357` (unseen sign, non-zero).
- Context `a`: followers are b×3, EOS×1, so `c(a) = 4`, `T(a) = 2`, escape weight `2/6`.
  - `p2(b|a) = (3 + 2·0.1786)/6 = 0.5595`
  - `p2(c|a) = (0 + 2·0.1310)/6 = 0.0437` (unseen pair, both signs seen)
  - `p2(f|a) = (0 + 2·0.0357)/6 = 0.0119` (unseen pair, unseen sign)
  - `p2(a|f) = p1(a) = 0.2262` (unseen context)
- Every row sums to 1 over the 8 events (asserted to 1e-12 for all 8 contexts).
- Perplexity of held-out text `abc`: 4.04 counting the EOS prediction, 2.72 not counting it. Held-out text `af`: 9.90 (finite).

**Reference implementation (the code that produced the numbers above)**
```python
from collections import Counter, defaultdict
import math
BOS, EOS = "<s>", "</s>"

def fit(texts, V):                       # V = closed sign list, without BOS/EOS
    uni, big = Counter(), defaultdict(Counter)
    for t in texts:
        s = [BOS] + list(t) + [EOS]
        uni.update(s[1:])                # BOS is never a predicted event
        for a, b in zip(s, s[1:]):
            big[a][b] += 1
    n_events = len(V) + 1                # signs + EOS
    N, T = sum(uni.values()), len(uni)
    def p1(w):
        return (uni[w] + T / n_events) / (N + T)
    def p2(w, h):
        c = big.get(h)
        if not c:
            return p1(w)                 # unseen context
        return (c[w] + len(c) * p1(w)) / (sum(c.values()) + len(c))
    return p1, p2

def perplexity(texts, p2, count_eos=True):
    ll = n = 0
    for t in texts:
        s = [BOS] + list(t) + [EOS]
        pairs = list(zip(s, s[1:]))
        if not count_eos:
            pairs = pairs[:-1]
        ll += sum(math.log2(p2(b, a)) for a, b in pairs)
        n += len(pairs)
    return 2 ** (-ll / n)
```
— formulas from [Chen & Goodman 1999, eq. 6–8 and section 4.1](https://u.cs.biu.ac.il/~yogo/courses/mt2013/papers/chen-goodman-99.pdf)

### Inferences
- For this project the transition matrix is 418 × 418: rows are the contexts BOS + 417 signs, columns are the events 417 signs + EOS. Each row must sum to 1; that row-sum assertion is the single most useful unit test.
- With `T = 417` (all signs seen in the full corpus) the uniform term contributes almost nothing on the full data, but inside a cross-validation fold a large share of the 112 hapax signs will be missing from training, and the uniform floor is the only thing keeping their probability above zero.
- A sign that occurs once as a context gets `c = 1, T = 1`, so half of its row mass goes to the unigram distribution. Rows for rare signs are therefore mostly unigram; do not interpret them as evidence about that sign's syntax.
- Pitfalls specific to hand-written versions: (a) counting BOS in the unigram distribution, which leaks probability to an event that never occurs; (b) using the raw unigram count of `v` instead of the context count `Σ_w c(v,w)` as the denominator; (c) using `|V| = 417` rather than 418 in the uniform term when EOS is predicted; (d) deriving the vocabulary from the training fold instead of the fixed 417-sign list.
- Reading direction (texts are conventionally read right to left) determines which sign is "previous"; fix it once in preprocessing and state it, since it changes every bigram count.

### Gaps
- Witten & Bell 1991 was not re-opened; the method A/B/C/P/X description is from memory. The Chen & Goodman footnote confirming "method C" was fetched.
- The Jurafsky & Martin 2nd-edition equal-share variant is from memory and the cited URL is the 3rd-edition draft, which does not contain it.
- The SRI-hosted man page could not be fetched (connection reset); the formulas come from a third-party transcription of it.

---

## 2. Witten-Bell versus modified Kneser-Ney, absolute discounting, Jelinek-Mercer and add-k on small corpora

### Takeaway
Chen & Goodman found modified Kneser-Ney best overall and Witten-Bell among the weakest methods on small training sets (the backoff variant "extremely poor", and the interpolated variant consistently worse than the backoff one). Those results come from word corpora with very large vocabularies, so their transfer to a 417-sign closed vocabulary is an inference, and the honest move is to compare two or three smoothers under the same cross-validation rather than assert that one is correct.

### Cited Findings
- "we see kneser-ney and kneser-ney-mod consistently outperforming the other algorithms." [fetched] — [Chen & Goodman 1999](https://u.cs.biu.ac.il/~yogo/courses/mt2013/papers/chen-goodman-99.pdf)
- "the method witten-bell-backoff performs extremely poorly for small training sets but competitively on very large training sets"; it "performs poorly, much worse than the baseline, for smaller datasets". The baseline is a simple Jelinek-Mercer implementation. [fetched] — [Chen & Goodman 1999](https://u.cs.biu.ac.il/~yogo/courses/mt2013/papers/chen-goodman-99.pdf)
- "The method abs-disc-interp generally outperforms the baseline algorithm, though not for very small datasets." [fetched] — [Chen & Goodman 1999](https://u.cs.biu.ac.il/~yogo/courses/mt2013/papers/chen-goodman-99.pdf)
- "For Witten–Bell smoothing, the backoff version consistently outperforms the interpolated version; for modified Kneser–Ney smoothing, the interpolated version consistently outperforms the backoff version; and for absolute discounting, the interpolated version works better on small datasets but worse on large datasets." [fetched] — [Chen & Goodman 1999](https://u.cs.biu.ac.il/~yogo/courses/mt2013/papers/chen-goodman-99.pdf)
- On higher-order models: "katz, abs-disc-interp, and witten-bell-backoff perform about as well or worse than the baseline algorithm except for the largest datasets. On the other hand, jelinek-mercer consistently outperforms the baseline algorithm." [fetched] — [Chen & Goodman 1999](https://u.cs.biu.ac.il/~yogo/courses/mt2013/papers/chen-goodman-99.pdf)
- Methodological warning: "the relative performance of smoothing techniques can vary dramatically over training set size, n-gram order, and training corpus ... it is not sufficient to run experiments on one or two datasets for a single training set size". [fetched] — [Chen & Goodman 1999](https://u.cs.biu.ac.il/~yogo/courses/mt2013/papers/chen-goodman-99.pdf)
- Formulas for the alternatives (SRILM notation: `a_z` an n-gram, `c` counts, `n(a_*)` distinct continuations, `n1..n4` counts-of-counts) [fetched] — [SRILM ngram smoothing notes](http://www.denizyuret.com/2007/12/srilm-ngram-smoothing-notes.html):
  - Add-k: `p(a_z) = (c(a_z) + D) / (c(a_) + D·n(*))`.
  - Absolute discounting, interpolated: `g(a_z) = max(0, c(a_z) − D)/c(a_)`, `bow(a_) = D·n(a_*)/c(a_)`, suggested `D = n1/(n1 + 2·n2)`.
  - Kneser-Ney: same top-level form, but the lower-order distribution uses continuation counts, `f(_z) = (n(*_z) − D1)/n(*_*)`.
  - Modified Kneser-Ney discounts: `Y = n1/(n1+2·n2)`, `D1 = 1 − 2Y·n2/n1`, `D2 = 2 − 3Y·n3/n2`, `D3+ = 3 − 4Y·n4/n3`.
- Jelinek-Mercer: `p(w|h) = λ_h·p_ML(w|h) + (1−λ_h)·p(w|h')` with `λ` trained on held-out data; Witten-Bell is the special case where `λ_h` is set in closed form by eq. 7. — [Chen & Goodman 1999, eq. 4 and section 2.5](https://u.cs.biu.ac.il/~yogo/courses/mt2013/papers/chen-goodman-99.pdf)
- Kneser-Ney's closed-form discounts break on small or unusual data: KenLM raises `BadDiscountException` ("Is this small or artificial data?") when the needed counts-of-counts are missing, and offers `--discount_fallback` to substitute fixed discounts. — [kenlm issue #82](https://github.com/kpu/kenlm/issues/82); [kenlm issue #424](https://github.com/kpu/kenlm/issues/424)
- NLTK's `KneserNeyInterpolated` uses a single fixed discount (default `0.1`), not the three count-dependent discounts of modified Kneser-Ney. [fetched source] — [nltk/lm/models.py](https://github.com/nltk/nltk/blob/develop/nltk/lm/models.py)

### Inferences
- Chen & Goodman do **not** support Witten-Bell for small data. Taken together, their two statements (backoff Witten-Bell is extremely poor on small training sets; backoff consistently beats interpolated Witten-Bell) imply the interpolated form was at least as weak in their experiments. The case for interpolated Witten-Bell here is only that the brief requires it and that it has no tunable parameters to overfit on 13k tokens.
- Their ranking comes from word corpora with vocabularies of tens of thousands and may not transfer to a bigram model over 417 signs. That uncertainty is the reason to run Kneser-Ney or absolute discounting through the same folds rather than assume either outcome.
- Kneser-Ney's advantage comes from the continuation-count unigram, which matters when a symbol is frequent but only in a few contexts. Whether that holds for Indus signs is an empirical question the project can answer cheaply: run add-k (with `k` tuned inside the training folds), Witten-Bell and interpolated absolute discounting or Kneser-Ney through the same folds and report all of them. Expect differences of a few percent in perplexity at bigram order; a large difference more likely indicates a bug than a real effect.
- With a closed and known vocabulary there is no unknown-word problem to solve: every one of the 417 signs must receive non-zero probability in every context, and the uniform base case does that. An `<UNK>` class is unnecessary and distorts perplexity (section 4).
- Add-one (Laplace) on a 417 × 417 table with about 13k transitions adds 417 pseudo-counts to every row, more than the real count of nearly every context, so it flattens the matrix badly; if add-k is used as a baseline, `k` should be small and tuned.

### Gaps
- No published comparison of these smoothers on a corpus of this size and vocabulary was found; the recommendation to compare is an inference.
- The smoothing method used in the closest Indus precedent, Yadav et al. 2010 ([PLoS ONE 5(3): e9506](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0009506)), could not be confirmed: only the abstract was retrievable, which says model performance was "evaluated using information-theoretic measures and cross-validation" without naming the smoother. The full text should be checked for the smoothing method, the handling of duplicate texts, and reported perplexities.

---

## 3. NLTK `nltk.lm` status and alternatives

### Takeaway
The reviewer's claim is half confirmed. The **infinite-perplexity** problem is real, present in the current release (3.10.3), and is a design choice rather than an unfixed bug: NLTK ends the Witten-Bell recursion in the maximum-likelihood unigram, so any sign absent from the training fold gets probability zero. The **crash** claim rests on one still-open 2021 issue that I could not reproduce on 3.10.3. A hand-written implementation is the right call for this corpus.

### Cited Findings

**Issues (all fetched)**
- **#2727**, "Unable to evaluate perplexity and entropy of KneserNeyInterpolated and WittenBellInterpolated Language Model". Opened 11 June 2021 against NLTK 3.6.2; status **open**. Error: `AttributeError: 'int' object has no attribute 'N'`, raised in `nltk/lm/smoothing.py` in `alpha_gamma` at `prefix_counts.N()`. Reporter notes "MLE and Laplace works fine". No fix or linked pull request is recorded on the issue page. — [nltk issue #2727](https://github.com/nltk/nltk/issues/2727)
- **#3065**, "Perplexity of a count based language model is always infinity". Opened 24 October 2022; status **closed**. Concerns `KneserNeyInterpolated`: the start token `<s>` gets a score of zero because its continuation count is zero, and "entropy is the mean of the ngram logscores, the result is infinity if any ngram has a score of zero". — [nltk issue #3065](https://github.com/nltk/nltk/issues/3065)
- **#3211**, "KneserNeyInterpolated has problem with OOV words during testing and perplexity is always inf". Opened 7 December 2023 against NLTK 3.8.1; status **closed**. Perplexity is infinite when an unseen word is the predicted (second) word of a bigram, finite when it is the context. The issue does not mention Witten-Bell. — [nltk issue #3211](https://github.com/nltk/nltk/issues/3211)
- **#2495**: `WittenBellInterpolated` recomputed `len(self.vocab)` and `self.counts[len(context)+1].N()` repeatedly, a reported 100–1000× slowdown (performance, not correctness). **#3317**: `KneserNeyInterpolated` text generation took over 3.5 minutes where Witten-Bell took 2 seconds. Both from search-result summaries, not opened. — [nltk issue #2495](https://github.com/nltk/nltk/issues/2495); [nltk issue #3317](https://github.com/nltk/nltk/issues/3317)

**Source code on the `develop` branch (fetched)**
- `WittenBell.unigram_score` returns `self.counts.unigrams.freq(word)`, the maximum-likelihood unigram. `_gamma` is `n_plus / (n_plus + self.counts[context].N())`, which matches Chen & Goodman eq. 7. So the bigram level is right and only the base case differs from `witten-bell-interp`. — [nltk/lm/smoothing.py](https://github.com/nltk/nltk/blob/develop/nltk/lm/smoothing.py)
- `InterpolatedLanguageModel.unmasked_score`: if the context is unseen, `alpha, gamma = 0, 1` (full fall-through to the lower order); result is `alpha + gamma * unmasked_score(word, context[1:])`. — [nltk/lm/models.py](https://github.com/nltk/nltk/blob/develop/nltk/lm/models.py)
- NLTK's own documentation asserts the infinite result as intended behaviour. Under "Issue 167": "Sentence containing an unseen word should result in infinite entropy because Witten-Bell is based ultimately on MLE, which cannot handle unseen ngrams." The doctest expects `inf`, and a finite value once unseen terms are removed. — [NLTK lm howto](https://www.nltk.org/howto/lm.html)

**Versions**
- NLTK 3.10.3 is the current release (released 12 August 2026 per PyPI-derived search results; `pip install nltk` on 2026-10-02 gave 3.10.3). The 3.9.2 (Oct 2025), 3.9.3 (Feb 2026) and 3.10.0 (June 2026) notes found concern security hardening and unrelated fixes; no `nltk.lm` change was found. — [nltk on PyPI](https://pypi.org/project/nltk/); [NLTK release notes](https://www.nltk.org/news.html)

**Local tests on NLTK 3.10.3** (train `abca`, `abd`, `cab`, `e`; bigram models built with `padded_everygram_pipeline`) — behaviour of the code at [nltk/lm/smoothing.py](https://github.com/nltk/nltk/blob/develop/nltk/lm/smoothing.py)

| Model | held-out `abc` | held-out `af` (`f` unseen in training) | P(`<s>` after `a`) |
|---|---|---|---|
| `WittenBellInterpolated`, default vocabulary | 4.205 | **inf** | 0.070 |
| `WittenBellInterpolated`, explicit closed `Vocabulary` including `f`, `g` | 4.205 | **inf** | 0.070 |
| `KneserNeyInterpolated` (either vocabulary) | 4.693 | **inf** | 0.0 |
| `Laplace`, default vocabulary | 5.069 | 7.268 | 0.083 |
| Hand-written Witten-Bell (section 1) | 4.042 | 9.901 | not an event |

- Supplying the full closed vocabulary does **not** fix the infinity: `f` is in the vocabulary but has zero training count, so its unigram MLE is zero.
- No exception was raised in any of these runs, nor when calling `entropy`/`perplexity` with a list of bigram tuples, a list of lists, a flat list of strings, a raw string, or mixed-order everygrams. **#2727's crash did not reproduce on 3.10.3 with these inputs**; I cannot say whether a fix landed or whether the reporter's exact input was different.
- Silent wrong answers that did occur:
  - A flat list of token strings (`lm.entropy(["the", "cat"])`) is interpreted as a list of n-grams. It returned `inf`, not an error (observed). Mechanism as read from the source, not separately tested: each string is treated as a sequence whose last character is the word and whose preceding characters are the context, truncated to the model order, so in a bigram model a sign identifier stored as `"342"` would be scored as `'2'` given `'4'`.
  - Passing everygrams (the workaround suggested in #3065) averages unigram and bigram log-scores together: 2.327 bits versus the bigram-only log2(4.205) = 2.07 bits per event on the same text, i.e. a different quantity, not a perplexity of the bigram model.
  - NLTK Witten-Bell and Laplace assign probability to `<s>` as a predicted symbol (0.070 and 0.083 after `a`), mass that can never be earned in evaluation. NLTK's Witten-Bell scored 4.205 where the hand-written version scored 4.042 on identical data; the gap comes partly from this leak and partly from the different base case (ML unigram counting `<s>` versus uniform-interpolated unigram). The two effects were not separated.

**Alternatives**
- SRILM supports Witten-Bell directly: `ngram-count -text corpus.txt -order 2 -wbdiscount -interpolate -lm wb.lm`; `-wbdiscount n` applies it to order `n`. — [ngram-count(1)](http://www.speech.sri.com/projects/srilm/manpages/ngram-count.1.html); [SRILM ngram smoothing notes](http://www.denizyuret.com/2007/12/srilm-ngram-smoothing-notes.html)
- KenLM's `lmplz` estimates modified Kneser-Ney only, and fails with `BadDiscountException` on small or artificial data unless `--discount_fallback` is given. It cannot produce a Witten-Bell model. — [kenlm issue #82](https://github.com/kpu/kenlm/issues/82); [kenlm issue #424](https://github.com/kpu/kenlm/issues/424)

### Inferences
- Verdict on the reviewer's claim: "infinite perplexity" confirmed (issues #3065 and #3211 for Kneser-Ney; the documented "Issue 167" doctest and my tests for Witten-Bell). "Crash" confirmed only as an open, unresolved 2021 report (#2727) that I could not reproduce on the current release. The recommendation to hand-write about 20 lines stands, but for the infinite-perplexity and probability-leak reasons, not the crash.
- For this corpus the NLTK failure is not an edge case. With 112 hapax signs, a 5-fold split puts roughly a fifth of them (about 22) in each test fold with no training occurrence, so every fold's NLTK Witten-Bell perplexity would be `inf`.
- Calling NLTK's behaviour a "bug" overstates it; it is a documented design choice that differs from Chen & Goodman's formulation. The report should say "NLTK's Witten-Bell backs off to an unsmoothed unigram and therefore does not meet the brief's requirement of removing zero probabilities".
- NLTK could be patched by subclassing `WittenBell` and overriding `unigram_score` with the uniform-interpolated unigram, but that leaves the `<s>` leak and the input-shape traps in place (untested). The hand-written version is shorter and easier to verify.
- SRILM is trustworthy for Witten-Bell but requires a research licence and a C++ build; useful as an independent cross-check of the hand-written numbers, not as a dependency. Its unigram base case and unknown-word handling should be checked against the hand-written model before comparing numbers.

### Gaps
- Whether #2727 was fixed by a specific commit is unknown: the issue is open with no linked pull request, and I did not read the NLTK git history.
- The version in which `nltk.lm` was introduced and in which Kneser-Ney was rewritten were not verified.
- Third-party "kneser-ney" Python packages on PyPI were not evaluated; I found no evidence either way about their correctness and would not rely on one without the row-sum test.
- SRILM's exact treatment of the unigram base case under `-wbdiscount` (uniform floor or not, `-unk` behaviour) was not confirmed from the primary man page.

---

## 4. Evaluation protocol for a tiny corpus

### Takeaway
Perplexity on this corpus is determined as much by protocol as by the model: whether the end-of-text prediction is counted (about 18% of all predicted events), whether duplicates straddle the train/test split, and whether rare signs are collapsed into an unknown class. Fix each choice, state it, and report the mean and spread across folds with a uniform and a unigram baseline alongside.

### Cited Findings
- Chen & Goodman's convention: "we include the end-of-sentence token ⟨EOS⟩ when computing W_T, but not the beginning-of-sentence" token, where `W_T` is the number of tokens the cross-entropy is normalised by. [fetched] — [Chen & Goodman 1999, footnote 2](https://u.cs.biu.ac.il/~yogo/courses/mt2013/papers/chen-goodman-99.pdf)
- Same convention in the R `kgrams` documentation: cross-entropy is normalised by the total number of words "where the End-Of-Sentence tokens are included, but not the Begin-Of-Sentence tokens". [search summary] — [kgrams perplexity reference](https://vgherard.github.io/kgrams/reference/perplexity.html)
- SRILM's `ngram -ppl` reports two normalisations, one counting all tokens and one excluding end-of-sentence tags (`ppl` and `ppl1`). [search summary] — [ngram(1)](http://www.speech.sri.com/projects/srilm/manpages/ngram.1.html)
- Chen & Goodman handled out-of-vocabulary items by mapping them to a single token: "For all corpora, any out-of-vocabulary words were mapped to a distinguished token". [fetched, sentence truncated in extraction] — [Chen & Goodman 1999](https://u.cs.biu.ac.il/~yogo/courses/mt2013/papers/chen-goodman-99.pdf)
- Definitions: cross-entropy `H = −(1/W_T) Σ log2 p(w_i | h_i)`, perplexity `= 2^H`. — [Jurafsky & Martin, SLP3 n-gram chapter](https://web.stanford.edu/~jurafsky/slp3/3.pdf) [memory]
- Effect of the EOS choice in the toy example: perplexity of `abc` is 4.04 with the EOS prediction counted and 2.72 without. [local test] — formulas per [Chen & Goodman 1999](https://u.cs.biu.ac.il/~yogo/courses/mt2013/papers/chen-goodman-99.pdf)
- NLTK's padding and scoring conventions are undocumented enough that a user had to improvise: "No documentation clarifies the proper methodology for perplexity calculation with these models or how padding tokens should be handled." [fetched summary of the issue] — [nltk issue #3065](https://github.com/nltk/nltk/issues/3065)
- Grouped k-fold splitting, which keeps every member of a group in the same fold, is available as scikit-learn's `GroupKFold`. — [scikit-learn GroupKFold](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GroupKFold.html) [memory]
- KenLM's own error message for small data recommends deduplicating the input. [search summary] — [kenlm issue #82](https://github.com/kpu/kenlm/issues/82)

### Inferences

**Recommended protocol**
1. **Padding.** One BOS and one EOS per text for a bigram model (`n−1` BOS symbols for order `n`). BOS is a context only; EOS is a predicted event only.
2. **Count EOS, report both.** Headline perplexity counts the EOS prediction (the Chen & Goodman / SRILM `ppl` convention) because a model that does not predict where texts end is not a proper distribution over texts. Also report the sign-only figure. With 13,372 sign tokens and about 2,900 texts there are about 16,270 predicted events, of which EOS is about 18%; since text-final signs are highly predictable in this script, including EOS will lower perplexity noticeably. Never compare a with-EOS number against a without-EOS number from another paper.
3. **Vocabulary.** Fix the 417-sign list before splitting. With a uniform base case every held-out sign has non-zero probability, so there is no out-of-vocabulary case at all. Do not map hapax signs to `<UNK>`: that merges 112 signs into one easy-to-predict class and lowers perplexity for a reason unrelated to model quality. If an `<UNK>` variant is reported for comparison with other work, label it as a different task.
4. **Deduplicate or group before splitting.** Exact-duplicate texts placed on both sides of a split let the model be tested on its own training data, which lowers perplexity and favours higher-order and less-smoothed models. Two defensible options, and the report should say which: (a) type-level evaluation, where each distinct text appears once; (b) token-level evaluation with grouped folds, where all copies of a text go to the same fold. They answer different questions (the grammar of distinct texts versus the corpus as attested). Reporting both, plus the naive ungrouped split, shows the size of the leak.
5. **k-fold over texts, never over tokens or bigrams.** Use 5 or 10 folds, with all counts, vocabulary statistics and any tuned parameter (for example add-k's `k`) computed from the training folds only. Repeat with several random seeds.
6. **Aggregate correctly.** Compute per-fold cross-entropy in bits per event, then report the mean and standard deviation across folds (and across repeats), converting the mean cross-entropy to perplexity at the end. Averaging perplexities directly overweights bad folds. Fold estimates are correlated because training sets overlap, so the across-fold standard deviation is a descriptive spread, not a formal standard error.
7. **Baselines on the same folds.** Uniform (perplexity exactly 418 whether or not the EOS events are scored, because the model still spreads mass over 418 events; 417 only for a separately built model that never predicts EOS) and smoothed unigram. A bigram model's result is only meaningful relative to the unigram one; compare them paired, fold by fold.
8. **Sanity checks.** Row sums equal 1; held-out perplexity is finite and below 418; training-set perplexity is lower than held-out perplexity; fold results do not change when the order of texts is shuffled.

**How perplexity moves with each choice (direction of the effect)**
- Counting EOS: lowers it.
- Leaving duplicates across the split: lowers it, and more so for higher orders.
- Mapping rare signs to `<UNK>`: lowers it.
- Ending Witten-Bell in the MLE unigram (NLTK): makes it infinite whenever a held-out sign is unseen in training.
- Dropping test events with unseen signs to "fix" the infinity: lowers it and is invalid, because the hardest events are removed.
- Leaking mass to BOS as a predicted event: raises it slightly.

**Other notes**
- Damaged or partially illegible texts need an explicit rule (drop the text, or split it at the gap and do not pad the broken ends with BOS/EOS). Padding a fragment as though it were complete inflates the BOS and EOS counts of signs that were not actually initial or final.
- Many texts are length 1 or 2; a length-1 text contributes only a BOS→sign and a sign→EOS event. Report the length distribution alongside perplexity.

### Gaps
- No source was found that quantifies the effect of duplicate inscriptions on held-out perplexity for Indus or comparable corpora; the direction of the effect is a standard leakage argument, not a cited measurement.
- The Jurafsky & Martin wording on including `</s>` but not `<s>` in the token count was not fetched; the convention is confirmed by the Chen & Goodman footnote.
- The SRILM `ppl`/`ppl1` description and the `kgrams` wording come from search-result summaries, not from opening the pages.
- How Yadav et al. 2010 handled duplicates and padding in their cross-validation is unknown (full text not retrieved).

---

## 5. Statistics for a sparse transition matrix: pair significance, multiple comparisons, entropy estimates

### Takeaway
Use the log-likelihood ratio (G²) or Fisher's exact test for sign-pair association, not chi-square or raw PMI, correct for the number of pairs tested, and treat plug-in conditional entropy as biased downward by an amount that is not negligible here. Bootstrap over whole texts, never over tokens.

### Cited Findings

**Pair association**
- Dunning (1993) argued that tests assuming normality (z-scores, Pearson chi-square) overstate the significance of rare events in text and proposed the likelihood-ratio statistic `−2 log λ`, which is asymptotically chi-square distributed and behaves much better at small counts. For a 2×2 table with observed counts `O_ij` and expected counts `E_ij = (row_i · col_j)/N`: `G² = 2 Σ_ij O_ij ln(O_ij / E_ij)`, 1 degree of freedom. — [Dunning 1993, Computational Linguistics 19(1):61–74](https://aclanthology.org/J93-1003/) [memory]
- For a sign pair (x, y) the table is: `O11 = c(x,y)`; `O12 = c(x,·) − c(x,y)`; `O21 = c(·,y) − c(x,y)`; `O22 = N − O11 − O12 − O21`, where `N` is the number of bigram events. — [Dunning 1993](https://aclanthology.org/J93-1003/) [memory]
- Moore (2004) "empirically compares results obtained using log-likelihood-ratios and Fisher's exact test" for rare associations and reports that an LLR score above 10 is a highly reliable indicator of a significant association (under 1% expected noise). [search summary of the paper] — [Moore 2004, EMNLP](https://aclanthology.org/W04-3243/)
- Pointwise mutual information: `PMI(x,y) = log2( p(x,y) / (p(x)·p(y)) )`. Church & Hanks note the ratio is unstable at small counts and applied a frequency threshold. — [Church & Hanks 1990, Computational Linguistics 16(1):22–29](https://aclanthology.org/J90-1003/) [memory]
- PMI's rare-event bias: for a perfectly associated pair `PMI = −log2 p(x,y)`, so the rarer the pair the higher the score; a pair of hapax signs that happen to be adjacent gets the maximum possible PMI in the corpus. — [Manning & Schütze 1999, Foundations of Statistical NLP, ch. 5](https://nlp.stanford.edu/fsnlp/) [memory]

**Multiple comparisons**
- The Benjamini-Hochberg procedure controls the false discovery rate: sort the `m` p-values, find the largest `k` with `p_(k) ≤ (k/m)·q`, reject hypotheses 1..k. Bonferroni (`p ≤ α/m`) controls the family-wise error rate and is more conservative. — [Benjamini & Hochberg 1995, JRSS B 57(1):289–300](https://doi.org/10.1111/j.2517-6161.1995.tb02031.x) [memory]

**Entropy estimation**
- The plug-in (maximum-likelihood) entropy estimator is biased downward; the Miller-Madow correction adds `(K − 1)/(2N)` (in nats), with `K` the number of non-empty bins and `N` the sample size. [search summary] — [infomeasure discrete entropy guide](https://infomeasure.readthedocs.io/en/0.5.0/guide/entropy/discrete/); original analysis in [Paninski 2003, Neural Computation 15(6):1191–1253](https://doi.org/10.1162/089976603321780272) [memory]
- "The NSB algorithm (Nemenman et al., 2002) and the Chao-Shen estimator (Chao and Shen, 2003) are now widely considered as benchmarks for small-sample entropy estimation"; in Hausser & Strimmer's simulations NSB, Chao-Shen and their shrinkage estimator had small mean squared error across scenarios. [search summary] — [Hausser & Strimmer 2009, JMLR 10:1469–1484](https://arxiv.org/abs/0811.3579)
- Chao-Shen estimator: estimate sample coverage `C = 1 − f1/N` (`f1` = number of singletons), set `p̃_i = C·p̂_i`, then `H = −Σ_i p̃_i ln p̃_i / (1 − (1 − p̃_i)^N)` (a Horvitz-Thompson correction for unseen species). — [Chao & Shen 2003, Environmental and Ecological Statistics 10:429–443](https://doi.org/10.1023/A:1026096204727) [memory]
- NSB: a Bayesian estimator using a mixture of Dirichlet priors chosen so the induced prior on entropy is approximately flat; it returns a posterior mean and standard deviation and needs the alphabet size as input. — [Nemenman, Shafee & Bialek 2002, "Entropy and inference, revisited"](https://arxiv.org/abs/physics/0108025) [memory]
- Python: the `ndd` package is "a Bayesian replacement for scipy.stats.entropy" implementing NSB for the regime where samples are far fewer than classes, and also provides Plugin, Miller-Madow, asymptotic NSB and Grassberger estimators. [search summary] — [ndd on GitHub](https://github.com/simomarsili/ndd)
- A recent treatment aimed at linguistic data: — [Arora, Meister & Cotterell 2022, "Estimating the Entropy of Linguistic Distributions", ACL](https://aclanthology.org/2022.acl-short.20.pdf) (found in search, not opened)

### Inferences
- **Scale of the sparsity.** There are 417² = 173,889 possible sign pairs and about 10,500 sign-to-sign transitions (13,372 tokens minus about 2,900 text-initial signs), so the mean expected count per cell is about 0.06. Almost every cell is empty by chance alone; an empty cell is not evidence of a prohibited pairing.
- **What to test.** Test only pairs that were observed, or pairs whose expected count under independence is large enough that absence would be informative (for "avoided" pairs, roughly `E ≥ 5`). State `m`, the number of tests, explicitly. Apply Benjamini-Hochberg at a stated `q`; for pairs with any cell expectation below about 5, use Fisher's exact test p-values instead of the chi-square approximation to G².
- **Report effect size and evidence separately.** G² (or the exact p-value) for evidence; PMI or the observed/expected ratio for strength; and the raw count. Rank by G², not PMI. If PMI is shown, apply a minimum count (for example `c(x,y) ≥ 5`).
- **Independence caveat.** Bigram events inside one text are not independent, and duplicated texts multiply identical events. Run the association tests on the deduplicated corpus, or state that p-values on the full corpus are anti-conservative.
- **Unigram entropy is safe; conditional entropy is not.** For the unigram distribution the Miller-Madow correction is `416/(2·13,372) ≈ 0.016` nats ≈ 0.022 bits, which is negligible. For the conditional entropy, writing `H(Y|X) = H(X,Y) − H(X)` and correcting each term gives a first-order bias of about `(K_xy − K_x)/(2N)` nats, where `K_xy` is the number of distinct observed bigrams. If a few thousand distinct bigrams are observed among about 13,000 events, that is on the order of 0.1 nats or more, and the first-order correction itself is unreliable in that regime. Plug-in conditional entropy therefore understates the true value and overstates how predictable the script is; the same bias inflates plug-in mutual information.
- **Practical recipe.** Report plug-in, Miller-Madow and NSB (or Chao-Shen) side by side for `H(X)`, `H(X,Y)` and the derived `H(Y|X)`. Agreement means the estimate is trustworthy; a spread is itself the finding. Held-out cross-entropy of the smoothed model (section 4) is an upper-bound-style estimate of the same quantity and is a useful cross-check, since it errs in the opposite direction.
- **Bootstrap.** Resample texts with replacement (a cluster bootstrap), recompute the statistic, and take percentile intervals from 1,000 or more replicates. Resampling individual tokens or bigrams destroys within-text dependence and gives intervals that are too narrow. A bootstrap of the plug-in estimator inherits its bias (each resample has even fewer distinct types, so replicates sit below the point estimate), so intervals should be built around a bias-corrected estimator, or the NSB posterior standard deviation reported instead.
- **Comparisons with other scripts or random baselines** are only valid at matched sample size and alphabet size, because the bias depends on both; subsample the comparison corpus to 13k tokens before estimating.

### Gaps
- Dunning 1993, Church & Hanks 1990, Benjamini & Hochberg 1995, Paninski 2003, Chao & Shen 2003 and Nemenman et al. 2002 were cited from memory of well-known results, not re-read this session; formulas should be checked against the originals before publication.
- Moore's "LLR above 10" threshold comes from a search summary and was derived for bilingual word association; its transfer to sign bigrams is an assumption.
- I did not verify the current maintenance status or Python-version compatibility of `ndd` as of October 2026.
- The number of distinct observed bigrams in the actual corpus is not known to me, so the conditional-entropy bias figure is an order-of-magnitude illustration, not a computed value.

---

## 6. Markov-chain diagnostics that make sense for very short sequences

### Takeaway
The meaningful question is whether a first-order model predicts held-out texts better than a zero-order one, and whether it still does after accounting for sign position. Answer it with held-out perplexity and a permutation test. The classical asymptotic chi-square test of Markov order is invalid at this sparsity, and stationary-distribution or mixing-time analysis describes a process that 5-sign texts never exhibit.

### Cited Findings
- The classical likelihood-ratio test of independence (order 0) against a first-order chain uses `−2 ln Λ = 2 Σ_ij n_ij ln( n_ij · N / (n_i· · n_·j) )`, asymptotically chi-square with `(m − 1)²` degrees of freedom for `m` states. — [Anderson & Goodman 1957, Ann. Math. Statist. 28(1):89–110](https://doi.org/10.1214/aoms/1177707039) [memory]
- Likelihood-ratio and chi-square approximations are unreliable when expected cell counts are small, which is the motivation for Dunning's and Moore's work on rare events. — [Dunning 1993](https://aclanthology.org/J93-1003/) [memory]; [Moore 2004](https://aclanthology.org/W04-3243/)
- Chen & Goodman evaluate models by cross-entropy on held-out data and compare against a baseline across training sizes, which is the model-comparison approach recommended here. [fetched] — [Chen & Goodman 1999](https://u.cs.biu.ac.il/~yogo/courses/mt2013/papers/chen-goodman-99.pdf)
- Precedent on the Indus corpus: Yadav et al. apply n-gram Markov chains and report that "a quadrigram Markov chain saturates information theoretic measures against a held-out corpus", and that the model restores doubtfully read texts with about 75% accuracy. [search summary of abstract] — [Yadav et al. 2010, PLoS ONE 5(3): e9506](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0009506)

### Inferences
- **Why the asymptotic order test fails here.** With `m = 417` the test has `416² = 173,056` degrees of freedom against about 10,500 transitions, so expected counts are about 0.06 per cell. The chi-square reference distribution does not apply; the statistic would be dominated by singleton cells. Do not report it.
- **Diagnostic 1: held-out comparison.** Compute per-fold cross-entropy for the smoothed unigram and the smoothed bigram model on identical folds and report the paired difference in bits per event with its spread across folds and repeats. A consistently positive gain on every fold is the evidence for first-order dependence. Extend to trigram to see whether the gain saturates.
- **Diagnostic 2: permutation test.** Null hypothesis: sign order carries no information. Shuffle the signs within each text (preserving each text's length and sign multiset), recompute a statistic such as total G², plug-in mutual information between adjacent signs, or held-out bigram cross-entropy, and repeat 1,000 or more times. The p-value is the fraction of shuffles at least as extreme as the observed value. Because the null distribution is built from the same sparse data, the small-sample bias cancels, which the asymptotic test cannot offer. A second null that shuffles signs across the whole corpus (keeping only text lengths) additionally tests which signs co-occur in a text; report which null was used.
- **Diagnostic 3: position confound.** In texts of 4 to 5 signs, strong positional preferences (signs that are typically initial or typically final) produce apparent bigram dependence without any sign-to-sign syntax. Fit a position-aware zero-order baseline, `p(sign | position from start, position from end)` with the same smoothing, and check whether the bigram model still beats it on held-out data. Without this baseline, "first-order dependence" may only restate "position matters".
- **Diagnostic 4: BOS and EOS rows.** `p(· | BOS)` and `p(EOS | ·)` are the initial-sign and terminal-sign distributions; they are well estimated (about 2,900 observations each) and directly interpretable, unlike most interior rows.
- **Why stationary distributions and mixing times are not meaningful.** (a) With BOS and EOS the chain is absorbing: every text terminates, and the only stationary distribution of the generative model is the trivial absorbed one. (b) Making the chain ergodic requires an artificial EOS→BOS link, after which the "stationary distribution" is just the unigram frequency of signs, already known from counting. (c) Stationarity and mixing describe behaviour as sequence length grows; a text of 5 signs makes at most 4 transitions, far short of any mixing time, and the positional effects above show the process is not time-homogeneous within a text in the first place. (d) Eigenvalues of a 418 × 418 matrix estimated from about 13,000 events, most rows resting on a handful of observations, are determined largely by the smoothing method rather than the data.
- **What can be reported instead.** Per-row effective sample size (context count) beside any quoted transition probability; the fraction of rows with fewer than, say, 5 observations; and bootstrap intervals (over texts) for the handful of high-count transitions the write-up discusses.

### Gaps
- Anderson & Goodman 1957 was cited from memory; the statistic and degrees of freedom should be checked against the paper.
- I found no source that explicitly states that stationary or mixing-time analysis is inapplicable to short inscriptions; the argument above is my own reasoning from the structure of the model.
- The position-confound argument is an inference; I did not find a study quantifying how much of the Indus bigram signal a positional unigram model explains.
- Yadav et al.'s full method (which test of model order, which smoothing) was not retrieved.
