# PRD: remaining work (README "To do") and add-ons

Written 16 Nov 2026 for an implementer with no prior context. Tasks 1-3 are the README to-do list; tasks 5-12 are add-ons (there is no Task 4: detector retraining was dropped, see "Out of scope"). Do them in the order given. Each task is independent: finish, verify and commit one before starting the next. Work on a branch `remaining-todos` off `main`.

## 0. Rules that apply to every task

**Read first (only these):** `README.md`, the file named under "Pattern to copy" in the task, and the functions the task lists. Do not read `run_milestone2.py` or `app.py` in full; they are long and you only need the named functions.

**House style (match the existing code):**
- One flat script per feature, no classes unless state is needed, no new packages, no config files, no CLI options beyond `--quick`.
- Every script has a module docstring with its run commands, a `selfcheck()` of plain `assert`s that needs no data, and `if __name__ == "__main__":` that calls `selfcheck()` first.
- Every script writes `reports/milestone_N/<name>.json` and a Markdown report `reports/milestone_N/<name>_report.md` built by a `write_report(res)` function from that JSON only. Every number in the report comes from the JSON; no hand-typed results.
- Fixed seeds everywhere (`np.random.default_rng(<date-like int>)`).
- `--quick` must finish in under 1 minute and write to the same files with `"quick": true` in `settings`. Never commit a quick-run report; rerun in full before committing.
- Report prose: plain sentences, no bold claims. State what the test can and cannot show. If a result is null or negative, say so plainly.

**Hard constraints:**
- Never commit anything under `data/raw/`, `data/derived/`, `models/`, `runs/`. Reports contain aggregates only: no corpus texts, no key values, no per-text output.
- Do not edit `decipherment/prereg.json` or `decipherment/harness.py`. The harness refuses to run with an uncommitted prereg, and the pre-registered results must stay reproducible. New Milestone 4 work is labelled "exploratory, not pre-registered" in its report title, as `overfit.py` does.
- Do not change existing numbers in existing reports. New work adds new files.
- No new dependencies. Everything needed is in `requirements.txt`.
- Outbound requests use the existing `data.get(url)` helper (curl, generic user agent). Never put personal details in a request.

**Environment:** `source AI/bin/activate` from the repo root. Data already on disk: `data/raw/{im77,im417,yajnadevam,reference,decipherment}`, `models/{bank.pt,masked_sign_model.pt}`, `runs/yolov8n`. GPU: RTX 4060 laptop, 8 GB.

**Definition of done for each task:** full (non-quick) run completes; `selfcheck()` passes; `python -m pytest -q sequence_analysis` still passes; `python app.py` (self-check) still passes; README updated as the task says; one commit.

**Data and licences (owner's decision, 16 Nov 2026):** this is a non-profit student project, so any publicly reachable source may be used for local analysis whatever its licence, as the project already does with the unlicensed Yajnadevam files. Licence is never a reason to stop. The one rule that stays, because this repository is public and MIT: third-party data is fetched at run time into `data/raw/` and never committed, and reports carry aggregates plus a source line. Record each source's URL, pinned commit or fetch date, and stated licence (or "none stated") in the report.

**When to stop and report instead of improvising:** a data source has no machine-readable access; an existing assert fails; a task needs a change to `harness.py` or `prereg.json`.

---

## Task 1. Dictionary match-rate test (Milestone 4)

**Goal.** A second, independent score for a claimed sign-to-sound table: how much of the table's output is made of real Sanskrit words, compared with shuffled versions of the same table. The existing score measures letter order; this one measures word matches, which is what decipherers actually present as evidence.

**New file:** `decipherment/lexicon.py`. **Pattern to copy:** `decipherment/overfit.py` (structure, imports, report style, how it builds `Corpus` and loads keys).

**Reuse, do not rewrite:** from `harness`: `units`, `shuffled`, `freq_bins`, `p_value`, `encipher`, `corrupt`; from `data`: `dcs_lines`, `control_lines`, `skeleton`, `load_key`; from `corpus`: `load_wells`; from `stats`: `holm`.

**Definitions.**
- *Lexicon*: split every `data.dcs_lines("train")` line on spaces, take `skeleton(word)`, count. Keep skeletons with length >= 3 and count >= 2. This is a set of strings over the 16-letter alphabet. Record its size.
- *Output of a table on a text*: `skeleton("".join(value[s] for s in text))`.
- *Score A, whole-text match*: among scored texts whose output has length >= 3, the share whose output is in the lexicon.
- *Score B, coverage*: over all scored texts, share of output letters covered by a greedy left-to-right longest-match segmentation into lexicon entries (at each position take the longest lexicon entry that starts there; if none, skip one letter, uncovered). Cap entry length at 12.
- Short skeletons match by chance very often; that is why the comparison is with shuffled tables and never with zero.

**Procedure** (texts = Wells texts fully covered by the key, exactly as `harness.py` lines selecting `texts`):
1. For both keys (`xlits_2024-11-30.csv`, `xlits_2026-07-15.csv`): observed A and B; null A and B from `draws` shuffles with `shuffled(us, rng)` and with `shuffled(us, rng, freq_bins(us, cp.counts, 10))`. `draws` = 2000 (quick: 100). p-value with `p_value`. Holm over the family (2 keys x 2 scores x 2 nulls = 8).
2. *Positive control*: `encipher(data.dcs_lines("heldout"), current_key, lengths, rng)` gives a corpus where the key is true by construction. Run the same test on 20 replicates (quick: 3), 500 draws each, at corruption shares 0, 0.1, 0.2, 0.3 via `corrupt`. Report power = share of replicates with p < 0.05, per score, per corruption. Read `harness.positive_control` first and mirror it.
3. *Negative control*: a randomly shuffled key tested against its own shuffles, 100 runs (quick: 10), 199 draws; report the false-positive rate at 0.05 (should be near 0.05).
4. *Lexicon-language control*: repeat step 1 with a lexicon built the same way from `data.control_lines()` (Old Tamil). If the Sanskrit table matches the Tamil lexicon as well as the Sanskrit one, the score is not measuring Sanskrit.

**Performance.** Do not loop over texts in Python per draw if it is slow. Precompute per text the sign tuple; per draw build `value_skel = {sign: skeleton(value)}` once (cache `skeleton` per distinct value string, there are few) and join. About 2,000 texts x 2,000 draws x 2 nulls x 2 keys should run in a few minutes; if over 15 minutes, lower `draws` to 1000 and say so in `settings`.

**selfcheck.** Tiny lexicon `{"kta", "mn"}`-style cases: greedy coverage of a known string equals a hand-computed value; whole-text match true/false cases; a key applied to a text gives the expected skeleton; shuffling preserves the multiset of values.

**Outputs.** `reports/milestone_4/lexicon.json`, `reports/milestone_4/lexicon_report.md` titled "Dictionary match rate (exploratory, not pre-registered)". Report sections: what is measured (4 sentences); lexicon size; controls table (power by corruption, false-positive rate); claims table (observed, null mean and sd, percentile, raw p, Holm p for each key x score x null); Tamil-lexicon row; one-paragraph reading; limits (short skeletons, lexicon from 5 texts only, greedy segmentation, vowels ignored).

**README.** Add a short "Extension: dictionary match rate" subsection under Milestone 4 after the blind-search extension, with the run command and a 3-4 sentence result written from the JSON. Remove the matching bullet from "To do". Add one line to `docs/RUN_GUIDE.md` under the Milestone 4 commands.

**Acceptance.** Positive control power at corruption 0 is reported (whatever it is; if below 0.8 for a score, the report must say that score cannot detect a true table and its claim rows are uninformative). Negative-control false-positive rate within 0.01-0.10.

---

## Task 2. Milestone 3 like-for-like comparison with Yadav et al. 2010

**Context you must know.** Milestone 3 already runs on the EBUDS-filtered corpus, which reproduces Yadav et al.'s filter exactly (1548 texts, 7000 tokens, 377 signs; see `corpus.ebuds`). So the corpus is already like for like. What is not like for like is the *metric and protocol*: Yadav et al. report restoration as whether the true sign is inside the model's predicted set (about 74% for their bigram), and they do not state their train/test split. The Milestone 3 report gives that set-based figure for the bigram only (`hit_in_90pct_mass_set`, see `masked_model.per_trial` and the line in `write_report` mentioning "Yadav et al. 2010 (74%)").

**Verified 16 Nov 2026:** `masked_model.per_trial` and `summary` already compute `hit_in_90pct_mass_set` and `mean_90pct_mass_set_size` for every model, and they are already in `reports/milestone_3/results.json` under `models.<name>`. The published figure is `run_milestone2.PUBLISHED["restoration_90mass"]` = 0.74. So this is a reporting change only: no retraining and no 15-minute rerun.

**Change:** `write_report` in `sequence_analysis/masked_model.py` only.
1. Add a section "Comparison with Yadav et al. 2010" with a table: first row "Yadav et al. 2010, bigram (published)" with 74% in the set column and "-" elsewhere; then rows for `unigram`, `bigram_calibrated`, `trigram_calibrated`, `kn_trigram_calibrated`, `net_calibrated`, `mixture`, `residual_calibrated` (keys of `res["models"]`); columns = top-1, share of trials with the true sign in the 90%-mass set, mean set size. Calibrated rows are used because a mass set only means something for calibrated probabilities; say so in one sentence. Keep the existing uncalibrated-bigram sentence as it is.
2. Under the table, three sentences: the corpus is identical (1548 texts); their split is unstated so the 5-fold figure is not the same protocol; set size must be read next to hit rate, since a larger set always hits more often.

**Regenerate the report without retraining:**

```bash
cd sequence_analysis && python -c "import json, masked_model as m; m.write_report(json.loads(m.M2_RESULTS.parent.parent.joinpath('milestone_3/results.json').read_text()))"
```

`git diff reports/milestone_3/` must show only the new section. Do not change folds, seeds, models or any existing number.

**selfcheck.** Add one assert: on a toy distribution, the 90%-mass set of a known probability vector has the expected size and membership.

**README.** In the Milestone 3 section add one sentence with the like-for-like figures; remove the bullet from "To do".

---

## Task 3. Linear B positive control (Milestone 4)

**Goal.** Show whether the Milestone 4 method detects a decipherment that is known to be right: Linear B read as Greek. The existing positive control uses a table this project made up; this one uses a real one.

**Data (verified 16 Nov 2026).**
- *Linear B texts:* `https://raw.githubusercontent.com/j-luo93/NeuroDecipher/480bad2487820e3737fecfdd108214efa769e34b/data/linear_b-greek.cog`. TSV with a header line; column 1 is a Linear B word in Unicode Linear B characters, column 2 its Greek cognate(s), `|` between alternatives. 919 rows, every character is a `LINEAR B SYLLABLE`. No licence is stated in the repository; fetch at run time, never commit.
- *Sign value from the character:* `unicodedata.name(ch)` gives for example `LINEAR B SYLLABLE B001 DA`; the value is the last word lowercased (`da`). No hand-typed sign table.
- *Greek target:* Homer from PerseusDL `canonical-greekLit` at commit `dacb04909e43dc47738828d33cc3d2b0635bc54c` (CC BY-SA 4.0): `data/tlg0012/tlg001/tlg0012.tlg001.perseus-grc2.xml` (Iliad) and `data/tlg0012/tlg002/tlg0012.tlg002.perseus-grc2.xml` (Odyssey), about 2 MB each. TEI XML; take the text of the verse-line elements (`<l>`), strip tags with a regex as `data.control_lines` does for HTML. Do not use column 2 of the cognate file as the target: it contains the answers.
- DAMOS (https://damos.hf.uio.no/, CC BY-NC-SA 4.0) would give running tablet text, but it is a JavaScript application with no visible bulk export. Do not spend time on it; mention it in the report's limits as the better source.

**Because the source is a word list:** one text = one distinct Linear B word, and the Greek target is Homer split into words (each word one target line).

**New file:** `decipherment/linear_b.py`. **Pattern to copy:** `decipherment/data.py` for fetching and pinning (`get`, SHA-256 recorded in `sha256.json` on first fetch and asserted afterwards), `decipherment/overfit.py` for the script and report shape.

**Data handling.** Fetch the three files once with `data.get` into `data/raw/linear_b/` (gitignored via `data/raw/`), record SHA-256 on first fetch and assert it afterwards. Nothing fetched is committed; only aggregates are published, with a source line for each file in the report.

**Design: no change to `harness.py`.** The harness scores strings over a fixed 16-letter alphabet `kgcjtdnpbmyrlvsh` (`data.CONSONANTS`) and treats letters as arbitrary symbols. Encode Linear B's consonant series in a subset of those letters:

| Linear B series | harness letter | Greek letters mapped to it |
|---|---|---|
| d | d | δ |
| j | y | (none in classical spelling) |
| k | k | κ γ χ |
| m | m | μ |
| n | n | ν |
| p | p | π β φ |
| q | c | (none: labiovelars were lost) |
| r | r | ρ λ |
| s | s | σ ς |
| t | t | τ θ |
| w | v | (none: digamma lost) |
| z | j | ζ |

Also ξ -> `ks`, ψ -> `ps`. Vowels, breathings, accents and punctuation are dropped (strip diacritics with `unicodedata.normalize("NFD", ...)` and remove combining marks before mapping). Letters `g b l h` of the harness alphabet stay unused. These value strings pass through `data.skeleton` unchanged (check this in `selfcheck`).

- *Signs*: the Linear B characters in the file. Give each distinct character an integer id; its syllable (`da`, `ro2`, `pte` ...) comes from its Unicode name as above.
- *True key*: `{id: (syllabogram, letters)}` where `letters` is the series letter of the syllabogram's consonant via the table (`da` -> `d`, `qe` -> `c`, `wo` -> `v`, pure vowels `a e i o u` -> empty string, `pte` -> `pt`, `nwa` -> `nv`, `dwe` -> `dv`, `two` -> `tv`, `ra2` -> `r`, `ha`/`a2` -> empty, `ai` -> empty). Write this mapping as a rule (consonant letters of the transliteration, digits dropped, mapped through the table), not a hand-typed list. Units with an empty value are still shuffled with the rest.
- *Texts*: one text per distinct Linear B word of at least 2 signs: the sequence of its characters. Report the counts of rows, distinct words and sign tokens.
- *Target*: Homer's words -> letters via the table, one word per target line. Hold out 10% of verse lines by a stable hash, as `data.dcs_lines` does. In the cognate file's Greek column `f` stands for digamma; that column is not used.
- *Tables*: `lm_g = harness.lm_table(greek_train)`; `tables = ((lm_g, harness.lm_table(greek_train, order=1)), (lm_g, harness.lm_table(sanskrit)), harness.nb_tables(greek_train, sanskrit))` where `sanskrit = [skeleton(l) for l in data.dcs_lines("train")]` serves as the control language. The primary score is `sequence` (index 0), as pre-registered for the Indus test.

**Procedure.**
1. `harness.test_key(cp, true_key, tables, draws, rng, bin_counts=(10,))` with `draws` = 10,000 (quick: 300). Report observed score, null mean and sd, percentile and p for the value-shuffle and frequency-binned nulls, all three scorers.
2. *Graded corruption*: `harness.corrupt(true_key, share, rng)` at 0.1, 0.2, 0.3, 0.5; 20 corrupted keys each (quick: 3), 1000 draws; report the share with p < 0.05. This shows how wrong a table can be and still pass.
3. *Size comparison*: this corpus (about 919 words) is smaller than the Indus one (8,252 sign tokens in 1,798 Wells texts), so do not subsample. Report its size in texts and sign tokens next to the Indus figures; a pass on a smaller corpus is the stronger result, and a fail may only reflect the size, which the report must say.
4. *Wrong-language check*: score the true Linear B key with Sanskrit as target and Greek as control (swap the roles in `tables`). It should not be significant.

**selfcheck (no data needed).** Greek-to-letters on 3 known words including one with ξ and one with accents; syllabogram-to-letters for `da`, `qe`, `wo`, `pte`, `a`, `ro2`; `skeleton(v) == v` for every letter string the rule can produce; the cognate-file parser on 3 hand-written rows.

**Outputs.** `reports/milestone_4/linear_b.json`, `reports/milestone_4/linear_b_report.md` titled "Positive control on a deciphered script: Linear B as Greek (exploratory, not pre-registered)". Sections: why (3 sentences); data and licences with attribution; filter counts; the letter table above and its known losses; results for steps 1-4; reading; limits.

**Limits the report must state.** Classical/Homeric Greek is centuries later than Mycenaean; `w`, `q` and `j` series have no counterpart in the target, which works against the true key; Linear B spelling omits most syllable-final consonants, the Greek skeleton does not; consonant-only scoring throws away the vowels a syllabary records. A pass therefore shows the method works despite these; a fail does not show the method is useless, and the report must say which of the two happened without softening it.

**README.** Replace the first "To do" bullet with a short subsection under Milestone 4 ("Extension: a real positive control"), run command plus a 3-4 sentence result from the JSON. Add the fetch note to `data/README.md` (source, licence, never committed). Add the run line to `docs/RUN_GUIDE.md`.

**Acceptance.** Sources documented in the report's data section; full run under 20 minutes after the first fetch; results for steps 1-4 present; no fetched text in git (`git status` shows only the script, two report files and the doc edits).

---

## Out of scope (do not start)

- Retraining the Milestone 1 detectors (the README "Retrain" bullet). Owner's decision, 16 Nov 2026: without hand-labelled real seals there is nothing to measure a retrained detector against, so it waits for the labelled set. Leave the bullet in the README "To do" list.

- Collecting or hand-labelling real seal photographs, a "not a sign" class, copy-paste augmentation, any training on real data. These need manual labelling that does not exist yet.
- The two Milestone 2 items listed in README as "Planned once Milestone 1 is properly hand-labelled" (error propagation with a real confusion matrix; direction test from photographs). Blocked on the same labels.
- New tabs or controls in `app.py` beyond what Task 12 lists.
- Any change to `prereg.json`, `harness.py`, or published numbers.
- Refactoring, renaming, type hints, new test frameworks, CI changes beyond Task 12.

---

# Add-ons (tasks 5-12)

Same rules as section 0. Each is one new script plus its JSON and report, labelled exploratory. Suggested order: 5, 8, 9, 6, 11, 7, 10, 12. Tasks 5, 6, 8, 9, 11 need nothing that is not already on disk.

For every add-on the README change is the same: a short subsection under the milestone it extends (run command, 3-4 sentences written from the JSON), one line in `docs/RUN_GUIDE.md`, and a row in the README "Layout" block if a new report folder appears.

## Task 5. How large would the corpus have to be? (Milestone 4)

**Question.** `overfit.py` showed that on a corpus of Indus size a blind search passes even a true table. At what corpus size does a true table become distinguishable? This turns "not testable" into a number.

**New file:** `decipherment/corpus_size.py`. **Read first:** all of `decipherment/overfit.py` (310 lines); `harness.encipher`, `harness.test_key`. **Reuse:** `overfit.scoring_tables`, `overfit.climb`, `overfit.split_score`, `overfit.agreement`, `harness.encipher`, `harness.units`, `harness.Corpus`.

**Procedure.** True key = `yajnadevam-2026-07-15` (as in the existing positive control). For each multiplier `m` in (0.5, 1, 2, 5, 10, 20) and each of 5 replicates (quick: m in (1, 5), 1 replicate):
1. Build a ciphertext corpus with `encipher(plain, key, lengths_m, rng)`, where `lengths_m` is the Wells text-length list repeated or subsampled to `m` times its size. The key is true by construction.
2. Plaintext must not overlap the text the scoring trigram was trained on. Measured: the held-out DCS split has 44,621 skeleton consonants and the train split 342,803, while the Wells corpus is 8,252 sign tokens, so the standard held-out split cannot fill the larger multipliers. For this script only, split DCS 50/50 by chapter (`int(md5(chapter).hexdigest(), 16) % 2`, reading `data/raw/decipherment/dcs.txt` as `data.dcs_lines` does): trigram from one half, plaintext from the other. Before the loop, encipher once and measure consonants used per sign token; drop any multiplier the plaintext half cannot fill and record which were dropped. Never reuse plaintext within one replicate.
   Note `overfit.scoring_tables` returns `((lm, unigram), None)`, which only supports `cp.score(M, tables, per_text=True)`. Get a corpus score from it as `(s * n).sum() / n.sum()`, as `split_score` does; do not call `harness.test_key` with these tables. Write the permutation test here as a 10-line loop over `harness.shuffled`.
3. Measure: (a) permutation-test p of the true key, 500 draws; (b) 10 searches with `climb` from random tables, 2000 swaps (quick: 2 searches, 300 swaps), fitted on a random half of the texts and scored on the other half with `split_score`; (c) held-out score of the true key minus the best searched table's held-out score; (d) `agreement` between each searched table and the true key.
4. Per `m` report: power at p < 0.05, mean and sd of (c), share of replicates where the true key beats every searched table on held-out texts, mean agreement.

**Outputs.** `reports/milestone_4/corpus_size.{json,png}`, `corpus_size_report.md`. Figure: x = corpus size in sign tokens (log scale), y = held-out margin (c) with a line at 0 and a vertical line at the real Indus size. The headline sentence is the smallest `m` at which the true key beats all searched tables in at least 80% of replicates, or "not reached at 20 times the corpus" if it never does.

**Limits to state.** One key and one language; enciphered Sanskrit is cleaner than any real inscription; the search is a simple hill-climb, a better search needs an even larger corpus.

**selfcheck.** `lengths_m` has the right total for m = 0.5 and 2; on a toy 3-sign key and a long toy plaintext the true key's held-out score is above a random table's.

## Task 6. Do signs that look alike behave alike? (Milestones 1 and 2)

**Question.** If the sign list splits one sign into several drawn variants, those variants should look alike and also occur in the same contexts. This joins the two halves of the project and bears on the dispute between short and long sign lists.

**New file:** `sequence_analysis/visual_vs_context.py`. **Read first:** `grapheme_extraction/classifier.py` (134 lines), `synthetic_generator.load_glyphs`, `stats.js_divergence`.

**Definitions.**
- Signs considered: those with at least 20 tokens in the EBUDS texts (`corpus.ebuds(corpus.load_lines())`). Record how many.
- *Visual similarity*: cosine similarity of DINOv2 embeddings of the 417 clean glyph drawings (`load_glyphs()` then the classifier's `to_tensor`, `load_backbone`, `embed`). One embedding per sign, from the clean drawing, not from the augmented bank.
- *Context similarity*: for each sign three probability vectors: distribution of the sign before it (with a text-start symbol), the sign after it (with a text-end symbol), and its relative position in the text in 5 bins. Similarity of two signs = 1 minus the mean Jensen-Shannon divergence over the three.

**Tests.**
1. Spearman correlation between the two pairwise similarity matrices (upper triangle), with a Mantel test: permute the sign labels of one matrix 10,000 times (quick: 500).
2. For the 50 most visually similar pairs: mean context similarity against 10,000 random sets of 50 pairs.
3. Repeat 1 with a pixel baseline (cosine of the flattened 150x150 drawings) to show whether DINOv2 matters.
4. List the 20 pairs that are high on both (top 5% visually and top 5% contextually) as candidate variant pairs, by Mahadevan sign number. Sign numbers are not corpus text and may be published.
5. Sensitivity: thresholds of 10 and 50 tokens.

**Outputs.** `reports/milestone_2/visual_vs_context.{json,png}` (scatter of the two similarities with the candidate pairs marked) and `visual_vs_context_report.md`.

**Limits to state.** Compound signs contain other signs, so they look alike for a reason unrelated to variation; drawings are normalised sign-list forms, not carved forms; a correlation near zero would mean visual form does not predict use, which is also a result.

**selfcheck.** Context similarity of a sign with itself is 1; two signs with identical toy contexts score 1 and disjoint contexts score below 0.5; the Mantel permutation keeps the matrix symmetric.

## Task 7. More controls on real scripts (Milestone 4). Do after Task 3.

**Question.** One true decipherment is one data point. Add a second true one and a known-wrong one, so the method has a measured hit rate and false-alarm rate on real scripts.

**New file:** `decipherment/real_controls.py`, importing the fetch, Greek mapping and test steps from `linear_b.py` (move shared helpers nowhere; import them).

**Control A, true: Ugaritic read as Hebrew.** (Format verified 16 Nov 2026.)
- Data: `https://raw.githubusercontent.com/j-luo93/NeuroDecipher/480bad2487820e3737fecfdd108214efa769e34b/data/uga-heb.no_spe.cog` (43,951 rows; the 4,535-row `uga-heb.small.no_spe.cog` has the same format, use it for `--quick`). TSV with a header; column 1 a Ugaritic word, column 2 its Hebrew cognate(s), both in the same one-character-per-letter Latin transliteration; `|` separates alternatives and `_` means no cognate (5,053 rows). Ugaritic uses 30 symbols (`$&*<@HSTZabdghiklmnpqrstuvwxyz`), Hebrew 23 (`$&<HSTabdghklmnpqrstwyz`). No other source is needed.
- Split rows in two by a stable hash of the Ugaritic word. *Target* = Hebrew words (first alternative, `_` rows skipped) of half A. *Texts* = distinct Ugaritic words of half B, one text per word, one sign per letter.
- *True key*, built by rule from half A only: a Ugaritic symbol that also exists in the Hebrew inventory maps to itself; each of the 7 Ugaritic-only symbols (`* @ Z i u v x`) maps to the Hebrew symbol most often found at the same position in equal-length cognate pairs. Print the resulting 30-row table in the report.
- *Fitting 23 Hebrew symbols into the 16 harness letters:* count Hebrew symbol frequencies in half A; the 15 most frequent each get their own harness letter (assign in frequency order to `k g c j t d n p b m y r l v s`), all the rest share the 16th (`h`). Apply the same map to the target and to the key's values. Check in `selfcheck` that mapped strings pass through `data.skeleton` unchanged. This merge is lossy and works against the true key; say so.

**Control B, wrong: Linear A read with Linear B sound values as Greek.**
- Data (format verified 16 Nov 2026): `https://raw.githubusercontent.com/mwenge/lineara.xyz/9c81beba669a4ffec3f5b29d4e7748d02c060eff/LinearAInscriptions.js` (1.6 MB; the author disclaims copyright). It is JavaScript, not JSON: `var inscriptions = new Map([["HT1",{...}], ...])`. Do not try to parse the whole file; pull every `"transliteratedWords": [ ... ]` array with a regex and `json.loads` each one. Entries are words like `"QE-RA₂-U"` (syllabograms joined by `-`, subscript digits for variants), plus numerals, `"\n"`, and divider characters.
- Texts = distinct words made only of syllabograms: every `-`-separated part must match `[A-Z]+[₀-₉]*` and the word must have at least 2 parts. Anything else (numerals, logograms, `*301`-style unidentified signs, dividers) is dropped; report how many words were kept and dropped. One text per word, matching the Linear B control.
- Values: lowercase the part, strip the subscript, then the Task 3 consonant rule. Target and tables: the Homer tables from Task 3.
- Expected: not significant. Linear A is not thought to be Greek.

**Procedure for each control.** Task 3 step 1 on the full corpus. For Ugaritic, which is large, also subsample the texts to the Indus size (8,252 sign tokens), 20 subsamples (quick: 3), 1000 draws each, and report power at p < 0.05. Then one summary table over all real controls (Linear B from Task 3's JSON, Ugaritic, Linear A): script, language, true or wrong, signs, tokens, percentile, p at full size, power at Indus size.

**Outputs.** `reports/milestone_4/real_controls.json`, `real_controls_report.md`. State plainly whether each control came out as expected. If Linear A as Greek is significant, do not explain it away: report it and list it as an open problem for the method.

**Limits to state.** Ugaritic is an alphabet of 30 letters, far easier than a 400-sign system; word lists are not running text; three controls are still few.

## Task 8. The same claim against other languages (Milestone 4)

**Question.** Is Sanskrit special for the claimed table, or does any language do as well? And does the blind search reach the same height for any language?

**New file:** `decipherment/languages.py`. **Read first:** `overfit.py`; `data.tamil_to_slp1`, `data.control_lines`; `corpus.reference_systems` and `corpus.cuneiform` for how the on-disk reference texts are read.

**Targets (all already on disk).** Sanskrit (`data.dcs_lines("train")`), Old Tamil (`data.control_lines()`), English (`data/raw/reference/pride_and_prejudice.txt`, by line), Sumerian (from `data/raw/reference/cdliatf_unblocked.atf`; `corpus.cuneiform` returns one flat stream of lowercased sign readings such as `lugal`, `e2`, so copy its 15-line loop into this script and keep one string per text line instead, readings joined, digits removed). Each is reduced to the same 16-letter consonant skeleton. Sanskrit and Tamil use `data.skeleton`. For English and Sumerian write one explicit mapping each to the 16 letters (for example English c, q, x -> k, k, ks; f -> p; w -> v; z -> s; Sumerian sh -> s, h-breve -> h, ng -> n) and print both in the report. Subsample every target to the same number of skeleton letters (the smallest of the four) so the trigrams are equally well estimated.

**Procedure.** For each target `L`, with tables built as `((lm_L, unigram_L), (lm_L, lm_other), nb)` where `other` is the pooled other three:
1. Permutation test of both Yajnadevam keys with the `sequence` score: observed, null mean and sd, percentile, p (2000 draws; quick: 200). Holm over 4 languages x 2 keys.
2. Blind search with `overfit.climb`: 10 starts, 2000 swaps (quick: 2, 300), fitted on half the texts and scored on the other half. Report the held-out ceiling and its gap above the claimed key.
3. Cross-language check: take each language's best searched table and score it under every other language's trigram. A 4x4 table.

**Outputs.** `reports/milestone_4/languages.json`, `languages_report.md`. Headline: whether the claimed key's percentile is higher for Sanskrit than for the others, and whether the search ceiling differs by language by more than its spread over starts.

**Limits to state.** Scores in bits per phoneme are not comparable across languages in absolute terms, only as percentiles against each language's own null and as gaps to its own ceiling; consonant skeletons erase much of what separates languages; four languages.

**selfcheck.** Each mapping on 3 known words; every mapped string contains only harness letters; subsampling gives equal letter counts.

## Task 9. Do the stroke signs behave like numerals? (Milestone 2)

**Question.** Signs made of repeated short or long strokes are often called numerals. If they are, three things should hold, and each can fail.

**New file:** `sequence_analysis/numerals.py`. **Read first:** `stats.py` (`g2`, `benjamini_hochberg`, `entropy`), `synthetic_generator.load_glyphs`.

**Finding the stroke signs.** Do not hard-code a list. From the 417 glyph drawings, a stroke sign is one whose connected components (OpenCV `connectedComponentsWithStats` on the thresholded drawing) are all thin near-vertical bars of about the same size: every component has height at least 3 times its width, and component heights differ by under 20%. Its count = number of components; record whether strokes are short or long (bar height under or over half the image) and whether they sit in one row or two. Save a contact sheet of the selected drawings with their counts to the report folder and look at it: fix the rule if it picks non-stroke signs or misses obvious ones, and say in the report what was adjusted. Record the final list (sign number, count, short/long) in the JSON.

**Tests, on EBUDS texts, each against a null of other signs matched in frequency.**
1. *Counted things.* Numerals precede a limited set of counted items. For each stroke sign, entropy of the following sign, compared with the same for 200 frequency-matched non-stroke signs. Then the overlap (Jensen-Shannon) between the following-sign distributions of different stroke signs, against frequency-matched pairs: numerals of different value should share what follows them.
2. *Interchangeability.* Count the text pairs that differ only in which stroke sign occupies one position, against the same count for frequency-matched sets of non-stroke signs.
3. *Frequency by count.* Plot tokens against stroke count, short and long separately. Real numeral use falls with value and has bumps at round numbers; report the Spearman correlation and the counts, no model fitting.
4. *Position.* Share of stroke-sign tokens at text start, text end and inside, against all signs.
5. *No runs.* Count adjacent stroke-sign pairs; additive numeral systems would show them, positional ones would not.

All p-values in one Benjamini-Hochberg family.

**Outputs.** `reports/milestone_2/numerals.{json,png}`, `numerals_report.md`. The reading must go test by test: which of the numeral predictions held, which did not. Do not conclude that the signs are or are not numerals; conclude which numeral-like behaviours are present.

**selfcheck.** The stroke detector on 4 synthetic images (3 bars -> 3; a circle -> not a stroke sign; 2 rows of 3 bars -> 6, two rows; bars of very different height -> rejected). Interchangeability count on toy texts.

## Task 10. Zero-shot detection on real photographs (Milestone 1)

**Question.** The synthetic-trained detectors fail on real seals. Does a pretrained open-vocabulary detector or segmenter isolate signs with no training at all? If it does, it can draft boxes for the real labelled set.

**New file:** `grapheme_extraction/zero_shot.py`. **Read first:** `pipeline.py` (181 lines), `bootstrap_boxes.py` (52 lines).

**Models, no new dependency.** The pinned `ultralytics` ships YOLO-World (`from ultralytics import YOLOWorld`, `model.set_classes([...])`) and SAM / FastSAM. Verified 16 Nov 2026: `from ultralytics import YOLOWorld, SAM, FastSAM` imports in this environment. Weights download on first use.
1. YOLO-World with prompts tried in turn: "symbol", "glyph", "engraved character", "carved sign", "letter". Confidence 0.05 and 0.15.
2. SAM (or FastSAM) automatic mask generation, then a filter: keep masks whose box is between 2% and 25% of the image width, whose aspect ratio is between 0.2 and 5, and that sit in the upper band of the seal where the inscription is; drop masks that contain three or more other kept masks.

**Evaluation.** On the four `data/samples/met_*.jpg` photos, save annotated images to `reports/milestone_1_zero_shot/`. There are no labels: count by eye, per photo and method, boxes on single signs, boxes on several signs at once, boxes on the animal or seal edge, signs missed. Put the counts in a table, one row per photo and method. Also run both methods on 100 synthetic test seals, where boxes are known, and report recall and precision at IoU 0.5 with `pipeline.iou`, to show the zero-shot methods are not simply worse everywhere.

**If one method finds at least 70% of the signs on the real photos:** add a `--detector` choice to `bootstrap_boxes.py` so label drafts can come from it, and say so in `docs/milestone1_status.md`. Otherwise report that zero-shot does not solve it and leave `bootstrap_boxes.py` alone.

**Outputs.** `reports/milestone_1_zero_shot/zero_shot_report.md`, annotated images, `zero_shot.json`. Update `docs/milestone1_status.md`.

## Task 11. Does the text predict the object? (Milestone 2)

**Question.** Milestone 2 found differences by site, object type and field symbol. How much can actually be predicted from the signs alone?

**New file:** `sequence_analysis/predict_object.py`. **Read first:** `run_milestone2.breakdown`, `icon_group`, `corpus.ebuds_rows`, `corpus.physical` (the columns of `corpus.load_lines()` are `site`, `object_type`, `fs_category`, `fs_description`, `signs`, among others; field-symbol group = `icon_group(fs_description)`, which gives unicorn / no field symbol / other).

**Setup.** Rows = `corpus.ebuds_rows(lines)` (one per object, metadata kept). Three targets: site, object type, field-symbol group; for each keep classes with at least 30 rows and report what was dropped. Features: counts of signs and of adjacent sign pairs (`CountVectorizer` on space-joined sign numbers, `ngram_range=(1, 2)`, `token_pattern=r"\S+"`), plus text length. Model: `LogisticRegression(max_iter=2000, class_weight="balanced")`. No neural model.

**Protocol.** 5-fold `StratifiedGroupKFold` grouped by the sign tuple, so copies of one text never sit on both sides. Metric: balanced accuracy and macro F1, mean and sd over 5 repeats with different fold seeds. Baselines: majority class; text length only; and labels permuted within the training folds, 200 times, for a p-value. Also: sign counts only against signs plus pairs, to show whether order adds anything.

**Confound to handle.** Site, object type and field symbol are correlated with each other. For each target also report balanced accuracy within the largest class of each other variable (for example field symbol predicted among Mohenjo-daro seals only).

**Outputs.** `reports/milestone_2/predict_object.json`, `predict_object_report.md`, one confusion-matrix figure per target. List the 10 signs with the largest absolute weights per class (sign numbers only). Limits: imbalance, Mahadevan's metadata categories, texts are very short, so a ceiling well below 100% is expected.

**selfcheck.** On toy rows where one sign determines the label, balanced accuracy is 1 and the permuted-label baseline is near chance; grouping keeps duplicate texts in one fold.

## Task 12. Housekeeping. Do last.

1. **CI for Milestone 4.** Add `decipherment/test_decipherment.py` with one test that imports each `decipherment` module and calls its `selfcheck()` (all are data-free). Add `python -m pytest -q decipherment` to `.github/workflows/tests.yml`, and add `scikit-learn` to the grep pattern in the install line there since `harness.py` imports it. If any selfcheck turns out to need data, skip it with `pytest.importorskip`-style guards as the corpus tests do.
2. **Commit the Milestone 1 plan.** `docs/Milestone_1_Plan.md` is untracked while `docs/milestone1_status.md` says the plan "is not in this repository yet". Commit it, and replace that sentence with a link.
3. **Final summary report.** New script `reports/build_summary.py` that reads every `results.json` and extension JSON under `reports/` and writes `reports/SUMMARY.md`: for each milestone the question, the method in two sentences, one table of headline numbers, what the result does and does not show, and a link to the full report. Then a closing section "What this project found" of at most 10 sentences, each traceable to a number in a table above. Missing JSON files are skipped with a line saying so. Link it from the top of the README.
4. **Demo app.** In `app.py`'s Milestone 4 tab, where an uploaded table is tested (`test_table`), also show the dictionary match-rate scores from `lexicon.py` next to the sequence score, if `reports/milestone_4/lexicon.json` exists. Extend `selftest()` to cover it. Nothing else in the app.
5. **README "To do".** Rewrite it to list only what is still open: the real labelled seal set, the detector retraining and the two Milestone 2 items, all of which depend on it.

## Final check before opening the PR

```bash
for f in decipherment/{lexicon,linear_b,corpus_size,languages,real_controls}.py sequence_analysis/{visual_vs_context,numerals,predict_object}.py; do python $f --quick || echo "FAILED $f"; done
git checkout reports/   # discard the quick-run reports
python -m pytest -q sequence_analysis decipherment && python app.py && python reports/build_summary.py
git status   # only scripts, reports/ and docs; nothing under data/, models/, runs/
```

README "To do" should then list only items that are blocked on real labelled seals. Open one PR per task or small group of tasks, not one PR for everything.
