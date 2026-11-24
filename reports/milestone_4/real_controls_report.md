# More controls on real scripts (exploratory, not pre-registered)

One true decipherment is one data point. This adds a second true one (Ugaritic read as Hebrew) and a known-wrong one (Linear A read with Linear B sound values as Greek), and runs each through the unchanged Milestone 4 test: a table against shuffled versions of itself, primary score (order beyond letter frequency), 10,000 draws.

## Summary

| Script | Read as | Table is | Signs | Tokens | Binned null: percentile, p | Value shuffle: percentile, p | Power at Indus size |
|---|---|---|---|---|---|---|---|
| Linear B | Greek | true | 69 | 3,450 | 99.0, 0.0101 | 67.6, 0.3244 | - |
| Ugaritic | Hebrew | true | 30 | 14,312 | 100.0, 0.0001 | 100.0, 0.0001 | 100% (binned), 100% (value shuffle) |
| Linear A, Linear B values | Greek | wrong | 61 | 2,374 | 98.1, 0.0189 | 95.4, 0.0457 | - |

Linear B comes from `linear_b.json`. A true table should be significant and the wrong one should not.

## Reading

- Ugaritic as Hebrew (true): detected against frequency-matched shuffles (p 0.0001); at Indus size it is detected in 100% of subsamples.
- Linear A as Greek (wrong): significant, which is a false alarm and an open problem for the method (p 0.0189).
- With Linear B (detected), the method finds 2 of 2 true tables and raises 1 false alarm on 1 wrong table, against the frequency-matched null.

## Ugaritic as Hebrew

Source: `uga-heb.no_spe.cog` from NeuroDecipher (commit 480bad24, no licence stated), 43,951 rows. Rows are split in two by a hash of the Ugaritic word. The Hebrew target is the first cognate of each half-A row (37,805 words); the texts are the 3,592 distinct Ugaritic words of half B (14,312 letters), one sign per letter. The true table is built from half A only: a Ugaritic letter that exists in Hebrew maps to itself, and each of the 7 Ugaritic-only letters to the Hebrew letter most often found at the same position in equal-length cognate pairs.

| Ugaritic | Hebrew | Harness letter |
|---|---|---|
| `$` | `$` | `r` |
| `&` | `&` | `h` |
| `*` | `z` | `h` |
| `<` | `<` | `l` |
| `@` | `<` | `l` |
| `H` | `H` | `v` |
| `S` | `S` | `h` |
| `T` | `T` | `h` |
| `Z` | `S` | `h` |
| `a` | `a` | `y` |
| `b` | `b` | `p` |
| `d` | `d` | `s` |
| `g` | `g` | `h` |
| `h` | `h` | `j` |
| `i` | `a` | `y` |
| `k` | `k` | `m` |
| `l` | `l` | `d` |
| `m` | `m` | `c` |
| `n` | `n` | `b` |
| `p` | `p` | `h` |
| `q` | `q` | `h` |
| `r` | `r` | `n` |
| `s` | `s` | `h` |
| `t` | `t` | `t` |
| `u` | `a` | `y` |
| `v` | `$` | `r` |
| `w` | `w` | `k` |
| `x` | `H` | `v` |
| `y` | `y` | `g` |
| `z` | `z` | `h` |

The 23 Hebrew letters are fitted into the harness's 16 by frequency: the 15 commonest get their own letter and the other 8 share `h`. This merge is lossy and works against the true table.

Indus size: 20 random subsamples of the Ugaritic words with about as many sign tokens as the Wells corpus (8,252), 1,000 draws each; power is the share with p < 0.05.

## Linear A with Linear B values as Greek

Source: `LinearAInscriptions.js` from lineara.xyz (commit 9c81beba; the author disclaims copyright). Of 6,416 transliterated entries, 5,318 were dropped (numerals, logograms, unidentified signs, dividers, single syllabograms) and 781 distinct words of two or more syllabograms kept (2,374 tokens, 61 signs). Values: the Linear B consonant series of each syllabogram, as in `linear_b.py`. Linear A is not thought to be Greek, so this table is wrong by construction.

## Limits

- Ugaritic is an alphabet of 30 letters, far easier than a system of about 400 signs.
- All three controls are word lists, not running text.
- Three controls are still few for a hit rate and a false-alarm rate.
- The Linear A control is wrong in its language, not necessarily in its sound values: Linear B took its signs from Linear A, and many values are thought to carry over. So a significant score may mean the order test responds to roughly right sound values in a language with similar syllable shapes, which is itself a limit on what the test can say about a language.

Settings: seed 20261124, run time 1.2 minutes, quick run: False.
