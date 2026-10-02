# Methodological critiques of Indus-script statistics and preparation of the IM77 corpus (as of October 2026)

Reading notes on evidence quality used below:

- **[read in full]** = I read the primary PDF/HTML myself.
- **[abstract only]**, **[snippet only]**, **[via summariser]** = weaker access; treat numbers as needing a check against the original before quoting them in a paper.
- **[computed here]** = counted by me from the project's own IM77 export, not a published figure. File: `data/raw/im77/im77_corpus_lines.csv`, sha256 `385b82bd6a9629e0ab215af46db84ca0c9ec4f6044a2a2ea58a3c64121f26211`, downloaded from [the pinned commit](https://raw.githubusercontent.com/paragorth/Indus-/ed56b9dc293f26a4e6bddea36c9a394e6b211ff4/data/im77/im77_corpus_lines.csv).

## 1. The linguistic vs non-linguistic debate: what was claimed, what was shown, where it stands

### Takeaway
The 2009-2015 exchange established that sign-sequence statistics (conditional entropy, block entropy, Zipf fits, Markov structure) show the Indus inscriptions are non-random and structured, and that nobody has demonstrated these measures separate language from structured non-linguistic systems. Neither side conceded; the last published round (Language, 2014-2015) ended with Rao et al. "stand[ing] by our original findings" and Sproat maintaining his critique, so there is no consensus.

### Cited Findings

**Farmer, Sproat & Witzel 2004**
- Citation: Farmer, S., Sproat, R., Witzel, M. 2004. "The Collapse of the Indus-Script Thesis: The Myth of a Literate Harappan Civilization." *Electronic Journal of Vedic Studies* 11(2): 19-57. — [citation as given in Sproat 2010 reference list](https://aclanthology.org/J10-3012.pdf); PDF at [safarmer.com](https://safarmer.com/fsw2.pdf) (downloaded but not readable page-by-page in this session).
- The arguments include: statistics of sign frequencies and repetitions, texts too short to encode messages, too many rare signs (especially "singletons"), and lack of sign repetition within a text. **[snippet only]** — [search summary of the paper](https://www.researchgate.net/publication/216842497_The_Collapse_of_the_Indus-Script_Thesis_The_Myth_of_a_Literate_Harappan_Civilization)
- Rao et al. report that FSW claimed the properties of the Indus signs are also seen in "countless nonlinguistic sign systems" (FSW 2004, p. 21) and that the arguments were said to amount to a "proof" (FSW pp. 34, 37; Farmer 2005). **[read in full, Rao's characterisation]** — [Rao et al. 2010](https://aclanthology.org/J10-4016.pdf)
- Sproat's own description: FSW 2004 "reviews extensive support for that view from archaeological evidence and comparisons with other ancient symbol systems"; he says the arguments "have been accepted by many archaeologists and linguists" (in his reply Sproat cites Lawler 2004 in *Science*, who reports this view and quotes George Thompson and Frank Southworth). **[read in full]** — [Sproat 2010](https://aclanthology.org/J10-3012.pdf); [Sproat 2010 reply](https://aclanthology.org/J10-4017.pdf)
- Sproat states that structure in the Indus texts "has of course been known for decades; see Farmer, Sproat, and Witzel (2004)". Both sides therefore agree the system is non-random. — [Sproat 2010, fn. 4](https://aclanthology.org/J10-3012.pdf)
- Published rebuttals of FSW listed by Rao et al.: Parpola 2005, 2008; Vidale 2007 ("I see no acceptable scientific demonstration of the non-scriptural nature of the Indus sign system", p. 362); McIntosh 2008. Vidale compiled ten roughly contemporaneous Central/South Asian non-linguistic systems averaging about 44 distinct signs. **[read in full, as reported by Rao]** — [Rao et al. 2010](https://aclanthology.org/J10-4016.pdf)

**Rao et al. 2009 (Science)**
- Citation: Rao, R. P. N., Yadav, N., Vahia, M. N., Joglekar, H., Adhikari, R., Mahadevan, I. 2009. "Entropic Evidence for Linguistic Structure in the Indus Script." *Science* 324(5931): 1165. — [Science](https://www.science.org/doi/10.1126/science.1170391)
- Claim: bigram conditional entropy of the Indus signs, plotted over the N most frequent signs, lies close to natural-language samples (Sumerian, Old Tamil, English words and characters) and far from two "non-linguistic" curves. — [Sproat 2010](https://aclanthology.org/J10-3012.pdf)
- Final sentence, as quoted by Sproat: "Given the prior evidence for syntactic structure in the Indus script, our results increase the probability that the script represents language, complementing other arguments that have been made explicitly or implicitly in favor of the linguistic hypothesis." — [Sproat 2010 reply, sec. 2.2](https://aclanthology.org/J10-4017.pdf)
- "Type 1" and "Type 2": the paper describes Type 1 as systems without much sequential structure (linked to Vinča) and Type 2 as rigidly ordered systems (linked to kudurru deity symbols). Per the supplement, both were artificially generated: Type 1 assumes "each sign has an equal probability of following any other" (random and equiprobable); Type 2 assumes "each sign has a unique successor sign" or 2-3 successors. — [Sproat 2010](https://aclanthology.org/J10-3012.pdf); [Sproat 2010 reply, sec. 2.1](https://aclanthology.org/J10-4017.pdf)
- The corpus actually analysed was small: Rao et al. describe "about 1,550 lines of text and 7,000 sign occurrences"; Sproat gives "7,000 tokens (and about 400 types)" and mean length about 4.5 signs. This is the filtered EBUDS subset, not the full concordance. — [Rao et al. 2010](https://aclanthology.org/J10-4016.pdf); [Sproat 2010](https://aclanthology.org/J10-3012.pdf)

**Criticisms of Rao et al. 2009**
- Artificial controls: Sproat argues "there is little evidence that either of these types accurately characterized any ancient symbol system", that a completely rigid system "would convey no information whatsoever", and that the artificial status is learned "only if one reads the supplementary material". — [Sproat 2010](https://aclanthology.org/J10-3012.pdf)
- Farmer, Sproat & Witzel 2009 slides: "Invented data unlike any real-world non-linguistic system", and "'Conditional entropy' is simply a measure of the degree of order or disorder in any sequentially ordered system". They show a plot of 10 language corpora plus Indus bar seals and medieval heraldry in which curves overlap without regard to linguistic status (their own unreviewed analysis). — [Farmer, Sproat & Witzel 2009 slides](https://safarmer.com/more.on.Rao.pdf)
- The same slides assert results "vary widely with corpus size, with the genre of texts sampled, with the kinds of encoding used ... and with the 'smoothing' methods", and that the Rao result "can't be replicated even in principle unless you use exactly the same texts". This is an assertion, though the genre/encoding/corpus-size dependence is uncontroversial. — [same slides](https://safarmer.com/more.on.Rao.pdf)
- Memoryless generator reproduces the figure (demonstrated): Sproat got "a very good fit to [Rao et al.'s] results for the Indus corpus with a model that has 400 elements with a perfect Zipf distribution, with α = 1.5, and conditional independence for the bigrams"; at EMNLP 2009 he replicated the curve with a corpus matching only Mahadevan unigram frequencies. — [Sproat 2010](https://aclanthology.org/J10-3012.pdf)
- Liberman (Zipf exponent 1.7, no sequential constraints) and Shalizi (geometric distribution) did the same independently; Liberman: "this simulation matches the values in Fig 1A using a unigram model, without any sequential constraints at all." **[via summariser]** — [Language Log, 26 April 2009](https://languagelog.ldc.upenn.edu/nll/?p=1374)
- "Random is not the same thing as random and equiprobable" — Sproat's summary of the control-design error. — [Sproat 2010](https://aclanthology.org/J10-3012.pdf)
- No decision rule: "None of Rao and colleagues' work provides a statistical measure by which one can assert that something is inside the 'linguistic' range. Rather, they present plots and invite the reader to eyeball them." Fortran sits only somewhat below the language samples. — [Sproat 2010 reply](https://aclanthology.org/J10-4017.pdf)
- What would count as a valid test, per Sproat: "If one took a large sample of known linguistic and non-linguistic symbol systems, and showed that a particular set of measures could reliably distinguish between them with very high accuracy, then such measures could presumably be applied in the case of unknown systems". He also asks for a range of languages and genres ("prose, to poetry, to genealogies, to grafitti, to shopping lists"). — [Sproat 2010](https://aclanthology.org/J10-3012.pdf); [Sproat 2010 reply](https://aclanthology.org/J10-4017.pdf)
- Small-sample caution: n-gram models work in speech recognition because they are built from "millions or even billions of tokens"; one should not expect "reliable results by such techniques from corpora consisting of only a few hundred or few thousand symbols". — [Sproat 2010](https://aclanthology.org/J10-3012.pdf)

**Rao et al. 2010 reply (Computational Linguistics 36(4): 795-805)**
- They deny claiming sufficiency: "Does the similarity in block entropies with linguistic systems ... prove that the Indus script is linguistic? We do not believe so. In fact, we contend that barring a full decipherment, one cannot prove either the linguistic or nonlinguistic thesis". — [Rao et al. 2010](https://aclanthology.org/J10-4016.pdf)
- Inductive framing: the result is one property among several (linearity, directionality, diacritic-like modification and ligatures, Zipf-Mandelbrot law, positional/syntactic structure, diverse media, altered ordering in West Asian finds) raising the posterior of the linguistic hypothesis. Zipf-Mandelbrot is called "a necessary (though not sufficient) condition". — [Rao et al. 2010](https://aclanthology.org/J10-4016.pdf)
- On the artificial controls: "This objection is a red herring and does not change the result that the Indus script is entropically similar to linguistic scripts." They say DNA and protein are natural Type 1 sequences, and that Vidale's systems would be roughly Type 2 (asserted from descriptions; no entropy was computed because no corpora exist). — [Rao et al. 2010](https://aclanthology.org/J10-4016.pdf)
- Block entropy extension (N = 1..6, NSB estimator, log base = alphabet size): Indus stays near natural languages; DNA, protein and a Beethoven sonata are near maximum entropy; Fortran is lower. "Beyond N = 6, the entropy estimates become less reliable due to the small sample size". They say the Zipfian memoryless counterexamples "fail to exhibit the scaling of block entropies beyond unigrams and bigrams" (asserted in the reply; no figure of the counterexample is shown). — [Rao et al. 2010](https://aclanthology.org/J10-4016.pdf)
- First-order Markov log-likelihood of four West Asian texts is lower than the mean for 50 held-out Indus-region texts. — [Rao et al. 2010, Fig. 1(c)](https://aclanthology.org/J10-4016.pdf)

**Sproat's reply (Computational Linguistics 36(4): 807-816)**
- He accepts that Rao et al. never claimed conditional entropy settles the matter, but argues the "increase the probability" claim is unsupported under any reading because discriminative power was never measured: "When one proposes a new feature, one must demonstrate that this feature adds information, and increases the discriminative power of one's model." — [Sproat 2010 reply](https://aclanthology.org/J10-4017.pdf)
- DNA and proteins are not relevant comparisons: "the topic of discussion surely is symbol systems that were products of the human mind." — [Sproat 2010 reply](https://aclanthology.org/J10-4017.pdf)
- Kudurru deity symbols share most of Rao's "language-like" properties (linear arrangement, directionality, ligaturing, Zipfian distribution, positional preference, affinities between glyphs, use on more than one medium) yet are known not to be writing. — [Sproat 2010 reply](https://aclanthology.org/J10-4017.pdf)
- He says he had not analysed the block-entropy results, so that specific claim was not rebutted in this exchange. — [Sproat 2010 reply](https://aclanthology.org/J10-4017.pdf)

**Lee, Jonathan & Ziman 2010 (Pictish)**
- Citation: Lee, R., Jonathan, P., Ziman, P. 2010. "Pictish symbols revealed as a written language through application of Shannon entropy." *Proceedings of the Royal Society A* 466: 2545-2560. — [citation in Rao et al. 2010](https://aclanthology.org/J10-4016.pdf)
- Method as described by Sproat: Ur = F2 / log2(Nd/Nu) and Cr = Nd/Nu + a·Sd/Td (a = 7), fed to a decision tree; Cr ≥ 4.89 means linguistic; Ur then separates segmental (< 1.09), syllabic (< 1.37), else logographic. Controls were random-and-equiprobable texts. — [Sproat 2010](https://aclanthology.org/J10-3012.pdf)
- Demonstrated failures: a 545-token, 59-type kudurru corpus scores Cr = 8.0, Ur = 1.55 (logographic writing); 75 "texts" from tosses of seven six-sided dice (638 symbols) score Cr = 12.64, Ur = 1.18 (syllabic writing). — [Sproat 2010](https://aclanthology.org/J10-3012.pdf)
- Lee et al.'s response added a shuffle test (true conditional entropy lower than all randomised versions for 80% of scripts and for Pictish) and accepted that kudurrus score like Pictish, attributing this to a broader definition of "writing" (Powell 2009). Sproat: the shuffle test "seems to make sense as a way to spot memoryless non-equiprobable processes" but "would still fail to spot nonlinguistic systems that have structure", and the broad definition "reduces the claim ... to near vacuity". — [Sproat 2010 reply, sec. 3](https://aclanthology.org/J10-4017.pdf)
- Liberman's separate critique is at Language Log, 2 April 2010 (not read here). — [Language Log](https://languagelog.ldc.upenn.edu/nll/?p=2227)

**Sproat 2014 and the Language exchange**
- Citation: Sproat, R. 2014. "A statistical comparison of written language and nonlinguistic symbol systems." *Language* 90(2): 457-481. **[abstract only]** — [Project MUSE](https://muse.jhu.edu/article/547119/summary)
- Abstract-level result: with more non-linguistic and linguistic corpora than earlier studies, "none of the previously proposed methods are useful as published"; one of Lee et al.'s measures with a different cut-off, and a new measure based on repetition, classify well; both classify Indus and Pictish as non-linguistic. — [search summary of abstract](https://www.semanticscholar.org/paper/A-statistical-comparison-of-written-language-and-Sproat/f41ca4a2dd3612240f9c2b6328911d0d0acdd4c9)
- Press summary: non-linguistic corpora included Mesopotamian deity symbols, totem poles, barn stars, weather icons; 14 languages for comparison. Sproat's caveat: "a truly reliable demonstration that a collection of symbols exemplifies written language requires supporting empirical evidence, such as a credible decipherment or independent archeological evidence." **[press release]** — [ScienceDaily, 2 June 2014](https://sciencedaily.com/releases/2014/06/140602101559.htm)
- Corpora and tools are public (XML corpora, Python tools); errata note the test was a Wilcoxon rank sum test. — [Sproat's corpora page](https://richardsproat.com/data/non-linguistic-symbols/)
- Response: Rao, R. P. N., Lee, R., Yadav, N., Vahia, M., Jonathan, P., Ziman, P. 2015. "On statistical measures and ancient writing systems." *Language* 91(4): e198-e205. Abstract: Sproat's article "does not accurately present our methods and findings"; its entropic results "seem to favor, rather than contradict, the inductive hypothesis that the Indus script may represent writing"; "we stand by our original findings." **[abstract only]** — [Cambridge Core](https://www.cambridge.org/core/journals/language/article/abs/on-statistical-measures-and-ancient-writing-systems/D75CB72F35D752E4848CA69AE8E32888)
- Rejoinder: Sproat, R. 2015. "On misunderstandings and misrepresentations: A reply to Rao et al." *Language* 91(4): e206-e208; defends his linguistic/non-linguistic labelling against a circularity charge. **[snippet only]** — [Cambridge Core](https://www.cambridge.org/core/journals/language/article/abs/on-misunderstandings-and-misrepresentations-a-reply-to-rao-et-al/8A8B75F2A308664E9406E4527C742C2C)

### Inferences
- Safe to state: (a) Indus sign sequences depart from frequency-matched random order; (b) Indus conditional and block entropy values fall within the range of the small language samples Rao et al. plotted, a range that a memoryless Zipfian source also reproduces for conditional entropy; (c) a memoryless Zipfian source reproduces the Rao Fig. 1A curve, so that curve does not by itself show sequential structure; (d) at least one known non-linguistic system (kudurrus) passes the Lee et al. classifier. Not safe: any statement that entropy values indicate language, or indicate non-language.
- The dispute is partly about inference form. Rao et al. argue likelihood-style (feature raises P(language)); Sproat argues the likelihood ratio was never estimated because P(feature | structured non-language) was never measured. A student paper avoids the dispute by not making either inference.
- Sproat 2014's non-linguistic classification of the Indus is itself contested (Rao et al. 2015) and depends on which Indus corpus and which text units were used. It should be reported as one side's result, not as the settled outcome.

### Gaps
- Sproat 2014 was read only at abstract/press level: exact corpora list, sizes, the repetition measure's definition and threshold, and which Indus corpus he used are not verified here. Read the paper before citing specifics.
- FSW 2004 was not read page by page; its specific statistics (e.g. singleton percentages, repetition counts) are not reproduced here.
- Rao et al. 2009 supplement and Rao 2010 (*IEEE Computer* 43(4): 76-80) were not read directly; statements about them come from the 2010 exchange.
- Lee et al.'s 2010 response in Computational Linguistics was read only through Sproat's reply.

## 2. Work from 2020-2026 bearing on the debate

### Takeaway
Two 2026 arXiv preprints, both single-author and not peer reviewed, directly test whether Indus-style statistics are specific to language; one (Raghavendra) reports that a purpose-built meaningful but non-phonetic emblem system receives the same outcome as the Indus corpus on every test it could score exactly; the other (Nair) reports that neither of two calibrated non-linguistic generators reproduces the full Indus profile. I found no peer-reviewed 2020-2026 paper that settles or materially shifts the linguistic question.

### Cited Findings
- Nair, A. 2026. "How Non-Linguistic Is the Indus Sign System? A Synthetic-Baseline Scorecard." arXiv:2604.17828 [cs.CL], submitted 20 April 2026. Preprint. — [arXiv](https://arxiv.org/abs/2604.17828)
  - Corpus (per the preprint): ICIT-derived, 2,511 inscriptions of which "595 (24%) are exact duplicates", leaving 1,916 deduplicated inscriptions, 11,110 tokens, 584 sign types, 52 sites. **[via summariser]** — [arXiv HTML](https://arxiv.org/html/2604.17828v1)
  - Baselines: within-inscription permutation null (1,000 permutations, fixed seed); two synthetic generators (heraldic, administrative) calibrated on six attested non-linguistic corpora (Sproat's kudurrus, totem poles, barn stars, Pictish stones; SCA heraldic blazons; CDLI proto-cuneiform). — [arXiv HTML](https://arxiv.org/html/2604.17828v1)
  - Reported Indus values: mean length 4.42 signs; hapax rate 33.2%; 43 distinct repeated sequences of length ≥ 5; positional Cramér's V 0.15. — [arXiv HTML](https://arxiv.org/html/2604.17828v1)
  - Conclusion wording: "Neither baseline generator reproduces the full observed statistical profile"; "This does not settle the linguistic question. A more carefully tuned non-linguistic model might eventually match the full Indus profile." — [arXiv HTML](https://arxiv.org/html/2604.17828v1)
  - Stated limitations: no validated cross-concordance between sign codings; "If many codes are allographic variants, our hapax rate is inflated"; "all n-gram statistics are estimated from very short sequences"; code "available from the corresponding author ... upon request and will be released as a public repository upon acceptance". — [arXiv HTML](https://arxiv.org/html/2604.17828v1)
- Raghavendra, N. 2026. "On the Non-Specificity of Statistical Measures Used in Script Decipherment." arXiv:2608.02999 [cs.CL], submitted 4 August 2026. Preprint. — [arXiv](https://arxiv.org/abs/2608.02999)
  - Abstract: "Any such inference rests on specificity: the reported outcome must be unusual among plausible structured non-languages." SIGIL is "a purpose-built generative emblem system whose 3,000-text core corpus carries explicit compositional meanings although no sign has a phonological value"; "SIGIL receives the same category as the Indus corpus on every criterion scored this way, across repetition, directional-asymmetry, and lexical-distribution tests." — [arXiv](https://arxiv.org/abs/2608.02999)
  - Per the preprint, a registry of 54 published methods was compiled; only 6 could be scored exactly (Sproat 2014 repetition; two Ashraf & Sinha 2018 terminal-asymmetry tests; three Oakes 2019 LNRE fits). Rao 2009 conditional entropy, Rao 2010 block entropy, Yadav 2010 Zipf-Mandelbrot/perplexity/restoration, Sinha 2011 network measures and the Lee 2010 classifier were only reconstructed descriptively because the sources under-specify them. **[via summariser]** — [arXiv HTML](https://arxiv.org/html/2608.02999)
  - Reported descriptive matches (per the preprint): SIGIL bigram perplexity falls 64.22 → 25.80 against a published Indus 68.82 → 26.69; 84.88% of masked signs recovered within a 90%-mass candidate set against the published Indus 74%. **[via summariser]** — [arXiv HTML](https://arxiv.org/html/2608.02999)
  - Conclusion: "The measures detect organization, predictability, concentration, direction, and model misfit; they do not, on their own, establish encoded speech." — [arXiv HTML](https://arxiv.org/html/2608.02999)
  - Author's own limits: SIGIL "is a single synthetic system, deliberately constructed after the relevant literature was known"; the work "establishes compatibility between the tested outcomes and one defined non-language ... and it supplies no likelihood ratio between linguistic and non-linguistic hypotheses." — [arXiv HTML](https://arxiv.org/html/2608.02999)
- Daggumati, S., Revesz, P. Z. 2021. "A method of identifying allographs in undeciphered scripts and its application to the Indus Valley Script." *Humanities and Social Sciences Communications* 8, article 50 (17 February 2021). Proposes 50 allograph sign pairs (23 mirrored, 27 non-mirrored) from positional statistics, using an ICIT-derived sign list. Peer reviewed; relevant to sign-list sensitivity, not to the linguistic question. **[snippet only; full text blocked]** — [Nature/HSSC](https://www.nature.com/articles/s41599-021-00713-0)
- Fuls, A. 2022 corpus/catalogue volumes expand the ICIT material (see section 5). **[snippet only]** — [ResearchGate: A Catalog of Indus Signs](https://www.researchgate.net/publication/373522673_A_Catalog_of_Indus_Signs)

### Inferences
- The masked-sign result in Raghavendra's preprint is directly relevant to this project's later milestone: high restoration accuracy is reported for a system known to encode no speech. If that holds, masked-prediction accuracy measures predictability of the notation and cannot be offered as evidence about semantics or language. This matches the project's planned "limits of semantic claims" benchmark.
- The two preprints disagree on whether non-linguistic baselines match the Indus profile (Raghavendra: one constructed system does; Nair: two calibrated generators do not). They agree only that the question is open and that structured non-linguistic baselines are the necessary comparison. Neither shows the Indus system is linguistic or non-linguistic.

### Gaps
- Both 2026 items are unreviewed preprints by authors I could not otherwise verify; their numbers reached me through an automatic summariser that made at least one error (it expanded EBUDS wrongly). Cite them as preprints, quote only after reading the PDF, and do not rest a methods claim on them alone.
- Ashraf & Sinha 2018, Oakes 2019 and Sinha et al. 2011 are known to me only as named in the Raghavendra preprint; I did not locate or verify their full citations. An arXiv item "The 'handedness' of language: Directional symmetry breaking of sign usage in words" ([arXiv:1509.09121](https://arxiv.org/pdf/1509.09121)) appeared in search results and may be the Ashraf & Sinha work; unconfirmed.
- I found no peer-reviewed 2020-2026 article with the specific claim "invented or non-linguistic systems reproduce the same n-gram/entropy statistics". If the student has seen such a paper cited, it is most likely one of the two preprints above or Sproat 2014; I could not verify any other.
- Yajnadevam 2024 (a claimed Sanskrit decipherment) is mentioned in the Raghavendra preprint; not examined, out of scope.

## 3. Wording of conclusions and baselines careful authors use

### Takeaway
Defensible wording stops at "structured", "non-random", "predictable from context" and "consistent with [more than one hypothesis]"; every published claim beyond that was challenged for lacking a control that matches sign frequencies and a comparison set of known non-linguistic systems. Three baseline tiers are needed, and each licenses a different, limited statement.

### Cited Findings
- Wording used by the n-gram authors in their preprint: the script is "a structured sign system showing features of a formal language" and the analysis "cannot conclusively establish that it encodes natural language"; information content is "intermediate between that of a completely random and a completely fixed ordering". **[via summariser of abstract]** — [arXiv:0901.3017](https://arxiv.org/abs/0901.3017)
- The published PLoS ONE version ends more strongly: "Our results provide evidence in favor of the linguistic hypothesis for the script but additional work is required to reach a conclusive verdict." — [Yadav et al. 2010](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0009506)
- Rao et al.'s hedge: results "increase the probability that the script represents language"; and "barring a full decipherment, one cannot prove either the linguistic or nonlinguistic thesis". — [Sproat 2010 reply](https://aclanthology.org/J10-4017.pdf); [Rao et al. 2010](https://aclanthology.org/J10-4016.pdf)
- Sproat's limit on what the measures show: neither study showed "that one can distinguish structure that derives from linguistic constraints from structure that derives from some other kind of constraints"; such measures "certainly can tell us some things about a corpus of symbols". — [Sproat 2010](https://aclanthology.org/J10-3012.pdf)
- Raghavendra's wording: measures "detect organization, predictability, concentration, direction, and model misfit". Nair's: "This does not settle the linguistic question." — [arXiv:2608.02999](https://arxiv.org/html/2608.02999); [arXiv:2604.17828](https://arxiv.org/html/2604.17828v1)
- Baseline tier 1, uniform random and rigid order (Rao's Type 1 / Type 2): only bounds the range; criticised as corresponding to no real system. — [Sproat 2010](https://aclanthology.org/J10-3012.pdf)
- Baseline tier 2, frequency-matched memoryless sequences (Zipfian or empirical unigram frequencies, independent draws): the control critics demanded, because it reproduced Rao's Fig. 1A. — [Sproat 2010](https://aclanthology.org/J10-3012.pdf); [Language Log 2009](https://languagelog.ldc.upenn.edu/nll/?p=1374)
- Baseline tier 2b, shuffles of the real corpus. Nair uses a within-inscription permutation preserving each text's composition (1,000 permutations, fixed seed); Raghavendra stores within-text permutations; Lee et al.'s response used corpus randomisations whose scheme is not specified in Sproat's account. Sproat: catches memoryless sources, not structured non-linguistic ones. — [Sproat 2010 reply](https://aclanthology.org/J10-4017.pdf); [arXiv:2604.17828](https://arxiv.org/html/2604.17828v1)
- Baseline tier 3, real comparison corpora: known scripts across genres, and known non-linguistic systems. Sproat's public set includes kudurrus, totem poles, barn stars, weather icons and Pictish stones. — [Sproat corpora](https://richardsproat.com/data/non-linguistic-symbols/); [ScienceDaily 2014](https://sciencedaily.com/releases/2014/06/140602101559.htm)
- Comparison corpora must be matched in size to the Indus sample: Rao's comparison samples were small "because the Indus corpus against which all other symbol systems are compared is very small". — [Sproat 2010](https://aclanthology.org/J10-3012.pdf)
- A decision rule and an uncertainty estimate are expected, not a plot to "eyeball". — [Sproat 2010 reply](https://aclanthology.org/J10-4017.pdf)
- Held-out evaluation used by the n-gram authors: 5 folds, 80:20 split, sign restoration sensitivity 74% (about 75% in the abstract). — [Yadav et al. 2010](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0009506)

### Inferences
- Suggested claim-to-control mapping for the methods section:
  - "Sign order is non-random" needs tier 2 and 2b, reported as observed statistic against the null distribution with seeds and number of draws.
  - "Bigram/trigram context improves prediction" needs held-out perplexity or accuracy against a unigram model on the same split.
  - "Positional regularities exist" needs a shuffle that preserves text length and sign frequencies.
  - "The system resembles language" should not be claimed; if comparison corpora are shown, include non-linguistic ones and state that overlap with them is expected.
- Preferred verbs: "consistent with", "shows", "departs from the null by". Avoid: "syntax", "grammar", "words", "linguistic structure", "meaning". "Formal-language-like" is used by Yadav et al. but invites the same objection; "sequential and positional regularity" is neutral.
- Comparison to language corpora at matched size and with the same estimator is needed even for descriptive plots, since entropy estimates depend on corpus size, genre and encoding.

### Gaps
- I found no published checklist or community standard for these analyses; the tiering above is my synthesis of the cited critiques.
- I did not verify which non-linguistic corpora in Sproat's download are large enough for trigram statistics.

## 4. Corpus preparation practice: duplicates, damage, doubtful readings, multi-line texts, single signs, site imbalance; what EBUDS did

### Takeaway
The only documented preparation recipe for M77 is Yadav et al.'s EBUDS: drop every text with any lost, damaged, illegible or doubtful sign, drop multi-line texts, keep one copy of each repeated text, giving 1,548 texts and 377 signs; no intermediate counts were published. The project's export does not reproduce 1,548/377 under any straightforward reading of those rules, so the student must define and report their own filter rather than claim to use EBUDS.

### Cited Findings
**What EBUDS did and why (published)**
- Source corpus: M77 "records 417 unique signs in 3573 lines of 2906 texts". — [Yadav et al. 2010](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0009506)
- Rule: EBUDS "is created by removing from the concordance all texts containing lost, damaged, illegible, or doubtfully read parts ... Texts spread across multiple lines are also removed. For texts that occur more than once, only a single copy is retained." Stated purpose: "to remove ambiguity". (EBUDS = Extended Basic Unique Data Set.) — [Yadav et al. 2010](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0009506)
- Reason for dropping multi-line texts: "it is not clear if each line of a multi-line text is to be treated as a single text". **[via summariser quote]** — [Yadav et al. 2010](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0009506)
- Result: "The EBUDS dataset contains 1548 texts"; "40 signs out of 417 ... do not make their appearance" (377 signs). — [Yadav et al. 2010](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0009506)
- Representativeness claim: "the relative frequency distribution of signs in EBUDS is comparable to M77 and hence EBUDS is a true representation of M77". — [Yadav et al. 2010](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0009506)
- Boundaries: "We introduce two additional tokens # and $ which indicate the beginning and end of a text respectively." Texts of length 3 and 5 are most frequent. — [Yadav et al. 2010](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0009506)
- Not stated in the paper: counts removed per category, total EBUDS tokens, per-site breakdown, treatment of single-sign texts, sensitivity to the sign list. — [Yadav et al. 2010](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0009506)
- Size elsewhere: "about 1,550 lines of text and 7,000 sign occurrences". — [Rao et al. 2010](https://aclanthology.org/J10-4016.pdf)
- Damaged texts were not training data. The restoration figure (abstract: "about 75%"; cross-validation sensitivity 74%) comes from masking signs in held-out clean texts under 5-fold validation, with a hit counted when the true sign falls in the 90%-probability candidate set. It was not scored on actual doubtful readings, which have no ground truth. — [Yadav et al. 2010](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0009506)
- A different corpus, different practice: Nair removes exact duplicates only (595 of 2,511 ICIT inscriptions, 24%), noting duplicates "may represent genuinely repeated formulae or catalog artifacts". — [arXiv:2604.17828](https://arxiv.org/html/2604.17828v1)

**Counts from the project's export [computed here]** — all from [the pinned export](https://raw.githubusercontent.com/paragorth/Indus-/ed56b9dc293f26a4e6bddea36c9a394e6b211ff4/data/im77/im77_corpus_lines.csv), sha256 `385b82bd…f26211`
- The file has 3,916 data rows; 343 have no signs (direction "unrecorded"). The 3,573 non-empty rows match the published line count. 2,906 texts, 13,372 legible tokens, 781 illegible markers (sign `0`), 504 doubtful positions, 417 sign types: all match the brief.
- Lines per text (non-empty): 1 line 2,299 texts; 2 lines 560; 3 lines 39; 4 lines 5; 5 lines 2; 7 lines 1. So 607 texts (20.9%) are multi-line. 493 texts span more than one side.
- Text length (legible plus illegible signs, all lines): mean 4.87, median 5, max 26; 69 texts are a single sign. Line length: mean 3.61, max 14; 215 single-sign lines.
- Damage: 648 texts contain at least one illegible sign; 423 at least one doubtful reading; 890 (30.6%) either.
- Duplicates, defined as identical ordered sequences of non-empty lines: 2,339 distinct texts; 567 texts (19.5%) are redundant copies; 825 texts sit in 258 duplicate groups; largest groups 22, 19, 15, 11, 11.
- Share of texts in a duplicate group by object type: copper tablets 104/135 (77%); miniature tablets 146/272 (54%); sealings 236/511 (46%); ivory/bone rods 9/29 (31%); pottery graffiti 37/119 (31%); seals 290/1,814 (16%); bronze implements 0/11.
- Sites, by texts: Mohenjo-daro 1,540 (53.0%); Harappa 985 (33.9%); Lothal 165 (5.7%); Kalibangan 99 (3.4%); Chanhu-daro 66 (2.3%); other sites 34 (1.2%); West Asian finds 17 (0.6%). By legible tokens: 54.4%, 32.6%, 5.7%, 3.2%, 2.4%, 1.1%, 0.6%. Mohenjo-daro plus Harappa hold 87% of both.
- Object types: seal 1,814; sealing 511; miniature tablet 272; copper tablet 135; pottery graffito 119; ivory/bone rod 29; miscellaneous 15; bronze implement 11.
- Directions (rows): right-to-left 2,974; left-to-right 235; single sign 190; no text 155; doubtful 12; symmetrical 7; unrecorded 343 (the empty rows).
- 112 of 417 sign types occur once (26.9%); most frequent sign 342 with 1,395 tokens, then 99 (649), 59 (381), 267 (376), 87 (365).
- EBUDS-style filters applied to this export do not give 1,548 texts / 377 signs:

| Filter (then de-duplicate) | Units | Tokens | Sign types |
|---|---|---|---|
| Single-line texts, no illegible, no doubtful | 1,291 | 6,099 | 347 |
| Single-line texts, no illegible (doubtful kept) | 1,421 | 6,780 | 357 |
| All lines as units, no illegible, no doubtful | 1,673 | 7,448 | 393 |
| All lines as units, no illegible (doubtful kept) | 1,849 | 8,313 | 398 |
| Published EBUDS | 1,548 | about 7,000 | 377 |

### Inferences
- Types versus tokens is a substantive choice, not housekeeping. The 567 exact repeats are of two kinds: copies sharing one physical source (moulded tablets, sealings from one seal) and distinct objects carrying the same text, which are independent attestations of a formula (Nair's "repeated formulae or catalog artifacts"). The export does not separate the two. Keeping all repeats inflates the counts of a few sequences (one text occurs 22 times) and makes held-out evaluation leak, since a copy of a test text is in training. Report sequence statistics on de-duplicated texts as primary and on all tokens as a sensitivity check; split train/test by distinct text, never by copy.
- De-duplication also changes the object-type mix: it removes most copper and miniature tablets and few seals. Say so.
- Illegible signs, three options and their costs:
  - Drop the whole text (EBUDS). Cleanest, loses 22% of texts for illegibles alone.
  - Split at the gap and treat fragments as texts. This creates false text-initial and text-final positions, so it corrupts the positional statistics Milestone 2 depends on. Avoid, or mark fragment ends as unknown rather than as boundaries.
  - Keep the text with an explicit gap token that contributes no bigram across it and is excluded from position-from-end counts when the gap lies between the sign and that end. Keeps the most data; needs careful bookkeeping.
- Doubtful readings: EBUDS drops them. An alternative is to keep them and run every statistic with and without; the export's `doubtful_positions` field makes that cheap. These 423 texts are natural held-out material for the masked-sign model, but the "true" label is itself uncertain, so they cannot serve as ground truth.
- Multi-line and multi-sided texts: there is no evidence that reading continues across lines or sides, which is why EBUDS dropped them. The defensible default is one sequence per line with boundary tokens, and never to count a bigram across a line or side break. Dropping all 607 multi-line texts is the conservative variant; concatenation is not supported by any source found.
- Single-sign texts (69 texts, 215 lines) contribute to unigram and boundary counts but to no sign-to-sign transition. Keep them for frequency, exclude them from transition and positional analyses, and state the count.
- Site imbalance: with 87% of tokens from two cities, any "Indus" statistic is mostly a Mohenjo-daro and Harappa statistic. Report per-site results for those two, pool the rest, and avoid claims about regional variation from 17 West Asian texts. Yadav et al. give no per-site treatment.
- The mismatch table means either EBUDS used rules not fully stated, or the export differs from the file Yadav et al. used (segmentation of lines, what counts as doubtful). Either way the project cannot say it uses EBUDS; it can say "an EBUDS-style filter" and publish its own counts.

### Gaps
- No published per-category removal counts for EBUDS, and no published number of duplicate texts in M77, were found; the duplicate figures above are this project's own.
- Mahadevan's 1977 introduction was not accessible, so his own conventions for doubtful signs, line order, side numbering and what counts as one "text" are unverified. Check the export's field semantics (e.g. `side` = 0, line = 0) against [indusscript.in](https://indusscript.in/) or the printed concordance before relying on them.
- 155 rows carry direction "no text" yet contain signs; what that label means in the export is unresolved.
- The export is a third-party transcription; its agreement with the printed concordance beyond the headline totals is untested.
- I found no published study comparing drop / split / gap-token treatment of illegible signs on Indus data.
- The site counts above are computed, not quoted from Mahadevan; a published breakdown was not retrieved.

## 5. Sign-list issues: Mahadevan versus Parpola versus Wells/Fuls

### Takeaway
Published inventories range from about 386 to over 700 signs depending on how allographs and compound signs are treated, and every type-based statistic (hapax rate, entropy, vocabulary size, Zipf fit) moves with that choice. No systematic published sensitivity analysis of the n-gram or entropy results to the sign list was found.

### Cited Findings
- Inventory sizes as reported (sources disagree; not reconciled here):
  - Mahadevan 1977: 417 signs — [Yadav et al. 2010](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0009506); given as 419 signs with 459 variants, 2,911 objects, in Fuls's comparison table **[snippet only]** — [Fuls 2022, Corpus of Indus Inscriptions](https://www.scribd.com/document/826339135/Fuls2022-Corpus-of-Indus-Inscriptions-view-inside)
  - Parpola: 386 (1994) — [ResearchGate: A Catalog of Indus Signs](https://www.researchgate.net/publication/373522673_A_Catalog_of_Indus_Signs) **[snippet only]**; 394 signs with 281 variants (1982), 3,265 objects — [Fuls 2022](https://www.scribd.com/document/826339135/Fuls2022-Corpus-of-Indus-Inscriptions-view-inside) **[snippet only]**; "~398" P-codes — [arXiv:2604.17828](https://arxiv.org/html/2604.17828v1)
  - Wells: 610 signs (1999), 676 signs, 694 signs (ICIT 2015; 3,903 objects, 4,794 texts); Fuls 2022: 709 signs, 4,660 objects, 5,644 texts. **[snippet only]** — [Fuls 2022](https://www.scribd.com/document/826339135/Fuls2022-Corpus-of-Indus-Inscriptions-view-inside); [ResearchGate catalogue](https://www.researchgate.net/publication/373522673_A_Catalog_of_Indus_Signs)
  - ICIT at another date: 4,537 objects, 5,509 texts, 19,616 sign occurrences; site reported down as of June 2021. — [Digital Classicist Wiki](https://wiki.digitalclassicist.org/Interactive_Corpus_of_Indus_Texts)
- Wells built his list without reference to Mahadevan's or Parpola's, adding concordances afterwards; the argument for the long list is that, absent evidence that two forms are allographs, keeping them apart is "the most secure way forward, especially for low frequency signs". **[snippet only]** — [harappa.com, A Catalog of Indus Signs II](https://www.harappa.com/content/catalog-indus-signs-ii)
- Criticism of M77 from the ICIT side: "the merging of graphically similar but structurally distinct signs into one sign, which reduces the number of distinct signs in the sign list to only 417", plus a misleading artefact typology. **[snippet only; exact source document not confirmed, appears to be Fuls]** — [ResearchGate: Corpus of Indus Inscriptions](https://www.researchgate.net/publication/373522666_Corpus_of_Indus_Inscriptions)
- Parpola's list is said to be "in general accordance with the signs listed in M77 with a few exceptions". **[snippet only]** — [Yadav & Vahia, Indus Script: A Study of its Sign Design](https://www.harappa.com/sites/default/files/pdf/Indus-sign-design.pdf)
- "Nearly 27 percent" of the 417 signs occur only once. **[snippet only]** Cross-check [computed here]: 112/417 = 26.9% in the export. — [Yadav & Vahia](https://www.harappa.com/sites/default/files/pdf/Indus-sign-design.pdf); [export](https://raw.githubusercontent.com/paragorth/Indus-/ed56b9dc293f26a4e6bddea36c9a394e6b211ff4/data/im77/im77_corpus_lines.csv)
- Rao et al. treat ligatures and modifier marks as a script-like property ("Multiple symbols are sometimes combined ('ligatured') to form a single glyph"); in M77 a compound is one sign number. — [Rao et al. 2010](https://aclanthology.org/J10-4016.pdf)
- Sensitivity is acknowledged, not measured: Nair's 584 ICIT types "should not be directly compared to published counts from other schemes"; "No fully validated cross-concordance exists"; allograph conflation would inflate the hapax rate. — [arXiv:2604.17828](https://arxiv.org/html/2604.17828v1)
- Raghavendra: "The Indus corpora behind the source studies differ in cleaning, orientation, and inventory". — [arXiv:2608.02999](https://arxiv.org/html/2608.02999)
- Daggumati & Revesz 2021 propose 50 allograph pairs from positional behaviour, i.e. a data-driven merge that would lower the type count. **[snippet only]** — [HSSC 8:50](https://www.nature.com/articles/s41599-021-00713-0)
- Yadav et al. 2010 do not discuss sign-list or allograph sensitivity. — [Yadav et al. 2010](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0009506)
- Rao et al. normalise block entropy by log base 417, so the inventory size enters their headline plot directly. — [Rao et al. 2010, Fig. 1 caption](https://aclanthology.org/J10-4016.pdf)

### Inferences
- Direction of bias: over-splitting (Wells) raises type count, hapax share and unigram entropy and thins bigram counts; over-merging (the charge against M77) lowers them and can create spurious regularity by pooling distinct signs. Token-level results on frequent signs should be the most robust; anything about rare signs or vocabulary growth is the least.
- The project is tied to M77 numbering by its data. Honest handling is to state that results are conditional on the M77 sign list, and to add one cheap robustness check: merge a documented set of candidate allographs (or collapse signs below a frequency threshold into a rare-sign class) and report how each headline statistic moves.
- The project's CISI files carry a Parpola-to-Mahadevan cross-reference for 179 Mohenjo-daro artefacts; that is enough for a spot check of sign-list dependence on a subset, not for a full re-run.
- Because M77 codes each compound as one sign, sub-sign structure is invisible to sequence statistics here; say so rather than treat compounds as atomic by assumption.

### Gaps
- No published quantitative sensitivity analysis of Indus entropy or n-gram results across the Mahadevan, Parpola and Wells lists was found. This is a real gap in the literature as far as I could determine, not just in my search.
- A figure "133 compound signs, of which 28 are treated distinctly" appeared in a search summary with no identifiable source; unverified, do not cite.
- Wells 2015 and Fuls 2022 were not read directly; their numbers above come from search snippets and conflict in places (676 vs 694; 417 vs 419). Verify against the books.
- Mahadevan's own account of his allograph and compound-sign decisions (1977 introduction) was not accessed.
- ICIT access terms and current availability are unconfirmed.

## 6. Reproducibility expectations for a replication

### Takeaway
The debate itself supplies the reporting standard: the critics' central complaints were undisclosed artificial controls, unspecified corpora and filters, eyeballed plots, and results that "can't be replicated ... unless you use exactly the same texts". A replication should pin the corpus by hash, publish counts after every filter step, fix and report seeds, and state the estimator and unit of analysis.

### Cited Findings
- Controls disclosed only in a supplement drew criticism: the artificial nature of Type 1/Type 2 is learned "only if one reads the supplementary material". — [Sproat 2010](https://aclanthology.org/J10-3012.pdf)
- Exact-text dependence: results "can't be replicated even in principle unless you use exactly the same texts used in their plot". — [Farmer, Sproat & Witzel 2009 slides](https://safarmer.com/more.on.Rao.pdf)
- EBUDS was published without intermediate counts, token total or single-sign policy, and this project's export does not reproduce its 1,548/377 (section 4). — [Yadav et al. 2010](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0009506)
- In a 54-method registry only 6 methods could be scored exactly; the rest were under-specified: "Several analyses are reconstructions because the publications omit operational details ... another reasonable reconstruction could produce different continuous values." **[per the preprint]** — [arXiv:2608.02999](https://arxiv.org/html/2608.02999)
- Practices reported by recent work: fixed seed and stated permutation count (1,000) for the shuffle null; stored permutation and bootstrap draws; pinned generator seeds; a deterministic corpus artefact; external code cited by commit revision. — [arXiv:2604.17828](https://arxiv.org/html/2604.17828v1); [arXiv:2608.02999](https://arxiv.org/html/2608.02999)
- Nair's code is "available from the corresponding author ... upon request and will be released as a public repository upon acceptance"; not yet a public release as of the preprint. — [arXiv:2604.17828](https://arxiv.org/html/2604.17828v1)
- Released data and tools enable checking and correction: Sproat published corpora, scripts and errata (including a mis-named statistical test). — [Sproat corpora page](https://richardsproat.com/data/non-linguistic-symbols/)
- Estimator and sample-size disclosure: Rao et al. name the NSB estimator and state where estimates become unreliable (block size > 6); Sproat notes smoothing "can only do so much" at 7,000 tokens. — [Rao et al. 2010](https://aclanthology.org/J10-4016.pdf); [Sproat 2010](https://aclanthology.org/J10-3012.pdf)
- Evaluation protocol disclosure: 5-fold, 80:20. — [Yadav et al. 2010](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0009506)
- A decision rule stated in advance, with accuracy on known systems, is what Sproat asks for in place of visual comparison. — [Sproat 2010 reply](https://aclanthology.org/J10-4017.pdf)

### Inferences
Minimum reporting list for the methods section (my synthesis from the above):
- Corpus identity: source (third-party export of M77 from indusscript.in), URL, commit `ed56b9dc…`, file sha256 `385b82bd…f26211`, date retrieved, licence status, sign list (M77, 417 signs), and the meaning of sign `0`.
- Unit of analysis: line or text; how sides and lines are ordered; boundary tokens; reading-order normalisation applied (cross-refer to the direction notes written by the other researcher).
- Filter funnel, one row per step, each with texts, lines, tokens, types: raw; minus empty rows; minus illegible; minus doubtful; minus multi-line (if applied); de-duplicated; minus single-sign (for transition analyses). Publish the surviving text IDs.
- Duplicate definition in one sentence, and both type-level and token-level results.
- Per-site and per-object-type counts after filtering.
- Randomness: seed(s), number of shuffles or bootstrap draws, the exact null (what is preserved: text lengths, unigram frequencies, within-text composition), and the split assignment file for train/test, with splitting done on distinct texts.
- Estimators: name each entropy or probability estimator and its settings; give sample sizes next to every estimate; give intervals from bootstrap over texts.
- Comparison corpora: source, size, how truncated to Indus size, how tokenised.
- Negative and non-matching results, e.g. the failure to reproduce EBUDS's 1,548/377.
- Code and environment: repository commit, package versions, one command to regenerate every table.
- A statement of what is not claimed: no decipherment, no inference to language or to non-language.

### Gaps
- No field-specific reproducibility standard or checklist for computational epigraphy of undeciphered scripts was found; the list above is inferred from the critiques, not quoted from a guideline.
- The export's licence is unclear (project README says so), which limits redistributing filtered text lists; publishing text IDs and hashes rather than sign sequences is a workaround I have not seen discussed in the literature.
