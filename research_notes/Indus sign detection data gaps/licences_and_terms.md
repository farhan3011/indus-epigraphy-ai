# Licences and terms of use for Indus script data sources (checked 1 October 2026)

Factual reporting of stated terms, not legal advice. "Verified directly" means the licence file, API metadata or page was fetched on 2026-10-01. Pages that could not be opened are flagged.

## Which sources can be redistributed inside a public repo, which must be downloaded by each user, and which are only usable under research/fair-dealing exceptions?

### Takeaway
The items with an open licence and no third-party photographs are the ones that can go straight into the repo: mayig CISI JSON (MIT), DM-BiCLab and tpsatish95 code (CC0-1.0, Apache-2.0), the Hugging Face synthetic set (CC-BY-4.0), Met and Cleveland CC0 images, and Wikimedia Commons files under their individual licences. Two sources carry an open licence label over images whose upstream rights are not shown: IM-417-150 ("Apache 2.0", image provenance unstated) and the DM-BiCLab sample images (CC0, but the paper says the images come from CISI and Mahadevan 1977). For both, download-by-user is the cautious route. indusscript.in, Harappa.com, the British Museum, the CISI volumes and Mahadevan 1977 carry no redistribution grant and should not be committed.

### Cited Findings

**Summary table**

| Source | Stated licence (exact) | Where stated | Commit raw files to public MIT repo? | Attribution |
|---|---|---|---|---|
| Kaggle storesource/im-417-150 | "Apache 2.0" | Kaggle dataset metadata | Licence permits it; provenance of crops unstated, so fetch-by-user is safer | Apache-2.0 notice; dataset asks label citation to Mahadevan 1977 pp. 32-34 |
| mayig/indus-valley-script-corpus | "MIT License" (LICENSE); Cargo.toml says "MIT OR Apache-2.0" | LICENSE, Cargo.toml, README | Yes | Keep "Copyright (c) 2024 Michael Carlson" + MIT text |
| DM-BiCLab repo | "CC0 1.0 Universal" | LICENSE | Code: yes. Sample seal/grapheme images: licence label permits it, but paper says images derive from CISI and Mahadevan 1977, so fetch-by-user is safer | None required |
| Dixit et al. 2025, JCAA | "CC Attribution 4.0" (CC BY 4.0) | Article page | Yes for the text; figures reproducing CISI/Mahadevan seal photographs may not be the authors' to license | Required |
| tpsatish95/OCR-on-Indus-Seals | "Apache License 2.0" | LICENCE file | Yes (code) | Apache-2.0 notice |
| arXiv:1702.00523 | arXiv non-exclusive distribution licence 1.0 | arXiv abstract page | No (link to it) | n/a |
| HF hellosindh/indus-script-synthetic | "cc-by-4.0" | Dataset card YAML | Yes | Required |
| indusscript.in | None found | n/a | No | App gives a citation string |
| Harappa.com | "© Harappa.com 1995-2026 ... may not be used commercially without written permission" | About Us: Credits page | No | Ask permission |
| Wikimedia Commons | Per file: Public domain, CC BY-SA 4.0, CC BY-SA 3.0, CC0 | Each file page | Yes, per-file terms | Required for CC BY-SA; share-alike applies |
| Met Museum | CC0 (Open Access items) | API `isPublicDomain` flag | Yes | Not required |
| Cleveland Museum of Art | CC0 | API `share_license_status` | Yes | Not required |
| British Museum | "Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)" for content marked with the CC logo | BM copyright and permissions page (archived copy) | No (NC + SA conflict with MIT repo) | Required |
| CISI vols 1-3 | No open licence found | n/a | No | n/a |
| Mahadevan 1977 (ASI) | No open licence found | n/a | No | n/a |

**Kaggle storesource/im-417-150**
- Kaggle metadata gives `licenseName: "Apache 2.0"`; the Croissant record gives licence URL https://www.apache.org/licenses/LICENSE-2.0. Verified directly through the Kaggle API and Croissant endpoints. — [Kaggle API record](https://www.kaggle.com/api/v1/datasets/view/storesource/im-417-150); [dataset page](https://www.kaggle.com/datasets/storesource/im-417-150)
- Description, verbatim: "Original 417 characters listed in "The Indus Scripts: Texts, Concordance and Tables" compiled by Iravatham Mahadevan, hence calling this dataset "IM-417-150". Label Citation: Pages 32-34 from "The Indus Scripts: Texts, Concordance and Tables" compiled by Iravatham Mahadevan, Published by Archaeological Survey of India (1977). Characters were extracted from images of seals that belong to Indus Valley Civilisation." — [Kaggle API record](https://www.kaggle.com/api/v1/datasets/view/storesource/im-417-150)
- Kaggle's download API requires each user's own Kaggle account and API token, so a fetch script works per user. Version 2, last updated 2024-03-23, about 20.5 MB, PNG and CSV files, 672 downloads at time of check. — [Kaggle API record](https://www.kaggle.com/api/v1/datasets/view/storesource/im-417-150)
- The description does not say which seal images the crops came from or who photographed them. — [Kaggle API record](https://www.kaggle.com/api/v1/datasets/view/storesource/im-417-150)

**mayig/indus-valley-script-corpus (CISI digitisation)**
- GitHub detects "MIT License" (SPDX MIT). LICENSE reads "MIT License / Copyright (c) 2024 Michael Carlson". Last push 2025-04-16. — [LICENSE](https://github.com/mayig/indus-valley-script-corpus/blob/main/LICENSE)
- Cargo.toml declares `license = "MIT OR Apache-2.0"`; README says "By releasing this corpus under the MIT and/or Apache licenses, it is my hope that others can use this data in whatever way they wish". Only an MIT LICENSE file is present at top level. — [README](https://github.com/mayig/indus-valley-script-corpus/blob/main/README.md); [Cargo.toml](https://github.com/mayig/indus-valley-script-corpus/blob/main/Cargo.toml)
- README describes it as "a (currently WIP) digitization of the Corpus of Indus Seals and Inscriptions (CISI) by Parpola et al." in JSON, using CISI text numbers and Parpola (1982) and Wells (2015) sign lists. It is a transcription (sign codes plus feature vectors), not photographs. The README states no permission from the CISI publishers. — [README](https://github.com/mayig/indus-valley-script-corpus/blob/main/README.md)

**DM-BiCLab repo and Dixit et al. 2025**
- GitHub detects "Creative Commons Zero v1.0 Universal" (CC0-1.0); LICENSE opens "Creative Commons Legal Code / CC0 1.0 Universal". Last push 2025-02-23. — [LICENSE](https://github.com/DM-BiCLab/Deep-Learning-in-Archiving-Indus-Script-and-Motif-Information/blob/main/LICENSE)
- README: "The full data sets may be obtained from authors upon request. To facilitate understanding and reproducibility, we have provided sample data of each data type." The repo holds notebooks and a `Sample Data` folder (Sample_Grapheme, Sample_IVC_Seal, Sample_Motif), not the full dataset. — [README](https://github.com/DM-BiCLab/Deep-Learning-in-Archiving-Indus-Script-and-Motif-Information/blob/main/README.md)
- Paper: Dixit, Hussain, Basak, Atturu, Mitra, Bhattacharya, "Deep Learning in Archiving Indus Script and Motif Information", JCAA 8(1), pp. 156-169, published 2025-05-19, DOI 10.5334/jcaa.175. Licence shown on the article page: "CC Attribution 4.0". — [JCAA article](https://journal.caa-international.org/articles/10.5334/jcaa.175)
- The paper's seal images come from the CISI volumes (Joshi and Parpola 1987) and Mahadevan (1977): 1,004 images collected, 963 after de-duplication, named "FIT_ISI_Corpora". The article page as read does not describe permissions from those publishers. — [JCAA article](https://journal.caa-international.org/articles/10.5334/jcaa.175)

**tpsatish95/OCR-on-Indus-Seals and arXiv:1702.00523**
- GitHub detects "Apache License 2.0"; file is named `LICENCE`. Top level contains `code`, `paper`, `slides`. Last push 2017-02-04. — [repo](https://github.com/tpsatish95/OCR-on-Indus-Seals)
- README acknowledgements: "We acknowledge the generous assistance of Suresh Babu of the Indus Research Centre and G. Sundar of the Roja Muthiah Research Library in providing images of the Indus seals. We thank Harappa.com for their kind permission to use an image of the Indus seal in this paper." So the seal images were supplied privately by RMRL, and Harappa.com permission covered one image in the paper. — [README](https://github.com/tpsatish95/OCR-on-Indus-Seals/blob/master/README.md)
- The arXiv abstract page links the licence http://arxiv.org/licenses/nonexclusive-distrib/1.0/ (arXiv's non-exclusive licence to distribute), not a Creative Commons licence. — [arXiv:1702.00523](https://arxiv.org/abs/1702.00523)

**Hugging Face hellosindh/indus-script-synthetic**
- Dataset card YAML: `license: cc-by-4.0`; the card's BibTeX repeats `license = {CC-BY-4.0}`. Not gated, public. Created 2026-04-10. Files: `synthetic_indus_5k.csv`, `reference/sign_index.json`, README. — [dataset card](https://huggingface.co/datasets/hellosindh/indus-script-synthetic)
- The card says the 5,000 synthetic sequences were built from "3,310 real archaeological inscriptions" and that 752 generated sequences exactly matched real inscriptions and were separated out. The card does not name the source corpus of the 3,310 real sequences. — [dataset card](https://huggingface.co/datasets/hellosindh/indus-script-synthetic)
- The companion model repo hellosindh/indus-script-models also declares `license: cc-by-4.0`. — [model card](https://huggingface.co/hellosindh/indus-script-models)

**indusscript.in (RMRL Indus Research Centre, IM77/IDF80 web app)**
- No licence or terms text could be found. The site is a Flutter single-page app on Firebase; the app's "Terms and Conditions" link points to https://indusscript.in/terms.html, which returns only the app shell (3,502-byte HTML, title "The Indus Script(IM77)") with no terms text. — [terms.html](https://indusscript.in/terms.html)
- Text in the app bundle: "With the support of the Department of Archaeology, Govt. of Tamilnadu, the Roja Muthiah Research Library (RMRL) is publishing Thiru. Mahadevan's updated corpus (IM77/IDF80) as a web application for use by researchers, students and interested members of the general public." Suggested citation in the bundle: "Mahadevan, Iravatham & Roja Muthiah Research Library (2021), The Indus Script Web Application(IM77/IDF80). https://indusscript.in". Stated source: "Mahadevan, 1977. The Indus Script: Texts, Concordance and Tables, Memoirs of the ASI, No.77 & IDF80 (Unpublished)". — [app bundle main.dart.js](https://indusscript.in/main.dart.js)
- The bundle also says "No signup is required for the application", with optional Google login. — [app bundle main.dart.js](https://indusscript.in/main.dart.js)

**Harappa.com**
- About Us: Credits page (Wayback capture 2026-06-20): "Copyright: All images and content are © Harappa.com 1995-2026 and may not be used commercially without written permission." — [archived About Us: Credits](https://web.archive.org/web/20260620011806/https://www.harappa.com/content/about-us-credits)
- A search-engine summary of the same page adds that photographs are watermarked, that images "may not be used for any purpose without asking for permission", and that many photographs are credited "Courtesy, Dept. of Archaeology and Museums, Govt. of Pakistan". These extra sentences were not confirmed against the page text. — [harappa.com/about-us](https://harappa.com/about-us)

**Wikimedia Commons**
- Relevant categories that exist: "Indus Valley Civilization seals", "Seals of the Indus Valley Civilization", "Indus civilization seals", "Indus civilization seal impressions", "Indus Valley cylinder seals", "Unicorns on seals of the Indus Valley Civilization", "Pashupati seal", "Indus script tablets". — [Commons category](https://commons.wikimedia.org/wiki/Category:Indus_Valley_Civilization_seals)
- Category "Indus civilization seals" holds 51 files directly, tagged: CC BY-SA 3.0 14, Public domain 12, CC BY-SA 4.0 12, CC0 9, CC BY 3.0 2, CC BY 4.0 1, CC BY-SA 2.0 1 ([category API query](https://commons.wikimedia.org/w/api.php?action=query&generator=categorymembers&gcmtitle=Category:Indus_civilization_seals&gcmtype=file&gcmlimit=500&prop=imageinfo&iiprop=extmetadata&iiextmetadatafilter=LicenseShortName&format=json)). "Indus Valley Civilization seals" and "Seals of the Indus Valley Civilization" hold no files directly. A separate sample of 50 keyword-search hits gave: Public domain 17, CC BY-SA 4.0 14, CC BY-SA 3.0 10, CC0 9. Licence is per file and is exposed through the MediaWiki API (`prop=imageinfo&iiprop=extmetadata`, field `LicenseShortName`). — [Commons API](https://commons.wikimedia.org/w/api.php?action=query&generator=search&gsrnamespace=6&gsrsearch=Indus+seal+Harappa+OR+Mohenjo-daro+seal&gsrlimit=50&prop=imageinfo&iiprop=extmetadata&iiextmetadatafilter=LicenseShortName&format=json)

**CISI volumes and Mahadevan 1977**
- CISI vol. 1 (Joshi and Parpola 1987, Collections in India) was published by Suomalainen Tiedeakatemia in Annales Academiae Scientiarum Fennicae and doubles as a Memoir of the Archaeological Survey of India; vol. 3.1 (2010) is AASF Humaniora 359 / Memoirs of the ASI No. 96. — [Cambridge review of vol. 3.1](https://www.cambridge.org/core/journals/journal-of-the-royal-asiatic-society/article/abs/corpus-of-indus-seals-and-inscriptions-volume-3-new-material-untraced-objects-and-collections-outside-india-and-pakistan-part-1-edited-by-asko-parpola-b-m-pande-and-petteri-koskikallio-in-collaboration-with-richard-h-meadow-and-j-mark-kenoyer-with-the-assistance-of-erja-lahdenpera-jyrki-lyytikka-and-arto-vuohelainen-supplement-to-mohenjodaro-and-harappa-suomalaisen-tiedeakatemian-toimituksiaannales-academiae-scientiarum-fennicae-humaniora-359-memoirs-of-the-archaeological-survey-of-india-no-96-helsinki-suomalainen-tiedeakatemia-2010/B9CCB96DD82777E486D2FF40B10B0052)
- CISI vol. 1 is readable as an e-book on the Government of India's Indian Culture portal, sourced from the Archaeological Survey of India. Readable online is not a redistribution licence; no licence statement was found. — [Indian Culture portal](https://www.indianculture.gov.in/ebooks/corpus-indus-seals-and-inscriptions)
- Vol. 3.3 was published in 2022 (Parpola and Koskikallio eds). — [Academia.edu contents and preface](https://www.academia.edu/94861598/Parpola_A_and_P_Koskikallio_eds_2022_Corpus_of_Indus_Seals_and_Inscriptions_vol_3_3_contents_and_preface)
- Indian Copyright Act 1957, s. 28: for a government work where the government is first owner, copyright lasts "until sixty years from the beginning of the calendar year next following the year in which the work is first published". — [Copyright law of India](https://en.wikipedia.org/wiki/Copyright_law_of_India)
- Indian Copyright Act s. 52(1)(a) permits fair dealing with a work for "private or personal use, including research". — [Mondaq on s. 52](https://www.mondaq.com/india/copyright/1184954/exception-to-infringement-of-copyright-section-52-of-the-copyright-act-1957)

### Inferences
- Three tiers for the project:
  - **Commit to repo:** mayig JSON (keep its MIT notice), HF synthetic CSV (credit per CC BY 4.0), Met and Cleveland CC0 images, Commons files that are PD/CC0. Commons CC BY-SA files can be included but stay under BY-SA, so they need a separate licence note and cannot be relicensed as MIT.
  - **Download by each user via script:** IM-417-150 (Kaggle), DM-BiCLab sample images, British Museum images, any Commons BY-SA files if the repo is to stay licence-clean.
  - **Research / fair-dealing use only, nothing committed:** indusscript.in content, Harappa.com photographs, CISI plates, Mahadevan 1977 pages, the full FIT_ISI_Corpora (obtainable only on request from the authors).
- Mahadevan 1977 is an ASI Memoir. If it counts as a government work under s. 28, the 60-year term runs to the end of 2037. If ASI is not first owner, the ordinary term (author's life plus 60 years) applies and runs later still. Either way it is in copyright in 2026. Which branch applies was not confirmed from the book itself.
- CISI volumes date from 1987, 1991, 2010 and later, with named editors and photographers; all are well within any copyright term. No open licence was found for them.
- The licences on mayig (MIT), IM-417-150 (Apache 2.0) and DM-BiCLab (CC0) are grants by the uploaders over their own contribution. None states permission from the CISI publishers, ASI, or the photographers. mayig is transcription data (sign codes), not photographs. IM-417-150 is image crops of unstated photographic origin. DM-BiCLab's sample images have a known third-party origin (CISI and Mahadevan 1977 per the paper), which makes it the clearest case of an open label over material the uploader may not own.
- The HF synthetic set's CC BY 4.0 covers the generated sequences; the unnamed 3,310-sequence real corpus behind it is not itself released or licensed there.
- A MIT licence on the repo's code does not need to cover data. A `DATA_LICENSES.md` listing each bundled dataset with its own licence is the usual way to keep them separate.

### Gaps
- British Museum live terms page returned HTTP 403; its text was read from a Wayback copy. Met policy page (https://www.metmuseum.org/policies/image-resources) returned HTTP 429 and was not read; Met terms are reported from a metmuseum.org article and the live API flags.
- Iravatham Mahadevan's year of death was not sourced here, so no end date is given for a life-plus-60 term.
- Copyright page of the CISI volumes and of Mahadevan 1977 was not seen; no statement from Suomalainen Tiedeakatemia or ASI on reuse of the plates was found.
- The Kaggle uploader's source for the seal images is unstated; could not be determined.
- The full JCAA article text on image permissions was read through a summariser, not line by line.
- Whether the tpsatish95 repo's `code`, `paper` or `slides` folders contain seal images was not established (the GitHub tree listing returned empty, probably rate-limited). If they do, the RMRL and Harappa.com permissions were given to those authors; the repo's Apache-2.0 licence does not show that they pass to others. The "Yes (code)" entry in the table refers to source code only.

## Is scraping indusscript.in or Harappa.com prohibited by their terms or robots.txt?

### Takeaway
Neither site has a robots.txt rule that disallows crawling its content pages, and no terms text prohibiting scraping was found on either. Harappa.com asks for a 10-second crawl delay, blocks non-browser clients through Cloudflare, and claims copyright over all images; indusscript.in has no robots.txt and no readable terms at all.

### Cited Findings
- https://indusscript.in/robots.txt returns HTTP 200 with content-type text/html and the 3,502-byte Flutter app shell, the same response as any unknown path (`/terms`, `/about`, `/sitemap.xml`). There is no actual robots.txt file. `www.indusscript.in` does not resolve. — [indusscript.in/robots.txt](https://indusscript.in/robots.txt)
- The app's "Terms and Conditions" link target, https://indusscript.in/terms.html, also returns only the app shell. — [terms.html](https://indusscript.in/terms.html)
- The app's asset manifest lists 890 assets, mostly `images/N.jpg` and `images/N.png`, served without login. Data is otherwise loaded from a Firebase/Firestore backend (project `theindusscript`). — [AssetManifest.json](https://indusscript.in/assets/AssetManifest.json)
- Harappa.com live robots.txt could not be fetched: Cloudflare returned a "Sorry, you have been blocked" page to curl, including with a browser user-agent; WebFetch got HTTP 403. — [harappa.com/robots.txt](https://www.harappa.com/robots.txt)
- Archived Harappa.com robots.txt (Wayback, 2026 capture) is the stock Drupal file: `User-agent: *`, `Crawl-delay: 10`, Disallow for `/includes/`, `/misc/`, `/modules/`, `/profiles/`, `/scripts/`, `/themes/`, `/admin/`, `/search/`, `/user/login/` and similar admin paths. Content pages and image file paths are not disallowed. — [archived robots.txt](https://web.archive.org/web/2026id_/https://www.harappa.com/robots.txt)
- Harappa.com copyright line: "All images and content are © Harappa.com 1995-2026 and may not be used commercially without written permission." — [archived About Us: Credits](https://web.archive.org/web/20260620011806/https://www.harappa.com/content/about-us-credits)

### Inferences
- indusscript.in: absence of robots.txt and terms means no stated prohibition, and equally no stated permission. The content is Mahadevan's corpus published by RMRL "for use by researchers, students and interested members of the general public"; that describes use of the app, not bulk copying or republication. Pulling data straight from the Firestore backend would bypass the app interface. Asking RMRL (www.rmrl.in) for the data is the clean route; the earlier OCR project got its images that way.
- Harappa.com: robots.txt does not forbid crawling content, but the Cloudflare block shows automated clients are actively refused in practice, and `/search/` is disallowed. The copyright notice restricts commercial use explicitly and the site asks for permission for image use. Scraped images could be used locally for research but not republished.
- For both sites, scraped material belongs in the "research use only, not committed" tier regardless of robots.txt.

### Gaps
- indusscript.in in-app terms could not be rendered (JavaScript app); only strings in the compiled bundle were inspected. A terms dialog shown at login may exist and was not seen.
- Harappa.com live robots.txt and live About page not opened; archived copies used. The exact timestamp of the robots.txt capture was not recorded.
- No Harappa.com page stating a scraping or automated-access policy was found.

## Can annotations a student creates over copyrighted photographs be published without the photographs?

### Takeaway
None of the sources' stated terms address annotations. Bounding boxes, sign labels and transcriptions keyed to public identifiers (CISI numbers such as "M-2", Mahadevan sign numbers) contain none of the photograph, and there is published precedent for releasing exactly that under open licences. Trained model weights are a less settled case and no source speaks to them.

### Cited Findings
- mayig/indus-valley-script-corpus publishes a sign-by-sign transcription of CISI under MIT, keyed by CISI text numbers, without reproducing the photographs. — [README](https://github.com/mayig/indus-valley-script-corpus/blob/main/README.md)
- DM-BiCLab publishes code and trained-pipeline notebooks under CC0 1.0 and withholds the full image set ("may be obtained from authors upon request"), though the images came from CISI and Mahadevan 1977. — [README](https://github.com/DM-BiCLab/Deep-Learning-in-Archiving-Indus-Script-and-Motif-Information/blob/main/README.md); [JCAA article](https://journal.caa-international.org/articles/10.5334/jcaa.175)
- hellosindh publishes models and synthetic sequences trained on an unreleased real corpus under CC BY 4.0. — [model card](https://huggingface.co/hellosindh/indus-script-models)
- tpsatish95 publishes code under Apache-2.0 while the RMRL-supplied seal images are acknowledged as provided privately. — [README](https://github.com/tpsatish95/OCR-on-Indus-Seals/blob/master/README.md)
- Indian Copyright Act s. 52(1)(a) covers fair dealing for "private or personal use, including research". — [Mondaq on s. 52](https://www.mondaq.com/india/copyright/1184954/exception-to-infringement-of-copyright-section-52-of-the-copyright-act-1957)

### Inferences
- No stated term from any source covers derived annotations. The comparable projects above all released their own derived material (transcriptions, code, models) openly while keeping the photographs out. Annotation files holding coordinates, class labels and a source identifier (CISI number or URL) follow that same pattern; image crops or thumbnails do not, because they are pieces of the photograph.
- No stated term from any source covers model weights trained on the photographs either. hellosindh released models under CC BY 4.0; DM-BiCLab and tpsatish95 released code but no weights were confirmed in their repos.
- Annotations made over CC BY-SA Commons images are cleaner still, and over CC0 museum images are unrestricted.
- Research exceptions cover the student's own use of the photographs; they are not a licence to redistribute them.

### Gaps
- No source states an explicit position on derived annotations or model weights. No case law was researched; this is outside the factual scope.
- The applicable jurisdiction (India, UK, Finland, Pakistan for different rights holders) changes the exception that applies; not analysed.

## Which museum collections provide genuinely open (CC0) Indus seal photographs with an API?

### Takeaway
The Metropolitan Museum of Art and the Cleveland Museum of Art both offer CC0 images of Indus seals through keyless public APIs, but the numbers are small: three confirmed at Cleveland and a handful at the Met. The British Museum is not CC0 (CC BY-NC-SA 4.0).

### Cited Findings
- Cleveland Museum of Art: "High-resolution images and data related to the CMA's collection are made available under a Creative Commons Zero (CC0) designation." Public API needs no key; data also on GitHub in JSON and CSV. "Works designated as Creative Commons (CC0) do not require attribution or citation." — [CMA Open Access](https://www.clevelandart.org/open-access)
- CMA API query `q=Indus seal&has_image=1` returned 3 artworks, all `share_license_status: CC0`, culture "Pakistan, Indus Valley civilization": 1973.160 "Seal with Two-Horned Bull and Inscription", 1964.104 "Seal with Unicorn and Inscription", 1973.161 "Seal with Unicorn and Inscription". — [CMA API query](https://openaccess-api.clevelandart.org/api/artworks/?q=Indus%20seal&has_image=1&limit=20)
- CMA API field `share_license_status` takes values "CC0", "Copyrighted" or "Other"; only CC0 artworks provide CC0 images. — [CMA Open Access API docs](https://openaccess-api.clevelandart.org/)
- Met Museum: Open Access images and data are under CC0, served by The Met Collection API (no key). — [Met Collection API article](https://www.metmuseum.org/perspectives/met-collection-api-2)
- Met API objects confirmed `isPublicDomain: true` with an image, culture "Indus", period "Mature Harappan": 49.40.1 "Stamp seal and modern impression: unicorn and incense burner (?)", 49.40.2 "Stamp seal: buffalo with incense burner (?)", 49.40.3 "Stamp seal fragment: unicorn and incense-burner (?), inscription", 49.40.4 "Bar-shaped rectangular plaque seal". — [Met API object 324062](https://collectionapi.metmuseum.org/public/collection/v1/objects/324062)
- Met API search `q=Indus stamp seal&hasImages=true` returns 68 object IDs, but this is a loose keyword match: object 324070 (49.40.9) is tagged culture "Gandhara", not Indus. Each hit must be filtered on `culture` and `isPublicDomain`. — [Met API search](https://collectionapi.metmuseum.org/public/collection/v1/search?q=Indus%20stamp%20seal&hasImages=true)
- British Museum: "The British Museum publishes some of the content on our website under a Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)" licence; "The Creative Commons licence does not apply to everything on our website"; licensed content is marked with the CC logo. Read from a 2026 Wayback copy; the live page returned HTTP 403. — [archived BM copyright and permissions](https://web.archive.org/web/2026/https://www.britishmuseum.org/terms-use/copyright-and-permissions)
- Wikimedia Commons is the other open route with an API; a 50-file sample showed 9 CC0 and 17 public-domain files alongside CC BY-SA ones. — [Commons API](https://commons.wikimedia.org/w/api.php?action=query&generator=search&gsrnamespace=6&gsrsearch=Indus+seal+Harappa+OR+Mohenjo-daro+seal&gsrlimit=50&prop=imageinfo&iiprop=extmetadata&iiextmetadatafilter=LicenseShortName&format=json)

### Inferences
- CC0 museum images are useful as a clean demo or test set that can live in the repo, not as training data: fewer than about ten confirmed seals across the Met and Cleveland combined.
- British Museum images can be used for the student's non-commercial research with credit, but NC and ShareAlike make them unsuitable to bundle in an MIT-licensed public repo.
- The bulk of Indus seals are held in India and Pakistan (National Museum New Delhi, ASI, Pakistani museums) and reach the public mainly through CISI and Harappa.com, which are not open.

### Gaps
- The exact count of Indus seals at the Met was not established; only five objects were inspected.
- British Museum: no public API was verified, and the number of Indus seals with images was not checked.
- Other museums (Ashmolean, Louvre, Museum für Asiatische Kunst Berlin, National Museum New Delhi, Smithsonian, Walters, Yale) were not checked. The Commons category list shows items from the Museum of Asian Art (I 10136) and the Louvre (Sb 2425) exist there under Commons licences.
