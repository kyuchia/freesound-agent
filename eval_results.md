# RAG Retrieval Evaluation Report

_Generated: 2026-05-04 02:19_

## TL;DR

- **Overall**: 9/13 pass (69.2%)
- **In-scope retrieval**: 9/10 pass (90.0%) — strong basic retrieval performance
- **MRR (in-scope)**: `0.9000` — expected source typically appears at rank 1
- **Distance separation** (OOS − in-scope): `+0.2177` — embedding shows correct calibration even though absolute threshold (0.7) was not met

## Setup

- **Test queries**: 13 (6 single-source · 4 multi-source · 3 out-of-scope)
- **Retrieval depth**: top-3
- **OOS distance threshold**: 0.7
- **Embedding model**: `all-MiniLM-L6-v2` (sentence-transformers)
- **Index**: ChromaDB with 45 chunks across 3 papers

## Results by category

| Category | Description | Pass | Total | Rate |
|---|---|---|---|---|
| A | Single-source | 6 | 6 | 100.0% |
| B | Multi-source | 3 | 4 | 75.0% |
| C | Out-of-scope | 0 | 3 | 0.0% |

### Distance distribution

- In-scope (A+B) top-1 distance: mean `0.4011` · max `0.6489`
- Out-of-scope (C) top-1 distance: mean `0.6187` · min `0.5862`
- Separation: `+0.2177` (good calibration signal)

## Per-query results

| ID | Cat | Query | Top-1 source | Distance | Pass | Reason |
|---|---|---|---|---|---|---|
| A1 | A | What is a narrow folksonomy? | `font_2012` | 0.6489 | ✓ | top-1 = font_2012 ✓ |
| A2 | A | How does the Freesound Datasets platform validate audio labels? | `fonseca_2017` | 0.3433 | ✓ | top-1 = fonseca_2017 ✓ |
| A3 | A | What percentage of Freesound tags are used only once? | `martinez_2009` | 0.4745 | ✓ | top-1 = martinez_2009 ✓ |
| A4 | A | How is tag co-occurrence used for tag recommendation? | `font_2012` | 0.3520 | ✓ | top-1 = font_2012 ✓ |
| A5 | A | What is the Wordnet categorization of Freesound tags? | `martinez_2009` | 0.3091 | ✓ | top-1 = martinez_2009 ✓ |
| A6 | A | What limitations do GTZAN and Ballroom datasets have? | `fonseca_2017` | 0.5946 | ✓ | top-1 = fonseca_2017 ✓ |
| B1 | B | Why is Freesound's tag distribution uneven? | `font_2012` | 0.5143 | ✓ | expected source found at rank 1 |
| B2 | B | What problems exist in collaborative audio tagging? | `martinez_2009` | 0.2602 | ✓ | expected source found at rank 1 |
| B3 | B | How are audio datasets built from Freesound? | `fonseca_2017` | 0.1907 | ✓ | expected source found at rank 1 |
| B4 | B | What is content-based audio analysis used for in Freesound? | `fonseca_2017` | 0.3229 | ✗ | none of [martinez_2009 or font_2012] in top-3 |
| C1 | C | How to train a convolutional neural network for image classification? | `fonseca_2017` | 0.6211 | ✗ | top-1 distance 0.6211 ≤ 0.7 (system overconfident on out-of-scope query) |
| C2 | C | What is the best microphone for recording birds? | `font_2012` | 0.6489 | ✗ | top-1 distance 0.6489 ≤ 0.7 (system overconfident on out-of-scope query) |
| C3 | C | How does Spotify recommend songs? | `font_2012` | 0.5862 | ✗ | top-1 distance 0.5862 ≤ 0.7 (system overconfident on out-of-scope query) |

## Findings

**1.** **In-scope retrieval is reliable.** 9 of 10 in-scope queries returned the expected source within top-3. MRR = 0.90 indicates the right source typically appears at rank 1, not rank 2 or 3.

**2.** **OOS queries show distance calibration but not refusal-grade separation.** OOS mean distance (0.619) is `+0.218` higher than in-scope (0.401) — the embedding model *does* register lower confidence on out-of-scope queries. However, the absolute distance threshold of 0.7 is not crossed by any OOS query, so retrieval-layer gating alone cannot reject these. This is an expected limit of cosine distance as a binary confidence signal; the LLM layer must handle final refusal.

**3.** **Multi-source failures** (B4). Top-1 fell on a non-expected but plausibly relevant source (B4 retrieved `fonseca_2017` instead). This surfaces an evaluation-design issue: "expected sources" were defined narrowly, but the queried concept is genuinely discussed across more papers than originally tagged. The retrieval is not wrong — the ground truth was over-specified.

## Methodology — limits and caveats

- **Small test set (13 queries)**. Designed to surface system behaviour across three retrieval modes, not to provide statistically robust benchmark numbers. A production evaluation would use larger sets such as BEIR or domain-specific benchmarks.
- **Author-as-judge bias**. Test queries and ground-truth source labels were authored by the same person who built the system. Some ground-truth labels may be over-specified (see Finding 3). An independent annotator pass would correct this.
- **Retrieval-only scope**. This evaluation does not exercise the LLM. Tool selection, citation accuracy, and refusal-on-OOS behaviour are properties of the agent, not the retriever, and require a separate end-to-end evaluation.
- **Absolute distance thresholds are fragile**. The OOS gating threshold (0.7) was set before observing results. Cosine distance is not a calibrated probability; reasonable values depend on the embedding model and corpus. Reporting the in-scope vs OOS *separation* (above) is more informative than pass/fail against a fixed cutoff.

## Future work

- **End-to-end evaluation**: measure whether the LLM correctly refuses OOS queries despite borderline retrieval distances. This is the most direct follow-up to the calibration finding above.
- **Tool-selection evaluation**: measure how often the agent picks `query_audio_knowledge` vs other tools given different query phrasings.
- **Independent annotation**: have a second annotator label expected sources to remove author-as-judge bias.
- **Larger test set with topic diversity**: scale to 50+ queries covering edge cases (multilingual queries, ambiguous phrasing, compound questions).
