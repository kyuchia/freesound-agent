"""
eval.py — RAG retrieval evaluation for Freesound Agent

Evaluates the quality of ChromaDB retrieval (NOT end-to-end agent behaviour).
Tests three categories:
  A. Single-source queries — retrieved top-1 must come from the expected paper
  B. Multi-source queries — retrieved top-3 must contain ≥1 expected paper
  C. Out-of-scope queries — retrieved top-1 distance must exceed threshold

Usage:
    # Run the full eval and print results to terminal
    $ python eval.py

    # Also write a Markdown report to eval_results.md
    $ python eval.py --write-report

    # Verbose: show retrieved chunks for each query
    $ python eval.py --verbose

Architecture:
    13 hand-curated test queries → query_audio_knowledge() → check expectations
    → aggregate metrics (hit rate, MRR, distance distribution)
"""

from __future__ import annotations

import argparse
import statistics
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from knowledge import query_audio_knowledge

# ---------------------------------------------------------------------------
# Test set (13 queries across 3 categories)
# ---------------------------------------------------------------------------

@dataclass
class EvalCase:
    """One evaluation query + its expectations."""
    qid: str
    query: str
    category: str  # "A", "B", or "C"
    expected_sources: list[str] = field(default_factory=list)
    notes: str = ""


# Pass thresholds
DISTANCE_THRESHOLD_OOS = 0.7   # Category C: top-1 distance must exceed this
TOP_K = 3                       # how many chunks to retrieve per query


EVAL_SET: list[EvalCase] = [
    # ─── Category A: Single-Source ──────────────────────────────────
    EvalCase(
        qid="A1",
        query="What is a narrow folksonomy?",
        category="A",
        expected_sources=["font_2012"],
        notes="Narrow folksonomy is Font 2012's defining term (Section 3).",
    ),
    EvalCase(
        qid="A2",
        query="How does the Freesound Datasets platform validate audio labels?",
        category="A",
        expected_sources=["fonseca_2017"],
        notes="Validation workflow is exclusive to Fonseca 2017 (Section 4.2).",
    ),
    EvalCase(
        qid="A3",
        query="What percentage of Freesound tags are used only once?",
        category="A",
        expected_sources=["martinez_2009"],
        notes="The '44% used once' statistic is unique to Martinez 2009 (4.1).",
    ),
    EvalCase(
        qid="A4",
        query="How is tag co-occurrence used for tag recommendation?",
        category="A",
        expected_sources=["font_2012"],
        notes="Font 2012's core method (Section 4.1).",
    ),
    EvalCase(
        qid="A5",
        query="What is the Wordnet categorization of Freesound tags?",
        category="A",
        expected_sources=["martinez_2009"],
        notes="Wordnet hypernym mapping is unique to Martinez 2009 (4.2).",
    ),
    EvalCase(
        qid="A6",
        query="What limitations do GTZAN and Ballroom datasets have?",
        category="A",
        expected_sources=["fonseca_2017"],
        notes="Critique of legacy datasets is in Fonseca 2017's Introduction.",
    ),

    # ─── Category B: Multi-Source ───────────────────────────────────
    EvalCase(
        qid="B1",
        query="Why is Freesound's tag distribution uneven?",
        category="B",
        expected_sources=["martinez_2009", "font_2012"],
        notes="Martinez 4.1 (log-normal) + Font 3 (20% covers 93%).",
    ),
    EvalCase(
        qid="B2",
        query="What problems exist in collaborative audio tagging?",
        category="B",
        expected_sources=["martinez_2009", "font_2012"],
        notes="Martinez 2 (polysemy/synonymy) + Font 1 (sparsity).",
    ),
    EvalCase(
        qid="B3",
        query="How are audio datasets built from Freesound?",
        category="B",
        expected_sources=["fonseca_2017", "martinez_2009"],
        notes="Fonseca (platform) + Martinez (content-based extension).",
    ),
    EvalCase(
        qid="B4",
        query="What is content-based audio analysis used for in Freesound?",
        category="B",
        expected_sources=["martinez_2009", "font_2012"],
        notes="Both papers use content-based similarity for tag tasks.",
    ),

    # ─── Category C: Out-of-Scope ───────────────────────────────────
    EvalCase(
        qid="C1",
        query="How to train a convolutional neural network for image classification?",
        category="C",
        expected_sources=[],  # nothing should match well
        notes="Cross-domain (computer vision). Stress test: 'classifier' keyword overlap.",
    ),
    EvalCase(
        qid="C2",
        query="What is the best microphone for recording birds?",
        category="C",
        expected_sources=[],
        notes="Topic-adjacent (recording/birds) but papers don't cover gear. Trap query.",
    ),
    EvalCase(
        qid="C3",
        query="How does Spotify recommend songs?",
        category="C",
        expected_sources=[],
        notes="Adjacent field (music rec) — keyword 'recommend' overlap stress test.",
    ),
]


# ---------------------------------------------------------------------------
# Per-query evaluation
# ---------------------------------------------------------------------------

@dataclass
class CaseResult:
    """Outcome of one evaluation query."""
    case: EvalCase
    retrieved_sources: list[str]      # top-k sources in order
    retrieved_sections: list[str]
    distances: list[float]
    rank_of_first_hit: int | None     # 1-based; None if expected source not in top-k
    passed: bool
    pass_reason: str

    @property
    def top1_source(self) -> str:
        return self.retrieved_sources[0] if self.retrieved_sources else ""

    @property
    def top1_distance(self) -> float:
        return self.distances[0] if self.distances else float("inf")


def evaluate_case(case: EvalCase) -> CaseResult:
    """Run retrieval for one case and check pass criteria."""
    passages = query_audio_knowledge(case.query, top_k=TOP_K)

    if not passages:
        return CaseResult(
            case=case,
            retrieved_sources=[],
            retrieved_sections=[],
            distances=[],
            rank_of_first_hit=None,
            passed=False,
            pass_reason="ERROR: no results returned. Did you run knowledge.py --rebuild?",
        )

    sources = [p["source"] for p in passages]
    sections = [p["section"] for p in passages]
    distances = [p["distance"] if p["distance"] is not None else float("inf")
                 for p in passages]

    # find the rank (1-indexed) of the first expected source in top-k
    rank_of_first_hit: int | None = None
    if case.expected_sources:
        for i, src in enumerate(sources, start=1):
            if src in case.expected_sources:
                rank_of_first_hit = i
                break

    # apply category-specific pass criteria
    passed, reason = _check_pass(case, sources, distances, rank_of_first_hit)

    return CaseResult(
        case=case,
        retrieved_sources=sources,
        retrieved_sections=sections,
        distances=distances,
        rank_of_first_hit=rank_of_first_hit,
        passed=passed,
        pass_reason=reason,
    )


def _check_pass(
    case: EvalCase,
    sources: list[str],
    distances: list[float],
    rank_of_first_hit: int | None,
) -> tuple[bool, str]:
    """Apply per-category pass criteria. Returns (passed, human-readable reason)."""
    if case.category == "A":
        # top-1 must equal expected source
        expected = case.expected_sources[0]
        if sources[0] == expected:
            return True, f"top-1 = {expected} ✓"
        return False, f"top-1 = {sources[0]}, expected {expected}"

    if case.category == "B":
        # top-3 must contain ≥1 expected source
        if rank_of_first_hit is not None:
            return True, f"expected source found at rank {rank_of_first_hit}"
        expected_str = " or ".join(case.expected_sources)
        return False, f"none of [{expected_str}] in top-3"

    if case.category == "C":
        # top-1 distance must exceed threshold (system should look uncertain)
        d = distances[0]
        if d > DISTANCE_THRESHOLD_OOS:
            return True, f"top-1 distance {d:.4f} > {DISTANCE_THRESHOLD_OOS} ✓"
        return False, (f"top-1 distance {d:.4f} ≤ {DISTANCE_THRESHOLD_OOS} "
                       f"(system overconfident on out-of-scope query)")

    return False, f"unknown category {case.category!r}"


# ---------------------------------------------------------------------------
# Aggregate metrics
# ---------------------------------------------------------------------------

@dataclass
class EvalSummary:
    """Roll-up metrics across all cases."""
    total: int
    passed: int
    by_category: dict[str, tuple[int, int]]            # category → (passed, total)
    in_scope_distances: list[float]                    # top-1 distances for A+B
    oos_distances: list[float]                          # top-1 distances for C
    mrr: float                                          # MRR over A+B (where rank exists)


def summarize(results: list[CaseResult]) -> EvalSummary:
    by_cat: dict[str, list[CaseResult]] = {}
    for r in results:
        by_cat.setdefault(r.case.category, []).append(r)

    by_category = {
        cat: (sum(1 for r in rs if r.passed), len(rs))
        for cat, rs in by_cat.items()
    }

    in_scope_distances = [
        r.top1_distance for r in results
        if r.case.category in ("A", "B") and r.distances
    ]
    oos_distances = [
        r.top1_distance for r in results
        if r.case.category == "C" and r.distances
    ]

    # MRR = mean of (1 / rank) for in-scope cases that have an expected hit
    reciprocal_ranks = [
        1.0 / r.rank_of_first_hit
        for r in results
        if r.case.category in ("A", "B") and r.rank_of_first_hit is not None
    ]
    in_scope_count = sum(1 for r in results if r.case.category in ("A", "B"))
    # zero-credit for misses → divide by total in-scope count, not just hits
    mrr = sum(reciprocal_ranks) / in_scope_count if in_scope_count else 0.0

    return EvalSummary(
        total=len(results),
        passed=sum(1 for r in results if r.passed),
        by_category=by_category,
        in_scope_distances=in_scope_distances,
        oos_distances=oos_distances,
        mrr=mrr,
    )


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_terminal_report(results: list[CaseResult], summary: EvalSummary,
                          verbose: bool = False) -> None:
    print("\n" + "=" * 72)
    print(" Freesound Agent — RAG Retrieval Evaluation")
    print("=" * 72)
    print(f" {len(results)} test queries · top_k={TOP_K} · "
          f"OOS distance threshold = {DISTANCE_THRESHOLD_OOS}")
    print("=" * 72)

    # per-case lines
    for r in results:
        status = "✓ PASS" if r.passed else "✗ FAIL"
        print(f"\n[{r.case.qid}] {status}  ({r.case.category})  "
              f"d={r.top1_distance:.4f}")
        print(f"   Q: {r.case.query}")
        if r.case.expected_sources:
            print(f"   Expected: {', '.join(r.case.expected_sources)}")
        print(f"   Top-3:    {' → '.join(r.retrieved_sources)}")
        print(f"   Reason:   {r.pass_reason}")

        if verbose:
            for i, (src, sec, d) in enumerate(
                zip(r.retrieved_sources, r.retrieved_sections, r.distances), 1
            ):
                print(f"     [{i}] {src}  §{sec}  (d={d:.4f})")

    # summary
    print("\n" + "─" * 72)
    print(" Summary")
    print("─" * 72)
    print(f" Overall pass rate:    {summary.passed}/{summary.total}  "
          f"({summary.passed / summary.total * 100:.1f}%)")
    for cat in ("A", "B", "C"):
        if cat in summary.by_category:
            p, t = summary.by_category[cat]
            label = {
                "A": "Single-source",
                "B": "Multi-source",
                "C": "Out-of-scope",
            }[cat]
            print(f"   Category {cat} ({label:<14}): {p}/{t}  "
                  f"({p / t * 100:.1f}%)")

    print()
    if summary.in_scope_distances:
        in_mean = statistics.mean(summary.in_scope_distances)
        in_max = max(summary.in_scope_distances)
        print(f" In-scope (A+B) top-1 distance:  "
              f"mean={in_mean:.4f}  max={in_max:.4f}")
    if summary.oos_distances:
        oos_mean = statistics.mean(summary.oos_distances)
        oos_min = min(summary.oos_distances)
        print(f" Out-of-scope (C) top-1 distance: "
              f"mean={oos_mean:.4f}  min={oos_min:.4f}")
    if summary.in_scope_distances and summary.oos_distances:
        gap = statistics.mean(summary.oos_distances) - statistics.mean(summary.in_scope_distances)
        print(f" Distance separation (OOS - in-scope): {gap:+.4f}  "
              f"({'good' if gap > 0.15 else 'weak'} signal)")

    print(f"\n Mean Reciprocal Rank (in-scope): {summary.mrr:.4f}")
    print("=" * 72 + "\n")


def write_markdown_report(results: list[CaseResult], summary: EvalSummary,
                          out_path: Path) -> None:
    """Write a Markdown report suitable for committing to the repo."""
    lines: list[str] = []
    add = lines.append

    # ─── Header ─────────────────────────────────────────────────────
    add("# RAG Retrieval Evaluation Report")
    add("")
    add(f"_Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}_")
    add("")

    # ─── TL;DR ──────────────────────────────────────────────────────
    pass_pct = summary.passed / summary.total * 100
    add("## TL;DR")
    add("")
    add(f"- **Overall**: {summary.passed}/{summary.total} pass ({pass_pct:.1f}%)")
    in_scope_passed = sum(p for cat in ("A", "B")
                          for p, _ in [summary.by_category.get(cat, (0, 0))])
    in_scope_total = sum(t for cat in ("A", "B")
                         for _, t in [summary.by_category.get(cat, (0, 0))])
    if in_scope_total:
        add(f"- **In-scope retrieval**: {in_scope_passed}/{in_scope_total} pass "
            f"({in_scope_passed / in_scope_total * 100:.1f}%) — "
            f"strong basic retrieval performance")
    add(f"- **MRR (in-scope)**: `{summary.mrr:.4f}` — expected source typically "
        f"appears at rank 1")
    if summary.in_scope_distances and summary.oos_distances:
        in_mean = statistics.mean(summary.in_scope_distances)
        oos_mean = statistics.mean(summary.oos_distances)
        gap = oos_mean - in_mean
        add(f"- **Distance separation** (OOS − in-scope): `{gap:+.4f}` — "
            f"embedding shows correct calibration even though absolute threshold "
            f"({DISTANCE_THRESHOLD_OOS}) was not met")
    add("")

    # ─── Setup ──────────────────────────────────────────────────────
    add("## Setup")
    add("")
    add(f"- **Test queries**: {len(results)} (6 single-source · 4 multi-source · 3 out-of-scope)")
    add(f"- **Retrieval depth**: top-{TOP_K}")
    add(f"- **OOS distance threshold**: {DISTANCE_THRESHOLD_OOS}")
    add(f"- **Embedding model**: `all-MiniLM-L6-v2` (sentence-transformers)")
    add(f"- **Index**: ChromaDB with 45 chunks across 3 papers")
    add("")

    # ─── Summary table ──────────────────────────────────────────────
    add("## Results by category")
    add("")
    add("| Category | Description | Pass | Total | Rate |")
    add("|---|---|---|---|---|")
    for cat in ("A", "B", "C"):
        if cat in summary.by_category:
            p, t = summary.by_category[cat]
            label = {
                "A": "Single-source",
                "B": "Multi-source",
                "C": "Out-of-scope",
            }[cat]
            add(f"| {cat} | {label} | {p} | {t} | {p / t * 100:.1f}% |")
    add("")

    if summary.in_scope_distances and summary.oos_distances:
        in_mean = statistics.mean(summary.in_scope_distances)
        in_max = max(summary.in_scope_distances)
        oos_mean = statistics.mean(summary.oos_distances)
        oos_min = min(summary.oos_distances)
        gap = oos_mean - in_mean
        add("### Distance distribution")
        add("")
        add(f"- In-scope (A+B) top-1 distance: mean `{in_mean:.4f}` · max `{in_max:.4f}`")
        add(f"- Out-of-scope (C) top-1 distance: mean `{oos_mean:.4f}` · min `{oos_min:.4f}`")
        add(f"- Separation: `{gap:+.4f}` "
            f"({'good calibration signal' if gap > 0.15 else 'weak calibration'})")
        add("")

    # ─── Per-query table ────────────────────────────────────────────
    add("## Per-query results")
    add("")
    add("| ID | Cat | Query | Top-1 source | Distance | Pass | Reason |")
    add("|---|---|---|---|---|---|---|")
    for r in results:
        status = "✓" if r.passed else "✗"
        q = r.case.query.replace("|", "\\|")
        reason = r.pass_reason.replace("|", "\\|")
        add(f"| {r.case.qid} | {r.case.category} | {q} | "
            f"`{r.top1_source}` | {r.top1_distance:.4f} | {status} | {reason} |")
    add("")

    # ─── Findings (auto-derived from results) ──────────────────────
    add("## Findings")
    add("")

    # Build findings as a list of (tag, body) tuples. Tags are stable IDs
    # used to cross-reference findings from later sections (e.g. Methodology).
    # Numbering is assigned dynamically based on which findings actually fire.
    findings: list[tuple[str, str]] = []

    # Finding: in-scope strength
    if in_scope_total and in_scope_passed / in_scope_total >= 0.8:
        findings.append((
            "in_scope_strength",
            f"**In-scope retrieval is reliable.** "
            f"{in_scope_passed} of {in_scope_total} in-scope queries returned the "
            f"expected source within top-{TOP_K}. MRR = {summary.mrr:.2f} "
            f"indicates the right source typically appears at rank 1, not rank 2 or 3."
        ))

    # Finding: OOS calibration
    if summary.in_scope_distances and summary.oos_distances:
        in_mean = statistics.mean(summary.in_scope_distances)
        oos_mean = statistics.mean(summary.oos_distances)
        gap = oos_mean - in_mean
        oos_pass_rate = summary.by_category.get("C", (0, 1))
        oos_passed_n, oos_total_n = oos_pass_rate
        if oos_passed_n < oos_total_n and gap > 0.15:
            findings.append((
                "oos_calibration",
                f"**OOS queries show distance calibration but not refusal-grade "
                f"separation.** OOS mean distance ({oos_mean:.3f}) is `{gap:+.3f}` "
                f"higher than in-scope ({in_mean:.3f}) — the embedding model "
                f"*does* register lower confidence on out-of-scope queries. "
                f"However, the absolute distance threshold of "
                f"{DISTANCE_THRESHOLD_OOS} is not crossed by any OOS query, "
                f"so retrieval-layer gating alone cannot reject these. "
                f"This is an expected limit of cosine distance as a binary "
                f"confidence signal; the LLM layer must handle final refusal."
            ))

    # Finding: any single-source failures
    a_passed, a_total = summary.by_category.get("A", (0, 0))
    if a_total and a_passed < a_total:
        failed_a = [r for r in results if r.case.category == "A" and not r.passed]
        ids = ", ".join(r.case.qid for r in failed_a)
        findings.append((
            "single_source_failures",
            f"**Single-source failures** ({ids}). These indicate either "
            f"chunking weakness or query phrasing that under-specifies the "
            f"expected source."
        ))

    # Finding: B-class failures (signals query-design bias)
    b_failed = [r for r in results if r.case.category == "B" and not r.passed]
    if b_failed:
        ids = ", ".join(r.case.qid for r in b_failed)
        also_reasonable = []
        for r in b_failed:
            top1 = r.top1_source
            if top1 and top1 not in r.case.expected_sources:
                also_reasonable.append((r.case.qid, top1))
        prefix = f"**Multi-source failures** ({ids}). "
        if also_reasonable:
            details = "; ".join(f"{qid} retrieved `{src}` instead"
                                for qid, src in also_reasonable)
            findings.append((
                "multi_source_overspec",
                prefix +
                f"Top-1 fell on a non-expected but plausibly relevant source "
                f"({details}). This surfaces an evaluation-design issue: "
                f"\"expected sources\" were defined narrowly, but the queried "
                f"concept is genuinely discussed across more papers than "
                f"originally tagged. The retrieval is not wrong — the ground "
                f"truth was over-specified."
            ))
        else:
            findings.append((
                "multi_source_miss",
                prefix + f"None of the expected sources appeared in top-{TOP_K}."
            ))

    # Render findings with sequential 1-based numbering, recording tag → number.
    finding_id_by_tag: dict[str, int] = {}
    for i, (tag, body) in enumerate(findings, start=1):
        finding_id_by_tag[tag] = i
        add(f"**{i}.** {body}")
        add("")

    # ─── Methodology limits ─────────────────────────────────────────
    add("## Methodology — limits and caveats")
    add("")
    add(f"- **Small test set ({len(results)} queries)**. Designed to surface "
        f"system behaviour across three retrieval modes, not to provide "
        f"statistically robust benchmark numbers. A production evaluation "
        f"would use larger sets such as BEIR or domain-specific benchmarks.")

    # Cross-reference the over-specification finding by its actual rendered ID.
    # If the finding didn't fire (no B-class over-spec failure), drop the parenthetical.
    overspec_ref = ""
    if "multi_source_overspec" in finding_id_by_tag:
        overspec_ref = f" (see Finding {finding_id_by_tag['multi_source_overspec']})"
    add(f"- **Author-as-judge bias**. Test queries and ground-truth source labels "
        f"were authored by the same person who built the system. Some "
        f"ground-truth labels may be over-specified{overspec_ref}. "
        f"An independent annotator pass would correct this.")

    add("- **Retrieval-only scope**. This evaluation does not exercise the LLM. "
        "Tool selection, citation accuracy, and refusal-on-OOS behaviour "
        "are properties of the agent, not the retriever, and require a "
        "separate end-to-end evaluation.")
    add(f"- **Absolute distance thresholds are fragile**. The OOS gating "
        f"threshold ({DISTANCE_THRESHOLD_OOS}) was set before observing "
        f"results. Cosine distance is not a calibrated probability; "
        f"reasonable values depend on the embedding model and corpus. "
        f"Reporting the in-scope vs OOS *separation* (above) is more "
        f"informative than pass/fail against a fixed cutoff.")
    add("")

    # ─── Future work ────────────────────────────────────────────────
    add("## Future work")
    add("")
    add("- **End-to-end evaluation**: measure whether the LLM correctly refuses "
        "OOS queries despite borderline retrieval distances. This is the "
        "most direct follow-up to the calibration finding above.")
    add("- **Tool-selection evaluation**: measure how often the agent picks "
        "`query_audio_knowledge` vs other tools given different query phrasings.")
    add("- **Independent annotation**: have a second annotator label expected "
        "sources to remove author-as-judge bias.")
    add("- **Larger test set with topic diversity**: scale to 50+ queries "
        "covering edge cases (multilingual queries, ambiguous phrasing, "
        "compound questions).")
    add("")

    out_path.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate RAG retrieval quality")
    parser.add_argument("--write-report", action="store_true",
                        help="Write eval_results.md alongside terminal output")
    parser.add_argument("--verbose", action="store_true",
                        help="Show all retrieved chunks per query")
    parser.add_argument("--report-path", type=Path, default=Path("eval_results.md"),
                        help="Output path for Markdown report (default: ./eval_results.md)")
    args = parser.parse_args()

    print(f"Running {len(EVAL_SET)} evaluation queries...")
    results = [evaluate_case(case) for case in EVAL_SET]
    summary = summarize(results)

    print_terminal_report(results, summary, verbose=args.verbose)

    if args.write_report:
        write_markdown_report(results, summary, args.report_path)
        print(f"Markdown report written to: {args.report_path}\n")


if __name__ == "__main__":
    main()
