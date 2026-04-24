"""
knowledge.py — RAG module for Freesound Agent

Provides a query_audio_knowledge(question) function that retrieves relevant
passages from a ChromaDB index of Freesound-related academic papers.

Papers indexed:
- Martínez et al. 2009 (SMC): Extending the folksonomies of freesound.org
- Font et al. 2012 (ISMIR): Folksonomy-based tag recommendation
- Fonseca et al. 2017 (ISMIR): Freesound Datasets

Usage:
    # One-time indexing (run once after adding PDFs or changing chunking):
    $ python knowledge.py --rebuild

    # Query (used by agent.py):
    from knowledge import query_audio_knowledge
    results = query_audio_knowledge("why is Freesound's tag distribution uneven?")

Architecture:
    PDF → pypdf → section-aware chunks → sentence-transformers embed → ChromaDB
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

from pypdf import PdfReader

# chromadb imports are deferred to get_collection() so --inspect can run
# without these heavy dependencies installed.

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).parent
PAPERS_DIR = REPO_ROOT / "papers"
CHROMA_DIR = REPO_ROOT / "chroma_db"
COLLECTION_NAME = "freesound_papers"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Paper metadata — hand-curated since auto-extracting from PDFs is unreliable.
# Key is the filename stem; values populate ChromaDB metadata for each chunk.
PAPER_METADATA: dict[str, dict] = {
    "martinez_2009_extending_the_folksonomies_of_freesound_org_using_content_based_audio_analysis": {
        "source": "martinez_2009",
        "paper_title": "Extending the Folksonomies of Freesound.org using Content-Based Audio Analysis",
        "authors": "Martínez, Celma, Sordo, de Jong, Serra",
        "year": 2009,
        "venue": "SMC",
    },
    "font_2012_folksonomy_based_tag_recommendation_for_online_audio_clip_sharing": {
        "source": "font_2012",
        "paper_title": "Folksonomy-based Tag Recommendation for Online Audio Clip Sharing",
        "authors": "Font, Serrà, Serra",
        "year": 2012,
        "venue": "ISMIR",
    },
    "fonseca_2017_freesound_datasets_a_platform_for_the_creation_of_open_audio_datasets": {
        "source": "fonseca_2017",
        "paper_title": "Freesound Datasets: A Platform for the Creation of Open Audio Datasets",
        "authors": "Fonseca, Pons, Favory, Font, Bogdanov, Ferraro, Oramas, Porter, Serra",
        "year": 2017,
        "venue": "ISMIR",
    },
}

# Chunking parameters
MIN_CHUNK_CHARS = 150    # shorter chunks get merged with next
MAX_CHUNK_CHARS = 2500   # longer chunks get split by paragraph
TARGET_CHUNK_CHARS = 1500

# Section heading regex — matches patterns like:
#   "1. INTRODUCTION"
#   "1 INTRODUCTION"
#   "4.1 Tag behaviour"
#   "4.2 Tag categorization"
# Requirements:
#   - Section number: 1-2 digits, optional subsection (e.g. "4.1")
#   - Title: starts with uppercase letter, max 60 chars
#   - Anchored to start of line with optional whitespace
# Rejects:
#   - "488 Proceedings of..." (page numbers starting a line)
#   - Very long titles that are really body text
SECTION_REGEX = re.compile(
    r"^\s*(\d{1,2}(?:\.\d{1,2})?)\s*\.?\s+([A-Z][A-Za-z0-9 \-,:]{2,60}?)\s*$",
    re.MULTILINE,
)

# Additional validation: title must not start with common page-header words
# like "Proceedings", "International" that often follow page numbers.
SECTION_TITLE_BLOCKLIST = {
    "proceedings", "international", "conference", "society", "ismir",
    "smc", "since", "copyright",
}

# Lines to strip as noise (case-insensitive)
NOISE_PATTERNS = [
    re.compile(r"^\s*\d+\s*$"),                                    # bare page numbers
    re.compile(r"^\s*ISMIR.*\d{4}.*$", re.IGNORECASE),            # ISMIR conference headers
    re.compile(r"^\s*Proceedings of.*$", re.IGNORECASE),
    re.compile(r"^\s*\d+(?:st|nd|rd|th)\s+International\s+Society.*$", re.IGNORECASE),
    re.compile(r"^\s*©.*$"),                                       # copyright lines
    re.compile(r"^\s*c\s+.*Licensed under.*$", re.IGNORECASE),    # CC license lines
]

# pypdf 6.x sometimes splits uppercase words at mid-word character-spacing.
# E.g. "DATASETS" → "DA TASETS", "PLATFORM" → "PLA TFORM".
# Only words we've observed being broken are listed here — this prevents
# over-aggressive merging of legitimate uppercase phrases like "AUDIO DATASETS".
KNOWN_BROKEN_WORDS = {
    "DATASETS", "DATASET", "DATA",
    "PLATFORM", "PLATFORMS",
    "CREATION", "CREATE", "CREATIVE",
    "RECREATION",
    "DATABASE", "DATABASES",
    "METADATA",
}


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Chunk:
    """A single retrievable passage with attached metadata."""
    chunk_id: str
    text: str
    metadata: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# PDF → text extraction
# ---------------------------------------------------------------------------

def fix_broken_uppercase_words(text: str) -> str:
    """Rejoin uppercase words that pypdf split mid-word by character spacing.

    pypdf 6.x occasionally interprets wide character tracking within uppercase
    headings as a word break — e.g. "DATASETS" → "DA TASETS". This function
    merges only pairs that produce a word in KNOWN_BROKEN_WORDS, avoiding
    corruption of legitimate phrases like "AUDIO DATASETS" or "PLATFORM FOR".
    """
    def merger(match: re.Match) -> str:
        merged = match.group(1) + match.group(2)
        return merged if merged in KNOWN_BROKEN_WORDS else match.group(0)
    # Pattern: uppercase chunk (2-4 letters) + single space + uppercase chunk (2-8 letters)
    return re.sub(r"\b([A-Z]{2,4})\s([A-Z]{2,8})\b", merger, text)


def extract_text_from_pdf(pdf_path: Path) -> str:
    """Extract full text from a PDF using pypdf, then fix hyphenated line breaks.

    pypdf handles multi-column layouts better than pdfplumber for these papers
    (ISMIR/SMC academic format). Post-processing fixes common issues:
    - 'advance-\\nment' → 'advancement' (hyphenated word wraps)
    - 'ﬁeld' → 'field' (ligature normalisation)
    - 'DA TASETS' → 'DATASETS' (pypdf 6.x mid-word character spacing)
    """
    reader = PdfReader(str(pdf_path))
    pages = []
    for page in reader.pages:
        pages.append(page.extract_text() or "")
    text = "\n\n".join(pages)

    # normalise ligatures (pypdf preserves them)
    ligatures = {
        "ﬁ": "fi", "ﬂ": "fl", "ﬀ": "ff", "ﬃ": "ffi", "ﬄ": "ffl",
        "ﬆ": "st", "\u2010": "-", "\u2011": "-",
    }
    for old, new in ligatures.items():
        text = text.replace(old, new)

    # fix hyphenated line breaks: 'advance-\nment' → 'advancement'
    # only when the hyphen is preceded by a lowercase letter (real word break,
    # not a compound like 'content-based')
    text = re.sub(r"([a-z])-\n([a-z])", r"\1\2", text)

    # fix pypdf 6.x uppercase-word splitting
    text = fix_broken_uppercase_words(text)

    return text


def clean_text(text: str) -> str:
    """Remove headers, page numbers, and other noise."""
    lines = text.split("\n")
    cleaned = []
    for line in lines:
        # drop lines matching any noise pattern
        if any(pat.match(line) for pat in NOISE_PATTERNS):
            continue
        cleaned.append(line)
    # collapse 3+ consecutive blank lines into 2 (preserve paragraph breaks)
    result = "\n".join(cleaned)
    result = re.sub(r"\n{3,}", "\n\n", result)
    return result.strip()


# ---------------------------------------------------------------------------
# Section-aware chunking
# ---------------------------------------------------------------------------

def split_into_sections(text: str) -> list[tuple[str, str]]:
    """
    Split paper text into (section_label, section_body) pairs.

    Uses section headings like "1. INTRODUCTION", "4.1 Tag behaviour" as split
    points. Text before the first heading (abstract etc.) gets labeled "Abstract".
    """
    # find all candidate matches, then filter out false positives
    raw_matches = list(SECTION_REGEX.finditer(text))
    matches = []
    for m in raw_matches:
        title = m.group(2).strip()
        first_word = title.split()[0].lower() if title.split() else ""
        if first_word in SECTION_TITLE_BLOCKLIST:
            continue  # looks like a page-header, not a real section
        matches.append(m)

    if not matches:
        # no sections detected — treat whole doc as one section
        return [("Full text", text)]

    sections: list[tuple[str, str]] = []

    # everything before first heading → Abstract
    first_start = matches[0].start()
    if first_start > 100:  # non-trivial preamble
        preamble = text[:first_start].strip()
        if preamble:
            sections.append(("Abstract", preamble))

    # split by headings
    for i, match in enumerate(matches):
        number = match.group(1).strip()
        title = match.group(2).strip()
        label = f"{number} {title}"

        body_start = match.end()
        body_end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[body_start:body_end].strip()

        if body:  # skip empty sections (e.g. if heading immediately followed by subheading)
            sections.append((label, body))

    return sections


def split_long_section(body: str, max_chars: int = MAX_CHUNK_CHARS) -> list[str]:
    """
    Split a too-long section body into sub-chunks by paragraph boundaries.

    Tries to keep each sub-chunk around TARGET_CHUNK_CHARS, never exceeding
    MAX_CHUNK_CHARS unless a single paragraph is itself that long.
    """
    if len(body) <= max_chars:
        return [body]

    paragraphs = re.split(r"\n\s*\n", body)
    sub_chunks: list[str] = []
    current = ""

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        # if adding this paragraph would exceed target, flush current
        if current and len(current) + len(para) + 2 > TARGET_CHUNK_CHARS:
            sub_chunks.append(current.strip())
            current = para
        else:
            current = f"{current}\n\n{para}" if current else para

    if current.strip():
        sub_chunks.append(current.strip())

    return sub_chunks


def build_chunks_for_paper(pdf_path: Path) -> list[Chunk]:
    """Extract, clean, and chunk a single paper."""
    stem = pdf_path.stem
    if stem not in PAPER_METADATA:
        raise ValueError(
            f"No metadata registered for {stem}. Add it to PAPER_METADATA in knowledge.py."
        )
    paper_meta = PAPER_METADATA[stem]

    raw_text = extract_text_from_pdf(pdf_path)
    cleaned = clean_text(raw_text)
    sections = split_into_sections(cleaned)

    chunks: list[Chunk] = []

    # pending: chunks too short to stand alone, waiting to merge with next
    pending_text = ""
    pending_section = ""

    for section_label, section_body in sections:
        # stop at references / acknowledgments
        if re.match(r"^\d+\.?\s*(REFERENCES|ACKNOWLEDGMENTS|ACKNOWLEDGEMENTS)",
                    section_label, re.IGNORECASE):
            break

        sub_bodies = split_long_section(section_body)

        for sub_idx, sub_body in enumerate(sub_bodies):
            # merge with pending if there's anything too-short hanging
            if pending_text:
                sub_body = f"[{pending_section}]\n{pending_text}\n\n[{section_label}]\n{sub_body}"
                pending_text = ""
                pending_section = ""

            # check minimum size
            if len(sub_body) < MIN_CHUNK_CHARS:
                pending_text = sub_body
                pending_section = section_label
                continue

            # build chunk
            chunk_meta = {
                **paper_meta,
                "section": section_label,
            }
            # make chunk_id unique even when a section gets split into sub-chunks
            suffix = f"_{sub_idx}" if len(sub_bodies) > 1 else ""
            chunk_id = f"{paper_meta['source']}__{_slug(section_label)}{suffix}"

            chunks.append(Chunk(chunk_id=chunk_id, text=sub_body, metadata=chunk_meta))

    # flush any pending short chunk as its own
    if pending_text and len(pending_text) >= 50:
        chunk_meta = {**paper_meta, "section": pending_section}
        chunk_id = f"{paper_meta['source']}__{_slug(pending_section)}_tail"
        chunks.append(Chunk(chunk_id=chunk_id, text=pending_text, metadata=chunk_meta))

    return chunks


def _slug(s: str) -> str:
    """Turn 'Section 4.1 Tag behaviour' into '4_1_tag_behaviour' for chunk IDs."""
    s = s.lower()
    s = re.sub(r"[^\w\s]", "", s)
    s = re.sub(r"\s+", "_", s.strip())
    return s[:60]


# ---------------------------------------------------------------------------
# ChromaDB indexing
# ---------------------------------------------------------------------------

def get_collection(rebuild: bool = False):
    """Return a ChromaDB collection, optionally dropping and recreating first."""
    import chromadb
    from chromadb.utils import embedding_functions

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL
    )

    if rebuild:
        try:
            client.delete_collection(COLLECTION_NAME)
            print(f"  dropped existing collection '{COLLECTION_NAME}'")
        except Exception:
            pass  # didn't exist

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embed_fn,
        metadata={"description": "Freesound academic papers for RAG"},
    )
    return collection


def build_index(rebuild: bool = True) -> None:
    """Read all PDFs in papers/, chunk them, embed, and store in ChromaDB."""
    pdf_files = sorted(PAPERS_DIR.glob("*.pdf"))
    if not pdf_files:
        print(f"ERROR: no PDFs found in {PAPERS_DIR}", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(pdf_files)} PDF(s) in {PAPERS_DIR}")

    all_chunks: list[Chunk] = []
    for pdf in pdf_files:
        print(f"\n→ Processing {pdf.name}")
        chunks = build_chunks_for_paper(pdf)
        print(f"    {len(chunks)} chunks produced")
        all_chunks.extend(chunks)

    print(f"\nTotal chunks across all papers: {len(all_chunks)}")

    # Get collection (rebuild=True drops existing one)
    collection = get_collection(rebuild=rebuild)

    # Add to ChromaDB in a single batch — sentence-transformers handles embedding
    print(f"\nEmbedding and indexing with {EMBEDDING_MODEL}...")
    collection.add(
        ids=[c.chunk_id for c in all_chunks],
        documents=[c.text for c in all_chunks],
        metadatas=[c.metadata for c in all_chunks],
    )
    print(f"✓ Indexed {collection.count()} chunks into '{COLLECTION_NAME}'")
    print(f"  Stored at: {CHROMA_DIR}")


# ---------------------------------------------------------------------------
# Query interface (used by agent.py)
# ---------------------------------------------------------------------------

def query_audio_knowledge(question: str, top_k: int = 3) -> list[dict]:
    """
    Retrieve the top-k most relevant passages from the knowledge base.

    Args:
        question: natural-language query.
        top_k: number of passages to return (default 3).

    Returns:
        List of dicts with keys: text, source, paper_title, authors, year,
        venue, section, distance (lower = more relevant).
        Returns empty list if the index doesn't exist yet.
    """
    if not CHROMA_DIR.exists():
        return []

    try:
        collection = get_collection(rebuild=False)
    except Exception as e:
        print(f"[query_audio_knowledge] could not load collection: {e}", file=sys.stderr)
        return []

    if collection.count() == 0:
        return []

    results = collection.query(query_texts=[question], n_results=top_k)

    passages: list[dict] = []
    # ChromaDB returns parallel lists; since we queried one text, index [0] everywhere
    for i in range(len(results["ids"][0])):
        meta = results["metadatas"][0][i]
        passages.append({
            "text": results["documents"][0][i],
            "distance": results["distances"][0][i] if results.get("distances") else None,
            "source": meta.get("source", "unknown"),
            "paper_title": meta.get("paper_title", ""),
            "authors": meta.get("authors", ""),
            "year": meta.get("year"),
            "venue": meta.get("venue", ""),
            "section": meta.get("section", ""),
        })
    return passages


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Freesound RAG knowledge base")
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Rebuild the index from scratch (required on first run and after PDF changes)",
    )
    parser.add_argument(
        "--query",
        type=str,
        help="Run a test query and print results",
    )
    parser.add_argument(
        "--inspect",
        action="store_true",
        help="Print all chunks without embedding (for debugging chunking logic)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Number of results for --query (default 3)",
    )
    args = parser.parse_args()

    if args.inspect:
        pdf_files = sorted(PAPERS_DIR.glob("*.pdf"))
        for pdf in pdf_files:
            print(f"\n{'='*70}\n{pdf.name}\n{'='*70}")
            chunks = build_chunks_for_paper(pdf)
            for c in chunks:
                print(f"\n--- {c.chunk_id} ({len(c.text)} chars) ---")
                print(f"Section: {c.metadata.get('section')}")
                preview = c.text[:200].replace("\n", " ")
                print(f"Preview: {preview}...")
            print(f"\n[{pdf.name}: {len(chunks)} chunks total]")
        return

    if args.rebuild:
        build_index(rebuild=True)

    if args.query:
        results = query_audio_knowledge(args.query, top_k=args.top_k)
        if not results:
            print("No results. Did you run with --rebuild yet?")
            return
        print(f"\nTop {len(results)} results for: {args.query!r}\n")
        for i, r in enumerate(results, 1):
            print(f"{'─'*70}")
            print(f"[{i}] {r['source']} — {r['section']}  (distance={r['distance']:.4f})")
            print(f"    {r['paper_title']}")
            preview = r['text'][:400].replace("\n", " ")
            print(f"\n{preview}...\n")

    if not args.rebuild and not args.query and not args.inspect:
        parser.print_help()


if __name__ == "__main__":
    main()
