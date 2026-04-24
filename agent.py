"""
Freesound Agent — Layer 1: Intelligent Search + Audio Analysis + Academic RAG
Uses Claude API (native tool use) + Freesound API + ChromaDB knowledge base

Setup:
  1. Get Freesound API key: https://freesound.org/apiv2/apply/
  2. pip install -r requirements.txt
  3. export ANTHROPIC_API_KEY="your-key"
  4. export FREESOUND_API_KEY="your-key"
  5. python knowledge.py --rebuild   # one-time: build RAG index
  6. python agent.py
"""

import os
import json
import anthropic
import freesound
import librosa
import numpy as np
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()  # load .env before reading environment variables

from knowledge import query_audio_knowledge

# ── Config ──────────────────────────────────────────────────────────
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
FREESOUND_API_KEY = os.environ.get("FREESOUND_API_KEY")
MODEL = "claude-haiku-4-5-20251001"
DOWNLOAD_DIR = Path("./downloads")
DOWNLOAD_DIR.mkdir(exist_ok=True)

# ── Freesound Client ────────────────────────────────────────────────
fs_client = freesound.FreesoundClient()
fs_client.set_token(FREESOUND_API_KEY, "token")

# ── Tool Definitions (Claude API format) ────────────────────────────
TOOLS = [
    {
        "name": "search_sounds",
        "description": (
            "Search the Freesound database for sounds matching a query. "
            "Returns a list of sounds with id, name, tags, duration, and username. "
            "Use filter syntax like 'duration:[5 TO 30]' for duration filtering, "
            "'tag:rain' for tag filtering. Sort options: score, duration_desc, "
            "duration_asc, created_desc, created_asc, downloads_desc, rating_desc."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query text, e.g. 'rain forest ambience'"
                },
                "filter": {
                    "type": "string",
                    "description": "Optional Freesound filter, e.g. 'duration:[10 TO 60] tag:nature'"
                },
                "sort": {
                    "type": "string",
                    "description": "Sort order. Default: 'score'",
                    "enum": ["score", "duration_desc", "duration_asc",
                             "created_desc", "created_asc",
                             "downloads_desc", "rating_desc"]
                },
                "num_results": {
                    "type": "integer",
                    "description": "Number of results to return (1-15). Default: 5"
                }
            },
            "required": ["query"]
        }
    },
    {
        "name": "get_sound_details",
        "description": (
            "Get detailed metadata for a specific Freesound sound by ID. "
            "Returns name, description, tags, duration, sample rate, channels, "
            "license, uploader, download count, average rating, geotag if available."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "sound_id": {
                    "type": "integer",
                    "description": "Freesound sound ID"
                }
            },
            "required": ["sound_id"]
        }
    },
    {
        "name": "analyze_audio",
        "description": (
            "Download a sound preview from Freesound and analyze its audio features "
            "using librosa. Returns: duration, sample rate, spectral centroid (mean/std), "
            "spectral bandwidth, zero crossing rate, estimated tempo, RMS energy, "
            "and MFCC summary (mean of first 13 coefficients). "
            "Useful for comparing acoustic characteristics across sounds."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "sound_id": {
                    "type": "integer",
                    "description": "Freesound sound ID to analyze"
                }
            },
            "required": ["sound_id"]
        }
    },
    {
        "name": "analyze_tag_distribution",
        "description": (
            "Search for sounds with a given tag and analyze the distribution "
            "of co-occurring tags, uploaders, and durations. Useful for "
            "understanding taxonomic biases in Freesound's commons — "
            "e.g. what other tags appear alongside 'rain'? Who records rain sounds? "
            "How long are they typically?"
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "tag": {
                    "type": "string",
                    "description": "Tag to analyze, e.g. 'rain', 'bird', 'city'"
                },
                "sample_size": {
                    "type": "integer",
                    "description": "Number of sounds to sample for analysis (5-50). Default: 20"
                }
            },
            "required": ["tag"]
        }
    },
    {
        "name": "query_audio_knowledge",
        "description": (
            "Search a curated knowledge base of peer-reviewed academic papers about "
            "Freesound's folksonomy, tag quality, and dataset design. Use this tool "
            "when the user asks WHY something about Freesound is the way it is, asks "
            "about technical terms like 'folksonomy', 'tag sparsity', 'narrow folksonomy', "
            "or wants to ground observations about Freesound data in prior research. "
            "Returns the most relevant passages from three papers by the Freesound "
            "research team (UPF MTG): Martínez et al. 2009 (SMC), Font et al. 2012 "
            "(ISMIR), and Fonseca et al. 2017 (ISMIR). When you use results from this "
            "tool, cite the source paper and section in your response."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": (
                        "A natural-language question or keyword phrase. "
                        "Examples: 'why is the tag distribution uneven?', "
                        "'what is a folksonomy?', 'how are tags validated?', "
                        "'tag sparsity issues', 'narrow folksonomy definition'."
                    )
                },
                "top_k": {
                    "type": "integer",
                    "description": "Number of passages to return (1-5). Default: 3"
                }
            },
            "required": ["question"]
        }
    }
]

# ── Tool Implementations ────────────────────────────────────────────

def search_sounds(query: str, filter: str = "", sort: str = "score",
                  num_results: int = 5) -> str:
    """Search Freesound and return formatted results."""
    try:
        params = {
            "query": query,
            "fields": "id,name,tags,duration,username,avg_rating,num_downloads,previews",
            "page_size": min(num_results, 15),
            "sort": sort,
        }
        if filter:
            params["filter"] = filter

        results = fs_client.text_search(**params)
        sounds = []
        for sound in results:
            sounds.append({
                "id": sound.id,
                "name": sound.name,
                "tags": sound.tags[:10],  # limit tags for readability
                "duration_sec": round(sound.duration, 1),
                "uploader": sound.username,
                "avg_rating": getattr(sound, 'avg_rating', None),
                "downloads": getattr(sound, 'num_downloads', None),
                "preview_url": sound.previews.preview_hq_mp3 if hasattr(sound, 'previews') else None,
            })

        return json.dumps({
            "count": len(sounds),
            "query": query,
            "results": sounds
        }, indent=2)

    except Exception as e:
        return json.dumps({"error": str(e)})


def get_sound_details(sound_id: int) -> str:
    """Get detailed metadata for a specific sound."""
    try:
        sound = fs_client.get_sound(
            sound_id,
            fields="id,name,description,tags,duration,samplerate,channels,"
                   "license,username,num_downloads,avg_rating,geotag,created"
        )
        details = {
            "id": sound.id,
            "name": sound.name,
            "description": getattr(sound, 'description', '')[:500],
            "tags": sound.tags,
            "duration_sec": round(sound.duration, 1),
            "sample_rate": getattr(sound, 'samplerate', None),
            "channels": getattr(sound, 'channels', None),
            "license": getattr(sound, 'license', None),
            "uploader": sound.username,
            "downloads": getattr(sound, 'num_downloads', None),
            "avg_rating": getattr(sound, 'avg_rating', None),
            "geotag": getattr(sound, 'geotag', None),
            "created": getattr(sound, 'created', None),
        }
        return json.dumps(details, indent=2)

    except Exception as e:
        return json.dumps({"error": str(e)})


def analyze_audio(sound_id: int) -> str:
    """Download preview and analyze with librosa."""
    try:
        # Get sound info with preview URL
        sound = fs_client.get_sound(
            sound_id,
            fields="id,name,previews"
        )

        # Download preview
        preview_path = DOWNLOAD_DIR / f"{sound_id}.mp3"
        if not preview_path.exists():
            sound.retrieve_preview(str(DOWNLOAD_DIR), f"{sound_id}.mp3")

        # Load and analyze with librosa
        y, sr = librosa.load(str(preview_path), sr=None, mono=True)
        duration = librosa.get_duration(y=y, sr=sr)

        # Spectral features
        spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
        zcr = librosa.feature.zero_crossing_rate(y)[0]
        rms = librosa.feature.rms(y=y)[0]

        # Tempo
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)

        # MFCCs
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfcc_means = np.mean(mfccs, axis=1).tolist()

        analysis = {
            "sound_id": sound_id,
            "name": sound.name,
            "duration_sec": round(duration, 2),
            "sample_rate": sr,
            "spectral_centroid": {
                "mean": round(float(np.mean(spec_cent)), 1),
                "std": round(float(np.std(spec_cent)), 1)
            },
            "spectral_bandwidth_mean": round(float(np.mean(spec_bw)), 1),
            "zero_crossing_rate_mean": round(float(np.mean(zcr)), 4),
            "rms_energy_mean": round(float(np.mean(rms)), 4),
            "estimated_tempo": round(float(tempo), 1) if not isinstance(tempo, np.ndarray) else round(float(tempo[0]), 1),
            "mfcc_means": [round(m, 2) for m in mfcc_means],
        }
        return json.dumps(analysis, indent=2)

    except Exception as e:
        return json.dumps({"error": str(e)})


def analyze_tag_distribution(tag: str, sample_size: int = 20) -> str:
    """Analyze co-occurring tags, uploaders, durations for a given tag."""
    try:
        results = fs_client.text_search(
            query=tag,
            filter=f"tag:{tag}",
            fields="id,name,tags,duration,username",
            page_size=min(sample_size, 50),
            sort="score"
        )

        all_tags = {}
        uploaders = {}
        durations = []

        for sound in results:
            # Count co-occurring tags
            for t in sound.tags:
                t_lower = t.lower()
                if t_lower != tag.lower():
                    all_tags[t_lower] = all_tags.get(t_lower, 0) + 1

            # Count uploaders
            uploaders[sound.username] = uploaders.get(sound.username, 0) + 1

            # Collect durations
            durations.append(sound.duration)

        # Sort tags by frequency
        top_tags = sorted(all_tags.items(), key=lambda x: x[1], reverse=True)[:15]

        # Sort uploaders by frequency
        top_uploaders = sorted(uploaders.items(), key=lambda x: x[1], reverse=True)[:10]

        analysis = {
            "tag": tag,
            "sample_size": len(durations),
            "co_occurring_tags": [
                {"tag": t, "count": c} for t, c in top_tags
            ],
            "top_uploaders": [
                {"username": u, "count": c} for u, c in top_uploaders
            ],
            "duration_stats": {
                "min_sec": round(min(durations), 1) if durations else 0,
                "max_sec": round(max(durations), 1) if durations else 0,
                "mean_sec": round(np.mean(durations), 1) if durations else 0,
                "median_sec": round(float(np.median(durations)), 1) if durations else 0,
            },
            "observations": (
                f"Based on {len(durations)} samples with tag '{tag}': "
                f"The most common co-occurring tags reveal the recording culture "
                f"and taxonomic biases of Freesound contributors."
            )
        }
        return json.dumps(analysis, indent=2)

    except Exception as e:
        return json.dumps({"error": str(e)})


def query_knowledge_base(question: str, top_k: int = 3) -> str:
    """Retrieve relevant passages from the academic RAG knowledge base."""
    try:
        passages = query_audio_knowledge(question, top_k=top_k)

        if not passages:
            return json.dumps({
                "error": (
                    "Knowledge base is empty or not yet indexed. "
                    "Run `python knowledge.py --rebuild` first."
                ),
                "passages": []
            })

        # Shape the output for Claude: compact but with full citation metadata
        formatted = {
            "question": question,
            "num_passages": len(passages),
            "passages": [
                {
                    "rank": i + 1,
                    "source": p["source"],            # e.g. "martinez_2009"
                    "paper_title": p["paper_title"],
                    "authors": p["authors"],
                    "year": p["year"],
                    "venue": p["venue"],              # e.g. "ISMIR"
                    "section": p["section"],          # e.g. "4.1 Tag behaviour"
                    "distance": round(p["distance"], 4) if p["distance"] is not None else None,
                    "text": p["text"],
                }
                for i, p in enumerate(passages)
            ],
            "citation_hint": (
                "When using these passages in your response, cite as: "
                "(Author et al., Year, Section X.Y). For example: "
                "(Martínez et al., 2009, Section 4.1)."
            )
        }
        return json.dumps(formatted, indent=2, ensure_ascii=False)

    except Exception as e:
        return json.dumps({"error": str(e), "passages": []})


# ── Tool Dispatcher ─────────────────────────────────────────────────

def dispatch_tool(tool_name: str, tool_input: dict) -> str:
    """Route tool calls to implementations."""
    handlers = {
        "search_sounds": search_sounds,
        "get_sound_details": get_sound_details,
        "analyze_audio": analyze_audio,
        "analyze_tag_distribution": analyze_tag_distribution,
        "query_audio_knowledge": query_knowledge_base,
    }
    handler = handlers.get(tool_name)
    if handler:
        return handler(**tool_input)
    return json.dumps({"error": f"Unknown tool: {tool_name}"})


# ── Agent Loop ──────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are the Freesound Agent — an intelligent assistant for exploring, 
analyzing, and understanding the Freesound audio commons.

You have access to five tools:
1. search_sounds — Search Freesound using natural language queries and filters.
2. get_sound_details — Get detailed metadata about specific sounds.
3. analyze_audio — Analyze audio features (spectral, temporal, MFCC) using librosa.
4. analyze_tag_distribution — Analyze tag distributions to reveal taxonomic biases.
5. query_audio_knowledge — Retrieve passages from peer-reviewed papers about 
   Freesound's folksonomy, tag quality, and dataset design.

When users ask you to explore sounds, you should:
- Translate natural language into effective Freesound queries and filters
- Use audio analysis to compare acoustic characteristics across sounds
- Point out interesting patterns in tag distributions, uploader demographics, 
  and duration statistics that reveal the "materiological biases" of the commons
  (i.e., what gets recorded, by whom, and how that shapes the database)

When users ask WHY things are the way they are in Freesound — e.g. why tags are 
uneven, what folksonomy means, how dataset quality is controlled — use 
`query_audio_knowledge` FIRST to ground your answer in the actual research 
literature from the Freesound research team (UPF MTG). When you use passages 
from this tool, cite them explicitly: e.g. "According to Martínez et al. (2009, 
Section 4.1), ...". Do not paraphrase extensively — prefer short attributed 
summaries. If the knowledge base does not contain a relevant answer, say so 
rather than making one up.

You understand audio and music concepts deeply. When presenting analysis results,
explain them in terms a sound designer or audio researcher would appreciate —
reference spectral characteristics, timbral qualities, and acoustic properties.

Keep responses concise and insightful. When you spot interesting patterns or biases,
highlight them — that's the core value of this tool."""


def run_agent(user_message: str, conversation: list = None):
    """Run one turn of the agent loop with tool use."""
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

    if conversation is None:
        conversation = []

    conversation.append({"role": "user", "content": user_message})

    # Agent loop: keep going until Claude stops calling tools
    while True:
        response = client.messages.create(
            model=MODEL,
            max_tokens=4096,
            system=SYSTEM_PROMPT,
            tools=TOOLS,
            messages=conversation,
        )

        # Collect all content blocks
        assistant_content = response.content
        conversation.append({"role": "assistant", "content": assistant_content})

        # Check if there are tool calls
        tool_uses = [b for b in assistant_content if b.type == "tool_use"]

        if not tool_uses:
            # No more tool calls — extract final text and return
            text_blocks = [b.text for b in assistant_content if b.type == "text"]
            return "\n".join(text_blocks), conversation

        # Process each tool call
        tool_results = []
        for tool_use in tool_uses:
            print(f"  🔧 Calling: {tool_use.name}({json.dumps(tool_use.input, ensure_ascii=False)[:80]}...)")
            result = dispatch_tool(tool_use.name, tool_use.input)
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": tool_use.id,
                "content": result,
            })

        conversation.append({"role": "user", "content": tool_results})


# ── CLI Interface ───────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("🎵 Freesound Agent")
    print("   Explore, analyze, and understand the Freesound commons")
    print("=" * 60)
    print()
    print("Examples:")
    print('  > Find 5 rain recordings longer than 30 seconds')
    print('  > Analyze the audio features of sound #12345')
    print('  > What tags co-occur with "bird"? Any bias patterns?')
    print('  > Compare spectral features of thunder vs rain recordings')
    print('  > Why is Freesound\'s tag distribution uneven? (uses RAG)')
    print('  > What does "folksonomy" mean in Freesound research? (uses RAG)')
    print()
    print("Type 'quit' to exit.")
    print("-" * 60)

    conversation = []

    while True:
        try:
            user_input = input("\n🎤 You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            print("Bye!")
            break

        print("\n🤖 Agent thinking...")
        try:
            response, conversation = run_agent(user_input, conversation)
            print(f"\n🤖 Agent:\n{response}")
        except Exception as e:
            print(f"\n❌ Error: {e}")


if __name__ == "__main__":
    main()
