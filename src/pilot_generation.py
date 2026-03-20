"""
BNS Pilot Generation Test
============================
End-to-end validation: Retrieval → Context Injection → LLM Call → Section Extraction.
Runs 20 queries through the Structured (P2) prompt with LLaMA-3 via Ollama.

Output: results/pilot_generation_results.json

Environment: conda run -n bns_rag python src/pilot_generation.py
"""

import json
import os
import re
import sys
import time
import numpy as np
import faiss
import requests
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# ── Configuration ─────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHUNKS_PATH = os.path.join(BASE_DIR, "data", "bns_chunks.json")
QUERIES_PATH = os.path.join(BASE_DIR, "data", "evaluation_queries_augmented.json")
INDEX_DIR = os.path.join(BASE_DIR, "indexes")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

DENSE_MODEL = "sentence-transformers/all-mpnet-base-v2"
INDEX_FILE = "mpnet.index"
META_FILE = "chunk_metadata.json"

OLLAMA_URL = "http://localhost:11434/api/generate"
LLM_MODEL = "llama3"
TEMPERATURE = 0.0
TOP_K_RETRIEVAL = 5
PILOT_SIZE = 20
TIMEOUT = 120  # seconds per LLM call


# ── Prompt Template (P2 — Structured) ────────────────────────────────────

def prompt_p2(query: str, context_chunks: list) -> str:
    context = "\n\n".join(
        f"[Section {c['section_id']} — {c['title']}]\n{c['text']}"
        for c in context_chunks
    )
    return f"""You are a legal expert specializing in the Bharatiya Nyaya Sanhita (BNS), 2023.

Based ONLY on the legal provisions below, identify the most applicable BNS section for the query.

**Retrieved Legal Provisions:**
{context}

**Query:** {query}

**Instructions:**
1. Analyze each retrieved provision for relevance to the query.
2. Identify the single most applicable BNS section.
3. Provide your answer in EXACTLY this format:

**Applicable Section:** Section [NUMBER]
**Reasoning:** [Brief explanation of why this section applies]

Important: Your answer MUST reference a specific section number from the provisions above."""


# ── Section Extraction ───────────────────────────────────────────────────

def extract_section(response: str) -> str:
    """
    Robust regex extraction of BNS section number from LLM response.
    Handles: "Section 55", "Section 55 of BNS", "section 55(1)", "§55", just "55".
    """
    # Priority 1: "Applicable Section: Section NNN"
    m = re.search(r'applicable\s+section[:\s]*section\s+(\d+)', response, re.IGNORECASE)
    if m:
        return m.group(1)

    # Priority 2: "Section NNN" anywhere
    m = re.search(r'\bsection\s+(\d+)', response, re.IGNORECASE)
    if m:
        return m.group(1)

    # Priority 3: "§NNN"
    m = re.search(r'§\s*(\d+)', response)
    if m:
        return m.group(1)

    # Priority 4: "BNS NNN" or "S. NNN"
    m = re.search(r'(?:BNS|S\.)\s*(\d+)', response, re.IGNORECASE)
    if m:
        return m.group(1)

    # Priority 5: Standalone number (last resort)
    m = re.search(r'\b(\d{1,3})\b', response)
    if m:
        return m.group(1)

    return ""


def get_correct_sections(query: dict) -> set:
    cs = query["correct_section"]
    return {str(s) for s in cs} if isinstance(cs, list) else {str(cs)}


# ── Ollama API Call ──────────────────────────────────────────────────────

def call_ollama(prompt: str) -> str:
    """Call Ollama API with graceful error handling."""
    try:
        resp = requests.post(
            OLLAMA_URL,
            json={
                "model": LLM_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": TEMPERATURE},
            },
            timeout=TIMEOUT,
        )
        resp.raise_for_status()
        return resp.json().get("response", "")
    except requests.exceptions.Timeout:
        return "[ERROR: Ollama timeout]"
    except requests.exceptions.ConnectionError:
        return "[ERROR: Ollama connection failed]"
    except Exception as e:
        return f"[ERROR: {str(e)}]"


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("=" * 70)
    print("BNS PILOT GENERATION TEST (G1)")
    print(f"Queries: {PILOT_SIZE} | Model: {LLM_MODEL} | Temp: {TEMPERATURE}")
    print(f"Retrieval: {DENSE_MODEL} | Top-K: {TOP_K_RETRIEVAL}")
    print("=" * 70)

    # Load data
    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    with open(QUERIES_PATH, "r", encoding="utf-8") as f:
        queries = json.load(f)[:PILOT_SIZE]
    with open(os.path.join(INDEX_DIR, META_FILE), "r", encoding="utf-8") as f:
        metadata = json.load(f)

    # Load retrieval components
    faiss_index = faiss.read_index(os.path.join(INDEX_DIR, INDEX_FILE))
    model = SentenceTransformer(DENSE_MODEL)
    print(f"  ✅ Loaded {len(chunks)} chunks, FAISS {faiss_index.ntotal} vectors\n")

    # Verify Ollama
    try:
        r = requests.get("http://localhost:11434/api/tags", timeout=5)
        models = [m["name"] for m in r.json().get("models", [])]
        print(f"  ✅ Ollama running — models: {models}\n")
    except Exception as e:
        print(f"  ❌ Ollama not reachable: {e}")
        sys.exit(1)

    # Run pilot
    results = []
    correct_count = 0

    for qi, query in enumerate(tqdm(queries, desc="  Pilot")):
        # Retrieve top-K chunks
        q_emb = model.encode([query["query"]], normalize_embeddings=True)
        q_emb = np.array(q_emb, dtype=np.float32)
        scores, indices = faiss_index.search(q_emb, TOP_K_RETRIEVAL)

        context_chunks = []
        for idx in indices[0]:
            if 0 <= idx < len(metadata):
                meta = metadata[idx]
                chunk_text = chunks[idx]["text"]
                context_chunks.append({
                    "section_id": meta["section_id"],
                    "title": meta["title"],
                    "text": chunk_text,
                })

        # Build prompt and call LLM
        prompt = prompt_p2(query["query"], context_chunks)
        t0 = time.time()
        response = call_ollama(prompt)
        latency = time.time() - t0

        # Extract and evaluate
        predicted = extract_section(response)
        ground_truth = get_correct_sections(query)
        is_correct = predicted in ground_truth

        if is_correct:
            correct_count += 1

        result = {
            "query_id": query["query_id"],
            "query": query["query"],
            "category": query.get("category", "unknown"),
            "correct_section": query["correct_section"],
            "predicted_section": predicted,
            "is_correct": is_correct,
            "retrieved_sections": [c["section_id"] for c in context_chunks],
            "latency_s": round(latency, 2),
            "response_preview": response[:300],
        }
        results.append(result)

    # Summary
    accuracy = correct_count / len(queries)
    print(f"\n{'=' * 70}")
    print(f"PILOT RESULTS")
    print(f"{'=' * 70}")
    print(f"  Accuracy: {accuracy:.4f} ({correct_count}/{len(queries)})")
    print(f"  Avg latency: {sum(r['latency_s'] for r in results) / len(results):.1f}s")

    errors = [r for r in results if not r["is_correct"]]
    if errors:
        print(f"\n  Misses ({len(errors)}):")
        for e in errors:
            print(f"    {e['query_id']}: predicted={e['predicted_section']}, "
                  f"expected={e['correct_section']}")

    # Save
    out_path = os.path.join(RESULTS_DIR, "pilot_generation_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"accuracy": round(accuracy, 4), "total": len(queries),
                    "correct": correct_count, "results": results},
                  f, indent=2, ensure_ascii=False)
    print(f"\n  ✅ Saved to {out_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
