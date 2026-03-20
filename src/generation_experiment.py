"""
BNS LLM Generation Experiment
================================
Full 300-query inference with three prompt templates via LLaMA-3 (Ollama).

Templates:
  P1 (Basic):           Direct context-to-answer
  P2 (Structured):      Expert-guided with explicit reasoning
  P3 (Chain-of-Thought): Multi-step legal deduction

Output:
  - results/generation_results.csv
  - results/generation_details.json

Environment: conda run -n bns_rag python src/generation_experiment.py
"""

import json
import os
import re
import sys
import time
import csv
import numpy as np
import faiss
import requests
from collections import defaultdict
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
TIMEOUT = 120
MAX_RETRIES = 2

CATEGORIES = ["direct", "paraphrased", "scenario", "multi-section", "confusing"]
CAT_LABELS = {
    "direct": "Direct", "paraphrased": "Paraphrased", "scenario": "Scenario",
    "multi-section": "Multi-section", "confusing": "Confusing",
}


# ── Prompt Templates ─────────────────────────────────────────────────────

def build_context_block(context_chunks: list) -> str:
    """Format retrieved chunks into a context block."""
    return "\n\n".join(
        f"[Section {c['section_id']} — {c['title']}]\n{c['text']}"
        for c in context_chunks
    )


def prompt_p1(query: str, context_chunks: list) -> str:
    """P1 — Basic: Direct context-to-answer mapping."""
    context = build_context_block(context_chunks)
    return f"""Given the following legal provisions from the Bharatiya Nyaya Sanhita (BNS):

{context}

Query: {query}

Which BNS section is most applicable? Answer with just the section number in the format: Section [NUMBER]"""


def prompt_p2(query: str, context_chunks: list) -> str:
    """P2 — Structured: Expert-guided with reasoning."""
    context = build_context_block(context_chunks)
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


def prompt_p3(query: str, context_chunks: list) -> str:
    """P3 — Chain-of-Thought: Multi-step legal deduction."""
    context = build_context_block(context_chunks)
    return f"""You are a senior legal analyst performing step-by-step legal reasoning under the Bharatiya Nyaya Sanhita (BNS), 2023.

**Legal Provisions:**
{context}

**Legal Query:** {query}

**Perform the following analysis step by step:**

Step 1 — Identify the Legal Issue: What is the core legal question or act described in the query?

Step 2 — Map to Provisions: Which of the retrieved provisions address this legal issue? List the relevant section numbers.

Step 3 — Eliminate Irrelevant Sections: Which provisions can be ruled out, and why?

Step 4 — Final Determination: Based on the above analysis, which single section is the MOST applicable?

**Final Answer:** Section [NUMBER]

You MUST conclude with "Final Answer: Section [NUMBER]" using a specific section number."""


PROMPT_TEMPLATES = {
    "P1_Basic": prompt_p1,
    "P2_Structured": prompt_p2,
    "P3_ChainOfThought": prompt_p3,
}


# ── Section Extraction ───────────────────────────────────────────────────

def extract_section(response: str) -> str:
    """
    Robust regex extraction of BNS section number from LLM response.
    Multi-priority matching for various LLM output formats.
    """
    if not response or response.startswith("[ERROR"):
        return ""

    # Priority 1: "Final Answer: Section NNN"
    m = re.search(r'final\s+answer[:\s]*section\s+(\d+)', response, re.IGNORECASE)
    if m:
        return m.group(1)

    # Priority 2: "Applicable Section: Section NNN"
    m = re.search(r'applicable\s+section[:\s]*section\s+(\d+)', response, re.IGNORECASE)
    if m:
        return m.group(1)

    # Priority 3: "Section NNN of BNS" or "Section NNN"
    matches = re.findall(r'\bsection\s+(\d+)', response, re.IGNORECASE)
    if matches:
        # Return the last match (usually the conclusion)
        return matches[-1]

    # Priority 4: "§NNN"
    m = re.search(r'§\s*(\d+)', response)
    if m:
        return m.group(1)

    # Priority 5: "BNS NNN" or "S. NNN"
    m = re.search(r'(?:BNS|S\.)\s*(\d+)', response, re.IGNORECASE)
    if m:
        return m.group(1)

    # Priority 6: Standalone number near end of response (last resort)
    last_200 = response[-200:] if len(response) > 200 else response
    m = re.search(r'\b(\d{1,3})\b', last_200)
    if m:
        return m.group(1)

    return ""


# ── Ground Truth ──────────────────────────────────────────────────────────

def get_correct_sections(query: dict) -> set:
    cs = query["correct_section"]
    return {str(s) for s in cs} if isinstance(cs, list) else {str(cs)}


# ── Ollama API ────────────────────────────────────────────────────────────

def call_ollama(prompt: str) -> str:
    """Call Ollama with retries and graceful error handling."""
    for attempt in range(MAX_RETRIES + 1):
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
            if attempt < MAX_RETRIES:
                time.sleep(2)
                continue
            return "[ERROR: Ollama timeout after retries]"
        except requests.exceptions.ConnectionError:
            if attempt < MAX_RETRIES:
                time.sleep(5)
                continue
            return "[ERROR: Ollama connection failed after retries]"
        except Exception as e:
            return f"[ERROR: {str(e)}]"
    return "[ERROR: Unknown failure]"


# ── Retrieval ─────────────────────────────────────────────────────────────

def retrieve_chunks(query_text, model, faiss_index, chunks, metadata, top_k):
    """Retrieve top-K chunks for a query."""
    q_emb = model.encode([query_text], normalize_embeddings=True)
    q_emb = np.array(q_emb, dtype=np.float32)
    _, indices = faiss_index.search(q_emb, top_k)

    context = []
    for idx in indices[0]:
        if 0 <= idx < len(metadata):
            context.append({
                "section_id": str(metadata[idx]["section_id"]),
                "title": metadata[idx]["title"],
                "text": chunks[idx]["text"],
            })
    return context


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("=" * 75)
    print("BNS LLM GENERATION EXPERIMENT")
    print(f"Model: {LLM_MODEL} | Temp: {TEMPERATURE} | Top-K Retrieval: {TOP_K_RETRIEVAL}")
    print(f"Templates: {list(PROMPT_TEMPLATES.keys())}")
    print("=" * 75)

    # Load data
    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    with open(QUERIES_PATH, "r", encoding="utf-8") as f:
        queries = json.load(f)
    with open(os.path.join(INDEX_DIR, META_FILE), "r", encoding="utf-8") as f:
        metadata = json.load(f)

    faiss_index = faiss.read_index(os.path.join(INDEX_DIR, INDEX_FILE))
    dense_model = SentenceTransformer(DENSE_MODEL)
    print(f"  ✅ {len(chunks)} chunks, FAISS {faiss_index.ntotal} vectors")
    print(f"  ✅ {len(queries)} queries loaded\n")

    # Verify Ollama
    try:
        r = requests.get("http://localhost:11434/api/tags", timeout=5)
        print(f"  ✅ Ollama running\n")
    except Exception:
        print("  ❌ Ollama not reachable at localhost:11434")
        sys.exit(1)

    # Pre-retrieve contexts (shared across all templates)
    print("─── Pre-retrieving contexts ───")
    all_contexts = []
    for q in tqdm(queries, desc="  Retrieval"):
        ctx = retrieve_chunks(q["query"], dense_model, faiss_index, chunks, metadata,
                              TOP_K_RETRIEVAL)
        all_contexts.append(ctx)
    print()

    # Run each template
    csv_rows = []
    all_details = {}

    for template_name, template_fn in PROMPT_TEMPLATES.items():
        print(f"─── Template: {template_name} {'─' * (50 - len(template_name))}")

        correct_count = 0
        total_latency = 0.0
        cat_stats = defaultdict(lambda: {"total": 0, "correct": 0})
        template_details = []

        for qi, query in enumerate(tqdm(queries, desc=f"  {template_name}")):
            context = all_contexts[qi]
            prompt = template_fn(query["query"], context)

            t0 = time.time()
            response = call_ollama(prompt)
            latency = time.time() - t0
            total_latency += latency

            predicted = extract_section(response)
            ground_truth = get_correct_sections(query)
            is_correct = predicted in ground_truth
            cat = query.get("category", "unknown")

            cat_stats[cat]["total"] += 1
            if is_correct:
                correct_count += 1
                cat_stats[cat]["correct"] += 1

            template_details.append({
                "query_id": query["query_id"],
                "query": query["query"],
                "category": cat,
                "correct_section": query["correct_section"],
                "predicted_section": predicted,
                "is_correct": is_correct,
                "latency_s": round(latency, 2),
                "response_preview": response[:200] if response else "",
            })

        n = len(queries)
        accuracy = correct_count / n
        avg_latency = total_latency / n

        print(f"\n  📊 {template_name}:")
        print(f"     Generation Accuracy: {accuracy:.4f} ({correct_count}/{n})")
        print(f"     Avg Latency:         {avg_latency:.1f}s/query")

        # Category breakdown
        print(f"     Category breakdown:")
        for cat in CATEGORIES:
            s = cat_stats[cat]
            cat_acc = s["correct"] / s["total"] if s["total"] > 0 else 0.0
            print(f"       {CAT_LABELS.get(cat, cat):<15} "
                  f"{cat_acc:.4f} ({s['correct']}/{s['total']})")

        csv_rows.append({
            "Template": template_name,
            "Accuracy": round(accuracy, 4),
            "Avg_Latency_s": round(avg_latency, 2),
        })

        # Add per-category columns
        for cat in CATEGORIES:
            s = cat_stats[cat]
            cat_acc = s["correct"] / s["total"] if s["total"] > 0 else 0.0
            csv_rows[-1][f"Cat_{CAT_LABELS.get(cat, cat)}"] = round(cat_acc, 4)

        all_details[template_name] = template_details
        print()

    # ── Save CSV ──────────────────────────────────────────────────
    csv_path = os.path.join(RESULTS_DIR, "generation_results.csv")
    fieldnames = ["Template", "Accuracy", "Avg_Latency_s"] + \
                 [f"Cat_{CAT_LABELS[c]}" for c in CATEGORIES]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)

    # ── Save Details JSON ─────────────────────────────────────────
    details_path = os.path.join(RESULTS_DIR, "generation_details.json")
    with open(details_path, "w", encoding="utf-8") as f:
        json.dump(all_details, f, indent=2, ensure_ascii=False)

    # ── Final Summary ─────────────────────────────────────────────
    print("=" * 75)
    print("GENERATION EXPERIMENT — FINAL RESULTS")
    print("=" * 75)
    print(f"\n| {'Template':<22} | {'Accuracy':>10} | {'Latency':>10} |"
          + "".join(f" {CAT_LABELS[c]:>12} |" for c in CATEGORIES))
    print(f"|{'-'*24}|{'-'*12}|{'-'*12}|"
          + "".join(f"{'-'*14}|" for _ in CATEGORIES))
    for r in csv_rows:
        line = f"| {r['Template']:<22} | {r['Accuracy']:>10.4f} | {r['Avg_Latency_s']:>9.1f}s |"
        for cat in CATEGORIES:
            line += f" {r.get(f'Cat_{CAT_LABELS[cat]}', 0):>12.4f} |"
        print(line)

    print(f"\n  ✅ CSV:     {csv_path}")
    print(f"  ✅ Details: {details_path}")
    print("=" * 75)


if __name__ == "__main__":
    main()
