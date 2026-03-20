"""
BNS Full Matrix Runner
========================
36-configuration grid search: 2 Models × 3 Prompts × 2 Top-K × 3 Temps.

Grid:
  Models:  [llama3, mistral]
  Prompts: [P1_Basic, P2_Structured, P3_ChainOfThought]
  Top-K:   [3, 5]
  Temps:   [0.0, 0.3, 0.7]

Features:
  - Safety reframing prefix for clinical legal analysis
  - Resume capability (skips completed configs)
  - Tracks: is_correct, latency_ms, is_hallucination, safety_refusal
  - Full 300-query benchmark
  - GPU-accelerated embeddings (RTX 3050)
  - Connection-pooled HTTP for Ollama
  - Pre-compiled regex for fast extraction
  - Pre-cached context chunks per top-k

Output:
  - results/matrix_results.csv       (per-config aggregates)
  - results/matrix_details.json      (per-query details, append-safe)
  - results/matrix_checkpoint.json   (resume state)

Environment: conda run -n bns_rag python src/full_matrix_runner.py
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
TIMEOUT = 120
MAX_RETRIES = 2

# Grid parameters
MODELS = ["llama3:latest", "mistral:7b-instruct"]
TOP_K_VALUES = [3, 5]
TEMPERATURES = [0.0, 0.3, 0.7]

CATEGORIES = ["direct", "paraphrased", "scenario", "multi-section", "confusing"]
CAT_LABELS = {
    "direct": "Direct", "paraphrased": "Paraphrased", "scenario": "Scenario",
    "multi-section": "Multi-section", "confusing": "Confusing",
}

# Safety reframing prefix
SAFETY_PREFIX = (
    "LEGAL RESEARCH MODE: You are a clinical statutory index. "
    "Analyze the provided legal context and return the section number. "
    "Do not filter legal definitions.\n\n"
)

CHECKPOINT_FILE = os.path.join(RESULTS_DIR, "matrix_checkpoint.json")
DETAILS_FILE = os.path.join(RESULTS_DIR, "matrix_details.json")


# ── Pre-compiled Regex Patterns (compiled once, reused every call) ────────
_RE_FINAL_ANSWER = re.compile(r'final\s+answer[:\s]*section\s+(\d+)', re.IGNORECASE)
_RE_APPLICABLE   = re.compile(r'applicable\s+section[:\s]*section\s+(\d+)', re.IGNORECASE)
_RE_SECTION_ALL  = re.compile(r'\bsection\s+(\d+)', re.IGNORECASE)
_RE_BNS_SHORT    = re.compile(r'(?:BNS|S\.|§)\s*(\d+)', re.IGNORECASE)
_RE_LAST_NUMBER  = re.compile(r'\b(\d{1,3})\b')

_REFUSAL_RE_LIST = [
    re.compile(p, re.IGNORECASE) for p in [
        r"i cannot", r"i can't", r"i'm not able to", r"i am not able to",
        r"not appropriate", r"i must decline", r"cannot provide",
        r"unable to assist", r"not comfortable", r"refrain from",
        r"against my guidelines", r"i apologize", r"i won't",
    ]
]


# ── Prompt Templates ─────────────────────────────────────────────────────

def build_context(context_chunks):
    return "\n\n".join(
        f"[Section {c['section_id']} - {c['title']}]\n{c['text']}"
        for c in context_chunks
    )


def prompt_p1(query, context_chunks):
    ctx = build_context(context_chunks)
    return (
        f"{SAFETY_PREFIX}"
        f"Given the following legal provisions from the Bharatiya Nyaya Sanhita (BNS):\n\n"
        f"{ctx}\n\n"
        f"Query: {query}\n\n"
        f"Which BNS section is most applicable? Answer with just the section number "
        f"in the format: Section [NUMBER]"
    )


def prompt_p2(query, context_chunks):
    ctx = build_context(context_chunks)
    return (
        f"{SAFETY_PREFIX}"
        f"You are a legal expert specializing in the Bharatiya Nyaya Sanhita (BNS), 2023.\n\n"
        f"Based ONLY on the legal provisions below, identify the most applicable BNS section.\n\n"
        f"**Retrieved Legal Provisions:**\n{ctx}\n\n"
        f"**Query:** {query}\n\n"
        f"**Instructions:**\n"
        f"1. Analyze each retrieved provision for relevance.\n"
        f"2. Identify the single most applicable BNS section.\n"
        f"3. Answer in EXACTLY this format:\n\n"
        f"**Applicable Section:** Section [NUMBER]\n"
        f"**Reasoning:** [Brief explanation]\n\n"
        f"Important: Your answer MUST reference a specific section number."
    )


def prompt_p3(query, context_chunks):
    ctx = build_context(context_chunks)
    return (
        f"{SAFETY_PREFIX}"
        f"You are a senior legal analyst performing step-by-step reasoning under "
        f"the Bharatiya Nyaya Sanhita (BNS), 2023.\n\n"
        f"**Legal Provisions:**\n{ctx}\n\n"
        f"**Legal Query:** {query}\n\n"
        f"**Perform the following analysis step by step:**\n\n"
        f"Step 1 - Identify the Legal Issue: What is the core legal question?\n\n"
        f"Step 2 - Map to Provisions: Which provisions address this issue?\n\n"
        f"Step 3 - Eliminate Irrelevant Sections: Which can be ruled out?\n\n"
        f"Step 4 - Final Determination: Which single section is MOST applicable?\n\n"
        f"**Final Answer:** Section [NUMBER]\n\n"
        f"You MUST conclude with 'Final Answer: Section [NUMBER]'."
    )


PROMPT_FNS = {
    "P1_Basic": prompt_p1,
    "P2_Structured": prompt_p2,
    "P3_ChainOfThought": prompt_p3,
}


# ── Section Extraction (uses pre-compiled regex) ─────────────────────────

def extract_section(response):
    if not response or response.startswith("[ERROR"):
        return ""
    m = _RE_FINAL_ANSWER.search(response)
    if m:
        return m.group(1)
    m = _RE_APPLICABLE.search(response)
    if m:
        return m.group(1)
    matches = _RE_SECTION_ALL.findall(response)
    if matches:
        return matches[-1]
    m = _RE_BNS_SHORT.search(response)
    if m:
        return m.group(1)
    last = response[-200:] if len(response) > 200 else response
    m = _RE_LAST_NUMBER.search(last)
    return m.group(1) if m else ""


def get_correct(q):
    cs = q["correct_section"]
    return {str(s) for s in cs} if isinstance(cs, list) else {str(cs)}


def detect_safety_refusal(response):
    resp_lower = response.lower()
    return any(p.search(resp_lower) for p in _REFUSAL_RE_LIST)


def detect_hallucination(predicted, context_sections):
    """Takes a pre-computed set of context section IDs for speed."""
    if not predicted:
        return False
    return predicted not in context_sections


# ── Ollama API (connection-pooled session) ────────────────────────────────

def create_session():
    """Create a requests.Session with connection pooling for Ollama."""
    session = requests.Session()
    adapter = requests.adapters.HTTPAdapter(
        pool_connections=1,
        pool_maxsize=1,
        max_retries=0,  # we handle retries manually
    )
    session.mount("http://", adapter)
    return session


def call_ollama(session, prompt, model, temperature):
    payload = {
        "model": model, "prompt": prompt, "stream": False,
        "options": {"temperature": temperature},
    }
    for attempt in range(MAX_RETRIES + 1):
        try:
            resp = session.post(OLLAMA_URL, json=payload, timeout=TIMEOUT)
            resp.raise_for_status()
            return resp.json().get("response", "")
        except requests.exceptions.Timeout:
            if attempt < MAX_RETRIES:
                time.sleep(2)
                continue
            return "[ERROR: timeout]"
        except requests.exceptions.ConnectionError:
            if attempt < MAX_RETRIES:
                time.sleep(5)
                continue
            return "[ERROR: connection]"
        except Exception as e:
            return f"[ERROR: {e}]"
    return "[ERROR: unknown]"


# ── Resume Logic ──────────────────────────────────────────────────────────

def config_key(model, prompt, top_k, temp):
    return f"{model}|{prompt}|k{top_k}|t{temp}"


def load_checkpoint():
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, "r") as f:
            return json.load(f)
    return {"completed": [], "results": []}


def save_checkpoint(checkpoint):
    with open(CHECKPOINT_FILE, "w") as f:
        json.dump(checkpoint, f, indent=2)


# ── Helper: flush-safe print ─────────────────────────────────────────────
def log(msg=""):
    """Print + flush immediately so Windows terminals never buffer output."""
    print(msg, flush=True)


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Build config grid
    configs = []
    for model in MODELS:
        for prompt_name in PROMPT_FNS:
            for top_k in TOP_K_VALUES:
                for temp in TEMPERATURES:
                    configs.append({
                        "model": model, "prompt": prompt_name,
                        "top_k": top_k, "temp": temp,
                    })

    log("=" * 75)
    log("BNS FULL MATRIX RUNNER  (Optimized)")
    log(f"Total configs: {len(configs)} "
        f"({len(MODELS)} models × {len(PROMPT_FNS)} prompts × "
        f"{len(TOP_K_VALUES)} K × {len(TEMPERATURES)} temps)")
    log("=" * 75)

    # ── Step 1: Load data ─────────────────────────────────────────
    log("\n📂 Step 1/5: Loading data...")
    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    with open(QUERIES_PATH, "r", encoding="utf-8") as f:
        queries = json.load(f)
    with open(os.path.join(INDEX_DIR, META_FILE), "r", encoding="utf-8") as f:
        metadata = json.load(f)

    log(f"  ✅ Corpus: {len(chunks)} chunks | Queries: {len(queries)} | "
        f"Metadata: {len(metadata)} entries")

    # ── Step 2: Load models ───────────────────────────────────────
    log("\n🧠 Step 2/5: Loading FAISS index + embedding model (GPU if available)...")

    faiss_index = faiss.read_index(os.path.join(INDEX_DIR, INDEX_FILE))
    log(f"  ✅ FAISS index loaded ({faiss_index.ntotal} vectors)")

    # Use GPU for SentenceTransformer if CUDA is available
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"  📐 Embedding device: {device}"
        + (f" (RTX 3050 – {torch.cuda.get_device_name(0)})" if device == "cuda" else ""))
    dense_model = SentenceTransformer(DENSE_MODEL, device=device)

    # ── Step 3: Encode queries + retrieve ─────────────────────────
    log("\n🔍 Step 3/5: Encoding queries & pre-retrieving...")
    q_texts = [q["query"] for q in queries]
    q_embs = dense_model.encode(
        q_texts, show_progress_bar=True, batch_size=128,
        normalize_embeddings=True, device=device,
    )
    q_embs = np.array(q_embs, dtype=np.float32)

    max_k = max(TOP_K_VALUES)
    _, all_indices = faiss_index.search(q_embs, max_k)
    log(f"  ✅ Pre-retrieved top-{max_k} for all {len(queries)} queries")

    # ── Step 3b: Pre-cache context chunks per top-k ───────────────
    # Context only depends on (query_index, top_k), not on model/prompt/temp.
    # Building it once per top-k saves redundant work across 18 configs each.
    log("\n📦 Step 3b: Pre-caching context chunks for each top-k...")
    ctx_cache = {}  # (qi, top_k) -> list[dict]
    sec_cache = {}  # (qi, top_k) -> set of section IDs (for hallucination check)

    for top_k in TOP_K_VALUES:
        for qi in range(len(queries)):
            ctx_chunks = []
            sec_ids = set()
            for idx in all_indices[qi][:top_k]:
                if 0 <= idx < len(metadata):
                    sid = str(metadata[idx]["section_id"])
                    ctx_chunks.append({
                        "section_id": sid,
                        "title": metadata[idx]["title"],
                        "text": chunks[idx]["text"],
                    })
                    sec_ids.add(sid)
            ctx_cache[(qi, top_k)] = ctx_chunks
            sec_cache[(qi, top_k)] = sec_ids

    log(f"  ✅ Cached {len(ctx_cache)} context sets "
        f"({len(queries)} queries × {len(TOP_K_VALUES)} top-k values)")

    # Free embedding model from GPU memory – no longer needed
    del dense_model
    if device == "cuda":
        torch.cuda.empty_cache()
    log("  🧹 Freed embedding model from GPU memory")

    # ── Step 4: Resume state ──────────────────────────────────────
    log("\n♻️  Step 4/5: Checking resume state...")
    checkpoint = load_checkpoint()
    completed = set(checkpoint["completed"])
    all_results = checkpoint["results"]

    all_details = {}
    if os.path.exists(DETAILS_FILE):
        with open(DETAILS_FILE, "r", encoding="utf-8") as f:
            all_details = json.load(f)

    remaining = [c for c in configs if config_key(**c) not in completed]
    log(f"  Completed: {len(completed)} | Remaining: {len(remaining)}")

    if not remaining:
        log("\n  🎉 All configs already completed! Skipping to final output.\n")
    else:
        # ── Step 5: Run experiments ───────────────────────────────
        log(f"\n🚀 Step 5/5: Running {len(remaining)} configurations "
            f"({len(remaining) * len(queries)} total LLM calls)...\n")

        # Create a persistent HTTP session for Ollama
        session = create_session()

        # Pre-compute ground truth for all queries (avoids per-call overhead)
        gt_cache = [get_correct(q) for q in queries]

        # Outer progress bar: configs
        config_pbar = tqdm(
            remaining, desc="Configs", unit="cfg",
            position=0, leave=True, dynamic_ncols=True,
            file=sys.stdout,
        )

        for ci, cfg in enumerate(config_pbar):
            key = config_key(**cfg)
            model_name = cfg["model"]
            prompt_name = cfg["prompt"]
            top_k = cfg["top_k"]
            temp = cfg["temp"]
            prompt_fn = PROMPT_FNS[prompt_name]

            short_model = model_name.split(":")[0]
            config_pbar.set_description(
                f"Configs [{short_model}/{prompt_name}/k{top_k}/t{temp}]"
            )

            correct = 0
            total_latency = 0.0
            refusals = 0
            hallucinations = 0
            cat_stats = defaultdict(lambda: {"total": 0, "correct": 0})
            details = []

            # Inner progress bar: queries within this config
            query_pbar = tqdm(
                enumerate(queries), total=len(queries),
                desc=f"  Queries", unit="q",
                position=1, leave=False, dynamic_ncols=True,
                file=sys.stdout,
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] acc={postfix}",
            )
            query_pbar.set_postfix_str("0.00%")

            for qi, query_obj in query_pbar:
                gt = gt_cache[qi]
                cat = query_obj.get("category", "unknown")
                ctx_chunks = ctx_cache[(qi, top_k)]
                context_secs = sec_cache[(qi, top_k)]

                prompt = prompt_fn(query_obj["query"], ctx_chunks)

                t0 = time.time()
                response = call_ollama(session, prompt, model_name, temp)
                latency_ms = (time.time() - t0) * 1000

                predicted = extract_section(response)
                is_correct = predicted in gt
                is_refusal = detect_safety_refusal(response)
                is_halluc = detect_hallucination(predicted, context_secs)

                if is_correct:
                    correct += 1
                if is_refusal:
                    refusals += 1
                if is_halluc:
                    hallucinations += 1

                cat_stats[cat]["total"] += 1
                if is_correct:
                    cat_stats[cat]["correct"] += 1

                total_latency += latency_ms

                details.append({
                    "query_id": query_obj["query_id"], "category": cat,
                    "correct_section": query_obj["correct_section"],
                    "predicted": predicted, "is_correct": is_correct,
                    "is_refusal": is_refusal, "is_hallucination": is_halluc,
                    "latency_ms": round(latency_ms, 1),
                })

                # Live accuracy in the progress bar
                running_acc = correct / (qi + 1) * 100
                query_pbar.set_postfix_str(f"{running_acc:.1f}%")

            query_pbar.close()

            # ── Per-config summary ────────────────────────────────
            n = len(queries)
            acc = correct / n if n > 0 else 0.0
            avg_lat = total_latency / n if n > 0 else 0.0

            result_row = {
                "Model": model_name, "Prompt": prompt_name,
                "TopK": top_k, "Temperature": temp,
                "Accuracy": round(acc, 4),
                "Avg_Latency_ms": round(avg_lat, 1),
                "Safety_Refusals": refusals,
                "Refusal_Rate": round(refusals / n, 4) if n > 0 else 0.0,
                "Hallucinations": hallucinations,
                "Hallucination_Rate": round(hallucinations / n, 4) if n > 0 else 0.0,
            }
            for cat_key in CATEGORIES:
                s = cat_stats[cat_key]
                cat_acc = s["correct"] / s["total"] if s["total"] > 0 else 0.0
                result_row[f"Cat_{CAT_LABELS[cat_key]}"] = round(cat_acc, 4)

            all_results.append(result_row)
            all_details[key] = details

            tqdm.write(
                f"  ✅ {key}  →  Acc: {acc:.4f} | Refusals: {refusals} | "
                f"Halluc: {hallucinations} | Lat: {avg_lat:.0f}ms"
            )

            # Save checkpoint after each config
            checkpoint["completed"].append(key)
            checkpoint["results"] = all_results
            save_checkpoint(checkpoint)

            # Save details incrementally
            with open(DETAILS_FILE, "w", encoding="utf-8") as f:
                json.dump(all_details, f, ensure_ascii=False)

        config_pbar.close()
        session.close()

    # ── Final CSV ─────────────────────────────────────────────────
    csv_path = os.path.join(RESULTS_DIR, "matrix_results.csv")
    if all_results:
        fieldnames = list(all_results[0].keys())
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_results)

    # ── Summary Table ─────────────────────────────────────────────
    log("\n" + "=" * 75)
    log("MATRIX EXPERIMENT COMPLETE")
    log("=" * 75)
    log(f"  {'Model':<20} {'Prompt':<20} {'K':>2} {'T':>4} "
        f"{'Acc':>8} {'Ref':>5} {'Hal':>5} {'Lat':>8}")
    log(f"  {'─'*20} {'─'*20} {'─'*2} {'─'*4} {'─'*8} {'─'*5} {'─'*5} {'─'*8}")
    for r in all_results:
        log(f"  {r['Model']:<20} {r['Prompt']:<20} {r['TopK']:>2} "
            f"{r['Temperature']:>4} {r['Accuracy']:>8.4f} "
            f"{r['Safety_Refusals']:>5} {r['Hallucinations']:>5} "
            f"{r['Avg_Latency_ms']:>7.0f}ms")

    log(f"\n  ✅ CSV:     {csv_path} ({len(all_results)} rows)")
    log(f"  ✅ Details: {DETAILS_FILE}")
    log(f"  ✅ Ckpt:    {CHECKPOINT_FILE}")
    log("=" * 75)


if __name__ == "__main__":
    main()
