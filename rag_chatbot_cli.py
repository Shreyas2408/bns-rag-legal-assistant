"""
BNS RAG Legal Assistant — CLI Chatbot
=======================================
Terminal-based Retrieval-Augmented Generation chatbot for the
Bharatiya Nyaya Sanhita (BNS) legal dataset.

Pipeline:
  Query → MPNet Embedding → FAISS Retrieval → Prompt → Ollama LLM → Answer

Usage:
  conda run -n bns_rag python rag_chatbot_cli.py
"""

import json
import os
import re
import sys
import random
import time
import numpy as np
import faiss
import requests
from sentence_transformers import SentenceTransformer

# ── Paths ─────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHUNKS_PATH = os.path.join(BASE_DIR, "data", "bns_chunks.json")
QUERIES_PATH = os.path.join(BASE_DIR, "data", "evaluation_queries_augmented.json")
INDEX_PATH = os.path.join(BASE_DIR, "indexes", "mpnet.index")
META_PATH = os.path.join(BASE_DIR, "indexes", "chunk_metadata.json")

OLLAMA_URL = "http://localhost:11434/api/generate"
DENSE_MODEL = "sentence-transformers/all-mpnet-base-v2"
DEFAULT_TOP_K = 5
TIMEOUT = 120


# ── ANSI Colors for Terminal ──────────────────────────────────────────────
class C:
    HEADER = "\033[95m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RESET = "\033[0m"


# ── Data Loading ──────────────────────────────────────────────────────────

def load_data():
    """Load BNS chunks and evaluation queries from disk."""
    print(f"{C.DIM}  Loading BNS chunks...{C.RESET}", end=" ")
    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    print(f"{C.GREEN}✓{C.RESET} {len(chunks)} chunks")

    print(f"{C.DIM}  Loading evaluation queries...{C.RESET}", end=" ")
    with open(QUERIES_PATH, "r", encoding="utf-8") as f:
        queries = json.load(f)
    print(f"{C.GREEN}✓{C.RESET} {len(queries)} queries")

    return chunks, queries


def load_faiss_index():
    """Load the pre-built FAISS index and aligned chunk metadata."""
    print(f"{C.DIM}  Loading FAISS index...{C.RESET}", end=" ")
    index = faiss.read_index(INDEX_PATH)
    print(f"{C.GREEN}✓{C.RESET} {index.ntotal} vectors, dim={index.d}")

    print(f"{C.DIM}  Loading chunk metadata...{C.RESET}", end=" ")
    with open(META_PATH, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    print(f"{C.GREEN}✓{C.RESET} {len(metadata)} entries")

    return index, metadata


def load_embedding_model():
    """Load the MPNet sentence-transformer model."""
    print(f"{C.DIM}  Loading MPNet embedding model...{C.RESET}", end=" ", flush=True)
    model = SentenceTransformer(DENSE_MODEL)
    print(f"{C.GREEN}✓{C.RESET}")
    return model


# ── Retrieval ─────────────────────────────────────────────────────────────

def embed_query(query_text, model):
    """Encode a query string into a normalized embedding vector."""
    embedding = model.encode([query_text], normalize_embeddings=True)
    return np.array(embedding, dtype=np.float32)


def retrieve_chunks(query_embedding, faiss_index, chunks, metadata, top_k=DEFAULT_TOP_K):
    """
    Retrieve the top-K most relevant BNS chunks from the FAISS index.
    Returns a list of dicts with section_id, title, text, and similarity score.
    """
    scores, indices = faiss_index.search(query_embedding, top_k)

    results = []
    for rank, (score, idx) in enumerate(zip(scores[0], indices[0])):
        if 0 <= idx < len(metadata):
            meta = metadata[idx]
            results.append({
                "rank": rank + 1,
                "section_id": str(meta["section_id"]),
                "title": meta["title"],
                "text": chunks[idx]["text"],
                "score": float(score),
            })
    return results


# ── Prompt Engineering ────────────────────────────────────────────────────

def build_prompt(query, retrieved_chunks):
    """
    Build a structured legal analysis prompt with retrieved BNS context.
    The LLM is asked to produce THREE outputs:
      1. Applicable Section (number + title)
      2. Legal Explanation (why it applies)
      3. Punishment (penalty mentioned in the section)
    """
    # Format retrieved sections as context block
    context = "\n\n".join(
        f"[Section {c['section_id']} — {c['title']}]\n{c['text']}"
        for c in retrieved_chunks
    )

    prompt = (
        "LEGAL RESEARCH MODE: You are a clinical statutory index. "
        "Analyze the provided legal context and return the section number. "
        "Do not filter legal definitions.\n\n"
        "You are a legal expert specializing in the Bharatiya Nyaya Sanhita (BNS), 2023.\n\n"
        "Based ONLY on the legal provisions below, perform the following three tasks "
        "for the given query.\n\n"
        f"**Retrieved Legal Provisions:**\n{context}\n\n"
        f"**Query:** {query}\n\n"
        "**Instructions:**\n"
        "1. Identify the single most applicable BNS section.\n"
        "2. Briefly explain why this section applies to the query.\n"
        "3. Extract or summarize the punishment/penalty mentioned in that section.\n\n"
        "Provide your answer in EXACTLY this format (do not deviate):\n\n"
        "Applicable Section: Section [NUMBER] – [TITLE]\n\n"
        "Legal Explanation:\n[short explanation of why this section applies]\n\n"
        "Punishment:\n[punishment or penalty text from the section]\n\n"
        "Important: Your answer MUST reference a specific section number from the "
        "provisions above. Always include the punishment even if it is just a fine."
    )
    return prompt


# ── LLM Inference ─────────────────────────────────────────────────────────

def call_ollama(prompt, model_name):
    """
    Send a prompt to the Ollama API and return the LLM response.
    Handles timeouts and connection errors gracefully.
    """
    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": model_name,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.0},
            },
            timeout=TIMEOUT,
        )
        response.raise_for_status()
        return response.json().get("response", "")
    except requests.exceptions.Timeout:
        return "[ERROR: LLM request timed out. Please try again.]"
    except requests.exceptions.ConnectionError:
        return "[ERROR: Cannot connect to Ollama. Is it running at localhost:11434?]"
    except Exception as e:
        return f"[ERROR: {str(e)}]"


# ── Section & Field Extraction ───────────────────────────────────────────

def extract_predicted_section(response):
    """Extract the predicted BNS section number from the LLM response."""
    if not response or response.startswith("[ERROR"):
        return "N/A"

    # Priority 1: "Applicable Section: Section NNN"
    m = re.search(r'applicable\s+section[:\s]*section\s+(\d+)', response, re.IGNORECASE)
    if m:
        return m.group(1)

    # Priority 2: "Final Answer: Section NNN"
    m = re.search(r'final\s+answer[:\s]*section\s+(\d+)', response, re.IGNORECASE)
    if m:
        return m.group(1)

    # Priority 3: Any "Section NNN"
    matches = re.findall(r'\bsection\s+(\d+)', response, re.IGNORECASE)
    if matches:
        return matches[-1]

    # Priority 4: "§NNN" or "BNS NNN"
    m = re.search(r'(?:BNS|§)\s*(\d+)', response, re.IGNORECASE)
    if m:
        return m.group(1)

    return "N/A"


def extract_section_title(response):
    """Extract the section title from 'Section NNN – TITLE' format."""
    m = re.search(
        r'applicable\s+section[:\s]*section\s+\d+\s*[–\-—]\s*(.+)',
        response, re.IGNORECASE,
    )
    if m:
        return m.group(1).strip()
    return ""


def extract_legal_explanation(response):
    """
    Extract the 'Legal Explanation' block from the structured LLM output.
    Looks for text between 'Legal Explanation:' and the next section header.
    """
    m = re.search(
        r'legal\s+explanation[:\s]*\n?(.+?)(?=\n\s*punishment|\n\s*\*\*punishment|$)',
        response, re.IGNORECASE | re.DOTALL,
    )
    if m:
        text = m.group(1).strip()
        # Clean markdown artifacts
        text = re.sub(r'\*\*', '', text)
        return text if text else ""
    return ""


def extract_punishment(response, retrieved_chunks, predicted_section):
    """
    Extract the 'Punishment' block from the structured LLM output.
    Fallback: if the LLM didn't produce a punishment block, return the
    text snippet from the retrieved chunk that matches the predicted section.
    """
    # Try to extract from LLM response
    m = re.search(
        r'punishment[:\s]*\n?(.+?)(?=\n\s*important|\n\s*note|\n\s*disclaimer|$)',
        response, re.IGNORECASE | re.DOTALL,
    )
    if m:
        text = m.group(1).strip()
        text = re.sub(r'\*\*', '', text)
        if text and len(text) > 5:
            return text

    # Fallback: use the retrieved chunk text for the predicted section
    for c in retrieved_chunks:
        if str(c["section_id"]) == str(predicted_section):
            snippet = c["text"][:300].strip()
            return f"(From retrieved context) {snippet}"

    return "Punishment details not available in retrieved context."


# ── Display Helpers ───────────────────────────────────────────────────────

def print_banner():
    """Display the welcome banner."""
    print()
    print(f"{C.BOLD}{C.CYAN}╔══════════════════════════════════════════════════════════╗{C.RESET}")
    print(f"{C.BOLD}{C.CYAN}║     🏛️  BNS RAG Legal Assistant — CLI Chatbot  🏛️       ║{C.RESET}")
    print(f"{C.BOLD}{C.CYAN}║     Bharatiya Nyaya Sanhita (2023) Analysis Engine      ║{C.RESET}")
    print(f"{C.BOLD}{C.CYAN}╚══════════════════════════════════════════════════════════╝{C.RESET}")
    print()
    print(f"  {C.DIM}Powered by RAG: MPNet + FAISS + Ollama LLM{C.RESET}")
    print()


def print_divider(char="─", width=60):
    print(f"{C.DIM}{char * width}{C.RESET}")


def display_retrieved_context(retrieved_chunks):
    """Display the retrieved legal context in a formatted table."""
    print(f"\n{C.BOLD}{C.BLUE}📚 Retrieved Legal Context:{C.RESET}")
    print_divider()
    for c in retrieved_chunks:
        score_bar = "█" * int(c["score"] * 10)
        print(f"  {C.YELLOW}[{c['rank']}]{C.RESET} "
              f"{C.BOLD}Section {c['section_id']}{C.RESET} — {c['title']}")
        print(f"      {C.DIM}Score: {c['score']:.4f} {score_bar}{C.RESET}")
        # Show truncated text preview
        preview = c["text"][:150].replace("\n", " ")
        print(f"      {C.DIM}{preview}...{C.RESET}")
    print_divider()


def display_llm_response(response, predicted_section, retrieved_chunks,
                         ground_truth=None):
    """
    Display the structured LLM output:
      - Applicable Section (number + title)
      - Legal Explanation
      - Punishment / Penalty
      - Ground truth validation (if available)
      - Legal disclaimer
    """
    # Extract structured fields from LLM response
    section_title = extract_section_title(response)
    explanation = extract_legal_explanation(response)
    punishment = extract_punishment(response, retrieved_chunks, predicted_section)

    # ── Formatted Output ──────────────────────────────────────────
    print(f"\n{C.BOLD}{C.GREEN}🤖 LLM Analysis:{C.RESET}")
    print_divider("═", 60)

    # 1. Applicable Section
    title_part = f" – {section_title}" if section_title else ""
    print(f"\n  {C.BOLD}{C.CYAN}Applicable BNS Section:{C.RESET}")
    print(f"  Section {predicted_section}{title_part}")

    # 2. Legal Explanation
    print(f"\n  {C.BOLD}{C.YELLOW}Legal Explanation:{C.RESET}")
    if explanation:
        # Wrap long explanations for readability
        for line in explanation.split("\n"):
            print(f"  {line.strip()}")
    else:
        print(f"  {C.DIM}(See full LLM output below){C.RESET}")

    # 3. Punishment
    print(f"\n  {C.BOLD}{C.RED}Punishment:{C.RESET}")
    for line in punishment.split("\n"):
        print(f"  {line.strip()}")

    print_divider("═", 60)

    # Ground truth check (for benchmark queries)
    if ground_truth:
        gt_str = ground_truth if isinstance(ground_truth, str) else ", ".join(ground_truth)
        gt_set = {str(s) for s in ground_truth} if isinstance(ground_truth, list) else {str(ground_truth)}

        if predicted_section in gt_set:
            print(f"\n   {C.GREEN}✅ CORRECT{C.RESET} (Ground truth: Section {gt_str})")
        else:
            print(f"\n   {C.RED}❌ INCORRECT{C.RESET} (Ground truth: Section {gt_str})")

    # Legal disclaimer
    print(f"\n  {C.DIM}⚠ This assistant provides legal information only and does "
          f"not constitute legal advice (e.g., from an experienced lawyer).{C.RESET}")


# ── Model Selection ───────────────────────────────────────────────────────
 
def select_model():   # "llama3:latest", "mistral:7b-instruct"
    """Ask the user which LLM to use."""
    print(f"\n{C.BOLD}Select LLM Model:{C.RESET}")
    print(f"  {C.YELLOW}[1]{C.RESET} llama3")
    print(f"  {C.YELLOW}[2]{C.RESET} mistral")

    while True:
        choice = input(f"\n{C.BOLD}Enter choice (1/2): {C.RESET}").strip()
        if choice == "1":
            return "llama3:latest"    
        elif choice == "2":
            return "mistral:7b-instruct"
        print(f"  {C.RED}Invalid choice. Please enter 1 or 2.{C.RESET}")


# ── Query Input ───────────────────────────────────────────────────────────

def get_query(queries, mode=None):
    """
    Get a query from the user.
    Returns (query_text, ground_truth_or_None, category_or_None).
    """
    if mode is None:
        print(f"\n{C.BOLD}How would you like to provide a case?{C.RESET}")
        print(f"  {C.YELLOW}[A]{C.RESET} Write your own legal scenario")
        print(f"  {C.YELLOW}[B]{C.RESET} Random case from the 300-query benchmark")

        while True:
            choice = input(f"\n{C.BOLD}Enter choice (A/B): {C.RESET}").strip().upper()
            if choice in ("A", "B"):
                mode = choice
                break
            print(f"  {C.RED}Invalid choice. Please enter A or B.{C.RESET}")

    if mode == "A":
        print(f"\n{C.BOLD}Enter your legal scenario:{C.RESET}")
        query_text = input(f"  {C.CYAN}> {C.RESET}").strip()
        if not query_text:
            print(f"  {C.RED}Empty query. Using a random case instead.{C.RESET}")
            mode = "B"
        else:
            return query_text, None, None

    if mode == "B":
        case = random.choice(queries)
        print(f"\n{C.BOLD}🎲 Random Case Selected:{C.RESET}")
        print(f"  {C.DIM}ID:{C.RESET} {case['query_id']}")
        print(f"  {C.DIM}Category:{C.RESET} {case.get('category', 'N/A')}")
        print(f"  {C.BOLD}{case['query']}{C.RESET}")
        return case["query"], case["correct_section"], case.get("category")

    return "", None, None


# ── Main RAG Pipeline ─────────────────────────────────────────────────────

def run_rag_pipeline(query_text, model_name, embedding_model, faiss_index,
                     chunks, metadata, ground_truth=None):
    """Execute the full RAG pipeline: Embed → Retrieve → Prompt → LLM → Answer."""

    # Step 1: Embed the query
    print(f"\n{C.DIM}  ⏳ Embedding query with MPNet...{C.RESET}", end=" ", flush=True)
    t0 = time.time()
    query_embedding = embed_query(query_text, embedding_model)
    print(f"{C.GREEN}✓{C.RESET} ({time.time()-t0:.2f}s)")

    # Step 2: Retrieve top-K chunks from FAISS
    print(f"{C.DIM}  ⏳ Retrieving top-{DEFAULT_TOP_K} sections from FAISS...{C.RESET}",
          end=" ", flush=True)
    retrieved = retrieve_chunks(query_embedding, faiss_index, chunks, metadata, DEFAULT_TOP_K)
    print(f"{C.GREEN}✓{C.RESET} ({len(retrieved)} sections found)")

    # Display retrieved context
    display_retrieved_context(retrieved)

    # Step 3: Build prompt
    prompt = build_prompt(query_text, retrieved)

    # Step 4: Call LLM via Ollama
    print(f"\n{C.DIM}  ⏳ Querying {model_name} via Ollama...{C.RESET}", flush=True)
    t0 = time.time()
    response = call_ollama(prompt, model_name)
    latency = time.time() - t0
    print(f"{C.DIM}  ✓ Response received ({latency:.1f}s){C.RESET}")

    # Step 5: Extract predicted section and display structured output
    predicted = extract_predicted_section(response)
    display_llm_response(response, predicted, retrieved, ground_truth)

    print(f"\n{C.DIM}  Total pipeline time: {latency:.1f}s{C.RESET}")


# ── Chatbot Loop ──────────────────────────────────────────────────────────

def run_chatbot():
    """Main chatbot entry point with interactive loop."""

    print_banner()

    # ── Initialization ────────────────────────────────────────────
    print(f"{C.BOLD}Initializing components...{C.RESET}")
    print_divider()

    try:
        chunks, queries = load_data()
        faiss_index, metadata = load_faiss_index()
        embedding_model = load_embedding_model()
    except FileNotFoundError as e:
        print(f"\n{C.RED}❌ File not found: {e}{C.RESET}")
        print(f"{C.RED}   Make sure you've built the FAISS indices first:{C.RESET}")
        print(f"{C.RED}   conda run -n bns_rag python src/build_indices.py{C.RESET}")
        sys.exit(1)
    except Exception as e:
        print(f"\n{C.RED}❌ Initialization error: {e}{C.RESET}")
        sys.exit(1)

    # Verify Ollama
    print(f"{C.DIM}  Checking Ollama connection...{C.RESET}", end=" ")
    try:
        r = requests.get("http://localhost:11434/api/tags", timeout=5)
        models = [m["name"] for m in r.json().get("models", [])]
        print(f"{C.GREEN}✓{C.RESET} (models: {', '.join(models)})")
    except Exception:
        print(f"{C.RED}✗{C.RESET}")
        print(f"\n{C.RED}❌ Ollama is not running at localhost:11434{C.RESET}")
        print(f"{C.RED}   Start Ollama first: ollama serve{C.RESET}")
        sys.exit(1)

    print_divider()
    print(f"\n{C.GREEN}✅ All components loaded. Ready to assist!{C.RESET}")

    # ── Model Selection ───────────────────────────────────────────
    model_name = select_model()
    print(f"\n{C.GREEN}  Using model: {C.BOLD}{model_name}{C.RESET}")

    # ── Interaction Loop ──────────────────────────────────────────
    first_run = True
    while True:
        if first_run:
            query_text, ground_truth, category = get_query(queries)
            first_run = False
        else:
            print(f"\n{C.BOLD}What would you like to do next?{C.RESET}")
            print(f"  {C.YELLOW}[1]{C.RESET} New custom case")
            print(f"  {C.YELLOW}[2]{C.RESET} Random dataset case")
            print(f"  {C.YELLOW}[3]{C.RESET} Switch model")
            print(f"  {C.YELLOW}[4]{C.RESET} Exit")

            while True:
                choice = input(f"\n{C.BOLD}Enter choice (1/2/3/4): {C.RESET}").strip()
                if choice in ("1", "2", "3", "4"):
                    break
                print(f"  {C.RED}Invalid choice.{C.RESET}")

            if choice == "4":
                print(f"\n{C.CYAN}Thank you for using the BNS RAG Legal Assistant. Goodbye! 👋{C.RESET}\n")
                break
            elif choice == "3":
                model_name = select_model()
                print(f"\n{C.GREEN}  Switched to: {C.BOLD}{model_name}{C.RESET}")
                continue
            elif choice == "1":
                query_text, ground_truth, category = get_query(queries, mode="A")
            elif choice == "2":
                query_text, ground_truth, category = get_query(queries, mode="B")

        if not query_text:
            continue

        # Display the user's case
        print(f"\n{'═' * 60}")
        print(f"{C.BOLD}{C.HEADER}📋 User Case:{C.RESET}")
        print(f"  {query_text}")
        if category:
            print(f"  {C.DIM}Category: {category}{C.RESET}")
        print(f"{'═' * 60}")

        # Run the RAG pipeline
        run_rag_pipeline(
            query_text, model_name, embedding_model,
            faiss_index, chunks, metadata, ground_truth,
        )


# ── Entry Point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    try:
        run_chatbot()
    except KeyboardInterrupt:
        print(f"\n\n{C.CYAN}Session interrupted. Goodbye! 👋{C.RESET}\n")
        sys.exit(0)
