"""
BNS FAISS Index Builder
========================
Builds dual FAISS indices for the BNS legal corpus:
  Index A: intfloat/e5-large-v2       — "passage: " prefix, L2-norm, IndexFlatIP
  Index B: all-mpnet-base-v2           — No prefix, L2-norm, IndexFlatIP

Environment: conda run -n bns_rag python src/build_indices.py
"""

import json
import os
import sys
import time
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# ── Paths ─────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHUNKS_PATH = os.path.join(BASE_DIR, "data", "bns_chunks.json")
INDEX_DIR = os.path.join(BASE_DIR, "indexes")


# ── Pre-Flight Check ─────────────────────────────────────────────────────

def preflight_check():
    """Verify all runtime dependencies are importable."""
    print("=" * 60)
    print("PRE-FLIGHT CHECK")
    print("=" * 60)
    deps = {"faiss": faiss, "numpy": np, "sentence_transformers": None, "tqdm": None}
    for name in ["faiss", "numpy", "sentence_transformers", "tqdm"]:
        try:
            mod = __import__(name)
            ver = getattr(mod, "__version__", "OK")
            print(f"  ✅ {name:<25} v{ver}")
        except ImportError:
            print(f"  ❌ {name} — NOT FOUND")
            sys.exit(1)
    print()


# ── Index Builder ─────────────────────────────────────────────────────────

def build_index(model_name: str, texts: list, prefix: str, output_file: str):
    """
    Encode texts with a SentenceTransformer, L2-normalize, build IndexFlatIP.

    Args:
        model_name: HuggingFace model identifier
        texts:      Raw document texts
        prefix:     String to prepend (e.g., "passage: ") — empty string for none
        output_file: Path to save the .index file
    """
    display = model_name.split("/")[-1]
    print(f"─── {display} {'─' * (55 - len(display))}")
    print(f"  Model:  {model_name}")
    prefix_display = f'"{prefix}"' if prefix else "None"
    print(f"  Prefix: {prefix_display}")
    print(f"  Chunks: {len(texts)}")

    model = SentenceTransformer(model_name)

    # Apply prefix
    encoded_texts = [f"{prefix}{t}" for t in texts] if prefix else texts

    # Encode with L2 normalization
    print(f"  ⏳ Encoding {len(encoded_texts)} chunks...")
    t0 = time.time()
    embeddings = model.encode(
        encoded_texts,
        show_progress_bar=True,
        batch_size=32,
        normalize_embeddings=True,
    )
    embeddings = np.array(embeddings, dtype=np.float32)
    elapsed = time.time() - t0

    # Verify L2-normalization
    norms = np.linalg.norm(embeddings, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-4), f"Vectors not L2-normalized! Max deviation: {np.max(np.abs(norms - 1.0))}"

    print(f"  ✅ Encoded in {elapsed:.1f}s — shape: {embeddings.shape}")
    print(f"  ✅ L2-norm verified (mean: {norms.mean():.6f}, std: {norms.std():.8f})")

    # Build FAISS IndexFlatIP (inner product ≡ cosine on L2-normed vectors)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    faiss.write_index(index, output_file)
    file_mb = os.path.getsize(output_file) / (1024 * 1024)
    print(f"  ✅ Index saved: {output_file} ({file_mb:.1f} MB, {index.ntotal} vectors, dim={dim})")
    print()

    del model, embeddings
    return index.ntotal


def main():
    preflight_check()

    os.makedirs(INDEX_DIR, exist_ok=True)

    # Load corpus
    print("=" * 60)
    print("LOADING CORPUS")
    print("=" * 60)
    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    print(f"  Loaded {len(chunks)} chunks from bns_chunks.json\n")

    texts = [c["text"] for c in chunks]
    metadata = [{"chunk_id": c["chunk_id"], "section_id": c["section_id"], "title": c["title"]} for c in chunks]

    # Build dual indices
    print("=" * 60)
    print("BUILDING FAISS INDICES")
    print("=" * 60)

    n1 = build_index(
        model_name="intfloat/e5-large-v2",
        texts=texts,
        prefix="passage: ",
        output_file=os.path.join(INDEX_DIR, "e5_large.index"),
    )

    n2 = build_index(
        model_name="sentence-transformers/all-mpnet-base-v2",
        texts=texts,
        prefix="",
        output_file=os.path.join(INDEX_DIR, "mpnet.index"),
    )

    # Save chunk metadata (row-aligned with FAISS indices)
    meta_path = os.path.join(INDEX_DIR, "chunk_metadata.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print("=" * 60)
    print("BUILD COMPLETE")
    print("=" * 60)
    print(f"  E5-Large-v2:   {n1} vectors → indexes/e5_large.index")
    print(f"  MPNet-Base-v2: {n2} vectors → indexes/mpnet.index")
    print(f"  Metadata:      {len(metadata)} entries → indexes/chunk_metadata.json")
    print("=" * 60)


if __name__ == "__main__":
    main()
