import os
import json
import faiss
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

# --- MAC-SPECIFIC PATCH ---
# This stops the "Segmentation Fault" by forcing the math libraries 
# to stay on a single thread.
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Note: Check if your 'data' folder is one level up from 'src'
CHUNKS_PATH = os.path.join(os.path.dirname(BASE_DIR), "data", "bns_chunks.json")
OUTPUT_DIR = os.path.join(BASE_DIR, "indexes")
INDEX_PATH = os.path.join(OUTPUT_DIR, "mpnet.index")
META_PATH = os.path.join(OUTPUT_DIR, "chunk_metadata.json")

MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"

def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    print("⚖️ Loading Legal Chunks...")
    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    print(f"✅ Loaded {len(chunks)} chunks.")

    print(f"🧠 Loading Embedding Model: {MODEL_NAME}...")
    # Force the model to use the CPU to avoid Mac "MPS" crashes for now
    model = SentenceTransformer(MODEL_NAME, device="cpu")

    texts = [c["text"] for c in chunks]
    
    print("⏳ Generating Embeddings (This is the heavy part)...")
    # We use a smaller batch size and NO workers to prevent the crash
    embeddings = model.encode(
        texts, 
        batch_size=16, 
        show_progress_bar=True, 
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    # --- FAISS INDEXING ---
    dimension = embeddings.shape[1]
    # Simple Flat Index: Most accurate and least likely to crash
    index = faiss.IndexFlatIP(dimension) 
    index.add(embeddings.astype('float32'))

    print(f"💾 Saving Index to {INDEX_PATH}...")
    faiss.write_index(index, INDEX_PATH)

    # Save metadata so the backend knows which chunk is which
    metadata = []
    for c in chunks:
        metadata.append({
            "section_id": c["section_id"],
            "title": c["title"]
        })
    
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f,   indent=4, ensure_ascii=False)

    print("\n🎉 DONE! The 'Brain' is now built and Mac-compatible.")

if __name__ == "__main__":
    main()