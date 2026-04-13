from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os, json, faiss, requests
import numpy as np
from sentence_transformers import SentenceTransformer

app = FastAPI()

# Allow React to connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Load AI Logic ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INDEX_PATH = os.path.join(BASE_DIR, "src", "indexes", "mpnet.index")
CHUNKS_PATH = os.path.join(BASE_DIR, "data", "bns_chunks.json")
META_PATH = os.path.join(BASE_DIR, "src", "indexes", "chunk_metadata.json")

model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
index = faiss.read_index(INDEX_PATH)

with open(CHUNKS_PATH, "r") as f:
    chunks = json.load(f)

with open(META_PATH, "r") as f:
    metadata = json.load(f)

class ChatRequest(BaseModel):
    prompt: str


@app.post("/chat")
async def chat(request: ChatRequest):

    # 1. Embed & Retrieve
    emb = model.encode([request.prompt], normalize_embeddings=True)
    scores, indices = index.search(np.array(emb, dtype=np.float32), k=5)

    # ✅ FIX: properly inside function
    retrieved = []
    context = ""

    for i, idx in enumerate(indices[0]):
        section = metadata[idx]['section_id']
        text = chunks[idx]['text']
        score = float(scores[0][i])

        retrieved.append({
            "rank": i + 1,
            "section": section,
            "text": text[:200],
            "score": score
        })

        context += f"\nSection {section}: {text}\n"

    # 2. Call Ollama
    full_prompt = f"Context: {context}\n\nUser: {request.prompt}\nAssistant:"

    r = requests.post(
        "http://localhost:11434/api/generate",
        json={"model": "llama3", "prompt": full_prompt, "stream": False}
    )

    response_text = r.json().get("response", "AI Error")

    # 3. Return structured response
    return {
        "retrieved_context": retrieved,
        "analysis": {
            "raw": response_text
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)