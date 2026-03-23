from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os, json, faiss, requests
import numpy as np
from sentence_transformers import SentenceTransformer

app = FastAPI()

# Enable CORS so React can talk to us
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Initialize the "Brain" ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Check these paths! If indices are in 'src/indexes', update here:
INDEX_PATH = os.path.join(BASE_DIR, "src", "indexes", "mpnet.index")
CHUNKS_PATH = os.path.join(BASE_DIR, "data", "bns_chunks.json")
META_PATH = os.path.join(BASE_DIR, "src", "indexes", "chunk_metadata.json")

model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2", device="cpu")
index = faiss.read_index(INDEX_PATH)
with open(CHUNKS_PATH, "r") as f: chunks = json.load(f)
with open(META_PATH, "r") as f: metadata = json.load(f)

class ChatRequest(BaseModel):
    prompt: str # Matches the 'prompt' in your React fetch

@app.post("/chat") # Changed to /chat to match your frontend
async def ask_legal_bot(request: ChatRequest):
    try:
        # 1. Retrieval
        emb = model.encode([request.prompt], normalize_embeddings=True)
        scores, indices = index.search(np.array(emb, dtype=np.float32), k=3)
        
        context = ""
        for idx in indices[0]:
            context += f"\nSection {metadata[idx]['section_id']}: {chunks[idx]['text']}\n"

        # 2. Generation (Ollama)
        full_prompt = f"Use this BNS legal context: {context}\n\nQuestion: {request.prompt}\nAnswer:"
        r = requests.post("http://localhost:11434/api/generate", 
                          json={"model": "llama3", "prompt": full_prompt, "stream": False})
        
        return {"response": r.json().get("response", "No AI response.")}
    except Exception as e:
        return {"response": f"Backend Error: {str(e)}"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)