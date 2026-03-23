import json
from tqdm import tqdm

INPUT_PATH = "data/bns_sections.json"
OUTPUT_PATH = "data/bns_chunks.json"

CHUNK_SIZE = 350
OVERLAP = 50


def load_data(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_data(data, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def chunk_text(text, chunk_size=350, overlap=50):
    words = text.split()
    chunks = []

    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk_words = words[start:end]
        chunk_text = " ".join(chunk_words)
        chunks.append(chunk_text)

        start += chunk_size - overlap

    return chunks


def main():
    sections = load_data(INPUT_PATH)
    all_chunks = []

    print("Starting Chunking process...")
    for section in tqdm(sections):
        # Safety check: skip if content is missing or empty
        if "content" not in section or not section["content"]:
            continue
            
        chunks = chunk_text(section["content"], CHUNK_SIZE, OVERLAP)

        for idx, chunk in enumerate(chunks):
            all_chunks.append({
                "chunk_id": f"{section['section_id']}_{idx+1}",
                "section_id": section["section_id"],
                "title": section["title"],
                "text": chunk
            })

    print(f"\nSuccess! Total Chunks Created: {len(all_chunks)}")
    save_data(all_chunks, OUTPUT_PATH)

if __name__ == "__main__":
    main()