import fitz
import re
import json
import os

PDF_PATH = "data/250883_english_01042024.pdf"
OUTPUT_PATH = "data/bns_sections.json"

def extract_text(pdf_path):
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF not found at {pdf_path}")
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text("text") + "\n"
    return text

def clean_text(text):
    # Remove the side-notes like "Short title, commencement and application"
    # These often appear as separate lines in PDF extraction
    text = re.sub(r'\n{2,}', '\n\n', text)
    return text

def main():
    print("🚀 Starting Extraction...")
    raw_text = extract_text(PDF_PATH)
    
    # Split by CHAPTER
    chapters_raw = re.split(r'CHAPTER\s+[IVXLCDM]+', raw_text)
    all_sections_flat = []

    for idx, chap in enumerate(chapters_raw):
        if idx == 0: continue # Skip text before Chapter 1
        
        # Look for patterns like "1. (1)" or "2. " at the start of a line
        # The regex below looks for: Start of line -> Number -> Period -> Space
        sections_raw = re.split(r'\n(?=\d+\.\s)', chap)
        
        for sec_text in sections_raw:
            # Match: SectionNumber. Title/Content
            match = re.search(r'^(\d+)\.\s+(.*)', sec_text.strip(), re.DOTALL)
            if match:
                section_id = match.group(1)
                content = match.group(2).strip()
                
                # Use the first line as a temporary title
                title = content.split('\n')[0][:100] 
                
                all_sections_flat.append({
                    "section_id": section_id,
                    "title": title,
                    "content": content
                })

    print(f"✅ Extracted {len(all_sections_flat)} sections.")

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(all_sections_flat, f, indent=4, ensure_ascii=False)

if __name__ == "__main__":
    main()