import fitz
import re
import json

PDF_PATH = "data/250883_english_01042024.pdf"  # update path if needed
OUTPUT_PATH = "data/bns_sections.json"

def extract_text(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text("text") + "\n"
    return text

def clean_text(text):
    # Remove standalone page numbers
    text = re.sub(r'\n\d+\n', '\n', text)

    # Remove excess spaces
    text = re.sub(r'[ \t]+', ' ', text)

    # Normalize multiple line breaks
    text = re.sub(r'\n{2,}', '\n\n', text)

    return text

def remove_arrangement_section(text):
    start_index = text.find("CHAPTER I")
    return text[start_index:]

def split_chapters(text):
    # Split by CHAPTER headings
    chapters = re.split(r'\n(?=CHAPTER\s+[IVXLCDM]+)', text)
    return chapters

def split_sections(chapter_text):
    # Split by section numbers (e.g., "1. ", "2. ")
    sections = re.split(r'\n(?=\d+\.\s)', chapter_text)
    return sections

def parse_section(section_text):
    match = re.match(r'(\d+)\.\s+(.*)', section_text.strip())
    if match:
        section_id = match.group(1)
        rest = section_text.strip()[len(match.group(0)):]
        return section_id, match.group(2), rest.strip()
    return None, None, None

def main():
    raw_text = extract_text(PDF_PATH)
    cleaned_text = clean_text(raw_text)
    main_text = remove_arrangement_section(cleaned_text)
    chapters_raw = split_chapters(main_text)

    structured_chapters = []

    for chap in chapters_raw:
        lines = chap.strip().split("\n", 2)
        if len(lines) >= 2:
            chapter_heading = lines[0].strip()
            chapter_title = lines[1].strip()
        else:
            chapter_heading = lines[0].strip()
            chapter_title = ""

        sections_raw = split_sections(chap)
        structured_sections = []

        for section in sections_raw:
            section_id, title, content = parse_section(section)
            if section_id:
                structured_sections.append({
                    "section_id": section_id,
                    "title": title.strip(),
                    "content": content.strip()
                })

        structured_chapters.append({
            "chapter_heading": chapter_heading,
            "chapter_title": chapter_title,
            "sections": structured_sections
        })

    print(f"Total Chapters Extracted: {len(structured_chapters)}")
    total_sections = sum(len(ch["sections"]) for ch in structured_chapters)
    print(f"Total Sections Extracted: {total_sections}")

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(structured_chapters, f, indent=4, ensure_ascii=False)

if __name__ == "__main__":
    main()
