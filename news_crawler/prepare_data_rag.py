import json
import re
from pathlib import Path


def split_long_paragraph(para: str, max_chars: int = 900):
    """Split long paragraph by sentences (. ! ?), group into sub-paragraphs ≤ max_chars."""
    para = para.strip()
    if not para:
        return []

    sentences = re.split(r'(?<=[\.!?])\s+', para)
    chunks, current = [], ""

    for sent in sentences:
        sent = sent.strip()
        if not sent:
            continue
        if len(current) + len(sent) + 1 < max_chars:
            current += (" " + sent) if current else sent
        else:
            if current:
                chunks.append(current.strip())
            if len(sent) > max_chars:
                for i in range(0, len(sent), max_chars):
                    chunks.append(sent[i:i+max_chars].strip())
                current = ""
            else:
                current = sent
    if current:
        chunks.append(current.strip())
    return chunks


def chunk_text(text: str, max_chars: int = 900, overlap: int = 200):
    """Hybrid Paragraph Chunking (split by paragraphs, with overlap to maintain context)."""
    if not text:
        return []

    raw_paragraphs = re.split(r'\n\s*\n', text.strip())
    paragraphs = [p.strip() for p in raw_paragraphs if p.strip()]

    chunks, current = [], ""

    for para in paragraphs:
        subs = split_long_paragraph(para, max_chars=max_chars) if len(para) > max_chars else [para]
        for sp in subs:
            if len(current) + len(sp) + 2 < max_chars:
                current += sp + "\n\n"
            else:
                if current:
                    chunks.append(current.strip())
                if overlap > 0 and len(current) > overlap:
                    overlap_text = current[-overlap:]
                    current = overlap_text + "\n\n" + sp + "\n\n"
                else:
                    current = sp + "\n\n"
    if current:
        chunks.append(current.strip())
    return chunks



# PROCESS BILINGUAL FILES


INPUT_PATH = Path("data/raw/vnexpress_articles.json")
OUTPUT_VI = Path("data/processed/vnexpress_chunks_vi.jsonl")
OUTPUT_EN = Path("data/processed/vnexpress_chunks_en.jsonl")

def process_language(articles, lang="vi"):
    """Generate chunks for each language (vi or en)."""
    total_chunks = 0
    skipped = 0
    out_path = OUTPUT_VI if lang == "vi" else OUTPUT_EN

    with out_path.open("w", encoding="utf-8") as out_f:
        for i, art in enumerate(articles):
            content = art.get("content" if lang == "vi" else "content_en", "")
            title = art.get("title" if lang == "vi" else "title_en", "")
            url = art.get("url", "")
            date = art.get("date", "")

            if len(content.strip()) < 200:
                skipped += 1
                continue

            chunks = chunk_text(content, max_chars=900, overlap=200)
            for j, chunk in enumerate(chunks):
                record = {
                    "id": f"{lang}-vnexpress-{i}-{j}",
                    "text": chunk,
                    "metadata": {
                        "article_index": i,
                        "chunk_index": j,
                        "title": title,
                        "url": url,
                        "date": date,
                        "source": "vnexpress",
                        "lang": lang
                    }
                }
                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                total_chunks += 1

    print(f"[{lang.upper()}] Created {total_chunks} chunks, saved to {out_path.name}")
    print(f"[{lang.upper()}] Skipped {skipped} articles (too short)")


def main():
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"File not found: {INPUT_PATH.resolve()}")

    with INPUT_PATH.open("r", encoding="utf-8") as f:
        articles = json.load(f)

    print(f"Reading {len(articles)} articles from {INPUT_PATH.name}")

    # Process both languages in parallel
    process_language(articles, lang="vi")
    process_language(articles, lang="en")


if __name__ == "__main__":
    main()
