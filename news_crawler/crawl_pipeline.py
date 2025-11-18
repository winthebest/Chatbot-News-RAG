#!/usr/bin/env python
"""
Pipeline script to crawl and prepare RAG data from VNExpress.
Combines all steps: crawl URLs -> crawl content -> prepare chunks
"""
import argparse
import csv
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import List, Dict

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from deep_translator import GoogleTranslator

# ============================================================================
# Helper Functions
# ============================================================================

def configure_driver():
    """Configure the Selenium WebDriver."""
    chrome_options = Options()
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-extensions")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--disable-notifications")
    chrome_options.set_capability('pageLoadStrategy', 'none')
    chrome_options.add_argument('--headless')
    chrome_options.add_argument("--disable-features=ScriptStreaming")
    chrome_options.add_argument("--disable-features=PreloadMediaEngagementData")
    chrome_options.add_experimental_option("prefs", {
        "profile.managed_default_content_settings.images": 2,
        "profile.managed_default_content_settings.videos": 2
    })
    chrome_options.add_experimental_option('prefs', {'intl.accept_languages': 'en,en_US'})
    
    driver = webdriver.Chrome(options=chrome_options)
    return driver


def wait_for_page_load(driver, timeout=30):
    """Wait for web page to fully load."""
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            state = driver.execute_script("return document.readyState")
        except Exception:
            time.sleep(1)
            continue
        
        if state == "complete":
            return True
        time.sleep(1)
    print("Warning: Page did not fully load after timeout!")
    return False


# ============================================================================
# Step 1: Crawl URLs
# ============================================================================

def crawl_urls(start_page: int, end_page: int, output_file: Path) -> int:
    """
    Crawl URLs from VNExpress AI section.
    
    Args:
        start_page: Starting page number
        end_page: Ending page number
        output_file: CSV file to save URLs
    
    Returns:
        Number of URLs crawled
    """
    print(f"\n{'='*60}")
    print(f"STEP 1: Crawl URLs from page {start_page} to {end_page}")
    print(f"{'='*60}\n")
    
    driver = configure_driver()
    hrefs = set()
    
    try:
        temp = len(hrefs)
        for i in range(start_page, end_page + 1):
            url = f"https://vnexpress.net/cong-nghe/ai-p{i}" if i > 1 else "https://vnexpress.net/cong-nghe/ai"
            print(f"Crawling page {i}...")
            driver.get(url)
            time.sleep(3)
            
            links = driver.find_elements(By.XPATH, "//a[@data-medium and @href]")
            hrefs.update(
                link.get_attribute("href") 
                for link in links 
                if link.get_attribute("href") and link.get_attribute("href").endswith(".html")
            )
            
            if temp == len(hrefs):
                print("Collected enough links. Ending...")
                break
            temp = len(hrefs)
            print(f"  Collected {len(hrefs)} URLs...")
    finally:
        driver.quit()
    
    # Save to CSV
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["URL"])
        for href in sorted(hrefs):
            writer.writerow([href])
    
    print(f"\n✓ Saved {len(hrefs)} URLs to {output_file}")
    return len(hrefs)


# ============================================================================
# Step 2: Crawl Content
# ============================================================================

# Abbreviation dictionary
ABBREVIATION_DICT = {
    "AI": "Trí tuệ nhân tạo",
    "IoT": "Internet vạn vật",
    "ML": "Học máy",
    "NLP": "Xử lí ngôn ngữ tự nhiên"
}


def replace_abbreviations(text: str) -> str:
    """Replace abbreviations with full words."""
    if not text:
        return ""
    for abbr, full_form in ABBREVIATION_DICT.items():
        text = re.sub(rf'\b{abbr}\b', full_form, text, flags=re.IGNORECASE)
    return text


def translate_long_text(text: str, translator: GoogleTranslator, max_len: int = 4800) -> str:
    """Translate long text by splitting into chunks."""
    if not text:
        return ""
    
    text = replace_abbreviations(text)
    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
    chunks = []
    current = ""
    
    for p in paragraphs:
        if len(p) > max_len:
            for i in range(0, len(p), max_len):
                sub = p[i:i + max_len]
                if sub.strip():
                    chunks.append(sub.strip())
            continue
        
        if len(current) + len(p) + 1 <= max_len:
            current = (current + " " + p).strip()
        else:
            if current:
                chunks.append(current)
            current = p
    
    if current:
        chunks.append(current)
    
    translated_parts = []
    for chunk in chunks:
        try:
            translated_parts.append(translator.translate(chunk))
        except Exception as e:
            print(f"  Error translating chunk: {e}")
            continue
    
    return "\n".join(translated_parts)


def crawl_content(input_file: Path, output_file: Path) -> int:
    """
    Crawl content from saved URLs.
    
    Args:
        input_file: CSV file containing URLs
        output_file: JSON file to save articles
    
    Returns:
        Number of articles crawled
    """
    print(f"\n{'='*60}")
    print(f"STEP 2: Crawl content from URLs")
    print(f"{'='*60}\n")
    
    if not input_file.exists():
        raise FileNotFoundError(f"File not found: {input_file}")
    
    # Read URLs
    urls = []
    with input_file.open("r", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)  # Skip header
        urls = [row[0] for row in reader if row[0]]
    
    if not urls:
        print("No URLs to crawl!")
        return 0
    
    # Read existing data
    articles = []
    crawled_urls = set()
    if output_file.exists():
        try:
            with output_file.open("r", encoding="utf-8") as f:
                articles = json.load(f)
                crawled_urls = {article["url"] for article in articles}
        except (json.JSONDecodeError, KeyError):
            articles = []
            crawled_urls = set()
    
    # Filter uncrawled URLs
    new_urls = [url for url in urls if url not in crawled_urls]
    
    if not new_urls:
        print(f"All {len(urls)} URLs have been crawled. Nothing new.")
        return len(articles)
    
    print(f"Already have {len(articles)} articles. Will crawl {len(new_urls)} new URLs.\n")
    
    # Initialize translator and driver
    translator = GoogleTranslator(source="auto", target="en")
    driver = configure_driver()
    
    try:
        for idx, url in enumerate(new_urls, 1):
            print(f"[{idx}/{len(new_urls)}] Crawling: {url[:60]}...")
            
            try:
                driver.get(url)
                if not wait_for_page_load(driver):
                    print(f"  ⚠ Page did not fully load, skipping...")
                    continue
            except Exception as e:
                print(f"  ✗ Error loading page: {e}")
                continue
            
            try:
                title = driver.find_element(By.CLASS_NAME, "title-detail").text.strip()
                description = driver.find_element(By.CLASS_NAME, "description").text.strip()
                content = driver.find_element(By.CLASS_NAME, "fck_detail").text.strip()
                date = driver.find_element(By.CLASS_NAME, "date").text.strip()
                
                # Replace abbreviations
                title = replace_abbreviations(title)
                description = replace_abbreviations(description)
                content = replace_abbreviations(content)
                
                # Translate
                try:
                    title_en = translator.translate(title) if title else ""
                except Exception as e:
                    print(f"  ⚠ Error translating title: {e}")
                    title_en = ""
                
                try:
                    description_en = translator.translate(description) if description else ""
                except Exception as e:
                    print(f"  ⚠ Error translating description: {e}")
                    description_en = ""
                
                content_en = translate_long_text(content, translator)
                
                article = {
                    "url": url,
                    "title": title,
                    "title_en": title_en,
                    "description": description,
                    "description_en": description_en,
                    "content": content,
                    "content_en": content_en,
                    "date": date
                }
                
                articles.append(article)
                
                # Save after each article (backup)
                output_file.parent.mkdir(parents=True, exist_ok=True)
                with output_file.open("w", encoding="utf-8") as f:
                    json.dump(articles, f, ensure_ascii=False, indent=4)
                
                print(f"  ✓ Crawled and saved")
                
            except Exception as e:
                print(f"  ✗ Error extracting data: {e}")
                continue
                
    finally:
        driver.quit()
    
    print(f"\n✓ Complete! Total {len(articles)} articles in {output_file}")
    return len(articles)


# ============================================================================
# Step 3: Prepare RAG Data
# ============================================================================

def split_long_paragraph(para: str, max_chars: int = 900) -> List[str]:
    """Split long paragraph by sentences."""
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


def chunk_text(text: str, max_chars: int = 900, overlap: int = 200) -> List[str]:
    """Hybrid Paragraph Chunking with overlap."""
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


def prepare_rag_data(input_file: Path, output_vi: Path, output_en: Path) -> Dict[str, int]:
    """
    Prepare RAG data: chunk articles into JSONL.
    
    Args:
        input_file: JSON file containing articles
        output_vi: JSONL file for Vietnamese
        output_en: JSONL file for English
    
    Returns:
        Dict with number of chunks for each language
    """
    print(f"\n{'='*60}")
    print(f"STEP 3: Prepare RAG data (chunking)")
    print(f"{'='*60}\n")
    
    if not input_file.exists():
        raise FileNotFoundError(f"File not found: {input_file}")
    
    with input_file.open("r", encoding="utf-8") as f:
        articles = json.load(f)
    
    print(f"Reading {len(articles)} articles from {input_file.name}\n")
    
    # Process each language
    results = {}
    
    for lang in ["vi", "en"]:
        total_chunks = 0
        skipped = 0
        out_path = output_vi if lang == "vi" else output_en
        
        out_path.parent.mkdir(parents=True, exist_ok=True)
        
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
        results[lang] = total_chunks
    
    return results


# ============================================================================
# Main Pipeline
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Pipeline to crawl and prepare RAG data from VNExpress",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all steps
  python crawl_pipeline.py --all --start-page 1 --end-page 5
  
  # Only crawl URLs
  python crawl_pipeline.py --step urls --start-page 1 --end-page 3
  
  # Only crawl content
  python crawl_pipeline.py --step content
  
  # Only prepare RAG data
  python crawl_pipeline.py --step prepare
        """
    )
    
    parser.add_argument(
        "--step",
        choices=["urls", "content", "prepare", "all"],
        default="all",
        help="Step to run: urls, content, prepare, or all (default: all)"
    )
    
    parser.add_argument(
        "--start-page",
        type=int,
        default=1,
        help="Starting page to crawl URLs (default: 1)"
    )
    
    parser.add_argument(
        "--end-page",
        type=int,
        default=5,
        help="Ending page to crawl URLs (default: 5)"
    )
    
    parser.add_argument(
        "--urls-file",
        type=Path,
        default=Path("data/raw/vnexpress_links.csv"),
        help="CSV file containing URLs (default: data/raw/vnexpress_links.csv)"
    )
    
    parser.add_argument(
        "--articles-file",
        type=Path,
        default=Path("data/raw/vnexpress_articles.json"),
        help="JSON file containing articles (default: data/raw/vnexpress_articles.json)"
    )
    
    parser.add_argument(
        "--chunks-vi-file",
        type=Path,
        default=Path("data/processed/vnexpress_chunks_vi.jsonl"),
        help="JSONL file for Vietnamese chunks (default: data/processed/vnexpress_chunks_vi.jsonl)"
    )
    
    parser.add_argument(
        "--chunks-en-file",
        type=Path,
        default=Path("data/processed/vnexpress_chunks_en.jsonl"),
        help="JSONL file for English chunks (default: data/processed/vnexpress_chunks_en.jsonl)"
    )
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("VNEXPRESS CRAWL PIPELINE")
    print("="*60)
    
    try:
        if args.step in ["urls", "all"]:
            crawl_urls(args.start_page, args.end_page, args.urls_file)
        
        if args.step in ["content", "all"]:
            crawl_content(args.urls_file, args.articles_file)
        
        if args.step in ["prepare", "all"]:
            results = prepare_rag_data(args.articles_file, args.chunks_vi_file, args.chunks_en_file)
            print(f"\n✓ Complete! Total:")
            print(f"  - {results['vi']} Vietnamese chunks")
            print(f"  - {results['en']} English chunks")
        
        print("\n" + "="*60)
        print("✓ PIPELINE COMPLETE!")
        print("="*60 + "\n")
        
    except KeyboardInterrupt:
        print("\n\n⚠ Stopped by user (Ctrl+C)")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

