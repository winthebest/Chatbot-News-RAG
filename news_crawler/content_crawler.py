from selenium import webdriver
from selenium.webdriver.common.by import By
import time
import csv
import json
import re
from helper_functions import configure_driver
from deep_translator import GoogleTranslator

# Initialize Google Translate
translator = GoogleTranslator(source="auto", target="en")

# Abbreviation dictionary
abbreviation_dict = {
    "AI": "Trí tuệ nhân tạo",
    "IoT": "Internet vạn vật",
    "ML": "Học máy",
    "NLP": "Xử lí ngôn ngữ tự nhiên"
}

def replace_abbreviations(text):
    if not text:
        return ""
    for abbr, full_form in abbreviation_dict.items():
        text = re.sub(rf'\b{abbr}\b', full_form, text, flags=re.IGNORECASE)
    return text

# Safe translation function for long text (automatically splits < 5000 characters)
def translate_long_text(text, max_len=4800):
    if not text:
        return ""

    text = replace_abbreviations(text)

    # Split by newlines first
    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
    chunks = []
    current = ""

    for p in paragraphs:
        # If a paragraph is too long, split it roughly
        if len(p) > max_len:
            for i in range(0, len(p), max_len):
                sub = p[i:i + max_len]
                if sub.strip():
                    chunks.append(sub.strip())
            continue

        # Combine small paragraphs into one chunk until near max_len
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
            print(f"Error translating a content chunk: {e}")
            # If error, try to skip that chunk to avoid breaking the whole article
            continue

    return "\n".join(translated_parts)

# JSON file to save data
output_file = "data/raw/vnexpress_articles.json"

# Read existing data to avoid duplicate crawling
try:
    with open(output_file, "r", encoding="utf-8") as file:
        articles = json.load(file)
        crawled_urls = {article["url"] for article in articles}
except (FileNotFoundError, json.JSONDecodeError):
    articles = []
    crawled_urls = set()

# Read URL list from CSV file
input_file = "data/raw/vnexpress_links.csv"
urls = []
with open(input_file, "r", encoding="utf-8") as file:
    reader = csv.reader(file)
    next(reader)  # Skip header
    urls = [row[0] for row in reader if row[0] not in crawled_urls]

if not urls:
    print("All URLs have been crawled. Nothing new to collect.")
    exit()

# Initialize Selenium WebDriver
driver = configure_driver()

def wait_for_page_load(driver, timeout=30):
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            state = driver.execute_script("return document.readyState")
        except Exception:
            # If temporary error when calling JS, wait more
            time.sleep(1)
            continue

        if state == "complete":
            return True
        time.sleep(1)
    print("Warning: Page did not fully load after timeout!")
    return False

# Start crawling each URL
for url in urls:
    try:
        driver.get(url)
        if not wait_for_page_load(driver):
            print(f"Page {url} may not have fully loaded!")
            continue
    except Exception as e:
        print(f"Timeout loading {url}, error: {e}")
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

        # Translate title & description (short, no need to split)
        try:
            title_en = translator.translate(title) if title else ""
        except Exception as e:
            print(f"Error translating title {url}: {e}")
            title_en = ""

        try:
            description_en = translator.translate(description) if description else ""
        except Exception as e:
            print(f"Error translating description {url}: {e}")
            description_en = ""

        # Translate long content using split function
        content_en = translate_long_text(content)

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

        # Write data to JSON file after each article
        with open(output_file, "w", encoding="utf-8") as file:
            json.dump(articles, file, ensure_ascii=False, indent=4)

        print(f"Collected and saved: {url}")

    except Exception as e:
        print(f"Error extracting data from {url}: {e}")

# Close browser
driver.quit()
print(f"Completed crawling and saved to {output_file}")
