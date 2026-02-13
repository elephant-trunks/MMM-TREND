import os
import pandas as pd
import requests
from bs4 import BeautifulSoup
from newspaper import Article

# -----------------------------
# CONFIG
# -----------------------------
XLSX_PATH = "txts.xlsx"       # Excel file
URL_COLUMN = "File"           # Column with news URLs
OUTPUT_DIR = "articles_txt"   # Folder to save articles
TEXT_COLUMN = "text files"    # Column to store generated filenames
TIMEOUT = 10                  # Timeout for requests

# -----------------------------
# SETUP
# -----------------------------
os.makedirs(OUTPUT_DIR, exist_ok=True)
df = pd.read_excel(XLSX_PATH)
headers = {"User-Agent": "Mozilla/5.0"}

# Ensure the new column exists
df[TEXT_COLUMN] = ""

# -----------------------------
# FUNCTION: Extract article
# -----------------------------
def extract_article(url, lang="en"):
    """Extracts article text from a URL using newspaper3k, fallback to BeautifulSoup."""
    try:
        article = Article(url, language=lang)
        article.download()
        article.parse()
        text = article.text.strip()
        if len(text) > 50:
            return text
    except:
        pass

    try:
        response = requests.get(url, headers=headers, timeout=TIMEOUT)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "lxml")
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()
        text = "\n".join(line.strip() for line in soup.get_text().splitlines() if line.strip())
        return text
    except:
        return ""

# -----------------------------
# FETCH & SAVE
# -----------------------------
for idx, url in enumerate(df[URL_COLUMN], start=1):
    if pd.isna(url):
        print(f"Row {idx} skipped (empty URL)")
        continue

    try:
        text = extract_article(url, lang="en")

        if not text:
            print(f"Empty article at row {idx}, URL: {url}")
            continue

        file_name = f"txt_{idx}.txt"
        file_path = os.path.join(OUTPUT_DIR, file_name)

        with open(file_path, "w", encoding="utf-8") as f:
            f.write(text)

        # Save filename into the DataFrame
        df.at[idx-1, TEXT_COLUMN] = file_name

        print(f"Saved: {file_name}")

    except Exception as e:
        print(f"Failed at row {idx}, URL: {url}, Error: {e}")

# -----------------------------
# SAVE UPDATED EXCEL
# -----------------------------
df.to_excel(XLSX_PATH, index=False)
print(f"Updated Excel saved with '{TEXT_COLUMN}' column.")