import os
import pandas as pd
import requests
from bs4 import BeautifulSoup
from newspaper import Article

XLSX_PATH = "txts.xlsx"       
URL_COLUMN = "File"           
OUTPUT_DIR = "articles_txt"   
TEXT_COLUMN = "text files"    
TIMEOUT = 10                  


os.makedirs(OUTPUT_DIR, exist_ok=True)
df = pd.read_excel(XLSX_PATH)
headers = {"User-Agent": "Mozilla/5.0"}


df[TEXT_COLUMN] = ""


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

        df.at[idx-1, TEXT_COLUMN] = file_name

        print(f"Saved: {file_name}")

    except Exception as e:
        print(f"Failed at row {idx}, URL: {url}, Error: {e}")


df.to_excel(XLSX_PATH, index=False)
print(f"Updated Excel saved with '{TEXT_COLUMN}' column.")