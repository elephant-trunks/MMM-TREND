import os
import pandas as pd
from transformers import pipeline
from keybert import KeyBERT
from datetime import datetime
import re

# Initialize all models from Hugging Face
print("Initializing models... (This may take a while on the first run)")

# 1. Summarizer
summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device=-1)

# 2. Bengali to English Translator
bn_to_en_translator = pipeline("translation", model="Helsinki-NLP/opus-mt-bn-en", device=-1)

# 3. English to Bengali Translator (CORRECTED MODEL NAME)
# Using a reliable, publicly available model for this translation direction.
en_to_bn_translator = pipeline("translation", model="csebuetnlp/banglat5_nmt_en_bn", device=-1)

# 4. Keyword Extractor
kw_model = KeyBERT()

print("Models initialized.")

# --- HELPER FUNCTION ---
def split_text_for_translation(text, max_len=400):
    """
    Splits text into smaller, safer chunks to stay under the model's 512 token limit.
    """
    if not isinstance(text, str): return []
    sentences = re.split('(?<=[।?!])\s+', text)
    chunks, current_chunk = [], ""
    for sentence in sentences:
        if len(current_chunk) + len(sentence) + 1 < max_len:
            current_chunk += sentence + " "
        else:
            if current_chunk: chunks.append(current_chunk)
            current_chunk = sentence + " "
    if current_chunk: chunks.append(current_chunk)
    return chunks

# --- CORE PROCESSING FUNCTION (WITH BENGALI KEYWORD TRANSLATION) ---
def process_text(text, keyword_count=5):
    """
    Translates, summarizes, extracts English keywords, and translates them back to Bengali.
    """
    original_chunks = split_text_for_translation(text)
    if not original_chunks:
        return "Text could not be processed", [], []

    # Step 1: Translate (Bn -> En) and Summarize each chunk
    summarized_chunks = []
    print(f"    -> Translating and summarizing {len(original_chunks)} chunks...")
    try:
        for chunk in original_chunks:
            if not chunk.strip(): continue
            translated_result = bn_to_en_translator(chunk, max_length=512)
            english_chunk = translated_result[0]['translation_text']

            if len(english_chunk.split()) > 15:
                summary_result = summarizer(english_chunk, max_length=100, min_length=15, do_sample=False)
                summarized_chunks.append(summary_result[0]['summary_text'])
            else:
                summarized_chunks.append(english_chunk)
    except Exception as e:
        return f"Processing Error on Chunk: {e}", [], []

    # Step 2: Combine and create a final summary
    combined_summary = " ".join(summarized_chunks)
    final_summary = ""
    try:
        if len(combined_summary.split()) > 50:
            print("    -> Performing final summarization...")
            final_summary = summarizer(combined_summary, max_length=250, min_length=50, do_sample=False)[0]['summary_text']
        else:
            final_summary = combined_summary
    except Exception:
        final_summary = combined_summary

    # Step 3: Extract English Keywords
    if not final_summary:
        return "Failed to generate summary", [], []

    print("    -> Extracting English keywords...")
    english_keywords = []
    try:
        keywords = kw_model.extract_keywords(final_summary, keyphrase_ngram_range=(1, 2), stop_words='english', top_n=keyword_count)
        english_keywords = [kw[0] for kw in keywords if kw]
    except Exception:
        return final_summary, [], []

    # Step 4: Translate English Keywords back to Bengali
    if not english_keywords:
        return final_summary, [], []

    print("    -> Translating keywords to Bengali...")
    bengali_keywords = []
    try:
        # The new model works well with a list of keywords
        bengali_results = en_to_bn_translator(english_keywords)
        bengali_keywords = [item['translation_text'] for item in bengali_results]
    except Exception as e:
        print(f"    -> Could not translate keywords to Bengali: {e}")
        return final_summary, english_keywords, []

    return final_summary, english_keywords, bengali_keywords


# ---------- PROCESS EXCEL FILE (UPDATED) ----------
excel_file = r"C:\Users\DELL\Downloads\Task\tests\testd.xlsx"
text_folder = r"C:\Users\DELL\Downloads\Task\tests\text_files"

try:
    df = pd.read_excel(excel_file)
except FileNotFoundError:
    print(f"ERROR: Excel file not found at {excel_file}. Please check the path.")
    exit()

summaries, keywords_column, bengali_keywords_column = [], [], []
print("\nStarting the process...")

for index, row in df.iterrows():
    file_name = str(row["File"]).strip()
    file_path = os.path.join(text_folder, file_name)
    print(f"\nProcessing row {index + 1}/{len(df)}: {file_name}")

    if os.path.exists(file_path):
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                text_content = f.read()
            if not text_content.strip():
                summaries.append("Empty File")
                keywords_column.append("")
                bengali_keywords_column.append("")
                continue

            summary, eng_keywords, ben_keywords = process_text(text_content)
            summaries.append(summary)
            keywords_column.append(", ".join(eng_keywords))
            bengali_keywords_column.append(", ".join(ben_keywords)) # Add Bengali keywords
            print("    -> Successfully processed.")
        except Exception as e:
            print(f"    -> An unexpected error occurred: {e}")
            summaries.append("Major Processing Error")
            keywords_column.append("")
            bengali_keywords_column.append("")
    else:
        print(f"    -> File not found: {file_path}")
        summaries.append("File Not Found")
        keywords_column.append("")
        bengali_keywords_column.append("")

df["summary"] = summaries
df["keywords"] = keywords_column
df["keywords_in_bengali"] = bengali_keywords_column # Add the new column to the DataFrame

try:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_excel = excel_file.replace(".xlsx", f"_with_summary_keywords_{timestamp}.xlsx")
    df.to_excel(output_excel, index=False)
    print(f"\nExcel processing complete! Results saved to: {output_excel}")
except PermissionError:
    print(f"\nPERMISSION ERROR: Could not save Excel file. Is '{output_excel}' open?")