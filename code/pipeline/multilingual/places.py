import re
import os
import pandas as pd
from collections import Counter
from transformers import pipeline
from langdetect import detect, DetectorFactory

DetectorFactory.seed = 0

print("Loading Multilingual Models...")
ner_model = pipeline("ner", model="Davlan/xlm-roberta-base-ner-hrl", aggregation_strategy="simple")
translator = pipeline("translation", model="facebook/m2m100_418M")
print("Models loaded successfully!\n")


EXCEL_FILE = "datset.xlsx"  
SHEET_NAME = "Sheet1"             
FILE_NAME_COLUMN = "File"     
OUTPUT_COLUMN = "Locations" 
FOLDER_PATH = "articles_txt"       
SAVE_INTERVAL = 5                 
START_ROW = 2301                     
END_ROW = 28531                   

def get_places(file_path):
    if not os.path.exists(file_path):
        print("No file")
        return "File Not Found"

    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read().strip()

    if len(text) < 5: return "Text too short"

    try:
        
        lang_code = detect(text)

        clean_lang = lang_code.split('-')[0]

        ner_results = ner_model(text)
        raw_places = [ent["word"].strip() for ent in ner_results if ent["entity_group"] == "LOC"]

        cleaned_places = [re.sub(r'[^\w\s]', '', p) for p in raw_places if len(p) > 1]

        if not cleaned_places: return "No Locations Found"

        place_freq = Counter(cleaned_places)
        top_native = [p for p, _ in place_freq.most_common(2)]

        translator.tokenizer.src_lang = clean_lang
        translated = []
        for p in top_native:
            try:
                res = translator(p, forced_bos_token_id=translator.tokenizer.get_lang_id("en"))
                translated.append(res[0]["translation_text"])
            except:
                translated.append(p) 

        return ", ".join(translated)
    except Exception as e:
        return f"Processing Error: {str(e)[:20]}"


df = pd.read_excel(EXCEL_FILE, sheet_name=SHEET_NAME)

if OUTPUT_COLUMN not in df.columns:
    df[OUTPUT_COLUMN] = ""

print(f" Starting processing from row {START_ROW} to {END_ROW}...")

for i in range(START_ROW, min(END_ROW, len(df))):
    file_name = str(df.loc[i, FILE_NAME_COLUMN]).strip()
    full_path = os.path.join(FOLDER_PATH, file_name)

    print(f"[{i}] {file_name}...", end=" ")

    result = get_places(full_path)

    error_keywords = [
        "File Not Found",
        "No Locations Found",
        "Processing Error",
        "Text too short",
        "Translation requires"  
    ]

    is_error = any(keyword in result for keyword in error_keywords)

    if is_error:
        backup_location = str(df.loc[i, "Location"]).strip()

        if backup_location.lower() != "nan" and backup_location != "":
            result = backup_location
            print(f"Error detected. Used backup: {result}")
        else:
            print(f"Error detected and backup column was empty.")
    else:
        print(f"Success: {result}")

    df.at[i, OUTPUT_COLUMN] = result

    if (i + 1) % SAVE_INTERVAL == 0:
        df.to_excel(EXCEL_FILE, index=False)
        print(f"--- Progress saved at row {i} ---")


df.to_excel(EXCEL_FILE, index=False)
print("\n All tasks complete! Excel updated.")