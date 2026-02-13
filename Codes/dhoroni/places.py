import re
import os
import pandas as pd
from collections import Counter
from transformers import pipeline
from langdetect import detect, DetectorFactory

# Consistent language detection
DetectorFactory.seed = 0

# Load Multilingual Models
print("Loading Multilingual Models...")
ner_model = pipeline("ner", model="Davlan/xlm-roberta-base-ner-hrl", aggregation_strategy="simple")
translator = pipeline("translation", model="facebook/m2m100_418M")
print("Models loaded successfully!\n")

# --- SETTINGS ---
EXCEL_FILE = "datset.xlsx"  # Change this to your actual Excel name
SHEET_NAME = "Sheet1"             # Change to your sheet name
FILE_NAME_COLUMN = "File"     # The column containing 'txt_1.txt', etc.
OUTPUT_COLUMN = "Locations" # The column where we will save results
FOLDER_PATH = "articles_txt"       # Folder where text files live
SAVE_INTERVAL = 5                 # Save to Excel every 5 rows
START_ROW = 6744                     # Change this to start from a specific index
END_ROW = 6754                    # Change this to stop at a specific index
# ----------------
def get_places(file_path):
    if not os.path.exists(file_path):
        print("No file")
        return "File Not Found"

    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read().strip()
    
    if len(text) < 5: return "Text too short"

    try:
        # Detect Language
        lang_code = detect(text)
        
        # M2M100 uses 'zh' for Chinese; langdetect might use 'zh-cn'
        # This fix prevents the "Translation requires a src_lang" error
        clean_lang = lang_code.split('-')[0] 
        
        # Extract Locations
        ner_results = ner_model(text)
        raw_places = [ent["word"].strip() for ent in ner_results if ent["entity_group"] == "LOC"]
        
        # Filter: No numbers, no single characters, keep alphabets/scripts
        cleaned_places = [re.sub(r'[^\w\s]', '', p) for p in raw_places if len(p) > 1]

        if not cleaned_places: return "No Locations Found"

        # Translate Top 2
        place_freq = Counter(cleaned_places)
        top_native = [p for p, _ in place_freq.most_common(2)]
        
        # Set source language safely
        translator.tokenizer.src_lang = clean_lang 
        translated = []
        for p in top_native:
            # We wrap in try/except so one bad word doesn't break the whole row
            try:
                res = translator(p, forced_bos_token_id=translator.tokenizer.get_lang_id("en"))
                translated.append(res[0]["translation_text"])
            except:
                translated.append(p) # Fallback to original word
        
        return ", ".join(translated)
    except Exception as e:
        return f"Processing Error: {str(e)[:20]}"
    
# --- MAIN EXECUTION ---
# Load the Excel
df = pd.read_excel(EXCEL_FILE, sheet_name=SHEET_NAME)

# Ensure output column exists
if OUTPUT_COLUMN not in df.columns:
    df[OUTPUT_COLUMN] = ""

#Process within the specified range
print(f" Starting processing from row {START_ROW} to {END_ROW}...")

for i in range(START_ROW, min(END_ROW, len(df))):
    file_name = str(df.loc[i, FILE_NAME_COLUMN]).strip()
    full_path = os.path.join(FOLDER_PATH, file_name)
    
    print(f"[{i}] {file_name}...", end=" ")
    
    # 1️⃣ Run the extraction function
    result = get_places(full_path)
    
    # 2️⃣ Define all failure conditions (including your specific translation error)
    error_keywords = [
        "File Not Found", 
        "No Locations Found", 
        "Processing Error", 
        "Text too short",
        "Translation requires"  # This catches your specific error message
    ]
    
    # 3️⃣ Check if result is an error
    is_error = any(keyword in result for keyword in error_keywords)
    
    if is_error:
        # Fallback to the existing "Location" column
        backup_location = str(df.loc[i, "Location"]).strip()
        
        # Check if backup is valid (not 'nan' or empty)
        if backup_location.lower() != "nan" and backup_location != "":
            result = backup_location
            print(f"Error detected. Used backup: {result}")
        else:
            # If even the backup is empty, we keep the original error for your records
            print(f"Error detected and backup column was empty.")
    else:
        print(f"Success: {result}")

    # 4️⃣ Update the DataFrame
    df.at[i, OUTPUT_COLUMN] = result
    
    # # 5️⃣ Save every 5-10 rows
    if (i + 1) % SAVE_INTERVAL == 0:
        df.to_excel(EXCEL_FILE, index=False)
        print(f"--- Progress saved at row {i} ---")


# Final save for any remaining rows
df.to_excel(EXCEL_FILE, index=False)
print("\n All tasks complete! Excel updated.")