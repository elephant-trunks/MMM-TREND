import os
import pandas as pd
from transformers import pipeline
import re
import torch
import gc
from tqdm import tqdm
import sys
import time
import signal  # <--- NEW: Required for catching crashes/interrupts

# ---------------- CONFIGURATION ----------------
INPUT_FILE = r"/home/gudahvr_iitp/task2/textpro/txts.xlsx"
OUTPUT_FILE = r"/home/gudahvr_iitp/task2/txts_processed.xlsx"
TEXT_FOLDER = r"/home/gudahvr_iitp/task2/textpro/articles_txt"

# SAFETY SETTINGS
# MAX_ROWS_LIMIT = 5500      
SAVE_EVERY_N_ROWS = 500    # LOWERED: Saves more often to minimize data loss risk

# Check GPU
device = 0 if torch.cuda.is_available() else -1
device_name = torch.cuda.get_device_name(0) if device == 0 else "CPU"
print(f"Running on: {device_name}")

# ---------------- GLOBAL VARIABLES ----------------
translator = None
summarizer = None
df = None  # Global reference for the signal handler

def load_models():
    """Loads models safely."""
    global translator, summarizer
    print("(Re)Loading models...")
    
    if 'translator' in globals() and translator is not None: del translator
    if 'summarizer' in globals() and summarizer is not None: del summarizer
    clear_memory()
    
    try:
        print("   -> Loading Translator (NLLB)...")
        translator = pipeline(
            "translation", 
            model="facebook/nllb-200-distilled-600M", 
            src_lang="hin_Deva", 
            tgt_lang="eng_Latn", 
            device=device
        )
        
        print("   -> Loading Summarizer (DistilBART)...")
        summarizer = pipeline(
            "summarization", 
            model="sshleifer/distilbart-cnn-12-6", 
            device=device
        )
        print("Pipeline Ready.")
    except Exception as e:
        print(f"Model load failed: {e}")
        sys.exit(1)

def clear_memory():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()

# ---------------- SAFETY & ERROR HANDLING ----------------

def emergency_save(signum=None, frame=None):
    """
    Called when script is killed (Ctrl+C) or finishes.
    Ensures data is saved to disk before process dies.
    """
    print("\n STOPPING... Saving progress to disk immediately...")
    try:
        if df is not None:
            df.to_excel(OUTPUT_FILE, index=False)
            print(f"Emergency Save Successful: {OUTPUT_FILE}")
        else:
            print("Dataframe was empty, nothing to save.")
    except Exception as e:
        print(f"CRITICAL SAVE ERROR: {e}")
    
    if signum is not None:
        sys.exit(0)

# Register the signal handlers (Catches Ctrl+C and Kill signals)
signal.signal(signal.SIGINT, emergency_save)   # Handle Ctrl+C
signal.signal(signal.SIGTERM, emergency_save)  # Handle termination signals

def check_fatal_error(e, row_idx, df):
    """Detects fatal errors and auto-restarts."""
    error_str = str(e)
    fatal_errors = ["device-side assert", "CUDA error", "out of memory", "CUBLAS", "illegal memory access"]
    
    if any(x in error_str for x in fatal_errors):
        print(f"\nFATAL ERROR on Row {row_idx}. Initiating Auto-Recovery...")
        df.at[row_idx, "english_summary"] = "SKIPPED_CRASH"
        df.to_excel(OUTPUT_FILE, index=False)
        python = sys.executable
        os.execl(python, python, *sys.argv)

# ---------------- PROCESSING FUNCTIONS ----------------

def split_text(text, max_len=400):
    if not isinstance(text, str): return []
    sentences = re.split(r'(?<=[.!?।])\s+', text)
    chunks, current = [], ""
    for s in sentences:
        if len(current) + len(s) < max_len:
            current += s + " "
        else:
            chunks.append(current.strip())
            current = s + " "
    if current: chunks.append(current.strip())
    return chunks

def process_row(text, idx, df):
    if not isinstance(text, str) or len(text) < 5: return "Empty"
    
    # 1. Translate
    chunks = split_text(text)
    chunks = chunks[:30] # Hard cap
    
    english_chunks = []
    try:
        with torch.inference_mode():
            results = translator(chunks, max_length=512, batch_size=16)
            english_chunks = [res['translation_text'] for res in results]
    except Exception as e:
        return f"Translation Failed: {e}"

    full_english_text = " ".join(english_chunks)
    
    if len(full_english_text) < 700:
        return full_english_text

    # 2. Summarize
    safe_input = full_english_text[:3000]

    try:
        with torch.inference_mode():
            summary_result = summarizer(
                safe_input, 
                max_length=100, 
                min_length=25, 
                do_sample=False, 
                truncation=True
            )
        return summary_result[0]['summary_text']

    except Exception as e:
        check_fatal_error(e, idx, df)
        return "Summarization Failed"

# ---------------- MAIN LOGIC ----------------

load_models()

# Load Data
if os.path.exists(OUTPUT_FILE):
    print(f"Resuming from: {OUTPUT_FILE}")
    df = pd.read_excel(OUTPUT_FILE)
else:
    print(f"Starting fresh from: {INPUT_FILE}")
    if not os.path.exists(INPUT_FILE):
        print("Input file not found.")
        sys.exit(1)

# Ensure columns exist
if "english_summary" not in df.columns: df["english_summary"] = ""

# Identify Work
rows_to_process = []
for idx, row in df.iterrows():
    eng_summary = str(row.get("english_summary", "")).strip()
    if eng_summary == "SKIPPED_CRASH": continue
    
    summary_bad = (not eng_summary) or (eng_summary == "nan") or (eng_summary == "Read Error") or (eng_summary == "Summarization Failed") or (eng_summary.startswith("Translation Failed"))
    if summary_bad:
        rows_to_process.append(idx)

# Apply Limit
print(f"Total Pending Rows: {len(rows_to_process)}")
# if len(rows_to_process) > MAX_ROWS_LIMIT:
#     rows_to_process = rows_to_process[:MAX_ROWS_LIMIT]

# --- PROCESSING LOOP WITH SAFETY NET ---
try:
    pbar = tqdm(total=len(rows_to_process), desc="Processing", unit="row", mininterval=2.0)
    batch_save_counter = 0

    for idx in rows_to_process:
        row = df.loc[idx]

        # Process Row
        current_summary = str(row.get("english_summary", "")).strip()
        bad_status = ["nan", "Read Error", "Summarization Failed", "File Not Found"]
        
        if not current_summary or current_summary in bad_status or current_summary.startswith("Translation Failed"):
            file_base = str(row.get("text_files", "")).strip()
            if not file_base.lower().endswith(".txt"): file_base += ".txt"
            file_path = os.path.join(TEXT_FOLDER, file_base)

            if os.path.exists(file_path):
                try:
                    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                        content = f.read()
                    
                    if content.strip():
                        df.at[idx, "english_summary"] = process_row(content, idx, df)
                    else:
                        df.at[idx, "english_summary"] = "Empty File"
                except Exception as e:
                    df.at[idx, "english_summary"] = "Read Error"
            else:
                df.at[idx, "english_summary"] = "File Not Found"

        pbar.update(1)
        batch_save_counter += 1

        # Periodic Save (Now safer at 200 rows)
        if batch_save_counter >= SAVE_EVERY_N_ROWS:
            df.to_excel(OUTPUT_FILE, index=False)
            clear_memory()
            batch_save_counter = 0

    pbar.close()

finally:
    # THE SAFETY NET
    # This block ALWAYS runs, even if the loop crashes or you press Ctrl+C
    print("\nFinalizing and saving data...")
    df.to_excel(OUTPUT_FILE, index=False)
    print(f"All progress saved to: {OUTPUT_FILE}")