import os
import pandas as pd
import torch
import time
from PIL import Image

# --- COMBINED IMPORTS ---
# For BLIP
from transformers import BlipProcessor, BlipForImageTextRetrieval
# For CLIP
from sentence_transformers import SentenceTransformer
# ------------------------

# --- 1. Configuration ---
# --- Robust Path Configuration ---
# SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) # Uncomment if running as .py
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) # Use this if running in Jupyter/Colab
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

# --- THIS SCRIPT READS AND WRITES TO THE *SAME* UPDATED FILE ---
EXCEL_FILE_PATH = os.path.join(PROJECT_ROOT, "datasetttt.xlsx")

# --- Image Directory ---
IMAGE_OUTPUT_DIR = os.path.join(PROJECT_ROOT,"all_images")
# -----------------------


# ---
# --- Set your start and end rows ---
# ---
START_ROW = 2  # <-- SET THIS TO YOUR 'sno'
END_ROW = 2301   # <-- Set this to your desired end row
# -----------------------------------

NUM_IMAGES_TO_CHECK = 10 # Check for up to 10 image columns

# --- 2. Model and Processor Loading (BOTH MODELS) ---
print("Initializing models... (This may take a while on the first run)")
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# --- Set up device (GPU if available) ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- Load BLIP model and processor ---
print("Loading BLIP model (Salesforce/blip-itm-large-coco)...")
blip_model_id = "Salesforce/blip-itm-large-coco"
blip_processor = BlipProcessor.from_pretrained(blip_model_id)
blip_model = BlipForImageTextRetrieval.from_pretrained(blip_model_id).to(device)
blip_model.eval()
print("BLIP model loaded.")

# --- Load CLIP model ---
print("Loading CLIP model (clip-ViT-B-32)...")
# SentenceTransformer will automatically use the GPU (device) if available
clip_model = SentenceTransformer('clip-ViT-B-32')
clip_model.eval()
print("CLIP model loaded.")

print("\n--- All models initialized successfully! ---\n")


# --- 3. Helper Functions (Both) ---

def get_blip_scores(text_caption, image_paths, model, processor, device):
    """
    Computes BLIP Image-Text Matching (ITM) scores.
    Returns a list of scores (probabilities).
    """
    try:
        images = [Image.open(img_path).convert('RGB') for img_path in image_paths]
        texts = [text_caption] * len(images)

        inputs = processor(
            images=images,
            text=texts,
            return_tensors="pt",
            padding=True
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)

        itm_scores = outputs.itm_score
        match_probabilities = torch.nn.functional.softmax(itm_scores, dim=1)[:, 1]

        return match_probabilities.cpu().tolist()

    except Exception as e:
        print(f"   - Error calculating BLIP scores: {e}")
        return [None] * len(image_paths)

def get_clip_scores(text_caption, image_paths, model):
    """
    Computes CLIP similarity scores between a text and a list of images.
    Returns a list of scores.
    """
    try:
        images = [Image.open(img_path).convert('RGB') for img_path in image_paths]

        # Encode text (once)
        text_embedding = model.encode(text_caption, convert_to_tensor=True)

        # Encode images (in a batch)
        image_embeddings = model.encode(images, convert_to_tensor=True)

        # Compute cosine similarity
        cosine_scores = torch.nn.functional.cosine_similarity(text_embedding, image_embeddings)

        return cosine_scores.cpu().tolist()

    except Exception as e:
        print(f"   - Error calculating CLIP scores: {e}")
        return [None] * len(image_paths)

# --- 4. Main Execution Logic ---

if __name__ == "__main__":
    if not os.path.exists(IMAGE_OUTPUT_DIR):
        print(f"Error: Image directory not found at {IMAGE_OUTPUT_DIR}")
        print("Please check the path and try again.")
        exit()

    try:
        df = pd.read_excel(EXCEL_FILE_PATH)
    except FileNotFoundError:
        print(f"Error: The file was not found at '{EXCEL_FILE_PATH}'")
        exit()
    except Exception as e:
        print(f"An error occurred while reading the Excel file: {e}")
        exit()

    # --- Convert 1-based Excel rows to 0-based pandas index ---
    START_INDEX = START_ROW - 2
    END_INDEX = END_ROW - 2
    # -------------------------------------------------------------

    print(f"--- Starting SCORING process. Will process rows {START_ROW} to {END_ROW} (index {START_INDEX} to {END_INDEX}) ---")

    # --- Main Loop ---
    for index, row in df.iterrows():

        # --- Check if index is in the desired range ---
        if index < START_INDEX:
            continue # Skip rows before the start
        if index > END_INDEX:
            print(f"\n--- Reached end row {END_ROW}. Stopping process. ---")
            break # Stop the loop
        # --------------------------------------------------

        base_id = index + 2
        query = row.get('Title', '')+row.get('Summary','')

        print(f"\n--- Scoring Row {index + 1}/{len(df)} (ID: {base_id}) ---")
        print(f"Original Title: {str(query)[:70]}...")

        if not isinstance(query, str) or not query.strip():
            print("   - Skipping row due to missing original title.")
            continue

        # --- Stage 1: Collect all downloaded image paths for this row ---
        image_paths_to_score = []
        image_ids_in_row = []
        for i in range(1, NUM_IMAGES_TO_CHECK + 1):
            image_id_col = f'Image {i} ID'
            if image_id_col in row and pd.notna(row[image_id_col]):
                image_id = str(row[image_id_col])
                image_path = os.path.join(IMAGE_OUTPUT_DIR, image_id)

                if os.path.exists(image_path):
                    image_paths_to_score.append(image_path)
                    image_ids_in_row.append({'id': image_id, 'col_num': i})
                else:
                    print(f"   - Warning: Image not found (skipping): {image_id}")

        if not image_paths_to_score:
            print("   - No images found for this row. Moving to next.")
            continue

        # --- Stage 2: Score all images with BLIP ---
        print(f"   - Scoring {len(image_paths_to_score)} images with BLIP...")
        blip_scores = get_blip_scores(
            query,
            image_paths_to_score,
            blip_model,
            blip_processor,
            device
        )

        # --- Stage 3: Score all images with CLIP ---
        print(f"   - Scoring {len(image_paths_to_score)} images with CLIP...")
        clip_scores = get_clip_scores(
            query,
            image_paths_to_score,
            clip_model
        )

        # --- Stage 4: Assign scores and find the best match FOR BOTH ---
        row_blip_scores_tracking = []
        row_clip_scores_tracking = []

        # Loop once to assign all scores
        for i, image_info in enumerate(image_ids_in_row):
            image_id = image_info['id']
            col_num = image_info['col_num']

            # --- Process BLIP Score ---
            blip_score = blip_scores[i]
            if blip_score is not None:
                row_blip_scores_tracking.append({'id': image_id, 'score': blip_score})
                df.at[index, f'Image {col_num} BLIP Score'] = blip_score
                print(f"     - Image {col_num} BLIP: {blip_score:.4f}")

            # --- Process CLIP Score ---
            clip_score = clip_scores[i]
            if clip_score is not None:
                row_clip_scores_tracking.append({'id': image_id, 'score': clip_score})
                df.at[index, f'Image {col_num} CLIP Score'] = clip_score
                print(f"     - Image {col_num} CLIP: {clip_score:.4f}")


        # --- Stage 5: Find and save the best match for each model ---
        if row_blip_scores_tracking:
            best_blip_match = max(row_blip_scores_tracking, key=lambda x: x['score'])
            df.at[index, 'Best Match Image (BLIP)'] = best_blip_match['id']
            print(f"   - Best BLIP Match: {best_blip_match['id']} (Score: {best_blip_match['score']:.4f})")

        if row_clip_scores_tracking:
            best_clip_match = max(row_clip_scores_tracking, key=lambda x: x['score'])
            df.at[index, 'Best Match Image (CLIP)'] = best_clip_match['id']
            print(f"   - Best CLIP Match: {best_clip_match['id']} (Score: {best_clip_match['score']:.4f})")

        # Save progress periodically
        if (index + 1) % 10 == 0:
            df.to_excel(EXCEL_FILE_PATH, index=False)
            print(f"\nProgress saved after processing {index + 1} rows.")

    # Final save
    df.to_excel(EXCEL_FILE_PATH, index=False)
    print(f"\nScoring complete! The updated data has been saved to '{EXCEL_FILE_PATH}'")