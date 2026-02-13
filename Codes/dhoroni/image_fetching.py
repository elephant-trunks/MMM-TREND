import os
import pandas as pd
import requests
import time
from PIL import Image
from serpapi import GoogleSearch

# --- 1. Configuration ---
SERPAPI_API_KEYS = [

    # Add more keys here as needed
    # "your_next_api_key_here",
    # "another_key_here",
]

# --- Internal tracking ---
api_usage_count = {key: 0 for key in SERPAPI_API_KEYS}
current_api_index = 0  # which API key we’re currently using

# --- Robust Path Configuration ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

INPUT_EXCEL_PATH = os.path.join(PROJECT_ROOT, "dataset.xlsx")
OUTPUT_EXCEL_PATH = os.path.join(PROJECT_ROOT, "dataset_updated.xlsx")
IMAGE_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "img")

START_ROW = 2201
END_ROW = 2202
NUM_IMAGES_TO_DOWNLOAD = 10

print("--- Script running in Download-Only Mode (No Scoring) ---")


# --- Helper: get next working API key ---
def get_current_api_key():
    global current_api_index
    return SERPAPI_API_KEYS[current_api_index]


def rotate_api_key():
    global current_api_index
    current_api_index = (current_api_index + 1) % len(SERPAPI_API_KEYS)
    print(f"Switched to next API key (index {current_api_index + 1}/{len(SERPAPI_API_KEYS)}).")


# --- Enhanced download function ---
def download_google_images(query, output_dir, base_id, num_images=10, starting_index=1):
    """
    Downloads images using SerpAPI with key rotation if limit is hit.
    """
    global api_usage_count

    max_retries_per_key = 2
    downloaded_paths = []

    for attempt in range(max_retries_per_key * len(SERPAPI_API_KEYS)):
        api_key = get_current_api_key()
        params = {"q": query, "tbm": "isch", "api_key": api_key}

        try:
            search = GoogleSearch(params)
            results = search.get_dict()
            api_usage_count[api_key] += 1  # count successful call

            # Handle possible errors from SerpAPI
            if "error" in results:
                error_msg = results["error"].lower()
                print(f"SerpAPI error: {results['error']}")

                # Match all possible exhaustion messages
                if any(word in error_msg for word in ["limit", "exceeded", "invalid", "run out of searches", "quota"]):
                    print(f"API key {api_key[:10]}... exhausted or invalid, rotating...")
                    rotate_api_key()
                    time.sleep(1)
                    continue  # try again with next key
                else:
                    return []

            image_results = results.get("images_results", [])
            if not image_results:
                print(f"   - No image results found for query: '{query[:60]}...'")
                return []

            for i, image_data in enumerate(image_results[:num_images]):
                image_url = image_data.get("original") or image_data.get("thumbnail")
                if not image_url:
                    continue

                try:
                    response = requests.get(
                        image_url,
                        timeout=15,
                        headers={"User-Agent": "Mozilla/5.0"},
                        stream=True,
                    )
                    if response.status_code == 200 and "image" in response.headers.get("Content-Type", ""):
                        image_number = i + starting_index
                        image_id = f"IMG-{base_id}_{image_number}.jpg"
                        save_path = os.path.join(output_dir, image_id)
                        with open(save_path, "wb") as f:
                            for chunk in response.iter_content(1024):
                                f.write(chunk)

                        try:
                            Image.open(save_path).verify()
                            downloaded_paths.append(save_path)
                        except Exception:
                            os.remove(save_path)
                            print(f"   - Corrupted image skipped: {image_url}")
                    else:
                        print(f"   - Invalid response for {image_url}")
                except Exception as e:
                    print(f"   - Error downloading image {i + starting_index}: {e}")

                time.sleep(0.5)

            return downloaded_paths  # success, exit after first valid response

        except Exception as e:
            print(f"API key {api_key[:10]}... failed due to: {e}")
            rotate_api_key()
            time.sleep(2)
            continue

    print(f"All API keys failed for query: '{query[:60]}...'")
    return downloaded_paths


# --- 4. Main Execution Logic ---
if __name__ == "__main__":
    os.makedirs(IMAGE_OUTPUT_DIR, exist_ok=True)

    try:
        df = pd.read_excel(INPUT_EXCEL_PATH)
    except FileNotFoundError:
        print(f"Error: The file was not found at '{INPUT_EXCEL_PATH}'")
        exit()
    except Exception as e:
        print(f"An error occurred while reading the Excel file: {e}")
        exit()

    START_INDEX = START_ROW - 2
    END_INDEX = END_ROW - 2

    print(f"--- Starting process. Will process rows {START_ROW} to {END_ROW} ---")

    for index, row in df.iterrows():
        if index < START_INDEX:
            continue
        if index > END_INDEX:
            print(f"\n--- Reached end row {END_ROW}. Stopping process. ---")
            break

        base_id = index + 2
        original_bengali_title = row.get("Title", "")
        translated_title = row.get("Translated_title", "")
        focus = row.get("Focus", "")
        location_bengali = row.get("Location_bengali", "")

        print(f"\n--- Processing Row {index + 1}/{len(df)} (ID: {base_id}) ---")
        print(f"Original Title: {str(original_bengali_title)[:70]}...")

        if not isinstance(original_bengali_title, str) or not original_bengali_title.strip():
            print("   - Skipping row due to missing original title.")
            continue

        #Phase 1
        
        primary_query = f"{original_bengali_title} {location_bengali} {focus}"
        print(f"   - Pass 1: Searching with query: '{primary_query[:80]}...'")

        downloaded_images_pass1 = download_google_images(
            primary_query, IMAGE_OUTPUT_DIR, base_id, NUM_IMAGES_TO_DOWNLOAD, starting_index=1
        )

        all_downloaded_images = list(downloaded_images_pass1)
        num_downloaded = len(all_downloaded_images)

        if num_downloaded < NUM_IMAGES_TO_DOWNLOAD:
            remaining_needed = NUM_IMAGES_TO_DOWNLOAD - num_downloaded
            starting_index_pass2 = num_downloaded + 1
            fallback_query = f"{original_bengali_title} {location_bengali}"
            
        #Phase 2
            print(f"   - Pass 2: Fetching {remaining_needed} more images.")
            downloaded_images_pass2 = download_google_images(
                fallback_query,
                IMAGE_OUTPUT_DIR,
                base_id,
                remaining_needed,
                starting_index=starting_index_pass2,
            )
            all_downloaded_images.extend(downloaded_images_pass2)

        if not all_downloaded_images:
            print("   - No images downloaded from any query. Moving to next row.")
            continue

        print(f"   - Downloaded {len(all_downloaded_images)} images.")
        for i, img_path in enumerate(all_downloaded_images):
            image_id = os.path.basename(img_path)
            df.at[index, f"Image {i+1} ID"] = image_id
            print(f"   - Image Col {i+1}: {image_id}")

        if (index + 1) % 10 == 0:
            df.to_excel(OUTPUT_EXCEL_PATH, index=False)
            print(f"\nProgress saved after processing {index + 1} rows.")

    df.to_excel(OUTPUT_EXCEL_PATH, index=False)
    print(f"\nProcessing complete! The updated data has been saved to '{OUTPUT_EXCEL_PATH}'")

    # --- Final API usage report ---
    print("\n--- API USAGE SUMMARY ---")
    for key, count in api_usage_count.items():
        print(f"Key {key[:10]}... → {count} requests used.")
