import os
import requests
import pandas as pd
from urllib.parse import urlparse

# ----------------------------
# USER CONFIG
# ----------------------------
INPUT_EXCEL = "new dataset.xlsx"               # your input excel file
OUTPUT_EXCEL = "new dataset.xlsx"              # same file updated
IMAGE_URL_COLUMN = "Media_Link"           # column name containing URLs
DOWNLOAD_FOLDER = "downloaded_images"    # folder to store images
NEW_COLUMN = "stored_filename"           # new column to store filenames
# ----------------------------

# Create folder if not exists
os.makedirs(DOWNLOAD_FOLDER, exist_ok=True)

# Read excel
df = pd.read_excel(INPUT_EXCEL,sheet_name="A")

# Create empty new column
df[NEW_COLUMN] = ""
START_INDEX=-1
END_INDEX=26231
# Iterate rows
for index, row in df.iterrows():

    # --- Check if index is in the desired range ---
    if index < START_INDEX:
        continue # Skip rows before the start
    if index > END_INDEX:
        print(f"\n--- Reached end row {END_INDEX}. Stopping process. ---")
        break # Stop the loop
    # --------------------------------------------------

    url = str(row[IMAGE_URL_COLUMN]).strip()

    if not url.startswith("http"):
        continue  # skip invalid or empty cells

    try:
        # Extract file name from URL
        filename = f"img_{index+2}.jpg"

        file_path = os.path.join(DOWNLOAD_FOLDER, filename)

        # Download the image
        response = requests.get(url, timeout=10)
        response.raise_for_status()

        with open(file_path, "wb") as f:
            f.write(response.content)

        # Save filename into new column
        df.at[index, NEW_COLUMN] = filename

        print(f"Downloaded: {filename}")

    except Exception as e:
        print(f"Failed: {url} -> {e}")
        df.at[index, NEW_COLUMN] = "DOWNLOAD_FAILED"

# Save updated excel
df.to_excel(OUTPUT_EXCEL,sheet_name="A", index=False)
print("\nDone! Excel updated.")