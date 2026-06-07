import os
import requests
import pandas as pd
from urllib.parse import urlparse

INPUT_EXCEL = "new dataset.xlsx"               
OUTPUT_EXCEL = "new dataset.xlsx"              
IMAGE_URL_COLUMN = "Media_Link"           
DOWNLOAD_FOLDER = "downloaded_images"    
NEW_COLUMN = "stored_filename"           


os.makedirs(DOWNLOAD_FOLDER, exist_ok=True)

df = pd.read_excel(INPUT_EXCEL,sheet_name="A")

df[NEW_COLUMN] = ""
START_INDEX=-1
END_INDEX=26231

for index, row in df.iterrows():

    if index < START_INDEX:
        continue 
    if index > END_INDEX:
        print(f"\n--- Reached end row {END_INDEX}. Stopping process. ---")
        break 
    

    url = str(row[IMAGE_URL_COLUMN]).strip()

    if not url.startswith("http"):
        continue  

    try:
        filename = f"img_{index+2}.jpg"

        file_path = os.path.join(DOWNLOAD_FOLDER, filename)

        response = requests.get(url, timeout=10)
        response.raise_for_status()

        with open(file_path, "wb") as f:
            f.write(response.content)

        df.at[index, NEW_COLUMN] = filename

        print(f"Downloaded: {filename}")

    except Exception as e:
        print(f"Failed: {url} -> {e}")
        df.at[index, NEW_COLUMN] = "DOWNLOAD_FAILED"

df.to_excel(OUTPUT_EXCEL,sheet_name="A", index=False)
print("\nDone! Excel updated.")