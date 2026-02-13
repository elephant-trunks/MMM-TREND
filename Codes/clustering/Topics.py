import os
import sys
import matplotlib
matplotlib.use("Agg")  # Headless plotting

import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import textwrap
from PIL import Image, ImageFile
from torch.utils.data import Dataset, DataLoader
from transformers import SiglipModel, SiglipProcessor
from tqdm import tqdm
from sklearn.manifold import TSNE

# -----------------------------
# CONFIGURATION
# -----------------------------
INPUT_EXCEL = r"/home/gudahvr_iitp/task2/dataset_clustered_elbow.xlsx"
# If you already have coordinates, uncomment the line below for FAST plotting
# INPUT_EXCEL = "dataset_viz_ready.xlsx"

IMAGE_DIR = r"/home/gudahvr_iitp/task2/downloaded_images"
MODEL_PATH = "./siglip_model_offline"
OUTPUT_PLOT = "cluster_visualization_named_inside_legend.png"
OUTPUT_DATA_WITH_COORDS = "dataset_viz_ready.xlsx"

IMAGE_COLUMN_NAME = "Best Match Image (CLIP)"

# YOUR TOPIC MAPPING
TOPIC_MAP = {
    0: "General Social & Miscellaneous Claims",
    1: "Urban Governance & Civic Issues",
    2: "Political Statements & Public Discourse",
    3: "Science, Health & Environmental Misinformation",
    4: "Digital Media & Online Narratives",
    5: "Government Schemes & Policy Claims",
    6: "Climate, Weather & Air Pollution"
}

# -----------------------------
# SETUP & DATASET
# -----------------------------
ImageFile.LOAD_TRUNCATED_IMAGES = True
torch.set_float32_matmul_precision('high')
device = "cuda" if torch.cuda.is_available() else "cpu"

class MultimodalDataset(Dataset):
    def __init__(self, dataframe, img_dir):
        self.df = dataframe
        self.img_dir = img_dir
        self.texts = (
            self.df["Translated_title"].fillna("").astype(str) + " " +
            self.df["Summary"].fillna("").astype(str)
        ).tolist()

        if IMAGE_COLUMN_NAME in self.df.columns:
            self.img_ids = self.df[IMAGE_COLUMN_NAME].tolist()
        else:
            self.img_ids = [""] * len(self.df)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        text = self.texts[idx]
        img_id = self.img_ids[idx]
        image = None
        if isinstance(img_id, str) and img_id.strip():
            img_path = os.path.join(self.img_dir, img_id)
            if os.path.exists(img_path):
                try:
                    image = Image.open(img_path).convert("RGB")
                except:
                    pass
        if image is None:
            image = Image.new("RGB", (384, 384), color=(0, 0, 0))
            valid_image = False
        else:
            valid_image = True
        return text, image, valid_image

def collate_fn(batch):
    texts, images, valid_mask = zip(*batch)
    inputs = processor(text=list(texts), images=list(images), padding="longest", truncation=True, max_length=64, return_tensors="pt")
    return inputs, torch.tensor(valid_mask)

# -----------------------------
# MAIN PIPELINE
# -----------------------------
def main():
    print(f"Reading Data: {INPUT_EXCEL}")
    df = pd.read_excel(INPUT_EXCEL)

    if 'cluster_label' in df.columns:
        df['cluster_label'] = df['cluster_label'].fillna(-1).astype(int)

    # --- CHECK IF COORDINATES ALREADY EXIST ---
    if 'x' in df.columns and 'y' in df.columns:
        print("Found existing t-SNE coordinates! Skipping model generation...")
    else:
        print("No coordinates found. Loading SigLIP Model to generate embeddings...")
        global processor
        model = SiglipModel.from_pretrained(MODEL_PATH).to(device)
        processor = SiglipProcessor.from_pretrained(MODEL_PATH)
        model.eval()

        dataset = MultimodalDataset(df, IMAGE_DIR)
        dataloader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=2, collate_fn=collate_fn)

        all_embeddings = []
        with torch.no_grad():
            for inputs, valid_mask in tqdm(dataloader):
                inputs = {k: v.to(device) for k, v in inputs.items()}
                valid_mask = valid_mask.to(device)
                with torch.amp.autocast(device_type=("cuda" if device=="cuda" else "cpu"), enabled=(device=="cuda")):
                    outputs = model(**inputs)
                    text_embeds = outputs.text_embeds / outputs.text_embeds.norm(dim=-1, keepdim=True)
                    image_embeds = outputs.image_embeds / outputs.image_embeds.norm(dim=-1, keepdim=True)
                    mask_expanded = valid_mask.unsqueeze(1).float()
                    combined_embeds = (text_embeds + (image_embeds * mask_expanded)) / (1 + mask_expanded)
                all_embeddings.append(combined_embeds.float().cpu().numpy())

        print("Running t-SNE...")
        tsne = TSNE(n_components=2, perplexity=50, early_exaggeration=20, random_state=42)
        X_2d = tsne.fit_transform(np.vstack(all_embeddings))

        df['x'] = X_2d[:, 0]
        df['y'] = X_2d[:, 1]

        print(f"Saving coordinates to {OUTPUT_DATA_WITH_COORDS}...")
        df.to_excel(OUTPUT_DATA_WITH_COORDS, index=False)

    # --- PLOTTING ---
    print(f"Plotting Named Clusters...")
    plt.figure(figsize=(16, 12)) # Slightly smaller figure size as we save space

    unique_labels = sorted(df['cluster_label'].unique())
    cmap = plt.get_cmap('tab10') if len(unique_labels) <= 10 else plt.get_cmap('tab20')

    # Plot points
    for i, label in enumerate(unique_labels):
        if label == -1: continue
        cluster_data = df[df['cluster_label'] == label]
        plt.scatter(cluster_data['x'], cluster_data['y'], color=cmap(i), s=15, alpha=0.6)

    # Add Legend & Centroids
    legend_handles = []
    for i, label in enumerate(unique_labels):
        if label == -1: continue

        topic_name = TOPIC_MAP.get(label, f"Cluster {label}")

        # Legend Entry
        legend_handles.append(mpatches.Patch(color=cmap(i), label=f"{label}: {topic_name}"))

        # Map Annotation (Wrapped Text)
        mean_x = df[df['cluster_label'] == label]['x'].mean()
        mean_y = df[df['cluster_label'] == label]['y'].mean()
        wrapped_label = "\n".join(textwrap.wrap(topic_name, width=20))

        plt.text(mean_x, mean_y, wrapped_label, fontsize=9, weight='bold',
                 horizontalalignment='center', verticalalignment='center',
                 bbox=dict(facecolor='white', alpha=0.8, edgecolor='#333', boxstyle='round,pad=0.3'))

    # MODIFIED LEGEND POSITION: Placed inside the plot
    # loc="best" automatically finds the least crowded location
    plt.legend(handles=legend_handles, loc="best", fontsize=10, title="Identified Topics")

    plt.title('t-SNE Visualization with Topic Names', fontsize=18)
    plt.tight_layout()
    plt.savefig(OUTPUT_PLOT, dpi=300)
    print(f"Done! Plot saved to {OUTPUT_PLOT}")

if __name__ == "__main__":
    main()