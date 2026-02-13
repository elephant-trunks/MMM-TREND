import os
import sys
import matplotlib

# -----------------------------
# HPC / HEADLESS CONFIG
# -----------------------------
# These must be set before importing transformers or torch
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
matplotlib.use("Agg")  # Non-GUI backend for saving plots on HPC

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
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE

# -----------------------------
# CONFIGURATION
# -----------------------------
# PATHS (Update these if filenames change)
INPUT_RAW_EXCEL = r"/home/gudahvr_iitp/task2/datasetnew.xlsx"
IMAGE_DIR = r"/home/gudahvr_iitp/task2/downloaded_images"
MODEL_PATH = "./siglip_model_offline"

# OUTPUT FILES
OUTPUT_FINAL_EXCEL = "dataset_final_complete.xlsx"
PLOT_ELBOW = "1_elbow_method.png"
PLOT_CLUSTERS = "2_cluster_visualization.png"

# COLUMN SETTINGS
IMAGE_COLUMN_NAME = "Best Match Image (CLIP)"

# CLUSTERING PARAMETERS
MIN_CLUSTERS = 2
MAX_CLUSTERS = 20
BATCH_SIZE = 64

# TOPIC MAPPING (Valid ONLY if Optimal K = 7)
TOPIC_MAP_7 = {
    0: "General Social & Miscellaneous Claims",
    1: "Urban Governance & Civic Issues",
    2: "Political Statements & Public Discourse",
    3: "Science, Health & Environmental Misinformation",
    4: "Digital Media & Online Narratives",
    5: "Government Schemes & Policy Claims",
    6: "Climate, Weather & Air Pollution"
}

# -----------------------------
# DEVICE & DATASET SETUP
# -----------------------------
ImageFile.LOAD_TRUNCATED_IMAGES = True
torch.set_float32_matmul_precision('high')
device = "cuda" if torch.cuda.is_available() else "cpu"

# Initialize processor as None (will be loaded in main)
processor = None

class MultimodalDataset(Dataset):
    def __init__(self, dataframe, img_dir):
        self.df = dataframe
        self.img_dir = img_dir
        # Combine title and summary for richer text context
        self.texts = (
            self.df["Translated_title"].fillna("").astype(str) + " " +
            self.df["Summary"].fillna("").astype(str)
        ).tolist()

        if IMAGE_COLUMN_NAME in self.df.columns:
            self.img_ids = self.df[IMAGE_COLUMN_NAME].tolist()
        else:
            print(f"Warning: Column '{IMAGE_COLUMN_NAME}' not found. Using dummy images.")
            self.img_ids = [""] * len(self.df)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        text = self.texts[idx]
        img_id = self.img_ids[idx]
        image = None
        
        # Try to load image
        if isinstance(img_id, str) and img_id.strip():
            img_path = os.path.join(self.img_dir, img_id)
            if os.path.exists(img_path):
                try:
                    image = Image.open(img_path).convert("RGB")
                except:
                    pass # Keep image as None if corrupt

        # Handle missing images (Black square placeholder)
        if image is None:
            image = Image.new("RGB", (384, 384), color=(0, 0, 0))
            valid_image = False
        else:
            valid_image = True
            
        return text, image, valid_image

def collate_fn(batch):
    # Uses the global 'processor' loaded in main()
    texts, images, valid_mask = zip(*batch)
    inputs = processor(
        text=list(texts), 
        images=list(images), 
        padding="longest", 
        truncation=True, 
        max_length=64, 
        return_tensors="pt"
    )
    return inputs, torch.tensor(valid_mask)

# -----------------------------
# MAIN PIPELINE
# -----------------------------
def main():
    global processor
    print(f"Starting Unified Pipeline on Device: {device}")
    
    # --- 1. LOAD MODEL & DATA ---
    print(f"Loading Model from: {MODEL_PATH}")
    if not os.path.exists(MODEL_PATH):
        print("Error: Model path does not exist.")
        sys.exit(1)
        
    model = SiglipModel.from_pretrained(MODEL_PATH).to(device)
    processor = SiglipProcessor.from_pretrained(MODEL_PATH)
    model.eval()

    print(f"Reading Excel: {INPUT_RAW_EXCEL}")
    if not os.path.exists(INPUT_RAW_EXCEL):
        print("Error: Input Excel file not found.")
        sys.exit(1)
        
    df = pd.read_excel(INPUT_RAW_EXCEL)
    dataset = MultimodalDataset(df, IMAGE_DIR)
    
    # Determine safe number of workers
    num_workers = min(2, os.cpu_count() // 2)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, 
                            num_workers=num_workers, collate_fn=collate_fn)

    # --- 2. GENERATE EMBEDDINGS ---
    print(f"Generating Embeddings for {len(df)} records...")
    all_embeddings = []

    with torch.no_grad():
        for inputs, valid_mask in tqdm(dataloader, desc="Embedding"):
            inputs = {k: v.to(device) for k, v in inputs.items()}
            valid_mask = valid_mask.to(device)

            with torch.amp.autocast(device_type=("cuda" if device=="cuda" else "cpu"), enabled=(device=="cuda")):
                outputs = model(**inputs)
                
                # Normalize features
                text_embeds = outputs.text_embeds / outputs.text_embeds.norm(dim=-1, keepdim=True)
                image_embeds = outputs.image_embeds / outputs.image_embeds.norm(dim=-1, keepdim=True)

                # Multimodal Fusion: Average Text + Image (if image exists)
                mask_expanded = valid_mask.unsqueeze(1).float()
                combined_embeds = (text_embeds + (image_embeds * mask_expanded)) / (1 + mask_expanded)

            all_embeddings.append(combined_embeds.float().cpu().numpy())

    # Stack into a single matrix (N_samples x Hidden_dim)
    X = np.vstack(all_embeddings)
    
    # FREE GPU MEMORY
    print("Cleaning up GPU memory...")
    del model
    del inputs
    del combined_embeds
    torch.cuda.empty_cache()

    # --- 3. ELBOW METHOD ---
    print(f"\nCalculating Elbow (Inertia) for K={MIN_CLUSTERS} to {MAX_CLUSTERS}...")
    inertias = []
    K_range = range(MIN_CLUSTERS, MAX_CLUSTERS + 1)

    for k in tqdm(K_range, desc="Fitting K-Means"):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X)
        inertias.append(kmeans.inertia_)

    # --- Geometric Elbow Calculation ---
    x_points = np.array(list(K_range))
    y_points = np.array(inertias)
    p1 = np.array([x_points[0], y_points[0]])
    p2 = np.array([x_points[-1], y_points[-1]])

    distances = []
    for i in range(len(x_points)):
        p = np.array([x_points[i], y_points[i]])
        # Perpendicular distance from point to line P1-P2
        dist = np.abs(np.cross(p2-p1, p1-p)) / np.linalg.norm(p2-p1)
        distances.append(dist)

    optimal_k = x_points[np.argmax(distances)]
    print(f"\nOptimal K determined: {optimal_k}")

    # Plot Elbow Curve
    plt.figure(figsize=(10, 6))
    plt.plot(K_range, inertias, 'bo-', linewidth=2)
    plt.axvline(x=optimal_k, color='r', linestyle='--', label=f'Optimal K={optimal_k}')
    plt.title(f'Elbow Method Analysis (Optimal K={optimal_k})')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Inertia')
    plt.legend()
    plt.grid(True)
    plt.savefig(PLOT_ELBOW)
    print(f"Elbow plot saved to: {PLOT_ELBOW}")

    # --- 4. FINAL CLUSTERING ---
    print(f"Assigning Cluster Labels (K={optimal_k})...")
    final_kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    df["cluster_label"] = final_kmeans.fit_predict(X)

    # --- 5. T-SNE VISUALIZATION ---
    print("Running t-SNE (Dimensionality Reduction)...")
    tsne = TSNE(n_components=2, perplexity=50, early_exaggeration=20, random_state=42)
    X_2d = tsne.fit_transform(X)

    df['x'] = X_2d[:, 0]
    df['y'] = X_2d[:, 1]

    # Save Final Excel
    df.to_excel(OUTPUT_FINAL_EXCEL, index=False)
    print(f"Clustered data saved to: {OUTPUT_FINAL_EXCEL}")

    # --- 6. ADVANCED PLOTTING ---
    print("Generating Final Visualization...")
    plt.figure(figsize=(16, 12))
    
    unique_labels = sorted(df['cluster_label'].unique())
    
    # Choose colormap based on number of clusters
    cmap_name = 'tab10' if len(unique_labels) <= 10 else 'tab20'
    cmap = plt.get_cmap(cmap_name)

    legend_handles = []

    # Decide which labels to use
    if optimal_k == 7:
        current_map = TOPIC_MAP_7
        print("Using Pre-defined Topic Names (K=7 match)")
    else:
        current_map = {} # Empty map, falls back to "Cluster X"
        print(f"Using Generic Cluster Names (K={optimal_k} != 7)")

    for i, label in enumerate(unique_labels):
        cluster_data = df[df['cluster_label'] == label]
        
        # 1. Scatter Plot
        plt.scatter(cluster_data['x'], cluster_data['y'], color=cmap(i), s=15, alpha=0.6)

        # 2. Determine Name
        topic_name = current_map.get(label, f"Cluster {label}")

        # 3. Add to Legend
        legend_handles.append(mpatches.Patch(color=cmap(i), label=f"{label}: {topic_name}"))

        # 4. Add Annotation on Map (Centroid)
        mean_x = cluster_data['x'].mean()
        mean_y = cluster_data['y'].mean()
        wrapped_label = "\n".join(textwrap.wrap(topic_name, width=20))
        
        plt.text(mean_x, mean_y, wrapped_label, fontsize=9, weight='bold',
                 horizontalalignment='center', verticalalignment='center',
                 bbox=dict(facecolor='white', alpha=0.8, edgecolor='#333', boxstyle='round,pad=0.3'))

    # Final Plot Settings
    plt.legend(handles=legend_handles, loc="best", fontsize=10, title="Topics")
    plt.title(f'Cluster Visualization (K={optimal_k})', fontsize=18)
    plt.axis('off') # Cleaner look without axis numbers
    plt.tight_layout()
    plt.savefig(PLOT_CLUSTERS, dpi=300)
    print(f"Final Visualization saved to: {PLOT_CLUSTERS}")

if __name__ == "__main__":
    main()