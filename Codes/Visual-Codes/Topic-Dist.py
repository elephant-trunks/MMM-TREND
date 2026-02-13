import pandas as pd
import matplotlib.pyplot as plt

# -------------------------------------------------
# Load MMM-TREND dataset
# -------------------------------------------------
file_path = "complete_dataset_v1.1.xlsx"   # update if needed
df = pd.read_excel(file_path)

# -------------------------------------------------
# Topic name → Topic ID mapping (paper-aligned)
# -------------------------------------------------
TOPIC_MAP = {
    0: "General social narratives and mixed-content claims",
    1: "Urban governance, civic issues, and local administrative claims",
    2: "Political statements and public discourse-related-narratives",
    3: "Science, health, and environmental misinformation",
    4: "Digital media, online narratives, and platformdriven misinformation",
    5: "Government schemes, infrastructure projects, and policy-related claims",
    6: "Climate, weather, air pollution, and environmental condition reports"
}

# Inverse mapping: topic name → topic id
INVERSE_TOPIC_MAP = {v: k for k, v in TOPIC_MAP.items()}

# -------------------------------------------------
# Detect topic column
# -------------------------------------------------
if "Topics" in df.columns:
    topic_col = "Topics"
elif "cluster_label" in df.columns:
    topic_col = "cluster_label"
else:
    raise ValueError("Topic column not found in dataset.")

# -------------------------------------------------
# Map topic names to numeric Topic IDs
# -------------------------------------------------
df["Topic_ID"] = df[topic_col].map(INVERSE_TOPIC_MAP)

# Safety check: unmapped topics
if df["Topic_ID"].isnull().any():
    unmapped = df[df["Topic_ID"].isnull()][topic_col].unique()
    raise ValueError(f"Unmapped topic labels found: {unmapped}")

# -------------------------------------------------
# Topic-wise statistics
# -------------------------------------------------
topic_counts = df["Topic_ID"].value_counts().sort_index()
total_instances = len(df)

topic_table = pd.DataFrame({
    "Topic ID": topic_counts.index,
    "Count": topic_counts.values,
    "Percentage (%)": (topic_counts.values / total_instances * 100).round(2)
})

# -------------------------------------------------
# Print and save table
# -------------------------------------------------
print("\nTopic-wise Distribution Table:\n")
print(topic_table)

topic_table.to_csv("topic_wise_distribution.csv", index=False)

# -------------------------------------------------
# Topic-wise bar chart (NUMERIC IDs ONLY)
# -------------------------------------------------
plt.figure(figsize=(6, 4))
plt.bar(topic_table["Topic ID"].astype(str), topic_table["Count"])
plt.xlabel("Topic ID")
plt.ylabel("Number of Instances")
plt.title("Topic-wise Distribution in MMM-TREND Dataset")
plt.tight_layout()
plt.show()
