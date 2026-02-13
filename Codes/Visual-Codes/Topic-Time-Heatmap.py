import pandas as pd
import matplotlib.pyplot as plt

# -------------------------------------------------
# Load dataset
# -------------------------------------------------
file_path = "complete_dataset_v1.1.xlsx"   # update if needed
df = pd.read_excel(file_path)

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
# Detect time column
# -------------------------------------------------
time_candidates = [c for c in df.columns if "date" in c.lower() or "time" in c.lower()]
if len(time_candidates) == 0:
    raise ValueError("No temporal column found.")
time_col = time_candidates[0]

# -------------------------------------------------
# Topic name → Topic ID mapping
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
INVERSE_TOPIC_MAP = {v: k for k, v in TOPIC_MAP.items()}

df["Topic_ID"] = df[topic_col].map(INVERSE_TOPIC_MAP)

if df["Topic_ID"].isnull().any():
    unmapped = df[df["Topic_ID"].isnull()][topic_col].unique()
    raise ValueError(f"Unmapped topic labels found: {unmapped}")

# -------------------------------------------------
# Robust datetime parsing
# -------------------------------------------------
df[time_col] = pd.to_datetime(
    df[time_col],
    errors="coerce",
    dayfirst=True
)
df = df.dropna(subset=[time_col])

# -------------------------------------------------
# Year-wise aggregation
# -------------------------------------------------
df["Year"] = df[time_col].dt.year

topic_year = (
    df.groupby(["Topic_ID", "Year"])
      .size()
      .reset_index(name="Count")
)

# -------------------------------------------------
# Create GLOBAL year range (important fix)
# -------------------------------------------------
all_years = list(range(df["Year"].min(), df["Year"].max() + 1))

# -------------------------------------------------
# Small multiples with consistent year labels
# -------------------------------------------------
fig, axes = plt.subplots(2, 4, figsize=(12, 6), sharex=True, sharey=True)
axes = axes.flatten()

for topic_id in range(7):
    ax = axes[topic_id]

    data = topic_year[topic_year["Topic_ID"] == topic_id]
    data = (
        data.set_index("Year")
            .reindex(all_years, fill_value=0)
            .reset_index()
            .rename(columns={"index": "Year"})
    )

    ax.plot(data["Year"], data["Count"], marker="o")
    ax.set_title(f"Topic {topic_id}")
    ax.set_xlabel("Year")
    ax.set_xticks(all_years)
    ax.tick_params(axis="x", rotation=45)

# Hide unused subplot
axes[-1].axis("off")

fig.suptitle("Year-wise Topic Distribution in MMM-TREND Dataset", fontsize=12)
fig.text(0.04, 0.5, "Number of Instances", va="center", rotation="vertical")

plt.tight_layout(rect=[0.05, 0.05, 1, 0.95])
plt.show()
