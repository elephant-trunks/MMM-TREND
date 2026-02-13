import pandas as pd

# File paths
input_file = r"C:\Users\DELL\Downloads\Task\servey\Copy of dataset(imgs+clip+blip)(1).xlsx"
output_file = r"C:\Users\DELL\Downloads\Task\servey\output.xlsx"

# Number of random rows
N = 100   # change if needed

# Read input and output Excel
df_in = pd.read_excel(input_file)
df_out = pd.read_excel(output_file)

# Get only columns that exist in output file
required_columns = df_out.columns.intersection(df_in.columns)

# Select only required columns from input
df_filtered = df_in[required_columns]

# Randomly sample rows
sampled_df = df_filtered.sample(n=min(N, len(df_filtered)), random_state=11)

# Write directly to output Excel (overwrite)
sampled_df.to_excel(output_file, index=False)

print(f"{len(sampled_df)} random rows inserted into output Excel with selected columns only")
