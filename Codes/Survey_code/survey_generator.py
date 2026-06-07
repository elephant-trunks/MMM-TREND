import pandas as pd

input_file = r"C:\Users\DELL\Downloads\Task\servey\Copy of dataset(imgs+clip+blip)(1).xlsx"
output_file = r"C:\Users\DELL\Downloads\Task\servey\output.xlsx"

N = 100   

df_in = pd.read_excel(input_file)
df_out = pd.read_excel(output_file)

required_columns = df_out.columns.intersection(df_in.columns)

df_filtered = df_in[required_columns]

sampled_df = df_filtered.sample(n=min(N, len(df_filtered)), random_state=11)

sampled_df.to_excel(output_file, index=False)

print(f"{len(sampled_df)} random rows inserted into output Excel with selected columns only")
