import pandas as pd

# load clean set
df18 = pd.read_csv("../clean_data/cchs_2018_cleaned.csv")
if df18.empty:
    print("Warning: The 2018 dataset is empty. Please check the file path and contents.")
df22 = pd.read_csv("../clean_data/cchs_2022_cleaned.csv")
if df22.empty:
    print("Warning: The 2022 dataset is empty. Please check the file path and contents.")  

df18['year'] = 2018
df22['year'] = 2022

# reset the indices 
df1 = df18.reset_index(drop=True)
df2 = df22.reset_index(drop=True)

combined_summary = pd.concat([df1, df2], axis=0, ignore_index=True)

print("\nCombined Summary Stats:")
print(len(combined_summary), combined_summary.columns)

output_path = "../clean_data/cchs_2018_2022_combined.csv"
combined_summary.to_csv(output_path, index=False)
print(f"Successfully saved {len(combined_summary):,} rows to {output_path}")