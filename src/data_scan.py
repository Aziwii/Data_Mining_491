import pandas as pd
import numpy as np

file_path = '../raw_data/pumf_cchs.csv'

print("Loading dataset...")
df = pd.read_csv(file_path, low_memory=False)

# Explicit Row and Column counts
print(f"\n========================================")
print(f"       DATASET SHAPE & METRICS         ")
print(f"========================================")
print(f"Total Rows:    {df.shape[0]:,}")
print(f"Total Columns: {df.shape[1]:,}")
print(f"========================================\n")

# Function to identify and replace StatCan sentinel missing values (strings of 9s)
def clean_statcan_nulls(val):
    if pd.isna(val):
        return np.nan
    s = str(val).strip()
    if s.isdigit() and len(set(s)) == 1 and '9' in s:
        return np.nan
    return val

print("Cleaning StatCan missing codes (sentinel 9s)...")
for col in df.select_dtypes(include=['int64', 'float64']).columns:
    df[col] = df[col].apply(clean_statcan_nulls)

# Build true missingness summary
true_profile = pd.DataFrame({
    'Column_Name': df.columns,
    'Data_Type': df.dtypes.astype(str),
    'True_Null_Count': df.isnull().sum().values,
    'True_Null_Percentage': (df.isnull().mean() * 100).round(2).values
})

true_profile_sorted = true_profile.sort_values(by='True_Null_Count', ascending=False)

print("\n--- TRUE MISSINGNESS PROFILE (Highest Nulls First) ---")
print(true_profile_sorted.to_string(index=False))