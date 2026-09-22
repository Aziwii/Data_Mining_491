import pandas as pd
import numpy as np


def load_dataset(file_path):
    """
    Load the dataset from a CSV file.

    Parameters:
    file_path (str): The path to the CSV file.

    Returns:
    pd.DataFrame: Loaded DataFrame.
    """

    print("Loading dataset...")
    df = pd.read_csv(file_path, low_memory=False)

    # Explicit Row and Column counts
    print(f"\n========================================")
    print(f"       DATASET SHAPE & METRICS         ")
    print(f"========================================")
    print(f"Total Rows:    {df.shape[0]:,}")
    print(f"Total Columns: {df.shape[1]:,}")
    print(f"========================================\n")

    return df


def remove_columns(df, columns_to_keep):
    """
    Remove columns from the DataFrame that are not in the list of columns to keep.

    Parameters:
    df (pd.DataFrame): The input DataFrame.
    columns_to_keep (list): List of column names to keep.

    Returns:
    pd.DataFrame: DataFrame with only the specified columns.
    """
    df = df[columns_to_keep]

    # Explicit Row and Column counts
    print(f"\n========================================")
    print(f"Dataset after removing unnecessary columns")
    print(f"========================================")
    print(f"Total Rows:    {df.shape[0]:,}")
    print(f"Total Columns: {df.shape[1]:,}")
    print(f"========================================\n")
    return df

def main():
    # Load the dataset
    df = load_dataset('../raw_data/pumf_cchs.csv')

    # remove the columns that are not needed for the analysis
    columns_to_keep = [
    "DHHGAGE",
    "DHH_SEX",
    "MAC_05",
    "GEN_01",
    "GEN_05",
    "GEN_10",
    "GEN_15",
    "GEN_20",
    "LSM_01",
    "LSMDVSWL",
    "HWTDGISW",
    "HWTDGBCC",
    "WTP_50",
    "COV2_005",
    "CCC_80",
    "CCC_05",
    "CCC_90",
    "CCC_135",
    "CC1_140",
    "CC1_145",
    "CC1_155",
    "CSS_05",
    "CSS_15",
    "CSSG20",
    "ECV_05",
    "ECVG15",
    "ALC_05",
    "ALC_10",
    "ALC_15",
    "ALCDVTTM",
    "CAN_05C",
    "SDCDGIMM",
    "SDCDVABT",
    "SDCDVFLA",
    "INCDGHH",
]
    
    df = remove_columns(df, columns_to_keep)

if __name__ == "__main__":
    main()