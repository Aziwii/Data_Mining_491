import pandas as pd
import numpy as np

SELECTED_2022_DICT = {
    # Target
    "GEN_05": "perceived_mental_health",
    # Demographics
    "GEOGPRV": "province",
    "DHH_SEX": "sex",
    "DHHGAGE": "age_group",
    "INCDGHH": "household_income_group",
    "MAC_05": "main_activity",
    # Stress, Social & Physical Health
    "GEN_01": "perceived_general_health",
    "GEN_10": "perceived_life_stress",
    "GEN_20": "community_belonging",
    "HWTDGISW": "bmi_self_reported", #clear 6, 9
    # Lifestyle (Sleep & Screen Time)
    "SLPG005": "sleep_hours_per_night",
    "SBE_010": "screen_time_non_workday", # might have to do just this one
    "PAA_45A": "physical_activity_hours_7d", # clear 996, 999
    "PAADVAC2": "physical_activity_level", # could also use PAADVACV (worded indicator)
    # Substance Use
    "SMKDVSTY": "smoking_type",
    "ALC_15": "alcohol_frequency",
}

TARGET_COL = SELECTED_2022_DICT["GEN_05"]
RAW_COLS = list(SELECTED_2022_DICT.keys())

def load_raw_data(filepath): 
    """Loads the raw CSV file."""
    print("Loading raw dataset...")
    df = pd.read_csv(filepath, low_memory=False)
    print(f"Raw shape: {df.shape[0]:,} rows x {df.shape[1]:,} columns\n")
    return df

def check_module_flags(raw_df):
    """Check the subsamples based on the overlap of inclusion flags
        For ex. get the # of people who answered yes to each inclusion flag for 
        the lifestyle categories
    """
    print("=" * 50)
    print("1. INCLUSION FLAG CHECK (# of Overlap)")
    print("=" * 50)
    flag_cols = ["DOSLP", "DOSBE", "DOPAA"] # check all of the inclusions flags
    available_flags = [c for c in flag_cols if c in raw_df.columns] #check if flags exits

    for flag in available_flags:
        included_count = (raw_df[flag] == 1).sum() #get the sum of the inclusions for the flags and %
        print(f"{flag} : {included_count:6,} respondents ({(included_count/len(raw_df))*100:.1f})%")  

    if available_flags:
        lifestyle_flags = [
            c for c in flag_cols if c in raw_df.columns
        ]
        count_lifestyle = (raw_df[lifestyle_flags] == 1 ).all(axis=1).sum() # get all 1 answers then sum all the 
        print(f"\n---> people that answered {lifestyle_flags}: {count_lifestyle:,} respondents")
    print("=" * 50 + "\n")

def clean_and_prepare(raw_df, rename_dict):
    """Subsets, handles domain-specific valid skips, recodes missing values, and renames."""
    # filter only the available columns in the rename dict
    available_cols = [c for c in rename_dict.keys() if c in raw_df.columns] # filtering for cols that exist in df
    df = raw_df[available_cols].copy() # make a copy of cleaner df - removed other rows

    # HANDLE SPECIFIC RESPONSES
    #=========== TARGETS & DEMOS ======================
    # MENTAL HEALTH
    if "GEN_05" in df.columns:
        df["GEN_05"] = df["GEN_05"].replace({
            5:4, 9:np.nan,
            }) #combine the pool and fair mental health values to (4)

    # WORKING OR NOT
    if "MAC_05" in df.columns:
        df["MAC_05"] = df["MAC_05"].replace({1:1, 2:0, 6:0, 9:np.nan}) #workiing=1 not working=0
  
    #INCOME
    if "INCDGHH" in df.columns:
        df["INCDGHH"] = df["INCDGHH"].replace([9], np.nan) 

    #=========== PHYSCIAL & SLEEP & WEIGHT ======================
    # PHYS ACT LEVEL
    if "PAADVAC2" in df.columns:
        df["PAADVAC2"] = df["PAADVAC2"].replace([6,9], np.nan)

    # PHYS ACT HOURS
    if "PAA_45A" in df.columns:
        df["PAA_45A"] = df["PAA_45A"].replace([996, 999], np.nan)

    # BMI
    if "HWTDGISW" in df.columns:
        df["HWTDGISW"] = df["HWTDGISW"].replace({
            6:np.nan, 9:np.nan,
            }) #1=underweight/normal, 2=overweight/obeseClass1,2,3

    # SLEEP
    if "SLPG005" in df.columns:
        df["SLPG005"] = df["SLPG005"].replace([96, 99], np.nan)

    # SCREEN TIME NONWORKDAY
    if "SBE_010" in df.columns:
        df["SBE_010"] = df["SBE_010"].replace([6, 9], np.nan)

            
    #HEALTH COLS
    health_cols = ["GEN_01", "GEN_10", "GEN_20"]
    health_codes_to_clean = [7, 8, 9]
    #clean the lifestyle cols
    health_targets = [col for col in health_cols if col in df.columns]
    df[health_targets] = df[health_targets].replace(health_codes_to_clean, np.nan)

    #=========== ALC & SMK ======================
    # ALC
    if "ALC_15" in df.columns:
        df["ALC_15"] = df["ALC_15"].replace({
            96:0, 99:np.nan, #0 = never
            }) #1=<once a mo, 7=everyday
    # SMK
    if "SMKDVSTY" in df.columns: #1=daily, 6=never
        df["SMKDVSTY"] = df["SMKDVSTY"].replace([96,99], np.nan)

    # rename to english
    df.rename(columns=rename_dict, inplace=True) #applies to existing df

    return df

def generate_report(df_clean, target_name):
    """Prints a breakdown of complete cases, column completeness, and target balance."""
    print("=" * 50)
    print("2. CLEANED DATASET REPORT")
    print("=" * 50)

    valid_counts = df_clean.notna().sum() #count all non nulls
    pct_complete = (valid_counts/len(df_clean)) * 100 # % of valids

    report_df = pd.DataFrame(
        {
            "Feature": df_clean.columns,
            "Valid_Count": valid_counts,
            "Percent_Complete": pct_complete
        }
    )
    print(report_df.to_string(index=False)) # hide row numbers

    complete_cases = df_clean.dropna()
    print("-" * 50)
    print(f"Total Rows: {len(df_clean):,}")
    print(f"Non Null cases: {len(complete_cases):,} ({(len(complete_cases)/(len(df_clean))*100):.1f}%)" )

    if target_name in complete_cases:
        print("\n totals of answers for mental health (so we know what to expect):")
        target_dist = complete_cases[target_name].value_counts(sort=False)
        for val, count in target_dist.items():
            print(
                f"  Score {val}: {count:6,} ({count/len(complete_cases)*100:5.1f}%)"
            )
    print("=" * 50 + "\n")

    return complete_cases

# 3. main pipeline
def main():
    # 1. load raw files
    raw_df = load_raw_data("../raw_data/cchs_2022_raw.csv")
    # raw_df = load_raw_data("../raw_data/cchs_2022_raw.csv")

    # 2. check flags
    check_module_flags(raw_df)

    # 3. clean and recode
    df_clean = clean_and_prepare(raw_df, SELECTED_2022_DICT)

    # 4. create report and get rid of null values
    df_usable = generate_report(df_clean, TARGET_COL)

    #export the clean set to clean_data dir
    output_path = "../clean_data/cchs_2022_cleaned.csv"
    df_usable.to_csv(output_path, index=False)
    print(f"Successfully saved {len(df_usable):,} rows to {output_path}")


if __name__ == "__main__":
    main()