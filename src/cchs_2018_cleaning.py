import numpy as np
import pandas as pd

# 1. Dictionary mapping
SELECTED_CCHS_DICT = {
    # Target - done
    "GEN_015": "perceived_mental_health", # clear 7, 8, 9
    # Demographics - done
    "GEO_PRV": "province", # all good
    "DHH_SEX": "sex", # all good
    "DHHGAGE": "age_group", # 1-16 all good
    "INCDGHH": "household_income_group", #clear 9
    "MACG005": "main_activity", # clear 96,97, 98, 99
    # Stress, Social & Physical Health - done
    "GEN_005": "perceived_general_health", #clear 7, 8
    "GEN_020": "perceived_life_stress", #clear 7, 8
    "GEN_030": "community_belonging", #clear 7, 8, 9
    "HWTDGISW": "bmi_self_reported", #clear 6, 9
    # Lifestyle & Behaviors - done
    "SLPG005": "sleep_hours_per_night", #clear 96, 99
    "SBE_010": "screen_time_non_workday", #clear 7, 8
    "PAA_045": "physical_activity_hours_7d", # clear 996, 997, 998, 999
    "PAADVAC2": "physical_activity_level", # clear 6, 9
    # Substance Use - done
    "SMKDVSTY": "smoking_type", #clear 7, 8
    "ALC_015": "alcohol_frequency", # change 96 to 0 (never), and clear 97, 98, 99
}

RAW_COLS = list(SELECTED_CCHS_DICT.keys())
TARGET_COL = SELECTED_CCHS_DICT["GEN_015"]

def load_raw_data(file_path):
    """Loads the raw CSV file."""
    print("Loading raw dataset...")
    df = pd.read_csv(file_path, low_memory=False)
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
    flag_cols = ["DOSLP", "DOSBE", "DOPAA", "DODRG", "DOSWL", "DOSPS"] # check all of the inclusions flags - SPS, SWL, and DRG are extra for now
    available_flags = [c for c in flag_cols if c in raw_df.columns] # check if flag_cols exist

    for flag in available_flags: 
        #check the count of yes in each flag col
        included_count = (raw_df[flag] == 1).sum() # loop through every flag and see count if its a yes
        print(
            f"{flag:4s} : {included_count:6,} respondents ({included_count/len(raw_df)*100:.1f}%)"
        )

    if available_flags:
        # checking the main sub categories
        lifestyle_flags = [
            f for f in ["DOSLP", "DOSBE", "DOPAA"] if f in raw_df.columns # checking the cols existence
        ]
        all_lifestyle = (raw_df[lifestyle_flags] == 1).all(axis=1).sum() # row wise eval checking if all answered 1 to the flags
        print(f"\n--> # people that answered {lifestyle_flags}: {all_lifestyle:,} respondents")
    print("=" * 50 + "\n")


def clean_and_prepare(raw_df, rename_dict):
    """Subsets, handles domain-specific valid skips, recodes missing values, and renames."""
    # filter only the available columns in the rename dict
    available_cols = [c for c in rename_dict.keys() if c in raw_df.columns] # filtering for cols that exist in df
    df = raw_df[available_cols].copy() # make a copy of cleaner df - removed other rows

    # HANDLE SPECIFIC RESPONSES
    #=========== TARGETS & DEMOS ======================
    # MENTAL HEALTH
    if "GEN_015" in df.columns:
        df["GEN_015"] = df["GEN_015"].replace({
            5:4, 7:np.nan, 8:np.nan, 9:np.nan,
            }) #combine the pool and fair mental health values to (4)

    # AGE
    if "DHHGAGE" in df.columns:
        df["DHHGAGE"] = df["DHHGAGE"].replace({
            1:1, 2:1, #12-17
            3:2, 4:2, 5:2, 6:2, #18-34
            7:3, 8:3, 9:3, #35-49
            10:4, 11:4, 12:4, #50-64
            13:5, 14:5, 15:5, 16:5, #65+
            }) #combine the pool and fair mental health values to (4)
        
    #INCOME
    if "INCDGHH" in df.columns:
        df["INCDGHH"] = df["INCDGHH"].replace([9], np.nan) 

    # WORK OR NOT
    if "MACG005" in df.columns:
        df["MACG005"] = df["MACG005"].replace({
            1:1, 2:0, 3:0, 4:0, 5:0, 6:0, 97:np.nan, 98:np.nan, 99:np.nan,
            }) #working=1 notworking=0

    #=========== STRESS, SOCIAL, HEALTH ======================
    #HEALTH COLS
    health_cols = ["GEN_005", "GEN_020", "GEN_030", "SBE_010"]
    health_codes_to_clean = [7, 8, 9]
    #clean the lifestyle cols
    health_targets = [col for col in health_cols if col in df.columns]
    df[health_targets] = df[health_targets].replace(health_codes_to_clean, np.nan)

    #=========== PHYSCIAL & SLEEP & WEIGHT ======================
    # PHYS ACT LEVEL
    if "PAADVAC2" in df.columns:
        df["PAADVAC2"] = df["PAADVAC2"].replace([6,9], np.nan)

    # PHYS ACT HOURS
    if "PAA_045" in df.columns:
        df["PAA_045"] = df["PAA_045"].replace([996,997,998, 999], np.nan)

    # SLEEP
    if "SLPG005" in df.columns:
        df["SLPG005"] = df["SLPG005"].replace([96, 99], np.nan)

    # BMI
    if "HWTDGISW" in df.columns:
        df["HWTDGISW"] = df["HWTDGISW"].replace({
            1:1, 2:1, 3:2, 4:2, 6:np.nan, 9:np.nan,
            }) #1=underweight/normal, 2=overweight/obeseClass1,2,3

    #=========== ALC & SMK ======================
    # ALC
    if "ALC_015" in df.columns:
        df["ALC_015"] = df["ALC_015"].replace({
            96:0, 97:np.nan, 98:np.nan, 99:np.nan, #0=never
            }) #1=<once a mo, 7=everyday
        
    # SMK 
    if "SMKDVSTY" in df.columns: #1=daily, 6=never
        df["SMKDVSTY"] = df["SMKDVSTY"].replace([99], np.nan)

    # rename to english
    df.rename(columns=rename_dict, inplace=True) #applies to existing df

    return df


def generate_report(df_clean, target_name):
    """Prints a breakdown of complete cases, column completeness, and target balance."""
    print("=" * 50)
    print("2. CLEANED DATASET REPORT")
    print("=" * 50)

    # Column-by-column valid count
    valid_counts = df_clean.notna().sum() # count how many valid non nulls exist for each col
    pct_complete = (valid_counts / len(df_clean)) * 100 # % of valids

    #print out the report in a df table
    report_df = pd.DataFrame(
        {
            "Feature": df_clean.columns,
            "Valid_Count": valid_counts.values,
            "Percent_Complete": pct_complete.round(1).values,
        }
    )
    print(report_df.to_string(index=False)) #hide the row nums

    # drop all nan and check counts
    complete_cases = df_clean.dropna()
    print("-" * 50)
    print(f"Total Rows: {len(df_clean):,}")
    print(f"Non Null cases: {len(complete_cases):,} ({len(complete_cases)/len(df_clean)*100:.1f}%)")

    # totals for mental health answers 
    if target_name in complete_cases.columns:
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
    raw_df = load_raw_data("../raw_data/cchs_2018_raw.csv")
    # raw_df = load_raw_data("../raw_data/cchs_2022_raw.csv")

    # 2. check flags
    check_module_flags(raw_df)

    # 3. clean and recode
    df_clean = clean_and_prepare(raw_df, SELECTED_CCHS_DICT)

    # 4. create report and get rid of null values
    df_usable = generate_report(df_clean, TARGET_COL)

    #export the clean set to clean_data dir
    output_path = "../clean_data/cchs_2018_cleaned.csv"
    df_usable.to_csv(output_path, index=False)
    print(f"Successfully saved {len(df_usable):,} rows to {output_path}")


if __name__ == "__main__":
    main()