import pandas as pd
import numpy as np

SELECTED_2022_DICT = {
    # Target
    "GEN_05": "perceived_mental_health",
    # Demographics
    "GEO_PRV": "province",
    "DHH_SEX": "sex",
    "DHHGAGE": "age_group",
    "INCDGHH": "household_income_group",
    "MAC_05": "main_activity",
    # Stress, Social & Physical Health
    "GEN_01": "perceived_general_health",
    "GEN_10": "perceived_life_stress",
    "GEN_20": "community_belonging",
    "HWTDGISW": "perceived_weight",
    # Lifestyle (Sleep & Screen Time)
    "SLPG005": "sleep_hours_per_night",
    "SBE_005": "screen_time_workday",
    "SBE_010": "screen_time_non_workday",
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

    # handle specific cleaning for the responses
    #clean income
    if "INCDGHH" in df.columns:
        df["INCDGHH"] = df["INCDGHH"].replace([9], np.nan) 

    # Alcohol: 6 = Did not drink in past 12 months -> recode to 0
    if "ALC_015" in df.columns:
        df["ALC_015"] = df["ALC_015"].replace([96], 0).replace([97, 98, 99], np.nan)

    # Smoking: removing the nulls
    if "SMK_005" in df.columns:
        df["SMK_005"] = df["SMK_005"].replace([7,8], np.nan)

    # SCreen time workday: convert 6 and 96 to 0, not at work or school
    if "SBE_005" in df.columns:
        df["SBE_005"] = df["SBE_005"].replace([6, 96], 0) 

    # physical activity: remove nulls
    if "PAADVAC2" in df.columns:
        df["PAADVAC2"] = df["PAADVAC2"].replace([6,9], np.nan)

    #Clean the rest of the cols
    health_cols = ["GEN_005", "GEN_020", "GEN_030", "HWT_050", "GEN_015", "SBE_010"]
    health_codes_to_clean = [7, 8, 9]
    #clean the lifestyle cols
    health_targets = [col for col in health_cols if col in df.columns]
    df[health_targets] = df[health_targets].replace(health_codes_to_clean, np.nan)

    other_cols = ["SLPG005", "MACG005", "PAADVDYS"]
    other_codes_to_clean = [96, 97, 98, 99]
    other_targets = [col for col in other_cols if col in df.columns]
    df[other_targets] = df[other_targets].replace(other_codes_to_clean, np.nan)

    # rename to english
    df.rename(columns=rename_dict, inplace=True) #applies to existing df



def main():
    df = load_raw_data("../raw_data/cchs_2022_raw.csv")
    check_module_flags(df)

    df = clean_and_prepare(df, SELECTED_2022_DICT)

if __name__ == "__main__":
    main()