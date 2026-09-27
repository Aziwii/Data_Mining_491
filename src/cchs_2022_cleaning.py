import pandas as pd

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

# def check_module_flags(raw_df):


def main():
    load_raw_data("../raw_data/pumf_cchs.csv")

if __name__ == "__main__":
    main()