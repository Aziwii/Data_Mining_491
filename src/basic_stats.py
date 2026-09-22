import pandas as pd
import numpy as np


def load_dataset(file_path):
    """
    Load the dataset from a CSV file.
    """
    print("Loading dataset...")
    df = pd.read_csv(file_path, low_memory=False)

    # Explicit Row and Column counts
    print(f"\n========================================")
    print(f"RAW DATASET SHAPE & METRICS")
    print(f"Total Rows:    {df.shape[0]:,}")
    print(f"Total Columns: {df.shape[1]:,}")
    print(f"========================================\n")

    #rename dictionary for CCHS variables to more descriptive names
    CCHS_RENAME_DICT = {
    # --- Demographics & Identifiers ---
    "DHHGAGE": "age_group", # all
    "DHH_SEX": "sex", # all
    "INCDGHH": "household_income_group", # all
    "SDCDGIMM": "immigrant_status", # all
    "SDCDVABT": "aboriginal_identity", # all
    "SDCDVFLA": "first_official_language", # all
 

    # --- General Health & Stress ---
    "GEN_01": "perceived_health", # all
    "GEN_05": "perceived_mental_health",  # Target  # all
    "GEN_10": "perceived_life_stress", # all
    "GEN_15": "perceived_work_stress", # all
    "GEN_20": "perceived_sense_belonging_community", # all
    "MAC_05": "main_activity", # valid skip is school
    

    # --- Life Satisfaction & Well-being ---
    "LSM_01": "satisfaction_with_life", # all
    "LSMDVSWL": "derived_life_satisfaction_scale", # all same as above but with a different name

    # --- Height, Weight & Body Composition ---
    "HWTDGISW": "derived_bmi_class", #weight status based on BMI (over, under, normal, obese) # valid skip has 3000
    "HWTDGBCC": "body_mass_index_bmi", # same as above but with a different name # valid skip has 4000
    "WTP_50": "perceived_weight", # all

    # --- COVID-19 & General Context ---
    "COV2_005": "mental_health_before_or_during_covid", # all (answers are in terms of now, much better now ex)

    # --- Chronic Conditions ---
    "CCC_05": "has_asthma", #all  
    "CCC_80": "has_high_blood_pressure",#all 
    "CCC_90": "has_mood_disorder_depression_bipolar",#all
    "CCC_135": "has_diabetes",#all 
    "CC1_140": "has_heart_disease",#all  
    "CC1_145": "has_cancer",#all 
    "CC1_155": "has_arthritis",#all 

    # --- Screen Time & Sedentary (CSS) ---
    "CSSG20": "smoking_frequency_30d", # Valid skips are non smokers, never smoked

    # --- Electronic Cigarettes / Vaping (ECV) ---
    "ECVG15": "vaping_frequency", # valid skips are non vapers, never vaped

    # --- Alcohol Consumption (ALC) ---
    "ALC_15": "alcohol_frequency_past_12_months", # valid skips are non drinkers past 12 months
    "ALCDVTTM": "alcohol_total_drinks_per_week", #all (reg drinker, occasional drinker, none in 12 months)

    # --- Cannabis Use (CAN) ---
    "CAN_05C": "cannabis_use_past_12_months", # all 
}

    # filter the df to only include the columns that are in the dict
    available_cols = [col for col in CCHS_RENAME_DICT.keys() if col in df.columns]
    df_clean = df[available_cols].rename(columns=CCHS_RENAME_DICT)

    # Explicit Row and Column counts
    print(f"\n========================================")
    print(f"Dataset after removing unnecessary columns")
    print(f"Total Rows:    {df_clean.shape[0]:,}")
    print(f"Total Columns: {df_clean.shape[1]:,}")
    print(f"Columns: {list(df_clean.columns)}")
    print(f"\n========================================")

    return df_clean


def main():
    # Load the dataset
    load_dataset('../raw_data/pumf_cchs.csv')



if __name__ == "__main__":
    main()