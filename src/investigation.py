import pandas as pd

# load clean set
df18 = pd.read_csv("../clean_data/cchs_2018_cleaned.csv")
if df18.empty:
    print("Warning: The 2018 dataset is empty. Please check the file path and contents.")
df22 = pd.read_csv("../clean_data/cchs_2022_cleaned.csv")
if df22.empty:
    print("Warning: The 2022 dataset is empty. Please check the file path and contents.") 

SELECTED_2018_DICT = {
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

cols_2018 = SELECTED_2018_DICT.values()
cols_2022 = SELECTED_2022_DICT.values()

# summary stats
summary18 = df18[cols_2018].describe().T # pivot it to vertical
summary22 = df22[cols_2022].describe().T # pivot it to vertical
# add median and var
summary18["median"] = df18[cols_2018].median()
summary18["variance"] = df18[cols_2018].var()

summary22["median"] = df22[cols_2022].median()
summary22["variance"] = df22[cols_2022].var()

# Scale Direction Dictionary
FEATURE_SCALES = {
    "perceived_mental_health": "1 = Excellent  -->  4 = Fair/Poor  (HIGHER = WORSE)",
    "perceived_general_health": "1 = Excellent  -->  5 = Poor  (HIGHER = WORSE)",
    "perceived_life_stress": "1 = Not at all  -->  5 = Extreme  (HIGHER = MORE STRESS)",
    "community_belonging": "1 = Very strong  -->  4 = Very weak  (HIGHER = WEAKER)",
    "screen_time_non_workday": "1 = < 2 hours  -->  5 = 8+ hours  (HIGHER = MORE SCREEN TIME)",
    "sleep_hours_per_night": "1 = < 4 hours  -->  11 = 12+ hours  (HIGHER = MORE SLEEP)",
    "physical_activity_hours_7d": "0 hours  -->  70+ hours  (HIGHER = MORE EXERCISE)",
    "physical_activity_level": "1 = Active  -->  3 = Inactive  (HIGHER = LESS ACTIVE)",
    "smoking_type": "1 = Daily smoker  -->  6 = Never smoked  (HIGHER = LESS SMOKING)",
    "alcohol_frequency": "0 = Never  -->  7 = Daily  (HIGHER = MORE DRINKING)",
    "household_income_group": "1 = Lowest  -->  5 = Highest  (HIGHER = MORE INCOME)",
    "age_group": "1 = Youth (12-17)  -->  5 = Seniors (65+)  (HIGHER = OLDER)",
    "main_activity": "0 = Not working/Retired  -->  1 = Working  (HIGHER = EMPLOYED)",
    "bmi_self_reported": "1 = Normal/Under  -->  2 = Overweight/Obese  (HIGHER = HIGHER BMI)",
}

# Print the reference guide
print("\n" + "=" * 80)
print("FEATURE SCALE & DIRECTION REFERENCE GUIDE")
print("=" * 80)
for feature, scale_info in FEATURE_SCALES.items():
    print(f"{feature:28s} : {scale_info}")
print("=" * 80 + "\n")

# clean table viewing
print(summary18[["mean", "median", "std", "min", "25%", "75%", "max", "variance"]].round(2))
print("\n--------------------------------")
print(summary22[["mean", "median", "std", "min", "25%", "75%", "max", "variance"]].round(2))