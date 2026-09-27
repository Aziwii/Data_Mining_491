import pandas as pd

# load clean set
df = pd.read_csv("../clean_data/cchs_2018_cleaned.csv")

# most importtant cols
key_cols = [
    "perceived_mental_health", 
    "sleep_hours_per_night",   
    "screen_time_non_workday", 
    "active_days_past_week",  
    "perceived_life_stress", 
]

print(key_cols)
# summary stats
summary = df[key_cols].describe().T # pivot it to vertical
# add median and var
summary["median"] = df[key_cols].median()
summary["variance"] = df[key_cols].var()

# clean table viewing
print(summary[["mean", "median", "std", "min", "25%", "75%", "max", "variance"]].round(2))