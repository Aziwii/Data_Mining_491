import pandas as pd
import cchs_2018_cleaning
import cchs_2022_cleaning
import integration 

print("Cleaning 2018 dataset...")
cchs_2018_cleaning.main()

print("Cleaning 2022 dataset...")
cchs_2022_cleaning.main()

print("Integrating 2018 and 2022 datasets...")


try:
        
    df18 = pd.read_csv("../clean_data/cchs_2018_cleaned.csv")
    df22 = pd.read_csv("../clean_data/cchs_2022_cleaned.csv")
    if (not df18.empty) and (not df22.empty):
        integration.main()
        print("\n\nIntegration completed successfully.")
    else:
        print("Error: One or both of the cleaned datasets are empty. Please check the cleaning steps.")
except FileNotFoundError as e:
    print(f"Error: {e}. Please ensure that the cleaned datasets exist before integration.")

