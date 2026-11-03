import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


def main():
    df = pd.read_csv("../clean_data/cchs_2018_2022_combined.csv")

    df18 = df[df["year"]==2018]
    df22 = df[df["year"]==2022]

    life_features = [
            "perceived_life_stress",
            "community_belonging",
            "screen_time_non_workday",
            "sleep_hours_per_night",
            "physical_activity_hours_7d",
            "alcohol_frequency",
            "age_group",
            "household_income_group",
            "sex",
        ]

    # filter the existing cols

    features = [c for c in life_features if c in df.columns]
    X_18 = df18[features].copy()
    y_18 = (df18["perceived_mental_health"] == 4.0).astype(int)

    X_22 = df22[features].copy()
    y_22 = (df22["perceived_mental_health"] == 4.0).astype(int)


    # standard scalar for preprocessing
    scaler = StandardScaler()
    X_18_scaled = scaler.fit_transform(X_18)
    X_22_scaled = scaler.transform(X_22)

    #fit the models 
    model_18 = LogisticRegression(class_weight="balanced", random_state=42)
    model_22 = LogisticRegression(class_weight="balanced", random_state=42)

    #train the models
    model_18.fit(X_18_scaled, y_18)
    model_22.fit(X_22_scaled, y_22)

    #find the odds ratio
    or_18 = np.exp(model_18.coef_[0])
    or_22 = np.exp(model_22.coef_[0])

    # put them into a df
    comparison_df = pd.DataFrame(
        {
            "Feature": features,
            "Odds_Ratio_2018": or_18,
            "Odds_Ratio_2022": or_22,
            "Delta_Change": or_22 - or_18,
        }
    ).sort_values(by="Delta_Change", ascending=False)

    print("\n" + "=" * 70)
    print("PANDEMIC CONCEPT DRIFT: ODDS RATIO COMPARISON (2018 vs 2022)")
    print("=" * 70)
    print(comparison_df.round(3).to_string(index=False))

    # 1. Reset index to avoid any pandas alignment errors
    comparison_df = comparison_df.reset_index(drop=True)

    # 2. Simple, clean name replacement (no lambda needed)
    SEMANTIC_NAMES = {
        "perceived_life_stress": "Life Stress",
        "community_belonging": "Community Belonging (Higher = Weaker)",
        "screen_time_non_workday": "Screen Time",
        "sleep_hours_per_night": "Sleep Duration",
        "physical_activity_hours_7d": "Exercise Hours",
        "alcohol_frequency": "Alcohol Frequency",
        "household_income_group": "Household Income",
        "age_group": "Age Cohort",
        "sex": "Sex",
    }
    comparison_df["Display_Name"] = comparison_df["Feature"].replace(
        SEMANTIC_NAMES
    )

    # 3. Sort so top risk factors are at the top
    comparison_df = comparison_df.sort_values(
        by="Odds_Ratio_2018", ascending=True
    ).reset_index(drop=True)

    # 4. Plotting positions
    y_pos = np.arange(len(comparison_df))
    bar_height = 0.35

    plt.figure(figsize=(11, 7))

    # 2018 Bars (Pre-COVID)
    plt.barh(
        y_pos + bar_height / 2,
        comparison_df["Odds_Ratio_2018"],
        height=bar_height,
        color="#8ecae6",
        edgecolor="black",
        label="2018 (Pre-COVID)",
    )

    # 2022 Bars (Post-COVID)
    plt.barh(
        y_pos - bar_height / 2,
        comparison_df["Odds_Ratio_2022"],
        height=bar_height,
        color="#023047",
        edgecolor="black",
        label="2022 (Post-COVID)",
    )

    # Neutral Reference Line at OR = 1.0
    plt.axvline(
        1.0,
        color="red",
        linestyle="--",
        linewidth=1.5,
        label="Neutral (No Effect: OR = 1.0)",
    )

    # Annotate Zones
    plt.text(
        0.55,
        len(comparison_df) - 0.5,
        "← PROTECTIVE\n(Reduces Risk)",
        color="#2a9d8f",
        fontweight="bold",
        ha="center",
        fontsize=10,
    )
    plt.text(
        1.70,
        len(comparison_df) - 0.5,
        "RISK FACTOR →\n(Increases Risk)",
        color="#e76f51",
        fontweight="bold",
        ha="center",
        fontsize=10,
    )

    plt.yticks(y_pos, comparison_df["Display_Name"].tolist(), fontsize=10)
    plt.xlabel(
        "Odds Ratio (Values > 1 increase risk, Values < 1 decrease risk)",
        weight="bold",
        fontsize=11,
    )
    plt.title(
        "Pandemic Concept Drift: Odds Ratios Before vs. After COVID (2018 vs. 2022)",
        fontsize=13,
        weight="bold",
        pad=15,
    )
    plt.legend(loc="lower right", framealpha=0.9)
    plt.grid(axis="x", linestyle=":", alpha=0.6)
    plt.tight_layout()

    out_path = "../outputs/09_covid_concept_drift_comparison.png"
    plt.savefig(out_path, dpi=300)
    print(f"\nSaved intuitive comparison plot to: {out_path}")
    plt.show()
if __name__=="__main__":
    main()