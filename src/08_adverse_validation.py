import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from sklearn.model_selection import train_test_split

def main():
    # 1 load the dataset
    df = pd.read_csv("../clean_data/cchs_2018_2022_combined.csv")

    df = df.query("age_group == 2")
    print("\nBased on age_group 2:")

    print(
        df.groupby("year")[
            [
                "screen_time_non_workday",
                "alcohol_frequency",
                "community_belonging",
                "perceived_life_stress",
                "physical_activity_hours_7d",
                "perceived_life_stress",

            ]
        ].mean()
    )
    # feautes to analyze (drop mental health target and year)

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

    X = df[features].copy()
    #Synthentic target 1 = post 0 = pre 
    y = np.array(df["year"] == 2022, dtype=int)

    print(f"Total respondants: {len(df):,}")
    print(f"# of 2018: {(y==0).sum():,}")
    print(f"# of 2022: {(y==1).sum():,}\n")

    #test train split for adversarioal task
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    #2 train adversarial random forest - or test multiple to check the best one
    clf = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)
    clf.fit(X_train, y_train)

    prob = clf.predict_proba(X_test)[:,1] #all rows and col idx 1, just prob vector for y=1
    auc_score = roc_auc_score(y_test, prob)

    print(f"Adversarial Model ROC_AUC: {auc_score:.4f}")
    if auc_score > 0.65:
        print("Result: significant shift --- Pandemic created distinct lifestyle fingerprints")
    else: 
        print("Result: Minimal shift detected between 2018 and 2022")

    #3 extract feature importances
    importances = pd.Series(clf.feature_importances_, index=features).sort_values(ascending=True)

    NAMES = {
        "screen_time_non_workday": "Screen Time",
        "community_belonging": "Community Belonging",
        "perceived_life_stress": "Life Stress",
        "sleep_hours_per_night": "Sleep Duration",
        "physical_activity_hours_7d": "Exercise Hours",
        "alcohol_frequency": "Alcohol Frequency",
        "household_income_group": "Household Income",
        "age_group": "Age Cohort",
        "sex": "Sex",
    }

    importances.index = importances.index.map(lambda x: NAMES.get(x, x))

    print("\nTop Predictors of the post-covid:")
    for feat, imp in importances.sort_values(ascending=False).items():
        print(f"  {feat:25s}: {imp*100:5.2f}% importance")
    #4 plot the pandemic fingerprints
    plt.figure(figsize=(9, 5))
    importances.plot(kind="barh", color="#457b9d", edgecolor="black")
    plt.title(
        f"Adversarial Feature Importance: What Distinguishes 2022 from 2018?\n(Model ROC-AUC = {auc_score:.3f})",
        fontsize=12,
        weight="bold",
        pad=15,
    )
    plt.xlabel("Gini Feature Importance (%)", weight="bold")
    plt.grid(axis="x", linestyle=":", alpha=0.6)
    plt.tight_layout()

    out_path = "../outputs/08_adversarial_pandemic_fingerprints_age2.png"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=300)
    print(f"\nSaved adversarial plot to {out_path}")

if __name__ == "__main__":
    main()
