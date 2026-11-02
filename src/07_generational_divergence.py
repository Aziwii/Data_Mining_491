import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


def train_cohort_model(df_sub, name):
    features = [
        "perceived_life_stress",
        "community_belonging",
        "screen_time_non_workday",
        "household_income_group",
    ]
    X = df_sub[features]
    y = (df_sub["perceived_mental_health"] == 4.0).astype(int)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = LogisticRegression(class_weight="balanced", random_state=42)
    model.fit(X_scaled, y)

    odds_ratios = np.exp(model.coef_[0])
    return pd.Series(odds_ratios, index=features, name=name)


def main():
    df = pd.read_csv("../clean_data/cchs_2018_2022_combined.csv")

    # Split into Youth/Young Adults (12-34) and Older Adults (50+)
    youth = df[df["age_group"] <= 2].copy()
    seniors = df[df["age_group"] >= 4].copy()

    youth_or = train_cohort_model(youth, "Youth / Young Adults (12-34)")
    seniors_or = train_cohort_model(seniors, "Older Adults (50+)")

    comparison = pd.concat([youth_or, seniors_or], axis=1)

    print("\n" + "=" * 65)
    print("ODDS RATIO COMPARISON: YOUTH VS OLDER ADULTS")
    print("=" * 65)
    print(comparison.round(3))

    # Plot comparison bar chart
    comparison.plot(kind="barh", figsize=(10, 6), color=["#e76f51", "#2a9d8f"])
    plt.axvline(1.0, color="black", linestyle="--")
    plt.xlabel("Odds Ratio (OR)", weight="bold")
    plt.title(
        "Generational Divergence: How Lifestyle Impacts Youth vs. Seniors",
        fontsize=13,
        weight="bold",
        pad=15,
    )
    plt.grid(axis="x", linestyle=":", alpha=0.6)
    plt.tight_layout()

    output_path = "../outputs/07_generational_odds_comparison.png"
    plt.savefig(output_path, dpi=300)
    print(f"\nSaved comparison chart to {output_path}")


if __name__ == "__main__":
    main()