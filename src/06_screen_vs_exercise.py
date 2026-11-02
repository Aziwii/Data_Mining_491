import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def main():
    df = pd.read_csv("../clean_data/cchs_2018_2022_combined.csv")

    # Target: 1 = Fair/Poor Mental Health
    df["poor_mh"] = (df["perceived_mental_health"] == 4.0).astype(int)

    # bin physical activity Hours into intuitive human categories
    # 0 hrs, 1-3 hrs, 3-7 hrs, 7+ hrs
    bins = [-1, 0.5, 3.5, 7.5, 150]
    labels = ["None (0 hrs)", "Low (1-3 hrs)", "Moderate (4-7 hrs)", "High (7+ hrs)"]
    df["exercise_bracket"] = pd.cut(
        df["physical_activity_hours_7d"], bins=bins, labels=labels
    )

    # map screen time to names
    screen_map = {
        1.0: "< 2 hrs",
        2.0: "2 - 4 hrs",
        3.0: "4 - 6 hrs",
        4.0: "6 - 8 hrs",
        5.0: "8+ hrs",
    }
    df["screen_time_label"] = df["screen_time_non_workday"].map(screen_map)

    # create pivot table: % Poor Mental Health
    pivot = (
        df.groupby(["exercise_bracket", "screen_time_label"], observed=False)[
            "poor_mh"
        ].mean()
        * 100
    ).unstack()

    # reorder columns logically from low to high screens
    ordered_cols = ["< 2 hrs", "2 - 4 hrs", "4 - 6 hrs", "6 - 8 hrs", "8+ hrs"]
    pivot = pivot[ordered_cols]

    print("\n" + "=" * 60)
    print("PERCENTAGE OF POOR MENTAL HEALTH (%): EXERCISE VS SCREEN TIME")
    print("=" * 60)
    print(pivot.round(2))

    # plot seaborn heatmap
    plt.figure(figsize=(10, 6))
    sns.heatmap(
        pivot,
        annot=True,
        fmt=".1f",
        cmap="YlOrRd",
        cbar_kws={"label": "% Reporting Fair/Poor Mental Health"},
        linewidths=1,
    )

    plt.title(
        "The Exercise Buffer: Does Physical Activity Protect Against Screen Time?",
        fontsize=13,
        weight="bold",
        pad=15,
    )
    plt.xlabel("Daily Non-Work Screen Time", fontsize=11, weight="bold")
    plt.ylabel("Weekly Physical Activity", fontsize=11, weight="bold")
    plt.tight_layout()

    output_path = "../outputs/06_exercise_vs_screen_heatmap.png"
    plt.savefig(output_path, dpi=300)
    print(f"\nSaved heatmap to {output_path}")


if __name__ == "__main__":
    main()