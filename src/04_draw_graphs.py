from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


PROJECT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_DIR / "clustering_results"

MENTAL_HEALTH_ORDER = [
    "Excellent",
    "Very good",
    "Good",
    "Fair",
    "Poor",
]

MENTAL_HEALTH_COLORS = [
    "#2166ac",
    "#67a9cf",
    "#d9d9d9",
    "#ef8a62",
    "#b2182b",
]

def main():
    comparison_file = OUTPUT_DIR / "cluster_comparison.csv"
    mental_file = OUTPUT_DIR / "mental_health_percentages.csv"

    for path in [comparison_file, mental_file]:
        if not path.exists():
            raise FileNotFoundError(
                f"Cannot find:\n{path}\n"
                "Run the updated 03_clustering.py first."
            )

    comparison = pd.read_csv(comparison_file).sort_values("k")
    mental = pd.read_csv(mental_file, index_col="cluster")
    mental = mental.sort_index()[MENTAL_HEALTH_ORDER]

    # Figure 1: comparison of cluster counts.
    fig1, ax1 = plt.subplots(figsize=(9, 5))

    ax1.plot(
        comparison["k"],
        comparison["silhouette_score"],
        marker="o",
        color="#2166ac",
    )

    ax1.set_xticks(comparison["k"])
    ax1.set_xlabel("Number of clusters")
    ax1.set_ylabel("Silhouette score")
    ax1.set_title("Comparison of cluster counts")
    ax1.grid(alpha=0.3)

    fig1.tight_layout()
    fig1.savefig(
        OUTPUT_DIR / "cluster_comparison.png",
        dpi=300,
        bbox_inches="tight",
    )

    # Figure 2: all five mental-health categories within each cluster.
    fig2, ax2 = plt.subplots(figsize=(12, 7))

    mental.plot(
        kind="bar",
        stacked=True,
        color=MENTAL_HEALTH_COLORS,
        edgecolor="white",
        width=0.65,
        ax=ax2,
    )

    ax2.set_ylim(0, 100)
    ax2.set_xlabel("Cluster")
    ax2.set_ylabel("Respondents within cluster (%)")
    ax2.set_title(
        "Perceived mental health by cluster\n"
        "CCHS 2017–2018 selected sample — unweighted"
    )
    ax2.tick_params(axis="x", rotation=0)
    ax2.legend(
        title="Mental health",
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
    )

    fig2.tight_layout()
    fig2.savefig(
        OUTPUT_DIR / "mental_health_by_cluster.png",
        dpi=300,
        bbox_inches="tight",
    )

    print("\nSaved two figures:")
    print(OUTPUT_DIR / "cluster_comparison.png")
    print(OUTPUT_DIR / "mental_health_by_cluster.png")

    plt.show()


if __name__ == "__main__":
    main()
