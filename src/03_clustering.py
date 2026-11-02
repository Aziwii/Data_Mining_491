from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


# File path:
# CCHS_Project/
#   raw_data/cchs_2018_raw.csv
#   Data_Mining_491-main/
#     src/03_clustering.py
#     src/04_draw_graphs.py
#     clustering_results/

PROJECT_DIR = Path(__file__).resolve().parent.parent
INPUT_FILE = PROJECT_DIR.parent / "raw_data" / "cchs_2018_raw.csv"
OUTPUT_DIR = PROJECT_DIR / "clustering_results"

RANDOM_SEED = 20
CHOSEN_K = 7
K_VALUES = range(2, 9)

COLUMN_NAMES = {
    "GEN_015": "mental_health",
    "GEN_020": "life_stress",
    "GEN_030": "community_belonging",
    "SBE_010": "screen_time",
}

FEATURES = [
    "life_stress",
    "community_belonging",
    "screen_time",
]

VALID_CODES = {
    "mental_health": range(1, 6),
    "life_stress": range(1, 6),
    "community_belonging": range(1, 5),
    "screen_time": range(1, 6),
}

MENTAL_HEALTH_LABELS = {
    1: "Excellent",
    2: "Very good",
    3: "Good",
    4: "Fair",
    5: "Poor",
}


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Cannot find the raw data:\n{INPUT_FILE}")

    raw = pd.read_csv(INPUT_FILE, usecols=list(COLUMN_NAMES))
    raw = raw.rename(columns=COLUMN_NAMES)
    original_count = len(raw)

    # Missing-response codes for these four specific survey variables.
    raw = raw.replace([7, 8, 9], np.nan)

    print("\nMissing responses before removing rows:")
    print(raw.isna().sum())

    # Check that all remaining responses are documented category codes.
    for column, valid_codes in VALID_CODES.items():
        invalid = raw[column].notna() & ~raw[column].isin(valid_codes)
        if invalid.any():
            unexpected = raw.loc[invalid, column].unique().tolist()
            raise ValueError(
                f"Unexpected codes in {column}: {unexpected}"
            )

    df = raw.dropna(subset=list(VALID_CODES)).copy()
    df = df.astype({column: int for column in VALID_CODES})

    print(f"\nOriginal respondents: {original_count:,}")
    print(f"Removed respondents:  {original_count - len(df):,}")
    print(f"Retained respondents: {len(df):,}")

    print("\nMental-health counts:")
    print(
        df["mental_health"]
        .value_counts()
        .reindex(MENTAL_HEALTH_LABELS, fill_value=0)
        .rename(index=MENTAL_HEALTH_LABELS)
    )

    if len(df) <= max(K_VALUES):
        raise ValueError("Not enough respondents for the requested clusters.")

    if len(df[FEATURES].drop_duplicates()) < max(K_VALUES):
        raise ValueError("Not enough distinct predictor combinations.")

    # Only the three predictors form clusters.
    # Mental health is used afterward to describe the groups.
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[FEATURES])

    models = {}
    comparison_rows = []

    for k in K_VALUES:
        model = KMeans(
            n_clusters=k,
            random_state=RANDOM_SEED,
            n_init=100,
        )
        labels = model.fit_predict(X_scaled)

        score = silhouette_score(
            X_scaled,
            labels,
            sample_size=min(3000, len(df)),
            random_state=42,
        )

        sizes = pd.Series(labels).value_counts()
        models[k] = model

        comparison_rows.append({
            "k": k,
            "silhouette_score": score,
            "inertia": model.inertia_,
            "smallest_cluster": int(sizes.min()),
            "largest_cluster": int(sizes.max()),
        })

        print(f"k = {k}: silhouette score = {score:.4f}")

    comparison = pd.DataFrame(comparison_rows)

    print("\nCluster comparison:")
    print(comparison.round(4).to_string(index=False))

    if CHOSEN_K is None:
        best_row = comparison["silhouette_score"].idxmax()
        selected_k = int(comparison.loc[best_row, "k"])
    else:
        selected_k = CHOSEN_K

    if selected_k not in models:
        raise ValueError("CHOSEN_K must be included in K_VALUES.")

    df["cluster"] = models[selected_k].labels_ + 1

    print(f"\nSelected number of clusters: {selected_k}")
    print("Cluster numbers are labels, not health rankings.")

    sizes = df.groupby("cluster").size()
    profiles = df.groupby("cluster")[FEATURES].median()
    profiles = profiles.rename(
        columns={name: f"{name}_median" for name in FEATURES}
    )
    profiles.insert(0, "respondents", sizes)
    profiles.insert(
        1, "percent_of_sample", (sizes / len(df) * 100).round(2)
    )

    print("\nCluster profiles — original category codes:")
    print(profiles.to_string())

    for feature in FEATURES:
        percentages = pd.crosstab(
            df["cluster"],
            df[feature],
            normalize="index",
        ).reindex(columns=list(VALID_CODES[feature]), fill_value=0) * 100

        print(f"\n{feature}: category percentages within each cluster")
        print(percentages.round(2).to_string())

    mental_counts = pd.crosstab(
        df["cluster"], df["mental_health"]
    ).reindex(columns=list(MENTAL_HEALTH_LABELS), fill_value=0)

    mental_counts = mental_counts.rename(columns=MENTAL_HEALTH_LABELS)
    mental_percentages = mental_counts.div(
        mental_counts.sum(axis=1), axis=0
    ) * 100

    print("\nMental-health counts within each cluster:")
    print(mental_counts.to_string())

    print("\nMental-health percentages within each cluster:")
    print(mental_percentages.round(2).to_string())

    # Save only these three CSV files.
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    comparison_file = OUTPUT_DIR / "cluster_comparison.csv"
    mental_file = OUTPUT_DIR / "mental_health_percentages.csv"
    respondents_file = OUTPUT_DIR / "respondents_with_clusters.csv"

    comparison.to_csv(comparison_file, index=False)
    mental_percentages.to_csv(mental_file, index_label="cluster")

    df[
        FEATURES + ["mental_health", "cluster"]
    ].to_csv(respondents_file, index=False)

    print("\nSaved these three CSV files:")
    for path in [comparison_file, mental_file, respondents_file]:
        print(path)

    print("\nRun 04_draw_graphs.py to draw the graphs.")


if __name__ == "__main__":
    main()