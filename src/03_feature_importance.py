import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import os

def load_pipeline(filename):
    #to load the model in another file
    pipeline = joblib.load(filename)
    return pipeline

def extraction_of_features(
    pipeline, output_path="../outputs/03_odds_ratios_chart.png"
):
    preprocessor = pipeline.named_steps["preprocessor"]
    classifier = pipeline.named_steps["regression"]
    raw_coefs = classifier.coef_[0]

    feature_names = preprocessor.get_feature_names_out()
    exponential_coefs = np.exp(raw_coefs)

    coef_summary = pd.DataFrame(
        {
            "Feature": feature_names,
            "Raw_coef": raw_coefs,
            "Odds_Ratio": exponential_coefs,
        }
    ).sort_values(by="Odds_Ratio", ascending=False)

    SEMANTIC_LABELS = {
        "num__perceived_general_health": "Poorer Physical Health (Higher = Worse)",
        "num__perceived_life_stress": "Higher Life Stress (Higher = More Stress)",
        "num__community_belonging": "Weaker Community Belonging (Higher = More Isolated)",
        "num__screen_time_non_workday": "Higher Leisure Screen Time",
        "num__household_income_group": "Higher Household Income",
        "cat__sex_2": "Sex: Female",
        "cat__sex_1": "Sex: Male",
        "num__age_group": "Older Age Cohort (Higher = Older)",
    }

    coef_summary["Feature"] = coef_summary["Feature"].map(SEMANTIC_LABELS)

    print("\n" + "=" * 60)
    print("NARROW-SCOPED MODEL ODDS RATIOS")
    print("=" * 60)
    print(coef_summary.to_string(index=False))

    # Matplotlib diverging bar chart
    plt.figure(figsize=(9, 5))
    colors = [
        "#2a9d8f" if ratio < 1.0 else "#e76f51"
        for ratio in coef_summary["Odds_Ratio"]
    ]

    plt.barh(
        coef_summary["Feature"],
        coef_summary["Odds_Ratio"] - 1.0,
        left=1.0,
        color=colors,
        edgecolor="black",
        linewidth=0.5,
    )

    plt.axvline(1.0, color="black", linestyle="--", linewidth=1.2)
    plt.xlabel("Odds Ratio (OR)", fontsize=11, fontweight="bold")
    plt.title(
        "Predictors of Poor Mental Health",
        fontsize=12,
        fontweight="bold",
        pad=15,
    )
    
    caption_text = "Values > 1.0 Increase Risk  |  Values < 1.0 Reduce Risk"
    ax = plt.gca() 
    ax.text(
        0.5,             
        -0.15,           
        caption_text, 
        transform=ax.transAxes,
        horizontalalignment='center', 
        verticalalignment='top',
        fontsize=10, 
        style='italic',
        color='dimgray'  
    )
    # -----------------------------

    plt.grid(axis="x", linestyle=":", alpha=0.6)
    plt.tight_layout()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    plt.savefig(output_path, bbox_inches="tight")
    print(f"\nSaved updated plot to: {output_path}")





def main():
    pipeline = load_pipeline("../clean_data/winning_logistic_regression.joblib")
    extraction_of_features(pipeline)

if __name__ == "__main__":
    main()