import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os


def main():
    pipeline = joblib.load("../clean_data/winning_logistic_regression.joblib")

    # Baseline persona: High Stress, Isolated, Heavy Screen Time (Cluster 4 Archetype)
    baseline_person = pd.DataFrame(
        [
            {
                "perceived_life_stress": 4.0,  # quite stressful
                "community_belonging": 4.0,  # very weak
                "screen_time_non_workday": 5.0,  # 8+ hours
                "perceived_general_health": 3.0,  # fair/good
                "age_group": 2.0,  # young adult
                "household_income_group": 3.0,  # medium income
                "sex": 2.0,  # female
            }
        ]
    )

    baseline_prob = pipeline.predict_proba(baseline_person)[0][1]

    # create a dictionary to store the scenarios and their probabilities
    scenarios = {}
    scenarios["Baseline (Cluster 4 Archetype)"] = baseline_prob

    # 1 screen time intervention - reduce to < 4 hours
    sim_screen = baseline_person.copy()
    sim_screen["screen_time_non_workday"] = 2.0  # drop to < 4 hours
    scenarios["Intervention 1: Reduce Screen Time (8+ hrs -> < 4 hrs)"] = (
        pipeline.predict_proba(sim_screen)[0][1]
    )

    # 2. belonging intervention
    sim_belonging = baseline_person.copy()
    sim_belonging["community_belonging"] = 2.0  # very weak to somewhat strong belonging
    scenarios["Intervention 2: Strengthen Community Belonging (4 -> 2)"] = (
        pipeline.predict_proba(sim_belonging)[0][1]
    )

    # 3. life stress intervention to mild stress
    sim_stress = baseline_person.copy()
    sim_stress["perceived_life_stress"] = 2.0  # quite a bit to not very
    scenarios["Intervention 3: Reduce Life Stress to Mild (4 -> 2)"] = (
        pipeline.predict_proba(sim_stress)[0][1]
    )

    # 4. screen time + belonging intervention
    sim_combo = baseline_person.copy()
    sim_combo["screen_time_non_workday"] = 2.0
    sim_combo["community_belonging"] = 2.0
    scenarios["Intervention 4: Reduce Screens + Strengthen Belonging (4 -> 2)"] = (
        pipeline.predict_proba(sim_combo)[0][1]
    )

    # all three interventions together
    sim_all_three = baseline_person.copy()
    sim_all_three["screen_time_non_workday"] = 2.0  #cut screen time
    sim_all_three["community_belonging"] = 2.0  # stronger belonging
    sim_all_three["perceived_life_stress"] = 2.0  #lower stress
    scenarios["Intervention 5: Full Reset (Screens + Belonging + Stress)"] = (
        pipeline.predict_proba(sim_all_three)[0][1]
    )

    print("\n" + "=" * 70)
    print("INTERVENTION SIMULATION")
    print("=" * 70)

    for name, prob in scenarios.items():
        if name.startswith("Baseline"):
            print(f"{name:58s} : {prob*100:5.1f}% Risk")
        else:
            diff_abs = (prob - baseline_prob) * 100
            diff_rel = ((prob - baseline_prob) / baseline_prob) * 100
            print(
                f"{name:58s} : {prob*100:5.1f}% Risk | Abs Change: {diff_abs:+5.1f}% | Rel Drop: {abs(diff_rel):4.1f}%"
            )
    print("=" * 70)

    # SIMULTAION CHART ===========================
    labels = [
        "Baseline (High Risk / Cluster 4)",
        "1. Reduce Screen Time (8+ hrs -> < 4 hrs)",
        "2. Strengthen Community Belonging (4 -> 2)",
        "3. Reduce Life Stress to Mild (4 -> 2)",
        "4. Reduce Screens + Community",
        "5. Full Reset (Screens + Community + Stress)",
    ]

    risks = [prob * 100 for prob in scenarios.values()]

    # colors palette: red, orange, yellow, blue, violet, green
    colors = ["#d9534f", "#f0ad4e", "#f7e26b", "#5bc0de", "#9370db", "#5cb85c"]

    plt.figure(figsize=(11, 6))
    bars = plt.barh(labels, risks, color=colors, edgecolor="black", height=0.6)

    plt.xlim(0, 100)
    plt.xlabel(
        "Predicted Probability of Fair/Poor Mental Health (%)",
        fontsize=11,
        fontweight="bold",
    )
    plt.title(
        "Impact of Simulated Lifestyle Interventions on At-Risk Persona (Cluster 4)",
        fontsize=13,
        fontweight="bold",
        pad=15,
    )
    plt.gca().invert_yaxis()  # baseline on top

    # text to the right of each bar
    for bar in bars:
        width = bar.get_width()
        plt.text(
            width + 1.2,
            bar.get_y() + bar.get_height() / 2,
            f"{width:.1f}%",
            va="center",
            ha="left",
            fontsize=10,
            fontweight="bold",
        )

    plt.grid(axis="x", linestyle=":", alpha=0.6)
    plt.tight_layout()

    output_plot = "../outputs/04_lifestyle_intervention_simulation.png"
    os.makedirs(os.path.dirname(output_plot), exist_ok=True)
    plt.savefig(output_plot, dpi=300)
    print(f"\nSaved simulation chart to: {output_plot}")


if __name__ == "__main__":
    main()