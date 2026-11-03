# CMPT 491: Canadian Lifestyle & Mental Health Data Mining Study
**An End-to-End Analysis of Behavioral Archetypes, Intervention Simulations, and Covid Drift**

## Project Overview
This project investigates the relationship between daily lifestyle factors (screen time, sleep, exercise, substance use, and life stress) and self-reported mental health vulnerability using the Canadian Community Health Survey (CCHS) datasets (2018 and 2022). 

Our research pipeline follows a progressive data mining lifecycle:
1. **Unsupervised Discovery:** K-Means clustering ($k=7$) to identify natural behavioral personas (e.g., Thriving vs. Crisis archetypes).
2. **Supervised Modeling & Feature Ablation:** Benchmarking Logistic Regression, Random Forest, and XGBoost; pruning 27 features to an 8-feature core while maintaining predictive performance.
3. **Prescriptive Simulation:** Counterfactual "What-If" modeling measuring the exact risk reduction from targeted lifestyle interventions.
4. **Temporal Shock Analysis:** Testing for pandemic-induced **Covariate Shift** (Adversarial Validation) and **Concept Drift** (Odds Ratio rewiring) between pre-COVID (2018) and post-COVID (2022) cohorts.

---

## Repository Structure

```text
├── clean_data/        # Cleaned CCHS survey sets and serialized winning pipeline (.joblib)
├── outputs/           # Generated high-resolution plots, charts, and figures
└── src/               # Execution pipeline scripts
    ├── 01_clean.py                    # Orchestrator for data cleaning & integration
    ├── 02_train_and_eval.py           # Model benchmark, SMOTE pipeline, and 2022 temporal holdout
    ├── 03_clustering.py               # K-Means clustering & persona profiling
    ├── 03_feature_importance.py       # Odds Ratios extraction and semantic visualization
    ├── 04_simulation.py               # Counterfactual intervention simulator (What-If analysis)
    ├── 04_draw_graphs.py              # Visualizations for report deliverables
    ├── 06_screen_vs_exercise.py       # 2D heatmap matrix (The Exercise Buffer & Tipping Point)
    ├── 07_generational_divergence.py  # Stratified comparison: Youth (12-34) vs. Older Adults (50+)
    ├── 08_adverse_validation.py       # Adversarial validation detecting pandemic covariate shift
    ├── 09_covid_drift.py              # Concept drift analysis comparing 2018 vs. 2022 Odds Ratios
    ├── app.py                         # Interactive Streamlit dashboard for live risk simulation
    ├── cchs_2018_cleaning.py          # 2018 raw CCHS filtering and recoding
    ├── cchs_2022_cleaning.py          # 2022 raw CCHS filtering and recoding
    ├── integration.py                 # Merges cleaned 2018 and 2022 cohorts
    └── investigation.py               # Exploratory data analysis and summary statistics