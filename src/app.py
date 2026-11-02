import joblib
import numpy as np
import pandas as pd
import streamlit as st

#AI Generated code for a Streamlit app that predicts the probability of reporting Fair/Poor Mental Health based on user inputs.

st.set_page_config(page_title="Mental Health Risk Explorer", layout="wide")


# Load the trained model pipeline
@st.cache_resource
def load_pipeline():
    return joblib.load("../clean_data/winning_logistic_regression.joblib")


try:
    pipeline = load_pipeline()
except Exception as e:
    st.error(f"Could not load model pipeline: {e}")
    st.stop()

st.title("🧠 Lifestyle & Mental Health Risk Simulator")
st.markdown(
    "Explore how daily habits interact to influence the probability of reporting **Fair/Poor Mental Health**."
)

# Sidebar: Presets for quick exploration
st.sidebar.header("Archetype Presets")
preset = st.sidebar.selectbox(
    "Choose a starting persona:",
    [
        "Custom",
        "Crisis Archetype (Cluster 4: High Stress, Isolated, Heavy Screens)",
        "The Digital Thriver (Cluster 6: High Screens, High Belonging, Low Stress)",
        "The Thriving Baseline (Cluster 2: Healthy Habits)",
    ],
)

# Set defaults based on preset
if "Crisis" in preset:
    d_stress, d_belong, d_screen, d_age, d_income = 4, 4, 5, 2, 3
elif "Digital Thriver" in preset:
    d_stress, d_belong, d_screen, d_age, d_income = 2, 1, 4, 2, 4
elif "Thriving" in preset:
    d_stress, d_belong, d_screen, d_age, d_income = 1, 1, 1, 4, 5
else:
    d_stress, d_belong, d_screen, d_age, d_income = 3, 2, 3, 3, 3

# Main input columns
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Digital & Social Habits")
    screen_time = st.slider(
        "Leisure Screen Time",
        1,
        5,
        d_screen,
        help="1 = <2h, 2 = 2-4h, 3 = 4-6h, 4 = 6-8h, 5 = 8+h",
    )
    belonging = st.slider(
        "Community Belonging",
        1,
        4,
        d_belong,
        help="1 = Very strong, 2 = Somewhat strong, 3 = Somewhat weak, 4 = Very weak",
    )

with col2:
    st.subheader("Stress & Physical State")
    stress = st.slider(
        "Perceived Life Stress",
        1,
        5,
        d_stress,
        help="1 = Not at all, 5 = Extreme stress",
    )
    health = st.slider(
        "General Physical Health",
        1,
        5,
        3,
        help="1 = Excellent, 3 = Good, 5 = Poor",
    )

with col3:
    st.subheader("Demographics")
    age = st.slider(
        "Age Cohort",
        1,
        5,
        d_age,
        help="1 = Youth (12-17), 2 = 18-34, 3 = 35-49, 4 = 50-64, 5 = 65+",
    )
    income = st.slider(
        "Household Income Tier",
        1,
        5,
        d_income,
        help="1 = Lowest 20%, 5 = Highest 20%",
    )
    sex = st.radio("Sex", [1.0, 2.0], format_func=lambda x: "Male" if x == 1.0 else "Female")

# Build respondent dataframe
input_data = pd.DataFrame(
    [
        {
            "perceived_life_stress": float(stress),
            "community_belonging": float(belonging),
            "screen_time_non_workday": float(screen_time),
            "perceived_general_health": float(health),
            "age_group": float(age),
            "household_income_group": float(income),
            "sex": float(sex),
        }
    ]
)

# Run Inference
risk_prob = pipeline.predict_proba(input_data)[0][1] * 100

st.markdown("---")
st.header("Predicted Risk Assessment")

m_col1, m_col2 = st.columns([1, 2])

with m_col1:
    if risk_prob > 70:
        st.error(f"### High Vulnerability: **{risk_prob:.1f}%**")
    elif risk_prob > 40:
        st.warning(f"### Moderate Risk: **{risk_prob:.1f}%**")
    else:
        st.success(f"### Low Vulnerability: **{risk_prob:.1f}%**")

with m_col2:
    st.progress(int(risk_prob))
    st.caption("Probability of reporting Fair or Poor perceived mental health.")

# Live Counterfactual What-Ifs
st.markdown("---")
st.subheader("Instant Intervention Simulator")
c_col1, c_col2, c_col3 = st.columns(3)

# Simulation 1: Cut screen time to <2 hrs
data_screen = input_data.copy()
data_screen["screen_time_non_workday"] = 1.0
risk_screen = pipeline.predict_proba(data_screen)[0][1] * 100

# Simulation 2: Maximize community belonging
data_belong = input_data.copy()
data_belong["community_belonging"] = 1.0
risk_belong = pipeline.predict_proba(data_belong)[0][1] * 100

# Simulation 3: Stress reduction
data_stress = input_data.copy()
data_stress["perceived_life_stress"] = 1.0
risk_stress = pipeline.predict_proba(data_stress)[0][1] * 100

c_col1.metric("If Screen Time Dropped to <2h", f"{risk_screen:.1f}%", f"{risk_screen - risk_prob:.1f}%")
c_col2.metric("If Community Connection Maximized", f"{risk_belong:.1f}%", f"{risk_belong - risk_prob:.1f}%")
c_col3.metric("If Stress Reduced to Minimum", f"{risk_stress:.1f}%", f"{risk_stress - risk_prob:.1f}%")

"""
How to run
install streamlit if you haven't already:
pip install streamlit
Then run the app with: 
streamlit run app.py
"""