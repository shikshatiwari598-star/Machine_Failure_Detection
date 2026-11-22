import streamlit as st
import joblib
import numpy as np

# -----------------------------------------
# PAGE CONFIGURATION
# -----------------------------------------

st.set_page_config(
    page_title="Machine Failure AI",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------------------
# LOAD TRAINED MODEL
# -----------------------------------------

@st.cache_resource
def load_model():
    return joblib.load("Machine_Failure.pkl")

model = load_model()

# -----------------------------------------
# CUSTOM CSS
# -----------------------------------------

st.markdown("""
<style>

.stApp {
    background: linear-gradient(135deg, #080d1a, #111b30);
    color: #f1f5f9;
}

header {
    background: transparent !important;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1400px;
}

.main-title {
    font-size: 42px;
    font-weight: 800;
    color: #ffffff;
    letter-spacing: -1px;
}

.subtitle {
    color: #94a3b8;
    font-size: 16px;
    margin-bottom: 30px;
}

.section-title {
    color: #60a5fa;
    font-size: 21px;
    font-weight: 700;
    margin-bottom: 15px;
}

.metric-card {
    background: linear-gradient(145deg, #17243a, #101a2c);
    border: 1px solid #263b58;
    border-radius: 16px;
    padding: 22px;
    min-height: 115px;
}

.metric-label {
    color: #94a3b8;
    font-size: 14px;
}

.metric-value {
    color: #ffffff;
    font-size: 27px;
    font-weight: 800;
    margin-top: 8px;
}

.result-card {
    padding: 28px;
    border-radius: 18px;
    text-align: center;
    margin-top: 20px;
}

.success-card {
    background: linear-gradient(135deg, #063d35, #0b292d);
    border: 1px solid #10b981;
}

.danger-card {
    background: linear-gradient(135deg, #511b2a, #301622);
    border: 1px solid #ef4444;
}

.result-heading {
    font-size: 27px;
    font-weight: 800;
    color: white;
}

.result-description {
    color: #d1d5db;
    font-size: 15px;
}

div.stButton > button {
    width: 100%;
    height: 52px;
    border-radius: 12px;
    border: none;
    background: linear-gradient(90deg, #2563eb, #7c3aed);
    color: white;
    font-size: 17px;
    font-weight: 700;
    transition: 0.3s;
}

div.stButton > button:hover {
    background: linear-gradient(90deg, #1d4ed8, #6d28d9);
    border: none;
    color: white;
}

[data-testid="stSidebar"] {
    background: #0c1424;
    border-right: 1px solid #263b58;
}

[data-testid="stSidebar"] * {
    color: #e2e8f0;
}

div[data-baseweb="input"] input {
    background-color: #17243a;
    color: white;
}

div[data-baseweb="select"] > div {
    background-color: #17243a;
}

hr {
    border-color: #263b58;
}

</style>
""", unsafe_allow_html=True)


# -----------------------------------------
# SIDEBAR
# -----------------------------------------

with st.sidebar:

    st.markdown("## ⚙️ MACHINE AI")

    st.caption("INTELLIGENT FAILURE PREDICTION")

    st.divider()

    st.markdown("### Dashboard")

    st.markdown("""
    - 🏠 Machine Overview
    - 🔍 Failure Prediction
    - 📊 Model Insights
    """)

    st.divider()

    st.markdown("### About Model")

    st.info("""
    Algorithm: Logistic Regression

    Problem: Binary Classification

    Output: Machine Failure / No Failure
    """)

    st.caption("AI Powered Predictive Maintenance")


# -----------------------------------------
# HEADER
# -----------------------------------------

st.markdown(
    '<div class="main-title">⚙️ Machine Failure Detection</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">AI-powered predictive maintenance system | '
    'Predict machine failure before it happens.</div>',
    unsafe_allow_html=True
)

# -----------------------------------------
# TOP METRICS
# -----------------------------------------

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-label">MODEL ALGORITHM</div>
        <div class="metric-value">Logistic Regression</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-label">CLASSIFICATION TYPE</div>
        <div class="metric-value">Binary</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-label">INPUT FEATURES</div>
        <div class="metric-value">05 Features</div>
    </div>
    """, unsafe_allow_html=True)


st.divider()

# -----------------------------------------
# INPUT SECTION
# -----------------------------------------

st.markdown(
    '<div class="section-title">🔧 Machine Parameters</div>',
    unsafe_allow_html=True
)

st.write("Enter machine operating parameters to analyze its failure risk.")

col1, col2 = st.columns(2)

with col1:

    air_temp = st.number_input(
        "🌡️ Air Temperature (K)",
        min_value=250.0,
        max_value=350.0,
        value=298.1,
        step=0.1
    )

    process_temp = st.number_input(
        "🔥 Process Temperature (K)",
        min_value=250.0,
        max_value=400.0,
        value=308.6,
        step=0.1
    )

    rotational_speed = st.number_input(
        "⚙️ Rotational Speed (RPM)",
        min_value=0,
        max_value=5000,
        value=1500,
        step=10
    )

with col2:

    torque = st.number_input(
        "🔩 Torque (Nm)",
        min_value=0.0,
        max_value=150.0,
        value=40.0,
        step=0.5
    )

    tool_wear = st.number_input(
        "🛠️ Tool Wear (Minutes)",
        min_value=0,
        max_value=300,
        value=100,
        step=1
    )


st.divider()

# -----------------------------------------
# PREDICTION
# -----------------------------------------

st.markdown(
    '<div class="section-title">🚀 Failure Risk Analysis</div>',
    unsafe_allow_html=True
)

if st.button("🔍 ANALYZE MACHINE CONDITION"):

    # Feature order must match model training order
    input_data = np.array([[
        air_temp,
        process_temp,
        rotational_speed,
        torque,
        tool_wear
    ]])

    try:

        prediction = model.predict(input_data)[0]

        probabilities = model.predict_proba(input_data)[0]

        # Find probability of failure (class 1)
        failure_index = list(model.classes_).index(1)

        failure_probability = probabilities[failure_index] * 100

        normal_probability = 100 - failure_probability

        st.divider()

        result_col1, result_col2 = st.columns(2)

        with result_col1:

            st.markdown("""
            <div class="metric-card">
                <div class="metric-label">FAILURE PROBABILITY</div>
                <div class="metric-value">
            """ + f"{failure_probability:.2f}%" + """
                </div>
            </div>
            """, unsafe_allow_html=True)

        with result_col2:

            st.markdown("""
            <div class="metric-card">
                <div class="metric-label">NORMAL PROBABILITY</div>
                <div class="metric-value">
            """ + f"{normal_probability:.2f}%" + """
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("### 📊 Prediction Confidence")

        st.progress(float(failure_probability / 100))

        st.caption(
            f"Estimated failure probability: {failure_probability:.2f}%"
        )

        if prediction == 1:

            st.markdown(f"""
            <div class="result-card danger-card">
                <div class="result-heading">
                    ⚠️ MACHINE FAILURE DETECTED
                </div>
                <p class="result-description">
                    The model predicts a machine failure.
                    Immediate inspection is recommended.
                </p>
            </div>
            """, unsafe_allow_html=True)

            st.error("Recommended Action: Inspect the machine and its operating conditions.")

        else:

            st.markdown(f"""
            <div class="result-card success-card">
                <div class="result-heading">
                    ✅ MACHINE OPERATING NORMALLY
                </div>
                <p class="result-description">
                    The model predicts no machine failure
                    under the given operating parameters.
                </p>
            </div>
            """, unsafe_allow_html=True)

            st.success("Recommended Action: Continue regular monitoring and maintenance.")

        # -----------------------------------------
        # INPUT SUMMARY
        # -----------------------------------------

        st.divider()

        st.markdown("### 📋 Machine Input Summary")

        summary_col1, summary_col2 = st.columns(2)

        with summary_col1:
            st.write(f"Air Temperature: {air_temp} K")
            st.write(f"Process Temperature: {process_temp} K")
            st.write(f"Rotational Speed: {rotational_speed} RPM")

        with summary_col2:
            st.write(f"Torque: {torque} Nm")
            st.write(f"Tool Wear: {tool_wear} Minutes")

    except Exception as e:

        st.error(f"Prediction Error: {e}")


# -----------------------------------------
# FOOTER
# -----------------------------------------

st.divider()

st.markdown(
    "<center style='color:#64748b;'>"
    "Machine Failure Detection System | Powered by Machine Learning"
    "</center>",
    unsafe_allow_html=True
)
