import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go

# ---------------- PAGE CONFIG ----------------

st.set_page_config(
    page_title="Machine Failure Detection",
    page_icon="⚙️",
    layout="wide"
)

# ---------------- CUSTOM CSS ----------------

st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #0b1020, #111827);
    color: white;
}

.block-container {
    padding-top: 2rem;
    max-width: 1200px;
}

h1 {
    color: #60A5FA;
    font-weight: 800;
}

.subtitle {
    color: #94A3B8;
    font-size: 17px;
}

div[data-testid="stMetric"] {
    background: #1E293B;
    padding: 20px;
    border-radius: 15px;
    border: 1px solid #334155;
}

.stButton button {
    background: linear-gradient(90deg, #2563EB, #06B6D4);
    color: white;
    border: none;
    border-radius: 10px;
    height: 50px;
    font-weight: bold;
    width: 100%;
}

.stButton button:hover {
    background: #0891B2;
    color: white;
}

div[data-testid="stForm"] {
    background: #172033;
    padding: 25px;
    border-radius: 15px;
    border: 1px solid #334155;
}

</style>
""", unsafe_allow_html=True)

# ---------------- LOAD MODEL ----------------

@st.cache_resource
def load_models():
    model = joblib.load("Machine_Failure.pkl")
    scaler = joblib.load("Machine_Failure_Scaler.pkl")
    return model, scaler

try:
    model, scaler = load_models()
except Exception as e:
    st.error(f"Model loading error: {e}")
    st.stop()

# ---------------- HEADER ----------------

st.title("⚙️ Machine Failure Detection")

st.markdown(
    '<p class="subtitle">AI Powered Predictive Maintenance System | Logistic Regression</p>',
    unsafe_allow_html=True
)

st.divider()

# ---------------- SIDEBAR ----------------

with st.sidebar:
    st.title("⚙️ About Project")

    st.write("""
    This application predicts whether
    an industrial machine is likely
    to experience failure using
    sensor-based measurements.
    """)

    st.divider()

    st.subheader("Machine Learning")
    st.write("Algorithm: Logistic Regression")
    st.write("Dataset: Balanced Machine Failure")
    st.write("Input Features: 5")

    st.divider()

    st.caption("Predictive Maintenance System")

# ---------------- INPUT SECTION ----------------

st.subheader("📊 Machine Sensor Parameters")

st.write("Enter the current machine sensor readings below.")

with st.form("machine_form"):

    col1, col2 = st.columns(2)

    with col1:
        air_temp = st.number_input(
            "Air Temperature (K)",
            min_value=295.0,
            max_value=310.0,
            value=300.0,
            step=0.1
        )

        process_temp = st.number_input(
            "Process Temperature (K)",
            min_value=306.0,
            max_value=320.0,
            value=310.0,
            step=0.1
        )

        rotational_speed = st.number_input(
            "Rotational Speed (rpm)",
            min_value=500,
            max_value=4000,
            value=1500
        )

    with col2:
        torque = st.number_input(
            "Torque (Nm)",
            min_value=1.0,
            max_value=100.0,
            value=40.0,
            step=0.1
        )

        tool_wear = st.number_input(
            "Tool Wear (min)",
            min_value=0,
            max_value=300,
            value=100
        )

    submit = st.form_submit_button(
        "🔍 Predict Machine Condition"
    )

# ---------------- PREDICTION ----------------

if submit:

    feature_names = [
        'Air temperature [K]',
        'Process temperature [K]',
        'Rotational speed [rpm]',
        'Torque [Nm]',
        'Tool wear [min]'
    ]

    input_data = pd.DataFrame([[
        air_temp,
        process_temp,
        rotational_speed,
        torque,
        tool_wear
    ]], columns=feature_names)

    # Apply the same scaler used during training
    scaled_data = scaler.transform(input_data)

    # Predict machine condition
    prediction = int(model.predict(scaled_data)[0])

    # Get failure probability using correct class index
    failure_index = list(model.classes_).index(1)

    failure_probability = model.predict_proba(
        scaled_data
    )[0][failure_index] * 100

    normal_probability = 100 - failure_probability

    st.divider()
    st.subheader("📈 Prediction Results")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Failure Probability",
            f"{failure_probability:.2f}%"
        )

    with col2:
        st.metric(
            "Normal Probability",
            f"{normal_probability:.2f}%"
        )

    with col3:
        st.metric(
            "Model Prediction",
            "Failure" if prediction == 1 else "Normal"
        )

    # ---------------- RESULT CARD ----------------

    if prediction == 1:

        st.error(
            "⚠️ MACHINE FAILURE DETECTED"
        )

        st.warning(
            "The model predicts that the machine "
            "may experience a failure. Inspection is recommended."
        )

    else:

        st.success(
            "✅ NO MACHINE FAILURE PREDICTED"
        )

        st.info(
            "The model predicts that the machine "
            "is operating without a predicted failure."
        )

    # ---------------- PROBABILITY CHART ----------------

    st.subheader("Failure Probability Analysis")

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=failure_probability,
            number={"suffix": "%"},
            title={"text": "Failure Probability"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#38BDF8"},
                "steps": [
                    {"range": [0, 30], "color": "#166534"},
                    {"range": [30, 70], "color": "#854D0E"},
                    {"range": [70, 100], "color": "#991B1B"}
                ]
            }
        )
    )

    fig.update_layout(
        height=350,
        paper_bgcolor="#111827",
        font_color="white"
    )

    st.plotly_chart(fig, use_container_width=True)

    # ---------------- INPUT SUMMARY ----------------

    with st.expander("View Entered Sensor Details"):

        st.dataframe(
            input_data,
            use_container_width=True
        )

st.divider()

st.markdown(
    "<center style='color:#64748B'>"
    "Machine Failure Detection | Powered by Machine Learning"
    "</center>",
    unsafe_allow_html=True
)
