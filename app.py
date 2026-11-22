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

/* Main Background */
.stApp {
    background: linear-gradient(135deg, #0B1020, #111827);
    color: #FFFFFF;
}

.block-container {
    padding-top: 2rem;
    max-width: 1250px;
}

/* Main Heading */
h1 {
    color: #60A5FA !important;
    font-weight: 800 !important;
}

h2, h3 {
    color: #F8FAFC !important;
}

.subtitle {
    color: #94A3B8;
    font-size: 17px;
}

/* Metric Cards */
div[data-testid="stMetric"] {
    background: #1E293B !important;
    padding: 22px !important;
    border-radius: 16px !important;
    border: 1px solid #334155 !important;
}

/* Metric Labels */
div[data-testid="stMetricLabel"],
div[data-testid="stMetricLabel"] p {
    color: #CBD5E1 !important;
    font-size: 16px !important;
}

/* Metric Values */
div[data-testid="stMetricValue"],
div[data-testid="stMetricValue"] div {
    color: #FFFFFF !important;
    font-size: 36px !important;
    font-weight: 800 !important;
}

/* Metric Delta */
div[data-testid="stMetricDelta"] {
    color: #38BDF8 !important;
}

/* All Input Labels */
.stNumberInput label,
.stSelectbox label {
    color: #E2E8F0 !important;
}

/* Input Fields */
.stNumberInput input {
    background: #1E293B !important;
    color: #FFFFFF !important;
    border: 1px solid #475569 !important;
    border-radius: 8px !important;
}

/* Form Container */
div[data-testid="stForm"] {
    background: #172033;
    padding: 25px;
    border-radius: 16px;
    border: 1px solid #334155;
}

/* Buttons */
.stButton button,
.stFormSubmitButton button {
    background: linear-gradient(90deg, #2563EB, #06B6D4) !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: 10px !important;
    height: 50px !important;
    font-weight: bold !important;
    width: 100%;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #111827;
}

section[data-testid="stSidebar"] * {
    color: #E2E8F0;
}

/* Expanders */
div[data-testid="stExpander"] {
    background: #1E293B;
    border: 1px solid #334155;
    border-radius: 12px;
}

/* Dataframe */
div[data-testid="stDataFrame"] {
    border-radius: 10px;
}

/* Footer */
.footer {
    text-align: center;
    color: #64748B;
    padding: 20px;
    font-size: 14px;
}

</style>
""", unsafe_allow_html=True)


# ---------------- LOAD MODEL AND SCALER ----------------

@st.cache_resource
def load_models():

    model = joblib.load("Machine_Failure.pkl")
    scaler = joblib.load("Machine_Failure_Scaler.pkl")

    return model, scaler


try:
    model, scaler = load_models()

except Exception as e:
    st.error(f"Model Loading Error: {e}")
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

st.write(
    "Enter the machine sensor readings to predict its condition."
)

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

    input_data = pd.DataFrame(
        [[
            air_temp,
            process_temp,
            rotational_speed,
            torque,
            tool_wear
        ]],
        columns=feature_names
    )

    # Scale input using the trained scaler
    scaled_data = scaler.transform(input_data)

    # Predict machine condition
    prediction = int(model.predict(scaled_data)[0])

    # Calculate probabilities
    failure_index = list(model.classes_).index(1)

    failure_probability = (
        model.predict_proba(scaled_data)[0][failure_index] * 100
    )

    normal_probability = 100 - failure_probability


    # ---------------- RESULTS ----------------

    st.divider()

    st.subheader("📊 Prediction Results")

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


    # ---------------- STATUS ----------------

    if prediction == 1:

        st.error(
            "⚠️ MACHINE FAILURE DETECTED"
        )

        st.warning(
            "The model predicts a possible machine failure. "
            "Inspection and maintenance are recommended."
        )

    else:

        st.success(
            "✅ NO MACHINE FAILURE PREDICTED"
        )

        st.info(
            "The model predicts that the machine is operating "
            "without a predicted failure."
        )


    # ---------------- GAUGE CHART ----------------

    st.subheader("Failure Probability Analysis")

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=failure_probability,
            number={
                "suffix": "%",
                "font": {
                    "color": "#FFFFFF",
                    "size": 40
                }
            },
            title={
                "text": "Machine Failure Probability",
                "font": {
                    "color": "#E2E8F0",
                    "size": 20
                }
            },
            gauge={
                "axis": {
                    "range": [0, 100],
                    "tickcolor": "#FFFFFF"
                },
                "bar": {
                    "color": "#38BDF8"
                },
                "bgcolor": "#1E293B",
                "borderwidth": 1,
                "bordercolor": "#475569",
                "steps": [
                    {
                        "range": [0, 30],
                        "color": "#166534"
                    },
                    {
                        "range": [30, 70],
                        "color": "#854D0E"
                    },
                    {
                        "range": [70, 100],
                        "color": "#991B1B"
                    }
                ]
            }
        )
    )

    fig.update_layout(
        height=350,
        paper_bgcolor="#111827",
        font_color="#FFFFFF",
        margin=dict(t=60, b=20, l=30, r=30)
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


    # ---------------- INPUT SUMMARY ----------------

    with st.expander("📋 View Sensor Input Details"):

        st.dataframe(
            input_data,
            use_container_width=True,
            hide_index=True
        )


# ---------------- FOOTER ----------------

st.divider()

st.markdown(
    '<div class="footer">'
    'Machine Failure Detection | Powered by Machine Learning'
    '</div>',
    unsafe_allow_html=True
)
