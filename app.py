import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go

# ==========================================
# PAGE CONFIG
# ==========================================

st.set_page_config(
    page_title="Machine Failure Detection",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ==========================================
# CUSTOM CSS
# ==========================================

st.markdown("""
<style>

/* Hide Sidebar */
[data-testid="stSidebar"] {
    display: none;
}

/* Main Screen */
.stApp {
    background: #0B1120;
    color: #FFFFFF;
}

.block-container {
    max-width: 1450px;
    padding: 0.7rem 1.5rem 0.5rem 1.5rem;
}

/* Reduce vertical spacing */
div[data-testid="stVerticalBlock"] {
    gap: 0.55rem;
}

h1 {
    color: #60A5FA !important;
    font-size: 30px !important;
    margin-bottom: 0 !important;
}

h2, h3 {
    color: #F8FAFC !important;
}

p, label {
    color: #CBD5E1 !important;
}

/* Header */
.subtitle {
    color: #94A3B8 !important;
    font-size: 13px;
    margin-top: 2px;
}

/* Animated Machine */
.machine-animation {
    height: 75px;
    width: 100%;
    background: linear-gradient(120deg, #172554, #1E293B);
    border: 1px solid #334155;
    border-radius: 12px;
    position: relative;
    overflow: hidden;
}

.moving-machine {
    position: absolute;
    top: 13px;
    left: -65px;
    font-size: 31px;
    white-space: nowrap;
    animation: machineMove 7s linear infinite;
}

.machine-gear {
    display: inline-block;
    animation: gearRotate 2s linear infinite;
}

.machine-track {
    position: absolute;
    bottom: 12px;
    width: 200%;
    border-bottom: 2px dashed #38BDF8;
    animation: trackMove 2s linear infinite;
}

.machine-light {
    position: absolute;
    top: 9px;
    right: 10px;
    height: 8px;
    width: 8px;
    background: #22C55E;
    border-radius: 50%;
    box-shadow: 0 0 10px #22C55E;
    animation: blink 1s infinite alternate;
}

@keyframes machineMove {
    from { left: -65px; }
    to { left: 110%; }
}

@keyframes gearRotate {
    to { transform: rotate(360deg); }
}

@keyframes trackMove {
    to { transform: translateX(-50%); }
}

@keyframes blink {
    from { opacity: 0.3; }
    to { opacity: 1; }
}

/* Panels */
.panel {
    background: #172033;
    border: 1px solid #334155;
    border-radius: 14px;
    padding: 14px;
    min-height: 350px;
}

.panel-title {
    font-size: 18px;
    color: #F8FAFC;
    font-weight: 700;
    margin-bottom: 8px;
}

/* Input Labels */
.stNumberInput label {
    color: #CBD5E1 !important;
    font-size: 12px !important;
}

/* Input Boxes */
.stNumberInput input {
    background: #0F172A !important;
    color: #FFFFFF !important;
    -webkit-text-fill-color: #FFFFFF !important;
    border: 1px solid #475569 !important;
    border-radius: 7px !important;
    min-height: 35px !important;
}

/* Submit Button */
.stFormSubmitButton button {
    background: linear-gradient(90deg, #2563EB, #06B6D4) !important;
    color: white !important;
    -webkit-text-fill-color: white !important;
    border: none !important;
    border-radius: 8px !important;
    height: 42px !important;
    font-weight: bold !important;
    width: 100%;
    margin-top: 8px;
}

/* Result Cards */
.result-card {
    background: #1E293B;
    border: 1px solid #475569;
    border-radius: 10px;
    padding: 12px 8px;
    text-align: center;
    min-height: 85px;
}

.result-label {
    color: #CBD5E1 !important;
    font-size: 12px;
    margin-bottom: 9px;
}

.result-value {
    color: #FFFFFF !important;
    font-size: 22px;
    font-weight: 800;
}

/* Status */
.status-normal {
    background: #064E3B;
    color: #A7F3D0 !important;
    border: 1px solid #10B981;
    padding: 11px;
    border-radius: 9px;
    text-align: center;
    font-weight: bold;
}

.status-failure {
    background: #7F1D1D;
    color: #FECACA !important;
    border: 1px solid #EF4444;
    padding: 11px;
    border-radius: 9px;
    text-align: center;
    font-weight: bold;
}

.info-card {
    background: #1E293B;
    border: 1px solid #334155;
    padding: 15px;
    border-radius: 10px;
    color: #CBD5E1 !important;
    text-align: center;
    font-size: 13px;
}

/* Footer */
.footer {
    text-align: center;
    color: #64748B;
    font-size: 11px;
    padding-top: 4px;
}

/* Smaller screens */
@media (max-width: 700px) {
    .block-container {
        padding: 0.5rem;
    }

    h1 {
        font-size: 23px !important;
    }
}

</style>
""", unsafe_allow_html=True)


# ==========================================
# LOAD MODEL AND SCALER
# ==========================================

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


# ==========================================
# HEADER
# ==========================================

header1, header2 = st.columns([1.6, 1])

with header1:

    st.markdown(
        "<h1>⚙️ Machine Failure Detection</h1>",
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">'
        'AI Powered Predictive Maintenance | Logistic Regression'
        '</div>',
        unsafe_allow_html=True
    )

with header2:

    st.markdown(
        '<div class="machine-animation">'
        '<div class="machine-light"></div>'
        '<div class="moving-machine">'
        '🏭<span class="machine-gear">⚙️</span>'
        '</div>'
        '<div class="machine-track"></div>'
        '</div>',
        unsafe_allow_html=True
    )


st.divider()


# ==========================================
# MAIN TWO COLUMN LAYOUT
# ==========================================

left, right = st.columns([1, 1.25], gap="medium")


# ==========================================
# LEFT SIDE - SENSOR INPUT
# ==========================================

with left:

    st.markdown(
        '<div class="panel-title">📊 Machine Sensor Inputs</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">Enter machine operating parameters</div>',
        unsafe_allow_html=True
    )

    with st.form("machine_form"):

        input_col1, input_col2 = st.columns(2)

        with input_col1:

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

        with input_col2:

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

            st.markdown(
                '<div style="height:48px;"></div>',
                unsafe_allow_html=True
            )

        submit = st.form_submit_button(
            "🔍 ANALYZE MACHINE"
        )

    st.markdown(
        '<div class="subtitle">'
        'Algorithm: Logistic Regression | 5 Sensor Features'
        '</div>',
        unsafe_allow_html=True
    )


# ==========================================
# RIGHT SIDE - PREDICTION RESULTS
# ==========================================

with right:

    st.markdown(
        '<div class="panel-title">📈 Prediction Dashboard</div>',
        unsafe_allow_html=True
    )

    if submit:

        # Prepare input data in original training order

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

        # Apply trained scaler

        scaled_data = scaler.transform(input_data)

        # Predict machine failure

        prediction = int(model.predict(scaled_data)[0])

        # Calculate class probabilities

        probabilities = model.predict_proba(scaled_data)[0]

        failure_index = list(model.classes_).index(1)
        normal_index = list(model.classes_).index(0)

        failure_probability = probabilities[failure_index] * 100
        normal_probability = probabilities[normal_index] * 100

        # Result cards

        st.markdown(
            f"""
            <div style="display:flex; gap:8px; margin-top:12px;">

                <div class="result-card" style="flex:1;">
                    <div class="result-label">Failure Probability</div>
                    <div class="result-value">{failure_probability:.2f}%</div>
                </div>

                <div class="result-card" style="flex:1;">
                    <div class="result-label">Normal Probability</div>
                    <div class="result-value">{normal_probability:.2f}%</div>
                </div>

                <div class="result-card" style="flex:1;">
                    <div class="result-label">Prediction</div>
                    <div class="result-value">
                        {"Failure" if prediction == 1 else "Normal"}
                    </div>
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )

        # Machine status

        if prediction == 1:

            st.markdown(
                '<div class="status-failure">'
                '⚠️ MACHINE FAILURE DETECTED'
                '</div>',
                unsafe_allow_html=True
            )

        else:

            st.markdown(
                '<div class="status-normal">'
                '✅ NO MACHINE FAILURE PREDICTED'
                '</div>',
                unsafe_allow_html=True
            )

        # Gauge chart

        fig = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=failure_probability,

                number={
                    "suffix": "%",
                    "font": {
                        "size": 30,
                        "color": "#FFFFFF"
                    }
                },

                title={
                    "text": "Failure Probability",
                    "font": {
                        "size": 15,
                        "color": "#CBD5E1"
                    }
                },

                gauge={
                    "axis": {
                        "range": [0, 100],
                        "tickcolor": "#FFFFFF",
                        "tickfont": {
                            "color": "#CBD5E1",
                            "size": 9
                        }
                    },

                    "bar": {
                        "color": "#38BDF8",
                        "thickness": 0.25
                    },

                    "bgcolor": "#1E293B",
                    "borderwidth": 0,

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
            height=210,
            margin=dict(
                t=40,
                b=5,
                l=25,
                r=25
            ),
            paper_bgcolor="#172033",
            font_color="#FFFFFF"
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
            config={"displayModeBar": False}
        )

    
    else:

        st.markdown(
            '<div class="info-card">'
            '<div style="font-size:35px;">🏭</div>'
            '<h3 style="color:#60A5FA;">Machine Monitoring System</h3>'
            '<p>Enter sensor readings and click ANALYZE MACHINE '
            'to check the machine condition.</p>'
            '<div style="font-size:28px;margin-top:12px;">'
            '⚙️ &nbsp; 📊 &nbsp; 🔍'
            '</div>'
            '</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            '<div class="info-card" style="margin-top:10px;">'
            '<b style="color:#38BDF8;">5 Sensor Features</b>'
            '<br><br>'
            'Air Temperature • Process Temperature'
            '<br>'
            'Rotational Speed • Torque • Tool Wear'
            '</div>',
            unsafe_allow_html=True
        )


        st.markdown(
            """
            <div class="info-card" style="margin-top:10px;">
                <b style="color:#38BDF8;">5 Sensor Features</b>
                <br><br>
                Air Temperature • Process Temperature
                <br>
                Rotational Speed • Torque • Tool Wear
            </div>
            """,
            unsafe_allow_html=True
        )


# ==========================================
# FOOTER
# ==========================================

st.divider()

st.markdown(
    '<div class="footer">'
    '⚙️ Machine Failure Detection | '
    'Powered by Logistic Regression'
    '</div>',
    unsafe_allow_html=True
)
