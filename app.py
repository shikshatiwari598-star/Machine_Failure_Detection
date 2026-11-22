import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

# ---------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------
st.set_page_config(
    page_title="PredictIQ | Machine Failure Analytics",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------
# MODERN DARK UI
# ---------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}
.stApp {
    background:
      radial-gradient(circle at 10% 0%, rgba(37,99,235,.18), transparent 28%),
      radial-gradient(circle at 90% 5%, rgba(14,165,233,.12), transparent 25%),
      #080d18;
    color: #e5edf8;
}
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #101a2d 0%, #0b1220 100%);
    border-right: 1px solid rgba(148,163,184,.13);
}
.block-container { padding-top: 2rem; padding-bottom: 3rem; max-width: 1500px; }
h1, h2, h3 { font-family: 'Space Grotesk', sans-serif !important; color: #f8fafc; }
p, label, span { color: #cbd5e1; }
.hero {
    padding: 28px 32px;
    border: 1px solid rgba(96,165,250,.25);
    border-radius: 24px;
    background: linear-gradient(115deg, rgba(30,64,175,.34), rgba(15,23,42,.88) 55%, rgba(8,145,178,.18));
    box-shadow: 0 18px 55px rgba(0,0,0,.22);
    margin-bottom: 22px;
}
.eyebrow {
    color: #7dd3fc; text-transform: uppercase; letter-spacing: 2px;
    font-size: 11px; font-weight: 700;
}
.hero-title { font-family: 'Space Grotesk'; font-size: 36px; font-weight: 700; color: white; margin: 8px 0 5px; }
.hero-sub { color: #b8c7dc; font-size: 15px; }
.metric-card {
    background: linear-gradient(145deg, rgba(30,41,59,.92), rgba(15,23,42,.92));
    border: 1px solid rgba(148,163,184,.16);
    border-radius: 18px; padding: 18px 20px; min-height: 112px;
}
.metric-label { color: #94a3b8; font-size: 12px; text-transform: uppercase; letter-spacing: 1px; }
.metric-value { color: #f8fafc; font-family: 'Space Grotesk'; font-size: 29px; font-weight: 700; margin-top: 7px; }
.metric-note { color: #7dd3fc; font-size: 12px; margin-top: 3px; }
.section-card {
    background: rgba(15,23,42,.78); border: 1px solid rgba(148,163,184,.14);
    border-radius: 18px; padding: 20px;
}
div[data-testid="stForm"] {
    background: rgba(15,23,42,.75); border: 1px solid rgba(148,163,184,.16);
    padding: 22px; border-radius: 18px;
}
.stButton > button, div[data-testid="stFormSubmitButton"] > button {
    background: linear-gradient(90deg, #2563eb, #0891b2);
    color: white; border: 0; border-radius: 10px; font-weight: 700;
    padding: .55rem 1.2rem;
}
.stButton > button:hover, div[data-testid="stFormSubmitButton"] > button:hover {
    border: 0; color: white; filter: brightness(1.12);
}
div[data-testid="stDataFrame"] { border: 1px solid rgba(148,163,184,.18); border-radius: 12px; }
div[data-testid="stAlert"] { border-radius: 12px; }
hr { border-color: rgba(148,163,184,.16); }
.small-muted { color: #94a3b8; font-size: 12px; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# DATA + MODEL
# ---------------------------------------------------------
@st.cache_data
def load_data():
    data = pd.read_csv("Machine_Failure_Balanced.csv")
    return data

@st.cache_resource
def train_model(data):
    # IDs and failure-mode flags are excluded.
    # Failure-mode flags can reveal the target and cause target leakage.
    feature_cols = [
        "Type",
        "Air temperature [K]",
        "Process temperature [K]",
        "Rotational speed [rpm]",
        "Torque [Nm]",
        "Tool wear [min]"
    ]
    target_col = "Machine failure"

    X = data[feature_cols].copy()
    y = data[target_col].astype(int).copy()

    numeric_features = feature_cols[1:]
    categorical_features = ["Type"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), numeric_features),
            ("category", OneHotEncoder(handle_unknown="ignore"), categorical_features)
        ]
    )

    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(
            class_weight="balanced",
            max_iter=1500,
            random_state=42
        ))
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)
    probabilities = pipeline.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions, zero_division=0),
        "recall": recall_score(y_test, predictions, zero_division=0),
        "f1": f1_score(y_test, predictions, zero_division=0),
        "matrix": confusion_matrix(y_test, predictions),
        "y_test": y_test,
        "predictions": predictions,
        "probabilities": probabilities,
        "feature_cols": feature_cols
    }
    return pipeline, metrics

try:
    df = load_data()
except FileNotFoundError:
    st.error("Dataset file not found. Keep Machine_Failure_Balanced.csv in the same folder as app.py.")
    st.stop()

required_cols = [
    "Type", "Air temperature [K]", "Process temperature [K]",
    "Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]",
    "Machine failure"
]
missing = [col for col in required_cols if col not in df.columns]
if missing:
    st.error(f"Dataset is missing required columns: {missing}")
    st.stop()

model, results = train_model(df)

# ---------------------------------------------------------
# SIDEBAR NAVIGATION
# ---------------------------------------------------------
with st.sidebar:
    st.markdown("## ⚙️ PredictIQ")
    st.caption("INDUSTRIAL AI • MACHINE HEALTH")
    st.markdown("---")
    page = st.radio(
        "NAVIGATION",
        ["Overview", "Live Prediction", "Model Performance", "Dataset Explorer"],
        label_visibility="visible"
    )
    st.markdown("---")
    st.markdown("**Model**")
    st.caption("Logistic Regression")
    st.caption("Balanced training dataset")
    st.markdown("---")
    st.caption("PredictIQ • Predictive Maintenance")

# ---------------------------------------------------------
# HEADER
# ---------------------------------------------------------
st.markdown("""
<div class="hero">
  <div class="eyebrow">Predictive maintenance intelligence</div>
  <div class="hero-title">Machine Failure Detection</div>
  <div class="hero-sub">Turn machine sensor readings into clear, data-driven maintenance signals.</div>
</div>
""", unsafe_allow_html=True)

# Reusable KPI
def metric_card(label, value, note):
    st.markdown(
        f'<div class="metric-card"><div class="metric-label">{label}</div>'
        f'<div class="metric-value">{value}</div><div class="metric-note">{note}</div></div>',
        unsafe_allow_html=True
    )

# ---------------------------------------------------------
# OVERVIEW
# ---------------------------------------------------------
if page == "Overview":
    st.subheader("System overview")
    total = len(df)
    failures = int((df["Machine failure"] == 1).sum())
    normal = int((df["Machine failure"] == 0).sum())

    c1, c2, c3, c4 = st.columns(4)
    with c1: metric_card("Total records", f"{total:,}", "Balanced dataset")
    with c2: metric_card("Failure records", f"{failures:,}", "Class 1")
    with c3: metric_card("Normal records", f"{normal:,}", "Class 0")
    with c4: metric_card("Model accuracy", f"{results['accuracy']*100:.1f}%", "Hold-out test set")

    st.markdown("")
    left, right = st.columns([1.05, 1])
    with left:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown("### Class distribution")
        class_df = df["Machine failure"].value_counts().rename_axis("Class").reset_index(name="Records")
        class_df["Class"] = class_df["Class"].map({0: "No Failure", 1: "Failure"})
        fig = px.pie(class_df, names="Class", values="Records", hole=.68,
                     color="Class", color_discrete_map={"No Failure":"#38bdf8", "Failure":"#818cf8"})
        fig.update_traces(textposition="outside", textinfo="label+value")
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color="#dbeafe", legend_title_text=""
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with right:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown("### Sensor snapshot")
        sensor = df[["Air temperature [K]", "Process temperature [K]",
                     "Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]"]].describe().loc[["mean","min","max"]].T
        sensor.columns = ["Average", "Minimum", "Maximum"]
        st.dataframe(sensor.round(2), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.info("This dataset is intentionally balanced (339 No Failure + 339 Failure). Its class proportions do not represent the original real-world failure rate.")

# ---------------------------------------------------------
# LIVE PREDICTION
# ---------------------------------------------------------
elif page == "Live Prediction":
    st.subheader("Run a machine health prediction")
    st.write("Enter sensor readings below. The model estimates whether the machine is likely to fail.")
    st.markdown("")

    with st.form("prediction_form"):
        a, b = st.columns(2)
        with a:
            machine_type = st.selectbox("Machine type", ["L", "M", "H"], index=0)
            air_temp = st.number_input("Air temperature (K)", min_value=295.0, max_value=305.0, value=300.0, step=0.1)
            process_temp = st.number_input("Process temperature (K)", min_value=305.0, max_value=315.0, value=310.0, step=0.1)
        with b:
            rpm = st.number_input("Rotational speed (rpm)", min_value=1000, max_value=3000, value=1500, step=10)
            torque = st.number_input("Torque (Nm)", min_value=3.0, max_value=80.0, value=40.0, step=0.1)
            tool_wear = st.number_input("Tool wear (min)", min_value=0, max_value=300, value=100, step=1)

        submitted = st.form_submit_button("Analyze machine", use_container_width=True)

    if submitted:
        input_df = pd.DataFrame([{
            "Type": machine_type,
            "Air temperature [K]": air_temp,
            "Process temperature [K]": process_temp,
            "Rotational speed [rpm]": rpm,
            "Torque [Nm]": torque,
            "Tool wear [min]": tool_wear
        }])
        probability = float(model.predict_proba(input_df)[0, 1])
        prediction = int(probability >= 0.5)

        st.markdown("---")
        r1, r2 = st.columns([1, 1.2])
        with r1:
            if prediction == 1:
                st.error("⚠️ Potential machine failure")
                st.metric("Estimated failure probability", f"{probability*100:.1f}%")
                st.write("Recommended action: inspect the machine and review its sensor readings.")
            else:
                st.success("✓ No failure predicted")
                st.metric("Estimated failure probability", f"{probability*100:.1f}%")
                st.write("Recommended action: continue routine monitoring and scheduled maintenance.")
        with r2:
            gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=probability*100,
                number={"suffix":"%", "font":{"color":"#f8fafc","size":34}},
                title={"text":"Failure probability","font":{"color":"#cbd5e1","size":16}},
                gauge={
                    "axis":{"range":[0,100], "tickcolor":"#94a3b8"},
                    "bar":{"color":"#38bdf8"},
                    "bgcolor":"rgba(15,23,42,.5)",
                    "bordercolor":"rgba(148,163,184,.25)",
                    "steps":[
                        {"range":[0,50],"color":"rgba(14,165,233,.12)"},
                        {"range":[50,100],"color":"rgba(244,63,94,.16)"}
                    ],
                    "threshold":{"line":{"color":"#f97316","width":4},"thickness":.8,"value":50}
                }
            ))
            gauge.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color="#e2e8f0", height=260, margin=dict(l=25,r=25,t=55,b=10))
            st.plotly_chart(gauge, use_container_width=True)

    st.caption("Prediction uses a 0.50 probability threshold. This is an educational model, not a certified industrial safety system.")

# ---------------------------------------------------------
# MODEL PERFORMANCE
# ---------------------------------------------------------
elif page == "Model Performance":
    st.subheader("Model performance")
    st.caption("Metrics are calculated on a stratified 20% hold-out test split.")
    m1, m2, m3, m4 = st.columns(4)
    with m1: metric_card("Accuracy", f"{results['accuracy']*100:.2f}%", "Overall correct predictions")
    with m2: metric_card("Precision", f"{results['precision']*100:.2f}%", "Predicted failures that were correct")
    with m3: metric_card("Recall", f"{results['recall']*100:.2f}%", "Actual failures detected")
    with m4: metric_card("F1 score", f"{results['f1']*100:.2f}%", "Precision-recall balance")

    st.markdown("")
    left, right = st.columns([1, 1])
    with left:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown("### Confusion matrix")
        cm = results["matrix"]
        cm_fig = px.imshow(
            cm, text_auto=True, aspect="auto",
            x=["Predicted: No Failure", "Predicted: Failure"],
            y=["Actual: No Failure", "Actual: Failure"],
            color_continuous_scale="Blues"
        )
        cm_fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color="#e2e8f0", coloraxis_showscale=False,
            xaxis_title="", yaxis_title=""
        )
        st.plotly_chart(cm_fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with right:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown("### Classification report")
        report = classification_report(
            results["y_test"], results["predictions"],
            target_names=["No Failure", "Failure"], output_dict=True, zero_division=0
        )
        report_df = pd.DataFrame(report).T
        st.dataframe(report_df.round(3), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.warning("Because this model is evaluated on an artificially balanced sample, do not interpret these metrics as expected production performance. Validate on data that preserves the real-world failure rate before deployment.")

# ---------------------------------------------------------
# DATASET EXPLORER
# ---------------------------------------------------------
elif page == "Dataset Explorer":
    st.subheader("Dataset explorer")
    st.write("Explore the balanced machine sensor dataset used by this application.")

    search = st.text_input("Search Product ID or Type", placeholder="e.g. M or L...")
    view_df = df.copy()
    if search:
        view_df = view_df[
            view_df["Product ID"].astype(str).str.contains(search, case=False, na=False) |
            view_df["Type"].astype(str).str.contains(search, case=False, na=False)
        ]

    class_filter = st.multiselect(
        "Filter by machine status",
        ["No Failure", "Failure"],
        default=["No Failure", "Failure"]
    )
    selected_values = []
    if "No Failure" in class_filter: selected_values.append(0)
    if "Failure" in class_filter: selected_values.append(1)
    view_df = view_df[view_df["Machine failure"].isin(selected_values)]

    st.caption(f"Showing {len(view_df):,} records")
    st.dataframe(view_df, use_container_width=True, height=430)

    csv = view_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download filtered CSV",
        data=csv,
        file_name="machine_failure_filtered.csv",
        mime="text/csv"
    )

st.markdown("---")
st.markdown(
    '<div class="small-muted">PredictIQ • Machine Failure Detection • Built with Streamlit and Logistic Regression</div>',
    unsafe_allow_html=True
)
