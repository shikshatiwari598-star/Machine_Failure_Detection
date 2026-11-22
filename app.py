import os
import warnings

import joblib
import numpy as np
import plotly.graph_objects as go
import streamlit as st

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────
#  PAGE CONFIG
# ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Machine Failure AI",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ──────────────────────────────────────────────────────────────
#  MODEL + SCALING CONSTANTS
#  Model (LogisticRegression) standardized data par train hua hai, par scaler
#  pkl me save nahi hai. Neeche AI4I-2020 dataset ke mean/std use kiye gaye hain.
#  Agar aapke training data ke mean/std alag hain to yahan badal do.
# ──────────────────────────────────────────────────────────────
MODEL_CANDIDATES = ["Machine_Failure__1_.pkl", "Machine_Failure.pkl", "model.pkl"]

FEATURES = [
    # key, label, icon, unit, min, max, default, step, mean, std
    ("air",     "Air Temp",      "🌡️", "K",   295.0, 305.0, 299.0, 0.1, 300.00, 2.000),
    ("process", "Process Temp",  "🔥", "K",   305.0, 314.0, 309.5, 0.1, 310.00, 1.484),
    ("rpm",     "Rotation",      "⚙️", "rpm", 1100,  2900,  1500,  10,  1538.8, 179.28),
    ("torque",  "Torque",        "🔩", "Nm",  3.0,   77.0,  36.0,  0.5, 39.99,  9.97),
    ("wear",    "Tool Wear",     "🛠️", "min", 0,     255,   60,    1,   107.95, 63.65),
]
KEYS = [f[0] for f in FEATURES]
MEAN = np.array([f[8] for f in FEATURES])
STD = np.array([f[9] for f in FEATURES])

PRESETS = {
    "✅ Normal":   dict(air=299.0, process=309.5, rpm=1500, torque=36.0, wear=60),
    "⚠️ Stress":   dict(air=300.0, process=310.0, rpm=1540, torque=40.0, wear=108),
    "🚨 Critical": dict(air=303.5, process=311.5, rpm=1250, torque=70.0, wear=235),
}


@st.cache_resource
def load_model():
    here = os.path.dirname(os.path.abspath(__file__))
    for name in MODEL_CANDIDATES:
        path = os.path.join(here, name)
        if os.path.exists(path):
            return joblib.load(path)
    raise FileNotFoundError(f"Model file nahi mila. Ye naam try kiye: {MODEL_CANDIDATES}")


model = load_model()

# ──────────────────────────────────────────────────────────────
#  STYLE + ANIMATIONS
# ──────────────────────────────────────────────────────────────
CSS = """
<style>
header, footer, #MainMenu {visibility:hidden; height:0;}
.block-container {padding:0.6rem 1.2rem 0.2rem 1.2rem !important; max-width:100% !important;}
.stApp {
  background: radial-gradient(circle at 15% 10%, #16233f 0%, #0b1020 45%, #070a14 100%);
  color:#e6ecff;
}
/* moving grid overlay */
.stApp::before {
  content:""; position:fixed; inset:0; pointer-events:none; opacity:.07;
  background-image: linear-gradient(#5b8cff 1px, transparent 1px), linear-gradient(90deg,#5b8cff 1px, transparent 1px);
  background-size:42px 42px; animation: gridMove 14s linear infinite;
}
@keyframes gridMove {from{background-position:0 0;} to{background-position:42px 42px;}}
@keyframes fadeUp {from{opacity:0; transform:translateY(14px);} to{opacity:1; transform:none;}}
@keyframes shine {0%{background-position:0% 50%;} 100%{background-position:200% 50%;}}
@keyframes floaty {0%,100%{transform:translateY(0);} 50%{transform:translateY(-3px);}}
@keyframes spin {to{transform:rotate(360deg);}}
@keyframes pulseG {0%,100%{box-shadow:0 0 0 0 rgba(34,211,138,.45);} 50%{box-shadow:0 0 22px 4px rgba(34,211,138,.25);}}
@keyframes pulseA {0%,100%{box-shadow:0 0 0 0 rgba(251,191,36,.45);} 50%{box-shadow:0 0 22px 4px rgba(251,191,36,.28);}}
@keyframes pulseR {0%,100%{box-shadow:0 0 0 0 rgba(255,71,87,.6);}  50%{box-shadow:0 0 30px 8px rgba(255,71,87,.35);}}
@keyframes blink {50%{opacity:.25;}}

.hero {display:flex; align-items:center; justify-content:space-between; animation:fadeUp .6s ease both; margin-bottom:6px;}
.hero h1 {
  font-size:1.55rem; margin:0; font-weight:800; letter-spacing:.3px;
  background:linear-gradient(90deg,#5b8cff,#22d3ee,#a78bfa,#5b8cff); background-size:200% auto;
  -webkit-background-clip:text; -webkit-text-fill-color:transparent; animation:shine 5s linear infinite;
}
.hero small {color:#8fa3d6;}
.gear {display:inline-block; animation:spin 6s linear infinite; margin-right:6px; -webkit-text-fill-color:initial;}
.live {font-size:.72rem; color:#22d3ee; border:1px solid #22d3ee55; padding:2px 10px; border-radius:20px;}
.live i {display:inline-block; width:7px; height:7px; background:#22d3ee; border-radius:50%; margin-right:6px; animation:blink 1.2s infinite;}

.card {
  background:linear-gradient(145deg, rgba(255,255,255,.06), rgba(255,255,255,.02));
  border:1px solid rgba(120,150,255,.18); border-radius:16px; padding:10px 14px;
  backdrop-filter: blur(8px); animation:fadeUp .7s ease both;
}
.card h4 {margin:0 0 4px 0; font-size:.78rem; letter-spacing:1.4px; text-transform:uppercase; color:#8fa3d6;}

.frow {display:flex; justify-content:space-between; align-items:center; font-size:.78rem; margin-top:4px; margin-bottom:-14px;}
.frow .ic {display:inline-block; font-size:.95rem; margin-right:5px; animation:floaty 3s ease-in-out infinite;}
.frow b {color:#22d3ee; font-variant-numeric:tabular-nums;}

.status {border-radius:14px; padding:8px 12px; text-align:center; font-weight:800; font-size:1.05rem; letter-spacing:.5px;}
.status.safe {background:rgba(34,211,138,.12); border:1px solid #22d38a; color:#22d38a; animation:pulseG 2s infinite;}
.status.warn {background:rgba(251,191,36,.12); border:1px solid #fbbf24; color:#fbbf24; animation:pulseA 1.6s infinite;}
.status.crit {background:rgba(255,71,87,.14);  border:1px solid #ff4757; color:#ff4757; animation:pulseR 1s infinite;}
.status span {display:block; font-size:.72rem; font-weight:500; color:#c7d2f5; letter-spacing:0;}

.tiles {display:grid; grid-template-columns:repeat(5,1fr); gap:8px; margin-top:6px;}
.tile {
  background:rgba(255,255,255,.045); border:1px solid rgba(120,150,255,.15); border-radius:12px;
  padding:6px 10px; display:flex; align-items:center; gap:8px; animation:fadeUp .8s ease both;
  transition:transform .25s, border-color .25s;
}
.tile:hover {transform:translateY(-3px) scale(1.02); border-color:#5b8cff;}
.tile .ic {font-size:1.1rem; animation:floaty 3s ease-in-out infinite;}
.tile .v {font-weight:800; font-size:.95rem; line-height:1;}
.tile .l {font-size:.64rem; color:#8fa3d6; text-transform:uppercase; letter-spacing:.8px;}
.tile .z {font-size:.66rem; margin-left:auto; padding:1px 7px; border-radius:10px;}
.z.ok {background:#22d38a22; color:#22d38a;} .z.mid {background:#fbbf2422; color:#fbbf24;} .z.bad {background:#ff475722; color:#ff4757;}

.tip {font-size:.74rem; color:#b9c6ee; line-height:1.35; margin-top:4px;}

/* streamlit widgets */
div[data-testid="stSlider"] {padding-top:0; margin-bottom:-6px;}
div[data-testid="stSlider"] [role="slider"] {box-shadow:0 0 10px #5b8cff;}
.stButton>button {
  width:100%; border-radius:10px; border:1px solid rgba(120,150,255,.35); padding:2px 4px;
  background:rgba(91,140,255,.12); color:#dbe5ff; font-size:.78rem; transition:all .25s;
}
.stButton>button:hover {background:#5b8cff; color:white; transform:translateY(-2px); border-color:#5b8cff;}
div[data-testid="stVerticalBlock"] {gap:0.45rem;}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────
#  SESSION STATE
# ──────────────────────────────────────────────────────────────
for f in FEATURES:
    st.session_state.setdefault(f[0], f[6])
st.session_state.setdefault("history", [])
st.session_state.setdefault("last_inputs", None)


def apply_preset(name):
    for k, v in PRESETS[name].items():
        st.session_state[k] = v


# ──────────────────────────────────────────────────────────────
#  HEADER
# ──────────────────────────────────────────────────────────────
st.markdown(
    """
<div class="hero">
  <div>
    <h1><span class="gear">⚙️</span>Machine Failure AI · Predictive Maintenance</h1>
    <small>Logistic Regression model · real-time failure risk dashboard</small>
  </div>
  <div class="live"><i></i>LIVE PREDICTION</div>
</div>
""",
    unsafe_allow_html=True,
)

left, mid, right = st.columns([1.05, 1.25, 1.45], gap="small")

# ──────────────────────────────────────────────────────────────
#  LEFT : INPUTS
# ──────────────────────────────────────────────────────────────
with left:
    st.markdown('<div class="card"><h4>🎛️ Sensor Inputs</h4></div>', unsafe_allow_html=True)
    for (key, label, icon, unit, lo, hi, _d, step, _m, _s) in FEATURES:
        cur = st.session_state[key]
        shown = f"{cur:.1f}" if isinstance(step, float) else f"{cur}"
        st.markdown(
            f'<div class="frow"><span><span class="ic">{icon}</span>{label}</span>'
            f"<b>{shown} {unit}</b></div>",
            unsafe_allow_html=True,
        )
        st.slider(label, lo, hi, key=key, step=step, label_visibility="collapsed")

    st.markdown('<div class="tip">⚡ Quick presets</div>', unsafe_allow_html=True)
    pc = st.columns(3)
    for col, name in zip(pc, PRESETS):
        col.button(name, on_click=apply_preset, args=(name,), key=f"p_{name}")

# ──────────────────────────────────────────────────────────────
#  PREDICTION
# ──────────────────────────────────────────────────────────────
raw = np.array([st.session_state[k] for k in KEYS], dtype=float)
z = (raw - MEAN) / STD
prob = float(model.predict_proba([z])[0][1])
pred = int(prob >= 0.5)
contrib = model.coef_[0] * z  # har feature ka risk par asar

if prob < 0.35:
    level, cls, title, advice = "safe", "safe", "✅ MACHINE SAFE", "Sab parameters normal hain. Routine maintenance kaafi hai."
elif prob < 0.65:
    level, cls, title, advice = "warn", "warn", "⚠️ WARNING", "Risk badh raha hai. Inspection plan karo aur torque / wear monitor karo."
else:
    level, cls, title, advice = "crit", "crit", "🚨 FAILURE LIKELY", "Turant maintenance karo! Machine ko slow ya band karne par vichar karo."

if st.session_state["last_inputs"] != tuple(raw):
    st.session_state["history"].append(round(prob * 100, 1))
    st.session_state["history"] = st.session_state["history"][-25:]
    st.session_state["last_inputs"] = tuple(raw)

# ──────────────────────────────────────────────────────────────
#  MIDDLE : GAUGE + STATUS
# ──────────────────────────────────────────────────────────────
with mid:
    gauge_color = {"safe": "#22d38a", "warn": "#fbbf24", "crit": "#ff4757"}[level]
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=prob * 100,
            number={"suffix": "%", "font": {"size": 44, "color": gauge_color}},
            title={"text": "Failure Probability", "font": {"size": 13, "color": "#8fa3d6"}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": "#8fa3d6", "tickfont": {"size": 9}},
                "bar": {"color": gauge_color, "thickness": 0.28},
                "bgcolor": "rgba(0,0,0,0)",
                "borderwidth": 0,
                "steps": [
                    {"range": [0, 35], "color": "rgba(34,211,138,.18)"},
                    {"range": [35, 65], "color": "rgba(251,191,36,.18)"},
                    {"range": [65, 100], "color": "rgba(255,71,87,.20)"},
                ],
                "threshold": {"line": {"color": "white", "width": 3}, "thickness": 0.8, "value": prob * 100},
            },
        )
    )
    fig.update_layout(
        height=235, margin=dict(l=18, r=18, t=38, b=0),
        paper_bgcolor="rgba(0,0,0,0)", font={"color": "#e6ecff"},
        transition={"duration": 700, "easing": "cubic-in-out"},
    )
    st.markdown('<div class="card"><h4>🎯 Risk Meter</h4></div>', unsafe_allow_html=True)
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    st.markdown(
        f'<div class="status {cls}">{title}<span>Model prediction: '
        f'{"FAILURE (1)" if pred else "NO FAILURE (0)"} · confidence {max(prob, 1 - prob) * 100:.1f}%</span></div>',
        unsafe_allow_html=True,
    )
    st.markdown(f'<div class="tip">💡 {advice}</div>', unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────
#  RIGHT : RADAR + CONTRIBUTION + HISTORY
# ──────────────────────────────────────────────────────────────
PLOT_BASE = dict(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    font={"color": "#c7d2f5", "size": 10}, margin=dict(l=10, r=10, t=8, b=8),
)

with right:
    st.markdown('<div class="card"><h4>📡 Sensor Profile · Risk Drivers · Trend</h4></div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2, gap="small")

    # Radar: 50 = machine average
    names = [f[2] + " " + f[1] for f in FEATURES]
    score = np.clip((z + 3) / 6 * 100, 0, 100)
    radar = go.Figure()
    radar.add_trace(go.Scatterpolar(
        r=[50] * 6, theta=names + [names[0]], name="Normal",
        line=dict(color="#22d38a", dash="dot", width=1), hoverinfo="skip"))
    radar.add_trace(go.Scatterpolar(
        r=list(score) + [score[0]], theta=names + [names[0]], name="Now",
        fill="toself", fillcolor="rgba(91,140,255,.30)", line=dict(color="#5b8cff", width=2)))
    radar.update_layout(
        **PLOT_BASE, height=215, showlegend=False,
        polar=dict(bgcolor="rgba(0,0,0,0)",
                   radialaxis=dict(visible=False, range=[0, 100]),
                   angularaxis=dict(gridcolor="rgba(120,150,255,.2)", tickfont=dict(size=9))),
        transition={"duration": 600},
    )
    c1.plotly_chart(radar, use_container_width=True, config={"displayModeBar": False})

    # Contribution bars
    order = np.argsort(np.abs(contrib))
    bar = go.Figure(go.Bar(
        x=contrib[order], y=[FEATURES[i][2] + " " + FEATURES[i][1] for i in order], orientation="h",
        marker=dict(color=["#ff4757" if c > 0 else "#22d38a" for c in contrib[order]]),
        text=[f"{c:+.2f}" for c in contrib[order]], textposition="outside", textfont=dict(size=9),
    ))
    bar.update_layout(
        **PLOT_BASE, height=215,
        xaxis=dict(showgrid=False, zeroline=True, zerolinecolor="#8fa3d6", showticklabels=False,
                   range=[contrib.min() - 1.2, contrib.max() + 1.2]),
        yaxis=dict(tickfont=dict(size=9)),
        transition={"duration": 600},
    )
    c2.plotly_chart(bar, use_container_width=True, config={"displayModeBar": False})

    # History trend
    hist = st.session_state["history"]
    line = go.Figure()
    line.add_hrect(y0=65, y1=100, fillcolor="rgba(255,71,87,.10)", line_width=0)
    line.add_hrect(y0=35, y1=65, fillcolor="rgba(251,191,36,.08)", line_width=0)
    line.add_trace(go.Scatter(
        y=hist, mode="lines+markers", line=dict(color="#22d3ee", width=2, shape="spline"),
        marker=dict(size=5, color="#22d3ee"), fill="tozeroy", fillcolor="rgba(34,211,238,.12)"))
    line.update_layout(
        **PLOT_BASE, height=120, showlegend=False,
        xaxis=dict(showgrid=False, showticklabels=False),
        yaxis=dict(range=[0, 100], gridcolor="rgba(120,150,255,.12)", ticksuffix="%", tickfont=dict(size=9)),
    )
    st.plotly_chart(line, use_container_width=True, config={"displayModeBar": False})

# ──────────────────────────────────────────────────────────────
#  BOTTOM : KPI TILES (chhote feature icons)
# ──────────────────────────────────────────────────────────────
tiles = ""
for i, (key, label, icon, unit, *_rest) in enumerate(FEATURES):
    zi = z[i]
    zc = "ok" if abs(zi) < 1 else ("mid" if abs(zi) < 2 else "bad")
    val = f"{raw[i]:.1f}" if isinstance(FEATURES[i][7], float) else f"{int(raw[i])}"
    tiles += (
        f'<div class="tile" style="animation-delay:{i * 0.08:.2f}s"><span class="ic">{icon}</span>'
        f'<div><div class="v">{val}<small style="font-size:.6rem;color:#8fa3d6"> {unit}</small></div>'
        f'<div class="l">{label}</div></div><span class="z {zc}">z {zi:+.1f}</span></div>'
    )
st.markdown(f'<div class="tiles">{tiles}</div>', unsafe_allow_html=True)
