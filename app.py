import streamlit as st
from streamlit_option_menu import option_menu
import pickle
import warnings
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import shap

warnings.filterwarnings("ignore")
    
# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="MedPredict — Maternal & Fetal Health",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# CSS — Soft Pink Medical Theme (White / Light)
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=DM+Serif+Display&display=swap');

/*To hide deploy button*/
button[kind="header"] {
    display: none !important;
}
                     
/* ── Base ── */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif !important;
}
[data-testid="stAppViewContainer"] {
    background-color: #FFF5F7 !important;
}
[data-testid="stHeader"] { background: transparent !important; }
#MainMenu, footer { visibility: hidden; }
.block-container { padding: 1.5rem 2rem 2rem 2rem !important; max-width: 1200px; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #FFFFFF !important;
    border-right: 1.5px solid #FADADD;
}
[data-testid="stSidebar"] .block-container { padding: 1rem !important; }

/* ── Typography ── */
h1, h2, h3, h4, h5, h6 { color: #111827 !important; font-weight: 700; }
p, span { color: #374151 !important; }

/* ── INPUT FIX: cursor + text visibility ──
   Root cause: Streamlit injects color:#transparent or
   webkit-text-fill-color that hides typed text.
   We override all three properties explicitly.        */
[data-testid="stNumberInput"] input,
[data-testid="stTextInput"]   input {
    background:               #FFFFFF !important;
    color:                    #111827 !important;
    -webkit-text-fill-color:  #111827 !important;
    caret-color:              #EC4899 !important;
    border:        1.5px solid #FBCFE8 !important;
    border-radius: 10px       !important;
    font-size:     14px       !important;
    font-weight:   500        !important;
    padding:       8px 12px   !important;
    box-shadow:    none       !important;
}
[data-testid="stNumberInput"] input:focus,
[data-testid="stTextInput"]   input:focus {
    border-color: #EC4899 !important;
    box-shadow:   0 0 0 3px rgba(236,72,153,0.12) !important;
    color:                   #111827 !important;
    -webkit-text-fill-color: #111827 !important;
}
[data-testid="stNumberInput"] input::placeholder,
[data-testid="stTextInput"]   input::placeholder {
    color:   #9CA3AF !important;
    opacity: 1       !important;
}

/* Number spinner buttons */
[data-testid="stNumberInput"] button {
    background: #FFF0F5 !important;
    border:     1px solid #FBCFE8 !important;
    color:      #EC4899 !important;
}

/* Input + select wrapper — kill any inherited dark bg */
[data-testid="stNumberInput"] > div,
[data-testid="stTextInput"]   > div {
    background: transparent !important;
}

/* Input labels */
[data-testid="stNumberInput"] label,
[data-testid="stTextInput"]   label,
.stNumberInput label,
.stTextInput   label {
    color:          #1F2937 !important;
    font-weight:    600     !important;
    font-size:      13px    !important;
    letter-spacing: 0.2px   !important;
}

/* ── Buttons ── */
.stButton > button {
    background:    linear-gradient(135deg, #EC4899, #F472B6) !important;
    color:         #FFFFFF   !important;
    border:        none      !important;
    border-radius: 10px      !important;
    font-weight:   600       !important;
    font-size:     14px      !important;
    padding:       10px 28px !important;
    width:         100%      !important;
    transition:    all 0.2s  !important;
}
.stButton > button:hover {
    transform:  translateY(-1px);
    box-shadow: 0 4px 14px rgba(236,72,153,0.35) !important;
}

/* ── Form submit button ── */
[data-testid="stFormSubmitButton"] > button {
    background:    linear-gradient(135deg, #EC4899, #F472B6) !important;
    color:         #FFFFFF   !important;
    border:        none      !important;
    border-radius: 10px      !important;
    font-weight:   700       !important;
    font-size:     15px      !important;
    padding:       12px 32px !important;
    width:         100%      !important;
    transition:    all 0.2s  !important;
}
[data-testid="stFormSubmitButton"] > button:hover {
    transform:  translateY(-1px);
    box-shadow: 0 6px 18px rgba(236,72,153,0.4) !important;
}

/* ── Cards ── */
.info-card {
    background:    #FFFFFF;
    border:        1px solid #FBCFE8;
    border-radius: 14px;
    padding:       20px 24px;
    margin-bottom: 16px;
    box-shadow:    0 2px 8px rgba(236,72,153,0.06);
}

/* ── Section headers ── */
.section-header {
    font-family: 'DM Serif Display', serif;
    font-size:   27px;
    font-weight: 400;
    color:       #9D174D !important;
    margin-bottom: 4px;
}
.section-sub {
    font-size:   14px;
    color:       #6B7280 !important;
    line-height: 1.6;
    margin-bottom: 22px;
}

/* ── Input group label ── */
.input-group-label {
    font-size:       11px;
    font-weight:     700;
    color:           #BE185D !important;
    text-transform:  uppercase;
    letter-spacing:  0.7px;
    margin-bottom:   4px;
}

/* ── Result boxes ── */
.result-low, .result-normal {
    background:    #ECFDF5;
    border:        2px solid #10B981;
    border-radius: 14px;
    padding:       22px 26px;
    text-align:    center;
    color:         #111827 !important;
}
.result-mid, .result-suspect {
    background:    #FFF7ED;
    border:        2px solid #F59E0B;
    border-radius: 14px;
    padding:       22px 26px;
    text-align:    center;
    color:         #111827 !important;
}
.result-high, .result-pathological {
    background:    #FFF1F2;
    border:        2px solid #F43F5E;
    border-radius: 14px;
    padding:       22px 26px;
    text-align:    center;
    color:         #111827 !important;
}
.result-title {
    font-family:   'DM Serif Display', serif;
    font-size:     22px;
    font-weight:   600;
    margin-bottom: 8px;
    margin-top:    6px;
}
.result-desc {
    font-size:   14px;
    color:       #374151 !important;
    line-height: 1.6;
    font-weight: 500;
}

/* ── Metric cards ── */
.metric-card {
    background:    #FFFFFF;
    border:        1px solid #F9A8D4;
    border-radius: 12px;
    padding:       18px 16px;
    text-align:    center;
    box-shadow:    0 4px 12px rgba(236,72,153,0.08);
}
.metric-value {
    font-size:   26px;
    font-weight: 700;
    color:       #BE185D !important;
}
.metric-label {
    font-size:  12px;
    color:      #6B7280 !important;
    margin-top: 2px;
}

/* ── Disclaimer ── */
.disclaimer {
    background:   #FFF1F2;
    border-left:  4px solid #E11D48;
    border-radius: 0 10px 10px 0;
    padding:       13px 16px;
    font-size:     13px;
    color:         #7F1D1D !important;
    font-weight:   500;
    margin-top:    16px;
}

/* ── Dataframe ── */
[data-testid="stDataFrame"] { border-radius: 10px; overflow: hidden; }
            
section[data-testid="stSidebar"] {
    width: 330px !important;
        }
section[data-testid="stSidebar"] > div {
    width: 330px !important;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# LOAD MODELS
# ─────────────────────────────────────────────
@st.cache_resource
def load_models():
    maternal_model  = pickle.load(open("model/finalized_maternal_model_v12.sav", "rb"))
    maternal_scaler = pickle.load(open("model/maternal_scaler_v12.sav", "rb"))
    fetal_model     = pickle.load(open("model/finalized_fetal_model_v1.sav", "rb"))
    fetal_scaler    = pickle.load(open("model/fetal_scaler_v1.sav", "rb"))
    return maternal_model, maternal_scaler, fetal_model, fetal_scaler

maternal_model, maternal_scaler, fetal_model, fetal_scaler = load_models()

# ─────────────────────────────────────────────
# FEATURE ENGINEERING — MATERNAL (26 features)
# ─────────────────────────────────────────────
def engineer_maternal_features(age, systolic_bp, diastolic_bp, bs, body_temp, heart_rate):
    d = {}
    d['Age']                  = age
    d['SystolicBP']           = systolic_bp
    d['DiastolicBP']          = diastolic_bp
    d['BS']                   = bs
    d['BodyTemp']              = body_temp
    d['HeartRate']             = heart_rate
    d['PulsePressure']         = systolic_bp - diastolic_bp
    d['MeanArterialPressure']  = diastolic_bp + (systolic_bp - diastolic_bp) / 3
    d['BPRatio']               = systolic_bp / (diastolic_bp + 1e-6)
    d['Hypertension']          = int(systolic_bp > 140)
    d['Hypotension']           = int(systolic_bp < 90)
    d['ShockIndex']            = heart_rate / (systolic_bp + 1e-6)
    d['Tachycardia']           = int(heart_rate > 100)
    d['Bradycardia']           = int(heart_rate < 60)
    d['HighBS']                = int(bs > 7.8)
    d['VeryHighBS']            = int(bs > 11.0)
    d['BS_HR_interaction']     = bs * heart_rate
    d['Age_BS_interaction']    = age * bs
    d['Fever']                 = int(body_temp > 37.5)
    d['HighFever']             = int(body_temp > 38.5)
    d['Hypothermia']           = int(body_temp < 36.0)
    d['AgeGroup']              = 0 if age < 20 else (1 if age < 35 else 2)
    d['TeenPregnancy']         = int(age < 20)
    d['ElderlyPregnancy']      = int(age > 35)
    d['RiskSignalCount']       = (d['Hypertension'] + d['HighBS'] + d['Fever'] +
                                  d['Tachycardia'] + d['TeenPregnancy'] + d['ElderlyPregnancy'])
    d['BP_BS_Stress']          = (systolic_bp * bs) / 100
    return np.array(list(d.values())).reshape(1, -1)

# ─────────────────────────────────────────────
# FEATURE ENGINEERING — FETAL (39 features)
# ─────────────────────────────────────────────
def engineer_fetal_features(baseline, accels, fetal_mov, uterine_cont,
                             light_decel, severe_decel, prolonged_decel,
                             abnormal_stv, mean_stv, pct_altv,
                             mean_ltv, hist_width, hist_min, hist_max,
                             hist_peaks, hist_zeroes, hist_mode,
                             hist_mean, hist_median, hist_variance, hist_tendency):
    d = {}
    d['baseline value']                                         = baseline
    d['accelerations']                                          = accels
    d['fetal_movement']                                         = fetal_mov
    d['uterine_contractions']                                   = uterine_cont
    d['light_decelerations']                                    = light_decel
    d['severe_decelerations']                                   = severe_decel
    d['prolongued_decelerations']                               = prolonged_decel
    d['abnormal_short_term_variability']                        = abnormal_stv
    d['mean_value_of_short_term_variability']                   = mean_stv
    d['percentage_of_time_with_abnormal_long_term_variability'] = pct_altv
    d['mean_value_of_long_term_variability']                    = mean_ltv
    d['histogram_width']                                        = hist_width
    d['histogram_min']                                          = hist_min
    d['histogram_max']                                          = hist_max
    d['histogram_number_of_peaks']                              = hist_peaks
    d['histogram_number_of_zeroes']                             = hist_zeroes
    d['histogram_mode']                                         = hist_mode
    d['histogram_mean']                                         = hist_mean
    d['histogram_median']                                       = hist_median
    d['histogram_variance']                                     = hist_variance
    d['histogram_tendency']                                     = hist_tendency
    d['TotalDecelerations']       = light_decel + severe_decel + prolonged_decel
    d['SevereDecelFlag']          = int(severe_decel > 0)
    d['ProlongedDecelFlag']       = int(prolonged_decel > 0)
    d['DecelerationRatio']        = severe_decel / (light_decel + 1e-6)
    d['VariabilityRatio']         = abnormal_stv / (mean_stv + 1e-6)
    d['LongTermVariabilityHigh']  = int(pct_altv > 50)
    d['VariabilityStress']        = abnormal_stv * pct_altv
    d['TachycardiaFlag']          = int(baseline > 160)
    d['BradycardiaFlag']          = int(baseline < 110)
    d['AccelDecelRatio']          = accels / (d['TotalDecelerations'] + 1e-6)
    d['AccelFlag']                = int(accels > 0.003)
    d['HistogramSkewness']        = hist_max - hist_min
    d['HistogramModeOffset']      = abs(hist_mode - hist_mean)
    d['HistogramSymmetry']        = hist_mean / (hist_mode + 1e-6)
    d['FetalRiskScore']           = (d['SevereDecelFlag'] + d['ProlongedDecelFlag'] +
                                     d['LongTermVariabilityHigh'] + d['TachycardiaFlag'] +
                                     d['BradycardiaFlag'])
    d['ContractionAccelResponse'] = accels / (uterine_cont + 1e-6)
    d['ContractionDecelResponse'] = d['TotalDecelerations'] / (uterine_cont + 1e-6)
    d['MovementAccelRatio']       = fetal_mov / (accels + 1e-6)
    return np.array(list(d.values())).reshape(1, -1)

# ─────────────────────────────────────────────
# SHAP EXPLAINABILITY HELPERS
# ─────────────────────────────────────────────
@st.cache_resource
def get_maternal_explainer(_model):
    """Cache the TreeExplainer — expensive to create, reuse across predictions."""
    return shap.TreeExplainer(_model)
 
def compute_maternal_shap(explainer, features_scaled):
    """Compute SHAP values for one sample."""
    shap_values = explainer.shap_values(features_scaled)
    return shap_values
 
def compute_fetal_shap(model, features_scaled):
    """KernelExplainer for Stacking — model agnostic, slower but works for any model."""
    background  = np.zeros((1, features_scaled.shape[1]))
    explainer   = shap.KernelExplainer(model.predict_proba, background)
    shap_values = explainer.shap_values(features_scaled, nsamples=100, silent=True)
    return shap_values
 
def render_shap_chart(shap_vals, feature_names, prediction_class, title, top_n=10):
    """
    Renders a horizontal bar chart of top SHAP feature contributions.
    Handles all possible SHAP output shapes robustly:
      - list of arrays (old TreeExplainer multiclass): list[n_classes] of (n_samples, n_features)
      - 3D array (new TreeExplainer multiclass):        (n_samples, n_features, n_classes)
      - 2D array (KernelExplainer multiclass):          list[n_classes] of (n_samples, n_features)
      - 1D array (binary):                              (n_features,)
    """
    try:
        if isinstance(shap_vals, list):
            # Old SHAP format: list of arrays per class
            raw = np.array(shap_vals[prediction_class])
            vals = raw.flatten()[:len(feature_names)]
        elif isinstance(shap_vals, np.ndarray):
            if shap_vals.ndim == 3:
                # New SHAP format: (n_samples, n_features, n_classes)
                vals = shap_vals[0, :, prediction_class]
            elif shap_vals.ndim == 2:
                # (n_samples, n_features)
                vals = shap_vals[0]
            else:
                # Already 1D
                vals = shap_vals
        else:
            vals = np.array(shap_vals).flatten()[:len(feature_names)]
 
        # Safety: trim to feature count
        vals = np.array(vals, dtype=float).flatten()[:len(feature_names)]
 
        df = pd.DataFrame({"Feature": feature_names[:len(vals)], "SHAP": vals})
        df["Abs"] = df["SHAP"].abs()
        df = df.nlargest(top_n, "Abs").sort_values("SHAP")
 
        #colors = ["#EF4444" if v > 0 else "#22C55E" for v in df["SHAP"]]
        colors = ["#EC4899" if v > 0 else "#3B82F6" for v in df["SHAP"]]
        labels = [f"+{v:.3f}" if v > 0 else f"{v:.3f}" for v in df["SHAP"]]
 
        fig = go.Figure(go.Bar(
            x=df["SHAP"], y=df["Feature"],
            orientation="h",
            marker_color=colors,
            text=labels,
            textposition="outside",
            textfont=dict(size=11, color="#374151"),
            hovertemplate="<b>%{y}</b><br>SHAP value: %{x:.4f}<extra></extra>"
        ))
        fig.add_vline(x=0, line_color="#9CA3AF", line_width=1.5)
        fig.update_layout(
            title=dict(text=title, font=dict(size=13, color="#9D174D"), x=0),
            plot_bgcolor="white", paper_bgcolor="white",
            margin=dict(l=10, r=60, t=36, b=10),
            height=340,
            xaxis=dict(
                title="SHAP Value  (← opposes prediction  |  supports prediction →)",
                gridcolor="#F3F4F6", zeroline=False,
                tickfont=dict(size=10, color="#6B7280")
            ),
            yaxis=dict(tickfont=dict(size=11, color="#374151"), automargin=True),
        )
        return fig

    except Exception as e:
        raise ValueError(f"render_shap_chart failed: {e}")

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding:16px 0 8px 0;'>
        <div style='font-family:"DM Serif Display",serif; font-size:26px; color:#9D174D;'>
            MedPredict
        </div>
        <div style='font-size:12px; color:#9CA3AF; margin-top:4px;'>
            Maternal Health Risk &amp; Fetal Health Monitoring System
        </div>
    </div>
    <hr style='border:none; border-top:1.5px solid #FADADD; margin:12px 0;'>
    """, unsafe_allow_html=True)

    selected = option_menu(
        menu_title=None,
        options=["About", "Maternal Risk", "Fetal Health", "System Performance Dashboard"],
        icons=["house-heart", "person-heart", "heart-pulse", "bar-chart"],
        default_index=0,
        styles={
            "container":         {"padding": "0", "background-color": "#FFFFFF"},
            "icon":              {"color": "#F9A8D4", "font-size": "15px"},
            "nav-link":          {"font-size": "14px", "font-weight": "500",
                                  "color": "#374151", "padding": "10px 14px",
                                  "border-radius": "10px", "margin": "2px 0"},
            "nav-link-selected": {"background": "linear-gradient(135deg,#EC4899,#F472B6)",
                                  "color": "#FFFFFF", "font-weight": "600"},
        }
    )

    st.markdown("""
    <hr style='border:none; border-top:1.5px solid #FADADD; margin:16px 0 12px 0;'>
    <div style='font-size:11px; font-weight:700; color:#F9A8D4; text-transform:uppercase;
                letter-spacing:0.8px; margin-bottom:10px;'>Model Accuracy</div>
    <div style='display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;
                background:#FFF5F7; padding:8px 10px; border-radius:8px;'>
        <span style='font-size:12px; color:#374151; font-weight:500;'>Maternal (LightGBM)</span>
        <span style='font-size:13px; font-weight:700; color:#16A34A;'>90.64%</span>
    </div>
    <div style='display:flex; justify-content:space-between; align-items:center;
                background:#FFF5F7; padding:8px 10px; border-radius:8px;'>
        <span style='font-size:12px; color:#374151; font-weight:500;'>Fetal (Stacking)</span>
        <span style='font-size:13px; font-weight:700; color:#16A34A;'>94.13%</span>
    </div>
    <hr style='border:none; border-top:1px solid #FADADD; margin:14px 0 8px 0;'>
    <div style='font-size:11px; color:#D1A8B8; text-align:center; line-height:1.6;'>
        JSS STU &nbsp;|&nbsp; Dept. of CS&amp;E<br>Academic Year 2025-26
    </div>
    """, unsafe_allow_html=True)


# ═════════════════════════════════════════════
# PAGE: ABOUT
# ═════════════════════════════════════════════
if selected == "About":
    st.markdown('<div class="section-header">Welcome to MedPredict 🏥</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">An end-to-end Machine Learning and Streamlit based clinical decision support system for maternal health risk assessment and fetal health classification. Built for healthcare professionals, nurses, and rural health workers.</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2, gap="large")
    with col1:
        st.markdown("""
        <div class="info-card">
            <div style='font-size:34px; margin-bottom:10px;'>🤱</div>
            <div style='font-size:17px; font-weight:700; color:#9D174D; margin-bottom:8px;'>
                Maternal Health Risk Prediction
            </div>
            <div style='font-size:14px; color:#4B5563; line-height:1.75;'>
                Analyzes 6 clinical parameters — age, blood pressure, blood glucose,
                body temperature, and heart rate — extended to <b>26 engineered features</b>
                to classify pregnancy risk as <b>Low</b>, <b>Mid</b>, or <b>High</b>.
                Powered by LightGBM with Optuna hyperparameter tuning.
            </div>
            <div style='margin-top:14px; display:flex; gap:8px; flex-wrap:wrap;'>
                <span style='background:#F0FDF4; color:#16A34A; font-size:11px; font-weight:600;
                             padding:3px 10px; border-radius:20px; border:1px solid #BBF7D0;'>
                    90.64% Accuracy</span>
                <span style='background:#FFF5F7; color:#BE185D; font-size:11px; font-weight:600;
                             padding:3px 10px; border-radius:20px; border:1px solid #FBCFE8;'>
                    LightGBM</span>
                <span style='background:#F5F3FF; color:#7C3AED; font-size:11px; font-weight:600;
                             padding:3px 10px; border-radius:20px; border:1px solid #DDD6FE;'>
                    26 Features</span>
                <span style='background:#EFF6FF; color:#2563EB; font-size:11px; font-weight:600;
                             padding:3px 10px; border-radius:20px; border:1px solid #BFDBFE;'>
                    3 Risk Classes</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="info-card">
            <div style='font-size:34px; margin-bottom:10px;'>👶</div>
            <div style='font-size:17px; font-weight:700; color:#9D174D; margin-bottom:8px;'>
                Fetal Health Classification
            </div>
            <div style='font-size:14px; color:#4B5563; line-height:1.75;'>
                Interprets 21 Cardiotocography (CTG) measurements extended to
                <b>39 features</b> to classify fetal condition as <b>Normal</b>,
                <b>Suspect</b>, or <b>Pathological</b>.
                Stacking Ensemble model with exceptional CV stability of ±0.59%.
            </div>
            <div style='margin-top:14px; display:flex; gap:8px; flex-wrap:wrap;'>
                <span style='background:#F0FDF4; color:#16A34A; font-size:11px; font-weight:600;
                             padding:3px 10px; border-radius:20px; border:1px solid #BBF7D0;'>
                    94.13% Accuracy</span>
                <span style='background:#FFF5F7; color:#BE185D; font-size:11px; font-weight:600;
                             padding:3px 10px; border-radius:20px; border:1px solid #FBCFE8;'>
                    Stacking Ensemble</span>
                <span style='background:#F5F3FF; color:#7C3AED; font-size:11px; font-weight:600;
                             padding:3px 10px; border-radius:20px; border:1px solid #DDD6FE;'>
                    39 Features</span>
                <span style='background:#EFF6FF; color:#2563EB; font-size:11px; font-weight:600;
                             padding:3px 10px; border-radius:20px; border:1px solid #BFDBFE;'>
                    3 Health Classes</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ── How It Works — 3 step visual ──
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div style='font-size:16px; font-weight:700; color:#9D174D;
                margin-bottom:16px; text-align:center; letter-spacing:0.3px;'>
        How It Works
    </div>
    """, unsafe_allow_html=True)
 
    s1, arr1, s2, arr2, s3 = st.columns([3, 0.6, 3, 0.6, 3], gap="small")
 
    with s1:
        st.markdown("""
        <div class="info-card" style='text-align:center; padding:24px 20px;'>
            <div style='width:52px; height:52px; background:linear-gradient(135deg,#EC4899,#F472B6);
                        border-radius:50%; display:flex; align-items:center; justify-content:center;
                        margin:0 auto 14px auto; font-size:22px;'>🩺</div>
            <div style='font-size:12px; font-weight:700; color:#EC4899; text-transform:uppercase;
                        letter-spacing:0.8px; margin-bottom:6px;'>Step 1</div>
            <div style='font-size:15px; font-weight:700; color:#1F2937; margin-bottom:8px;'>
                Enter Patient Data
            </div>
            <div style='font-size:13px; color:#6B7280; line-height:1.6;'>
                Healthcare professional enters clinical parameters — blood pressure,
                glucose, heart rate, or CTG measurements — into the system.
            </div>
        </div>
        """, unsafe_allow_html=True)
 
    with arr1:
        st.markdown("""
        <div style='display:flex; align-items:center; justify-content:center;
                    height:100%; padding-top:40px; font-size:28px; color:#FBCFE8;'>
            →
        </div>
        """, unsafe_allow_html=True)
 
    with s2:
        st.markdown("""
        <div class="info-card" style='text-align:center; padding:24px 20px;'>
            <div style='width:52px; height:52px; background:linear-gradient(135deg,#EC4899,#F472B6);
                        border-radius:50%; display:flex; align-items:center; justify-content:center;
                        margin:0 auto 14px auto; font-size:22px;'>⚙️</div>
            <div style='font-size:12px; font-weight:700; color:#EC4899; text-transform:uppercase;
                        letter-spacing:0.8px; margin-bottom:6px;'>Step 2</div>
            <div style='font-size:15px; font-weight:700; color:#1F2937; margin-bottom:8px;'>
                ML Model Processes
            </div>
            <div style='font-size:13px; color:#6B7280; line-height:1.6;'>
                The trained ML model applies feature engineering, scaling, and
                classification using LightGBM or Stacking Ensemble to generate a prediction.
            </div>
        </div>
        """, unsafe_allow_html=True)
 
    with arr2:
        st.markdown("""
        <div style='display:flex; align-items:center; justify-content:center;
                    height:100%; padding-top:40px; font-size:28px; color:#FBCFE8;'>
            →
        </div>
        """, unsafe_allow_html=True)
 
    with s3:
        st.markdown("""
        <div class="info-card" style='text-align:center; padding:24px 20px;'>
            <div style='width:52px; height:52px; background:linear-gradient(135deg,#EC4899,#F472B6);
                        border-radius:50%; display:flex; align-items:center; justify-content:center;
                        margin:0 auto 14px auto; font-size:22px;'>📋</div>
            <div style='font-size:12px; font-weight:700; color:#EC4899; text-transform:uppercase;
                        letter-spacing:0.8px; margin-bottom:6px;'>Step 3</div>
            <div style='font-size:15px; font-weight:700; color:#1F2937; margin-bottom:8px;'>
                Review & Act
            </div>
            <div style='font-size:13px; color:#6B7280; line-height:1.6;'>
                The doctor reviews the risk classification and clinical signal
                summary, then makes an informed decision on patient care and next steps.
            </div>
        </div>
        """, unsafe_allow_html=True)
 
    # ── Key Features ──
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div style='font-size:16px; font-weight:700; color:#9D174D;
                margin-bottom:16px; text-align:center;'>
        What Makes This System Unique
    </div>""", unsafe_allow_html=True)
 
    f1, f2, f3 = st.columns(3, gap="medium")
    for col, (icon, title, desc) in zip(
        [f1, f2, f3, f1, f2, f3],
        [
            ("🔬", "Feature Engineering",
             "Goes beyond raw inputs — 26 maternal and 39 fetal engineered clinical features "
             "derived from domain knowledge, giving models richer signals than raw parameters alone."),
            ("⚖️", "Class Imbalance Handled",
             "SMOTETomek balancing applied only on training data, correctly preventing data leakage. "
             "Rare but critical High Risk and Pathological cases are given equal representation."),
            ("🎯", "Dual Model System",
             "Two independently trained models — one for maternal risk, one for fetal health — "
             "each optimised separately with 100-trial Optuna tuning for maximum accuracy."),
            ("📊", "Rigorous Evaluation",
             "Models evaluated using Macro F1, per-class Precision and Recall, 10-Fold CV "
             "Mean ± Std, and overfitting diagnostics — not just accuracy alone."),
            ("🔍", "Confidence & Explainability",
             "Every prediction shows class-wise confidence scores and SHAP-based feature "
            "contributions — explaining exactly why the model made each decision, "
            "not just what it decided."),
        ]
    ):
        col.markdown(f"""
        <div class="info-card" style='margin-bottom:14px; min-height:155px;'>
            <div style='font-size:24px; margin-bottom:8px;'>{icon}</div>
            <div style='font-size:14px; font-weight:700; color:#9D174D; margin-bottom:6px;'>{title}</div>
            <div style='font-size:13px; color:#6B7280; line-height:1.6;'>{desc}</div>
        </div>""", unsafe_allow_html=True)
 
    # ── WHO Statistics ──
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div style='font-size:16px; font-weight:700; color:#9D174D;
                margin-bottom:16px; text-align:center;'>
        Why Early Risk Detection Matters
    </div>""", unsafe_allow_html=True)
 
    st.markdown("""
    <div class="info-card" style='padding:24px 28px;'>
        <div style='display:grid; grid-template-columns:repeat(4,1fr); gap:20px; text-align:center;'>
            <div style='padding:16px 8px; border-right:1px solid #FADADD;'>
                <div style='font-size:30px; font-weight:800; color:#BE185D; margin-bottom:6px;'>810</div>
                <div style='font-size:12px; font-weight:600; color:#374151; margin-bottom:4px;'>Women Die Every Day</div>
                <div style='font-size:11px; color:#9CA3AF; line-height:1.5;'>
                    due to preventable pregnancy or childbirth complications<br><i>(WHO, 2023)</i></div>
            </div>
            <div style='padding:16px 8px; border-right:1px solid #FADADD;'>
                <div style='font-size:30px; font-weight:800; color:#BE185D; margin-bottom:6px;'>94%</div>
                <div style='font-size:12px; font-weight:600; color:#374151; margin-bottom:4px;'>Deaths in Low-Income Countries</div>
                <div style='font-size:11px; color:#9CA3AF; line-height:1.5;'>
                    where access to specialist care and monitoring is severely limited<br><i>(WHO, 2023)</i></div>
            </div>
            <div style='padding:16px 8px; border-right:1px solid #FADADD;'>
                <div style='font-size:30px; font-weight:800; color:#BE185D; margin-bottom:6px;'>~1M</div>
                <div style='font-size:12px; font-weight:600; color:#374151; margin-bottom:4px;'>Newborns Die Within 24 Hours</div>
                <div style='font-size:11px; color:#9CA3AF; line-height:1.5;'>
                    from birth complications that early fetal monitoring can help prevent<br><i>(WHO, 2023)</i></div>
            </div>
            <div style='padding:16px 8px;'>
                <div style='font-size:30px; font-weight:800; color:#BE185D; margin-bottom:6px;'>2/3</div>
                <div style='font-size:12px; font-weight:600; color:#374151; margin-bottom:4px;'>Deaths Are Preventable</div>
                <div style='font-size:11px; color:#9CA3AF; line-height:1.5;'>
                    with timely risk assessment, early intervention, and skilled clinical support<br><i>(WHO, 2023)</i></div>
            </div>
        </div>
        <div style='margin-top:18px; padding-top:14px; border-top:1px solid #FADADD;
                    font-size:13px; color:#6B7280; text-align:center; line-height:1.6;'>
            MedPredict addresses this gap by bringing systematic ML-based risk screening
            to any clinical setting with a browser — from urban hospitals to rural health centres.
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="disclaimer" style="margin-top:24px;">
        ⚠️ <b>Clinical Disclaimer:</b> MedPredict is a decision support tool only.
        All predictions must be reviewed by a qualified healthcare professional before any clinical action.
        This system has not undergone clinical trial validation and is not approved for standalone diagnostic use.
    </div>
    """, unsafe_allow_html=True)


# ═════════════════════════════════════════════
# PAGE: MATERNAL RISK
# ═════════════════════════════════════════════
elif selected == "Maternal Risk":
    st.markdown('<div class="section-header">🤱 Maternal Health Risk Prediction</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Enter the patient\'s clinical parameters. The model classifies risk as Low, Mid, or High using LightGBM with Optuna-tuned hyperparameters (90.64% accuracy).</div>', unsafe_allow_html=True)

    with st.form("maternal_form"):
        col1, col2, col3 = st.columns(3, gap="medium")

        with col1:
            st.markdown('<div class="input-group-label">👤 Demographics</div>', unsafe_allow_html=True)
            age = st.number_input("Age (years)", min_value=10, max_value=70, value=30, step=1)

        with col2:
            st.markdown('<div class="input-group-label">🩸 Blood Pressure</div>', unsafe_allow_html=True)
            systolic_bp = st.number_input("Systolic BP (mmHg)", min_value=50, max_value=200, value=120, step=1)

        with col3:
            st.markdown('<div class="input-group-label">🩸 Blood Pressure</div>', unsafe_allow_html=True)
            diastolic_bp = st.number_input("Diastolic BP (mmHg)", min_value=30, max_value=130, value=80, step=1)

        with col1:
            st.markdown('<div class="input-group-label">🧪 Metabolic</div>', unsafe_allow_html=True)
            bs = st.number_input("Blood Glucose (mmol/L)", min_value=1.0, max_value=25.0,
                                 value=5.0, step=0.1, format="%.1f")
        with col2:
            st.markdown('<div class="input-group-label">🌡️ Vitals</div>', unsafe_allow_html=True)
            body_temp = st.number_input("Body Temperature (°C)", min_value=34.0, max_value=42.0,
                                        value=37.0, step=0.1, format="%.1f")
        with col3:
            st.markdown('<div class="input-group-label">❤️ Heart Rate</div>', unsafe_allow_html=True)
            heart_rate = st.number_input("Heart Rate (bpm)", min_value=40, max_value=160, value=75, step=1)

        st.markdown("<br>", unsafe_allow_html=True)
        _, mid, _ = st.columns([2, 1, 2])
        with mid:
            submitted = st.form_submit_button("🔍 Predict Maternal Risk", use_container_width=True)

    if submitted:
        try:
            features        = engineer_maternal_features(age, systolic_bp, diastolic_bp, bs, body_temp, heart_rate)
            features_scaled = maternal_scaler.transform(features)
            prediction      = maternal_model.predict(features_scaled)[0]

            probs = maternal_model.predict_proba(features_scaled)
            confidence = max(probs[0]) * 100

            st.markdown("<br>", unsafe_allow_html=True)
            r1, r2 = st.columns([1, 1], gap="large")

            with r1:
                if prediction == 0:
                    st.markdown(f"""
                    <div class="result-low">
                        <div style='font-size:35px;'>✅</div>
                        <div class="result-title" style='color:#059669;'>Low Risk</div>
                        <div style='margin-top:8px; font-size:14px; font-weight:600; color:#065F46;'>
                            Confidence: {confidence:.2f}%
                        </div>
                        <div class="result-desc">Parameters are within normal ranges.
                        Continue routine prenatal care and regular scheduled monitoring.
                        No immediate intervention required.</div>
                    </div>""", unsafe_allow_html=True)
                elif prediction == 1:
                    st.markdown(f"""
                    <div class="result-mid">
                        <div style='font-size:35px;'>⚠️</div>
                        <div class="result-title" style='color:#D97706;'>Mid Risk</div>      
                        <div style='margin-top:8px; font-size:14px; font-weight:600; color:#065F46;'>
                            Confidence: {confidence:.2f}%
                        </div>       
                        <div class="result-desc">Some parameters indicate elevated risk.
                        Enhanced monitoring and specialist consultation recommended.
                        Schedule follow-up within 1–2 weeks.</div>
                    </div>""", unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="result-high">
                        <div style='font-size:35px;'>🚨</div>
                        <div class="result-title" style='color:#E11D48;'>High Risk</div>      
                        <div style='margin-top:8px; font-size:14px; font-weight:600; color:#065F46;'>
                            Confidence: {confidence:.2f}%
                        </div>
                        <div class="result-desc">Critical parameters detected.
                        Immediate medical attention is required. Refer to a specialist
                        or hospital without delay.</div>
                    </div>""", unsafe_allow_html=True)

            # with r2:
            #     risk_signals = []
            #     if systolic_bp > 140: risk_signals.append(("⚠️", "Hypertension",    "SBP > 140 mmHg"))
            #     if systolic_bp < 90:  risk_signals.append(("⚠️", "Hypotension",     "SBP < 90 mmHg"))
            #     if bs > 7.8:          risk_signals.append(("⚠️", "High Glucose",    "BS > 7.8 mmol/L"))
            #     if body_temp > 37.5:  risk_signals.append(("🌡️", "Elevated Temp",   "Temp > 37.5°C"))
            #     if heart_rate > 100:  risk_signals.append(("💓", "Tachycardia",     "HR > 100 bpm"))
            #     if heart_rate < 60:   risk_signals.append(("💓", "Bradycardia",     "HR < 60 bpm"))
            #     if age < 20:          risk_signals.append(("👩", "Teen Pregnancy",  "Age < 20 years"))
            #     if age > 35:          risk_signals.append(("👩", "Advanced Age",    "Age > 35 years"))

            #     st.markdown("""
            #     <div class="info-card">
            #         <div style='font-size:14px; font-weight:700; color:#9D174D; margin-bottom:12px;'>
            #             📋 Clinical Signal Summary
            #         </div>
            #     """, unsafe_allow_html=True)

            #     if risk_signals:
            #         for icon, name, detail in risk_signals:
            #             st.markdown(f"""
            #             <div style='display:flex; align-items:center; gap:10px; padding:6px 0;
            #                         border-bottom:1px solid #FFF0F5;'>
            #                 <span style='font-size:16px;'>{icon}</span>
            #                 <div>
            #                     <span style='font-size:13px; font-weight:600; color:#B45309;'>{name}</span>
            #                     <span style='font-size:12px; color:#9CA3AF;'> — {detail}</span>
            #                 </div>
            #             </div>""", unsafe_allow_html=True)
            #     else:
            #         st.markdown("""
            #         <div style='display:flex; align-items:center; gap:10px; padding:8px 0;'>
            #             <span style='font-size:20px;'>✅</span>
            #             <span style='font-size:13px; font-weight:600; color:#059669;'>
            #                 All parameters within normal ranges
            #             </span>
            #         </div>""", unsafe_allow_html=True)

            #     st.markdown("</div>", unsafe_allow_html=True)

            st.markdown("""
            <div class="disclaimer">
                ⚠️ This prediction is for clinical decision support only.
                A qualified healthcare professional must review all results before any medical action.
            </div>""", unsafe_allow_html=True)

            # ── SHAP Explainability ──
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("""
            <div class="info-card">
                <div style='font-size:14px; font-weight:700; color:#9D174D; margin-bottom:4px;'>
                    🔍 Model Explainability — Why did the model predict this?
                </div>
                <div style='font-size:12px; color:#6B7280; margin-bottom:14px;'>
                    SHAP (SHapley Additive exPlanations) shows how much each feature
                    contributed to this specific prediction.
                    <b style='color:#EC4899;'>Pink bars</b> show factors that pushed the model toward this prediction.
                    <b style='color:#3B82F6;'>Blue bars</b> show factors that pushed the model away from this prediction.
                </div>
            """, unsafe_allow_html=True)
 
            with st.spinner("Computing SHAP values..."):
                try:
                    maternal_feature_names = [
                        'Age','SystolicBP','DiastolicBP','BS','BodyTemp','HeartRate',
                        'PulsePressure','MeanArterialPressure','BPRatio',
                        'Hypertension','Hypotension','ShockIndex',
                        'Tachycardia','Bradycardia','HighBS','VeryHighBS',
                        'BS_HR_interaction','Age_BS_interaction',
                        'Fever','HighFever','Hypothermia',
                        'AgeGroup','TeenPregnancy','ElderlyPregnancy',
                        'RiskSignalCount','BP_BS_Stress'
                    ]
                    explainer = get_maternal_explainer(maternal_model)
                    shap_vals = compute_maternal_shap(explainer, features_scaled)
                    fig_shap  = render_shap_chart(
                        shap_vals, maternal_feature_names,
                        prediction_class=int(prediction),
                        title=f"Top 10 Feature Contributions — Predicted Class: {['Low Risk','Mid Risk','High Risk'][int(prediction)]}"
                    )
                    st.plotly_chart(fig_shap, use_container_width=True)
                except Exception as shap_err:
                    st.info(f"SHAP explanation unavailable: {shap_err}")
 
            st.markdown("</div>", unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Prediction error: {e}")


# ═════════════════════════════════════════════
# PAGE: FETAL HEALTH
# ═════════════════════════════════════════════
elif selected == "Fetal Health":
    st.markdown('<div class="section-header">👶 Fetal Health Classification</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Enter CTG (Cardiotocography) measurements from the monitor. The Stacking Ensemble model classifies fetal condition as Normal, Suspect, or Pathological (94.13% accuracy).</div>', unsafe_allow_html=True)

    with st.form("fetal_form"):

        st.markdown("**📊 Fetal Heart Rate Parameters**")
        c1, c2, c3 = st.columns(3, gap="medium")
        baseline  = c1.number_input("Baseline FHR (bpm)",  min_value=50.0,  max_value=250.0, value=133.0, step=1.0)
        accels    = c2.number_input("Accelerations",        min_value=0.0,   max_value=0.1,   value=0.0015, step=0.001, format="%.3f")
        fetal_mov = c3.number_input("Fetal Movement",       min_value=0.0,   max_value=0.5,   value=0.01,   step=0.001, format="%.3f")

        st.markdown("---")
        st.markdown("**📉 Contractions & Decelerations**")
        c1, c2, c3 = st.columns(3, gap="medium")
        uterine_cont    = c1.number_input("Uterine Contractions",     min_value=0.0, max_value=0.02, value=0.004, step=0.001,  format="%.3f")
        light_decel     = c2.number_input("Light Decelerations",      min_value=0.0, max_value=0.02, value=0.0,   step=0.001,  format="%.3f")
        severe_decel    = c3.number_input("Severe Decelerations",     min_value=0.0, max_value=0.01, value=0.0,   step=0.0001, format="%.4f")
        c1b, _, _       = st.columns(3, gap="medium")
        prolonged_decel = c1b.number_input("Prolongued Decelerations",min_value=0.0, max_value=0.01, value=0.0,   step=0.0001, format="%.4f")

        st.markdown("---")
        st.markdown("**〰️ Short & Long Term Variability**")
        c1, c2, c3 = st.columns(3, gap="medium")
        abnormal_stv = c1.number_input("Abnormal STV (%)",      min_value=0.0,  max_value=100.0, value=47.0, step=1.0)
        mean_stv     = c2.number_input("Mean STV Value",         min_value=0.0,  max_value=10.0,  value=1.0,  step=0.1,  format="%.1f")
        pct_altv     = c3.number_input("% Time Abnormal LTV",   min_value=0.0,  max_value=100.0, value=10.0, step=1.0)
        c1c, _, _    = st.columns(3, gap="medium")
        mean_ltv     = c1c.number_input("Mean LTV Value",        min_value=0.0,  max_value=50.0,  value=7.0,  step=0.1,  format="%.1f")

        st.markdown("---")
        st.markdown("**📊 FHR Histogram Features**")
        c1, c2, c3 = st.columns(3, gap="medium")
        hist_width    = c1.number_input("Histogram Width",    min_value=0.0,   max_value=200.0, value=70.0,  step=1.0)
        hist_min      = c2.number_input("Histogram Min",      min_value=50.0,  max_value=200.0, value=93.0,  step=1.0)
        hist_max      = c3.number_input("Histogram Max",      min_value=100.0, max_value=300.0, value=162.0, step=1.0)
        hist_peaks    = c1.number_input("No. of Peaks",       min_value=0.0,   max_value=20.0,  value=4.0,   step=1.0)
        hist_zeroes   = c2.number_input("No. of Zeroes",      min_value=0.0,   max_value=20.0,  value=0.0,   step=1.0)
        hist_mode     = c3.number_input("Histogram Mode",     min_value=50.0,  max_value=200.0, value=137.0, step=1.0)
        hist_mean     = c1.number_input("Histogram Mean",     min_value=50.0,  max_value=200.0, value=134.0, step=1.0)
        hist_median   = c2.number_input("Histogram Median",   min_value=50.0,  max_value=200.0, value=138.0, step=1.0)
        hist_variance = c3.number_input("Histogram Variance", min_value=0.0,   max_value=300.0, value=25.0,  step=1.0)
        c1d, _, _     = st.columns(3, gap="medium")
        hist_tendency = c1d.number_input("Tendency (-1/0/1)", min_value=-1.0,  max_value=1.0,   value=0.0,   step=1.0)

        st.markdown("<br>", unsafe_allow_html=True)
        _, mid, _ = st.columns([2, 1, 2])
        with mid:
            submitted_fetal = st.form_submit_button("🔍  Classify Fetal Health", use_container_width=True)

    if submitted_fetal:
        try:
            features        = engineer_fetal_features(
                baseline, accels, fetal_mov, uterine_cont,
                light_decel, severe_decel, prolonged_decel,
                abnormal_stv, mean_stv, pct_altv,
                mean_ltv, hist_width, hist_min, hist_max,
                hist_peaks, hist_zeroes, hist_mode,
                hist_mean, hist_median, hist_variance, hist_tendency
            )
            features_scaled = fetal_scaler.transform(features)
            prediction      = fetal_model.predict(features_scaled)[0]

            probs = fetal_model.predict_proba(features_scaled)
            confidence = max(probs[0]) * 100

            st.markdown("<br>", unsafe_allow_html=True)
            r1, r2 = st.columns([1, 1], gap="large")

            with r1:
                if prediction == 0:
                    st.markdown(f"""
                    <div class="result-normal">
                        <div style='font-size:35px;'>✅</div>
                        <div class="result-title" style='color:#059669;'>Normal</div>
                        <div style='margin-top:8px; font-size:14px; font-weight:600; color:#065F46;'>
                            Confidence: {confidence:.2f}%
                        </div>
                        <div class="result-desc">CTG patterns are within normal limits.
                        Fetal heart rate and variability indicate healthy fetal status.
                        Continue routine monitoring.</div>
                    </div>""", unsafe_allow_html=True)
                elif prediction == 1:
                    st.markdown(f"""
                    <div class="result-suspect">
                        <div style='font-size:35px;'>⚠️</div>
                        <div class="result-title" style='color:#D97706;'>Suspect</div>
                        <div style='margin-top:8px; font-size:14px; font-weight:600; color:#065F46;'>
                            Confidence: {confidence:.2f}%
                        </div>
                        <div class="result-desc">Borderline CTG abnormalities detected.
                        Further clinical evaluation recommended.
                        Consider repeat CTG or additional investigation.</div>
                    </div>""", unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="result-pathological">
                        <div style='font-size:35px;'>🚨</div>
                        <div class="result-title" style='color:#E11D48;'>Pathological</div>
                        <div style='margin-top:8px; font-size:14px; font-weight:600; color:#065F46;'>
                            Confidence: {confidence:.2f}%
                        </div>
                        <div class="result-desc">Abnormal CTG patterns detected.
                        Immediate obstetric review required. This may indicate
                        fetal distress — urgent clinical intervention needed.</div>
                    </div>""", unsafe_allow_html=True)

            # with r2:
            #     risk_flags = []
            #     if severe_decel > 0:    risk_flags.append(("📉", "Severe Decelerations",   "Present in trace"))
            #     if prolonged_decel > 0: risk_flags.append(("📉", "Prolonged Decelerations","Present in trace"))
            #     if baseline > 160:      risk_flags.append(("💓", "Fetal Tachycardia",      "FHR > 160 bpm"))
            #     if baseline < 110:      risk_flags.append(("💓", "Fetal Bradycardia",      "FHR < 110 bpm"))
            #     if pct_altv > 50:       risk_flags.append(("〰️", "High Abnormal LTV",      "> 50% of time"))
            #     if abnormal_stv > 70:   risk_flags.append(("〰️", "High Abnormal STV",      "> 70%"))
            #     if accels < 0.001:      risk_flags.append(("📊", "Low Accelerations",      "< 0.001"))

            #     st.markdown("""
            #     <div class="info-card">
            #         <div style='font-size:14px; font-weight:700; color:#9D174D; margin-bottom:12px;'>
            #             📋 CTG Signal Flags
            #         </div>
            #     """, unsafe_allow_html=True)

            #     if risk_flags:
            #         for icon, name, detail in risk_flags:
            #             st.markdown(f"""
            #             <div style='display:flex; align-items:center; gap:10px; padding:6px 0;
            #                         border-bottom:1px solid #FFF0F5;'>
            #                 <span style='font-size:16px;'>{icon}</span>
            #                 <div>
            #                     <span style='font-size:13px; font-weight:600; color:#B45309;'>{name}</span>
            #                     <span style='font-size:12px; color:#9CA3AF;'> — {detail}</span>
            #                 </div>
            #             </div>""", unsafe_allow_html=True)
            #     else:
            #         st.markdown("""
            #         <div style='display:flex; align-items:center; gap:10px; padding:8px 0;'>
            #             <span style='font-size:20px;'>✅</span>
            #             <span style='font-size:13px; font-weight:600; color:#059669;'>
            #                 No critical CTG flags detected
            #             </span>
            #         </div>""", unsafe_allow_html=True)

            #     st.markdown("</div>", unsafe_allow_html=True)

            st.markdown("""
            <div class="disclaimer">
                ⚠️ CTG interpretation requires clinical expertise. All Suspect and Pathological
                results must be reviewed immediately by a qualified obstetrician.
            </div>""", unsafe_allow_html=True)

            # ── SHAP Explainability ──
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("""
            <div class="info-card">
                <div style='font-size:14px; font-weight:700; color:#9D174D; margin-bottom:4px;'>
                    🔍 Model Explainability — Why did the model predict this?
                </div>
                <div style='font-size:12px; color:#6B7280; margin-bottom:14px;'>
                    SHAP (SHapley Additive exPlanations) shows how much each CTG feature
                    contributed to this specific prediction.
                    <b style='color:#EC4899;'>Pink bars</b> show factors that pushed the model toward this prediction.
                    <b style='color:#3B82F6;'>Blue bars</b> show factors that pushed the model away from this prediction.
                    Note: KernelExplainer may take 10–15 seconds to compute.
                </div>
            """, unsafe_allow_html=True)

            with st.spinner("Computing SHAP values (this may take a few seconds)..."):
                try:
                    fetal_feature_names = [
                        'baseline value','accelerations','fetal_movement',
                        'uterine_contractions','light_decelerations','severe_decelerations',
                        'prolongued_decelerations','abnormal_short_term_variability',
                        'mean_value_of_short_term_variability',
                        'percentage_of_time_with_abnormal_long_term_variability',
                        'mean_value_of_long_term_variability','histogram_width',
                        'histogram_min','histogram_max','histogram_number_of_peaks',
                        'histogram_number_of_zeroes','histogram_mode','histogram_mean',
                        'histogram_median','histogram_variance','histogram_tendency',
                        'TotalDecelerations','SevereDecelFlag','ProlongedDecelFlag',
                        'DecelerationRatio','VariabilityRatio','LongTermVariabilityHigh',
                        'VariabilityStress','TachycardiaFlag','BradycardiaFlag',
                        'AccelDecelRatio','AccelFlag','HistogramSkewness',
                        'HistogramModeOffset','HistogramSymmetry','FetalRiskScore',
                        'ContractionAccelResponse','ContractionDecelResponse','MovementAccelRatio'
                    ]
                    shap_vals = compute_fetal_shap(
                        fetal_model, features_scaled
                    )
                    fig_shap = render_shap_chart(
                        shap_vals, fetal_feature_names,
                        prediction_class=int(prediction),
                        title=f"Top 10 Feature Contributions — Predicted Class: {['Normal','Suspect','Pathological'][prediction]}"
                    )
                    st.plotly_chart(fig_shap, use_container_width=True)
                except Exception as shap_err:
                    st.info(f"SHAP explanation unavailable: {shap_err}")

            st.markdown("</div>", unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Prediction error: {e}")


# ═════════════════════════════════════════════
# PAGE: DASHBOARD
# ═════════════════════════════════════════════
elif selected == "System Performance Dashboard":
    st.markdown('<div class="section-header">📊 Model Insights Dashboard</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Model performance metrics and key project statistics at a glance.</div>', unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    for col, (val, label, sub) in zip([c1, c2, c3, c4], [
        ("90.64%", "Maternal Test Accuracy", "LightGBM + Optuna"),
        ("94.13%", "Fetal Test Accuracy",    "Stacking Ensemble"),
        ("84.70%", "Maternal CV Mean",        "10-Fold ±0.89%"),
        ("98.49%", "Fetal CV Mean",           "10-Fold ±0.59%"),
    ]):
        col.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{val}</div>
            <div style='font-size:13px; font-weight:600; color:#374151; margin-top:5px;'>{label}</div>
            <div class="metric-label">{sub}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    ch1, ch2 = st.columns(2, gap="large")

    PINK  = "#EC4899"
    DPINK = "#9D174D"

    with ch1:
        st.markdown("#### Maternal Model — 12 Version Journey")
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(
            x=["V1","V2-V4","V5-V6","V7","V8","V9","V10-V11","Seed","V12"],
            y=[84.23, 84.73, 75.37, 74.38, 84.73, 76.51, 87.45, 91.13, 90.64],
            mode="lines+markers",
            line=dict(color=PINK, width=2.5),
            marker=dict(size=8, color=PINK, line=dict(color="white", width=2)),
            fill="tozeroy", fillcolor="rgba(236,72,153,0.07)",
            hovertemplate="<b>%{x}</b><br>Accuracy: %{y}%<extra></extra>"
        ))
        fig1.add_hline(y=90, line_dash="dot", line_color="#EF4444",
                       annotation_text="90% target", annotation_position="top right",
                       annotation_font_color="#EF4444")
        fig1.update_layout(
            plot_bgcolor="white", paper_bgcolor="white",
            margin=dict(l=10, r=10, t=10, b=10), height=300,
            yaxis=dict(range=[70, 95], title="Accuracy (%)", gridcolor="#FFF0F5",
                       tickfont=dict(color="#37393D")),
            xaxis=dict(gridcolor="#FFF0F5", tickfont=dict(color="#38393A", size=10))
        )
        st.plotly_chart(fig1, use_container_width=True)

    with ch2:
        st.markdown("#### Fetal Model — All Models Comparison")
        models_f = ["XGBoost", "LightGBM", "Voting", "Random Forest", "Stacking"]
        fig2 = go.Figure()
        fig2.add_trace(go.Bar(
            name="Test Accuracy", x=models_f,
            y=[92.96, 93.66, 93.66, 93.90, 94.13],
            marker_color=PINK,
            text=["92.96%","93.66%","93.66%","93.90%","94.13%"],
            textposition="outside", textfont=dict(size=10)
        ))
        fig2.add_trace(go.Bar(
            name="Macro F1", x=models_f,
            y=[87.10, 87.65, 87.65, 88.50, 88.66],
            marker_color="#F9A8D4",
            text=["87.10%","87.65%","87.65%","88.50%","88.66%"],
            textposition="outside", textfont=dict(size=10)
        ))
        fig2.update_layout(
            barmode="group", plot_bgcolor="white", paper_bgcolor="white",
            margin=dict(l=10, r=10, t=10, b=10), height=300,
            yaxis=dict(range=[80, 98], title="Score (%)", gridcolor="#FFF0F5",
                       tickfont=dict(color="#151616")),
            xaxis=dict(tickfont=dict(color="#1C1C1D", size=10)),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0,
                        font=dict(size=11))
        )
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("#### Fetal Stacking Ensemble — Per Class Breakdown")
    df = pd.DataFrame({
        "Class":     ["Normal (332)",  "Suspect (59)", "Pathological (35)", "Macro Avg"],
        "Precision": [95.87, 86.79, 88.24, 90.30],
        "Recall":    [97.89, 77.97, 85.71, 87.19],
        "F1 Score":  [96.87, 82.14, 86.96, 88.66],
    })
    st.dataframe(
        df.style
          .background_gradient(subset=["Precision","Recall","F1 Score"], cmap="Blues")
          .format({"Precision":"{:.2f}%","Recall":"{:.2f}%","F1 Score":"{:.2f}%"})
          .set_table_styles([
          # Header styling
          {
              "selector": "th",
              "props": [
                  ("background-color", "#EAF2FF"),  # light gray
                  ("color", "#1E3A8A"),             # dark text
                  ("font-weight", "bold"),
                  ("text-align", "center"),
                  ("border", "1px solid #D1D5DB"),
                  ("vertical-align", "middle")
              ]
          },
          # Table styling
          {
              "selector": "td",
              "props": [
                  ("border", "1px solid #E5E7EB"),
                  ("color", "#111827"),
                  ("text-align","center"),
                  ("vertical-align", "middle")
              ]
          }
      ])
      .set_properties(**{
          "background-color": "white",
          "color":"#111827",
          "text-align":"center"
      }),
        use_container_width=True, hide_index=True
    )