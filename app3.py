import streamlit as st
from streamlit_option_menu import option_menu
import pickle
import warnings
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import shap
import hashlib
import json
import os
from datetime import datetime

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
# AUTH HELPERS (session-state based, no backend)
# ─────────────────────────────────────────────
USERS_FILE = "users.json"

def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()

def load_users() -> dict:
    if os.path.exists(USERS_FILE):
        with open(USERS_FILE, "r") as f:
            return json.load(f)
    # Seed with default clinician account
    default = {
        "admin": {
            "name": "Admin Clinician",
            "password": hash_password("admin123"),
            "role": "clinician",
            "created": str(datetime.now().date())
        }
    }
    save_users(default)
    return default

def save_users(users: dict):
    with open(USERS_FILE, "w") as f:
        json.dump(users, f, indent=2)

def register_user(username, name, password, role) -> tuple[bool, str]:
    users = load_users()
    if username in users:
        return False, "Username already exists."
    if len(password) < 6:
        return False, "Password must be at least 6 characters."
    users[username] = {
        "name": name,
        "password": hash_password(password),
        "role": role,
        "created": str(datetime.now().date())
    }
    save_users(users)
    return True, "Account created successfully!"

def login_user(username, password) -> tuple[bool, str, dict]:
    users = load_users()
    if username not in users:
        return False, "User not found.", {}
    if users[username]["password"] != hash_password(password):
        return False, "Incorrect password.", {}
    return True, "Login successful!", users[username]

# ─────────────────────────────────────────────
# CSS — Soft Pink Medical Theme
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=DM+Serif+Display&display=swap');

button[kind="header"] { display: none !important; }

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif !important; }
[data-testid="stAppViewContainer"] { background-color: #FFF5F7 !important; }
[data-testid="stHeader"] { background: transparent !important; }
#MainMenu, footer { visibility: hidden; }
.block-container { padding: 1.5rem 2rem 2rem 2rem !important; max-width: 1200px; }

[data-testid="stSidebar"] { background: #FFFFFF !important; border-right: 1.5px solid #FADADD; }
[data-testid="stSidebar"] .block-container { padding: 1rem !important; }

h1, h2, h3, h4, h5, h6 { color: #111827 !important; font-weight: 700; }
p, span { color: #374151 !important; }

[data-testid="stNumberInput"] input,
[data-testid="stTextInput"] input {
    background: #FFFFFF !important;
    color: #111827 !important;
    -webkit-text-fill-color: #111827 !important;
    caret-color: #EC4899 !important;
    border: 1.5px solid #FBCFE8 !important;
    border-radius: 10px !important;
    font-size: 14px !important;
    font-weight: 500 !important;
    padding: 8px 12px !important;
    box-shadow: none !important;
}
[data-testid="stNumberInput"] input:focus,
[data-testid="stTextInput"] input:focus {
    border-color: #EC4899 !important;
    box-shadow: 0 0 0 3px rgba(236,72,153,0.12) !important;
    color: #111827 !important;
    -webkit-text-fill-color: #111827 !important;
}
[data-testid="stNumberInput"] input::placeholder,
[data-testid="stTextInput"] input::placeholder { color: #9CA3AF !important; opacity: 1 !important; }
[data-testid="stNumberInput"] button { background: #FFF0F5 !important; border: 1px solid #FBCFE8 !important; color: #EC4899 !important; }
[data-testid="stNumberInput"] > div, [data-testid="stTextInput"] > div { background: transparent !important; }
[data-testid="stNumberInput"] label, [data-testid="stTextInput"] label,
.stNumberInput label, .stTextInput label {
    color: #1F2937 !important; font-weight: 600 !important; font-size: 13px !important; letter-spacing: 0.2px !important;
}

/* Selectbox */
[data-testid="stSelectbox"] label { color: #1F2937 !important; font-weight: 600 !important; font-size: 13px !important; }
[data-testid="stSelectbox"] > div > div {
    background: #FFFFFF !important; border: 1.5px solid #FBCFE8 !important;
    border-radius: 10px !important; color: #111827 !important;
}

.stButton > button {
    background: linear-gradient(135deg, #EC4899, #F472B6) !important;
    color: #FFFFFF !important; border: none !important; border-radius: 10px !important;
    font-weight: 600 !important; font-size: 14px !important; padding: 10px 28px !important;
    width: 100% !important; transition: all 0.2s !important;
}
.stButton > button:hover { transform: translateY(-1px); box-shadow: 0 4px 14px rgba(236,72,153,0.35) !important; }

[data-testid="stFormSubmitButton"] > button {
    background: linear-gradient(135deg, #EC4899, #F472B6) !important;
    color: #FFFFFF !important; border: none !important; border-radius: 10px !important;
    font-weight: 700 !important; font-size: 15px !important; padding: 12px 32px !important;
    width: 100% !important; transition: all 0.2s !important;
}
[data-testid="stFormSubmitButton"] > button:hover { transform: translateY(-1px); box-shadow: 0 6px 18px rgba(236,72,153,0.4) !important; }

.info-card {
    background: #FFFFFF; border: 1px solid #FBCFE8; border-radius: 14px;
    padding: 20px 24px; margin-bottom: 16px; box-shadow: 0 2px 8px rgba(236,72,153,0.06);
}
.section-header { font-family: 'DM Serif Display', serif; font-size: 27px; font-weight: 400; color: #9D174D !important; margin-bottom: 4px; }
.section-sub { font-size: 14px; color: #6B7280 !important; line-height: 1.6; margin-bottom: 22px; }
.input-group-label { font-size: 11px; font-weight: 700; color: #BE185D !important; text-transform: uppercase; letter-spacing: 0.7px; margin-bottom: 4px; }

.result-low, .result-normal { background: #ECFDF5; border: 2px solid #10B981; border-radius: 14px; padding: 22px 26px; text-align: center; color: #111827 !important; }
.result-mid, .result-suspect { background: #FFF7ED; border: 2px solid #F59E0B; border-radius: 14px; padding: 22px 26px; text-align: center; color: #111827 !important; }
.result-high, .result-pathological { background: #FFF1F2; border: 2px solid #F43F5E; border-radius: 14px; padding: 22px 26px; text-align: center; color: #111827 !important; }
.result-title { font-family: 'DM Serif Display', serif; font-size: 22px; font-weight: 600; margin-bottom: 8px; margin-top: 6px; }
.result-desc { font-size: 14px; color: #374151 !important; line-height: 1.6; font-weight: 500; }

.metric-card { background: #FFFFFF; border: 1px solid #F9A8D4; border-radius: 12px; padding: 18px 16px; text-align: center; box-shadow: 0 4px 12px rgba(236,72,153,0.08); }
.metric-value { font-size: 26px; font-weight: 700; color: #BE185D !important; }
.metric-label { font-size: 12px; color: #6B7280 !important; margin-top: 2px; }

.disclaimer { background: #FFF1F2; border-left: 4px solid #E11D48; border-radius: 0 10px 10px 0; padding: 13px 16px; font-size: 13px; color: #7F1D1D !important; font-weight: 500; margin-top: 16px; }

[data-testid="stDataFrame"] { border-radius: 10px; overflow: hidden; }

section[data-testid="stSidebar"] { width: 330px !important; }
section[data-testid="stSidebar"] > div { width: 330px !important; }

/* Auth page styles */
.auth-card {
    background: #FFFFFF; border: 1.5px solid #FBCFE8; border-radius: 20px;
    padding: 40px 44px; box-shadow: 0 8px 32px rgba(236,72,153,0.10);
    max-width: 480px; margin: 0 auto;
}
.auth-title {
    font-family: 'DM Serif Display', serif; font-size: 30px; color: #9D174D;
    text-align: center; margin-bottom: 6px;
}
.auth-sub { font-size: 13px; color: #6B7280; text-align: center; margin-bottom: 28px; }
.role-badge-user {
    display:inline-block; background:#EFF6FF; color:#2563EB; font-size:11px;
    font-weight:700; padding:3px 10px; border-radius:20px; border:1px solid #BFDBFE;
}
.role-badge-clinician {
    display:inline-block; background:#FDF4FF; color:#7C3AED; font-size:11px;
    font-weight:700; padding:3px 10px; border-radius:20px; border:1px solid #E9D5FF;
}
.access-denied {
    background: #FFF1F2; border: 2px solid #F43F5E; border-radius: 14px;
    padding: 30px; text-align: center;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# SESSION STATE INIT
# ─────────────────────────────────────────────
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "user_info" not in st.session_state:
    st.session_state.user_info = {}
if "username" not in st.session_state:
    st.session_state.username = ""
if "auth_tab" not in st.session_state:
    st.session_state.auth_tab = "login"

# ─────────────────────────────────────────────
# AUTH PAGE (shown when not logged in)
# ─────────────────────────────────────────────
def show_auth_page():
    st.markdown("<br>", unsafe_allow_html=True)
    # Header
    col_l, col_c, col_r = st.columns([1, 2, 1])
    with col_c:
        st.markdown("""
        <div style='text-align:center; margin-bottom:32px;'>
            <div style='font-size:52px; margin-bottom:10px;'>🏥</div>
            <div style='font-family:"DM Serif Display",serif; font-size:36px; color:#9D174D;'>MedPredict</div>
            <div style='font-size:14px; color:#9CA3AF; margin-top:6px;'>Maternal &amp; Fetal Health Intelligence System</div>
        </div>
        """, unsafe_allow_html=True)

        # Tab switcher
        tab_l, tab_r = st.columns(2, gap="small")
        with tab_l:
            if st.button("🔐  Sign In", key="tab_login_btn", use_container_width=True):
                st.session_state.auth_tab = "login"
                st.rerun()
        with tab_r:
            if st.button("📝  Create Account", key="tab_signup_btn", use_container_width=True):
                st.session_state.auth_tab = "signup"
                st.rerun()

        st.markdown("<br>", unsafe_allow_html=True)

        if st.session_state.auth_tab == "login":
            st.markdown("""
            <div class='auth-card'>
                <div class='auth-title'>Welcome Back 👋</div>
                <div class='auth-sub'>Sign in to your MedPredict account</div>
            </div>
            """, unsafe_allow_html=True)

            with st.form("login_form"):
                username = st.text_input("Username", placeholder="Enter your username")
                password = st.text_input("Password", type="password", placeholder="Enter your password")
                st.markdown("<br>", unsafe_allow_html=True)
                login_btn = st.form_submit_button("🔐  Sign In", use_container_width=True)

            if login_btn:
                if not username or not password:
                    st.error("Please fill in all fields.")
                else:
                    ok, msg, info = login_user(username, password)
                    if ok:
                        st.session_state.logged_in = True
                        st.session_state.user_info = info
                        st.session_state.username = username
                        st.success(f"✅ Welcome back, {info['name']}!")
                        st.rerun()
                    else:
                        st.error(f"❌ {msg}")

            st.markdown("""
            <div style='text-align:center; margin-top:16px; font-size:12px; color:#9CA3AF;'>
                Demo clinician: <b>admin</b> / <b>admin123</b>
            </div>
            """, unsafe_allow_html=True)

        else:  # signup
            st.markdown("""
            <div class='auth-card'>
                <div class='auth-title'>Create Account ✨</div>
                <div class='auth-sub'>Join MedPredict — choose your role carefully</div>
            </div>
            """, unsafe_allow_html=True)

            with st.form("signup_form"):
                full_name = st.text_input("Full Name", placeholder="Enter your Full Name")
                username  = st.text_input("Username", placeholder="Choose a unique username")
                password  = st.text_input("Password", type="password", placeholder="Min. 6 characters")
                confirm   = st.text_input("Confirm Password", type="password", placeholder="Re-enter password")

                st.markdown("**Account Role**")
                role = st.selectbox(
                    "Select your role",
                    options=["user", "clinician"],
                    format_func=lambda r: "👤  Normal User — General health awareness" if r == "user"
                                          else "👩‍⚕️  Clinician / Expert — Full clinical access"
                )

                st.markdown("""
                <div style='background:#FFF5F7; border:1px solid #FBCFE8; border-radius:10px;
                            padding:12px 14px; margin:8px 0; font-size:12px; color:#6B7280; line-height:1.7;'>
                    <b style='color:#BE185D;'>👤 Normal User:</b> Access to Maternal Risk, Gestational Diabetes &amp; C-Section Prediction<br>
                    <b style='color:#7C3AED;'>👩‍⚕️ Clinician:</b> Full access including Fetal Health classification &amp; Fetal Birth Weight — requires CTG expertise
                </div>
                """, unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)
                signup_btn = st.form_submit_button("📝  Create Account", use_container_width=True)

            if signup_btn:
                if not all([full_name, username, password, confirm]):
                    st.error("Please fill in all fields.")
                elif password != confirm:
                    st.error("❌ Passwords do not match.")
                else:
                    ok, msg = register_user(username, full_name, password, role)
                    if ok:
                        st.success(f"✅ {msg} Please sign in.")
                        st.session_state.auth_tab = "login"
                        st.rerun()
                    else:
                        st.error(f"❌ {msg}")

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
# FEATURE ENGINEERING — GESTATIONAL DIABETES (30 features)
# ─────────────────────────────────────────────
def engineer_gdm_features(pregnancies, glucose, blood_pressure, skin_thickness,
                           insulin, bmi, diabetes_pedigree, age):
    d = {}
    # Raw features
    d['Pregnancies']        = pregnancies
    d['Glucose']            = glucose
    d['BloodPressure']      = blood_pressure
    d['SkinThickness']      = skin_thickness
    d['Insulin']            = insulin
    d['BMI']                = bmi
    d['DiabetesPedigree']   = diabetes_pedigree
    d['Age']                = age

    # Clinical thresholds (IADPSG criteria)
    d['HighGlucose']        = int(glucose >= 140)
    d['VeryHighGlucose']    = int(glucose >= 200)
    d['Obese']              = int(bmi >= 30)
    d['Overweight']         = int(25 <= bmi < 30)
    d['HighInsulin']        = int(insulin > 166)
    d['HighBP']             = int(blood_pressure > 90)
    d['ElderlyMother']      = int(age > 35)
    d['MultiParity']        = int(pregnancies >= 3)

    # Interaction & derived features
    d['GlucoseBMI']         = glucose * bmi / 100
    d['GlucoseAge']         = glucose * age / 100
    d['InsulinGlucoseRatio']= insulin / (glucose + 1e-6)
    d['BMIAge']             = bmi * age / 100
    d['PregnancyAge']       = pregnancies * age
    d['PedigreeAge']        = diabetes_pedigree * age
    d['GlucoseInsulin']     = glucose * insulin / 1000

    # Risk scoring
    d['GlucoseCategory']    = 0 if glucose < 100 else (1 if glucose < 126 else 2)
    d['BMICategory']        = 0 if bmi < 18.5 else (1 if bmi < 25 else (2 if bmi < 30 else 3))
    d['InsulinResistance']  = int((glucose > 140) and (insulin > 166))
    d['MetabolicRisk']      = int(d['HighGlucose'] + d['Obese'] + d['HighBP'] + d['HighInsulin'])
    d['RiskScore']          = (d['HighGlucose'] + d['Obese'] + d['HighBP'] +
                               d['ElderlyMother'] + d['MultiParity'] +
                               int(diabetes_pedigree > 0.5))

    # HOMA-IR approximation
    d['HOMA_IR']            = (glucose * insulin) / 405
    d['SkinBMIRatio']       = skin_thickness / (bmi + 1e-6)

    return np.array(list(d.values())).reshape(1, -1)

# ─────────────────────────────────────────────
# FEATURE ENGINEERING — C-SECTION (35 features)
# ─────────────────────────────────────────────
def engineer_csection_features(age, bmi, systolic_bp, diastolic_bp,
                                gestational_age, parity, prev_csection,
                                fetal_presentation, labor_duration,
                                cervical_dilation, fetal_weight_est,
                                maternal_height, amniotic_fluid):
    d = {}
    # Raw inputs
    d['Age']                = age
    d['BMI']                = bmi
    d['SystolicBP']         = systolic_bp
    d['DiastolicBP']        = diastolic_bp
    d['GestationalAge']     = gestational_age
    d['Parity']             = parity
    d['PreviousCSections']  = prev_csection
    d['FetalPresentation']  = fetal_presentation   # 0=cephalic, 1=breech, 2=transverse
    d['LaborDuration']      = labor_duration        # hours
    d['CervicalDilation']   = cervical_dilation     # cm
    d['EstFetalWeight']     = fetal_weight_est      # kg
    d['MaternalHeight']     = maternal_height       # cm
    d['AmnioticFluidIndex'] = amniotic_fluid        # AFI cm

    # Clinical flags
    d['PrevCSectionFlag']   = int(prev_csection > 0)
    d['MultiplePrevCS']     = int(prev_csection >= 2)
    d['BreechPresentation'] = int(fetal_presentation == 1)
    d['AbnormalPresent']    = int(fetal_presentation > 0)
    d['Macrosomia']         = int(fetal_weight_est >= 4.0)
    d['ObeseFlag']          = int(bmi >= 30)
    d['ElderlyMother']      = int(age > 35)
    d['PostTerm']           = int(gestational_age > 41)
    d['Preterm']            = int(gestational_age < 37)
    d['ProlongedLabor']     = int(labor_duration > 18)
    d['NulliparousFlag']    = int(parity == 0)
    d['HypertensionFlag']   = int(systolic_bp > 140)
    d['OligohydramniosFlag']= int(amniotic_fluid < 5)
    d['PolyhydramniosFlag'] = int(amniotic_fluid > 24)

    # Derived features
    d['PulsePressure']      = systolic_bp - diastolic_bp
    d['MAP']                = diastolic_bp + (systolic_bp - diastolic_bp) / 3
    d['BMI_EFW_Ratio']      = bmi / (fetal_weight_est + 1e-6)
    d['Pelvis_EFW_Ratio']   = maternal_height / (fetal_weight_est * 10 + 1e-6)
    d['LaborProgress']      = cervical_dilation / (labor_duration + 1e-6)   # cm/hr
    d['AgeParityRatio']     = age / (parity + 1)
    d['GestAge_EFW']        = gestational_age * fetal_weight_est

    # Risk score
    d['CSectionRiskScore']  = (d['PrevCSectionFlag'] + d['BreechPresentation'] +
                               d['Macrosomia'] + d['ProlongedLabor'] +
                               d['ElderlyMother'] + d['HypertensionFlag'])

    return np.array(list(d.values())).reshape(1, -1)

# ─────────────────────────────────────────────
# FEATURE ENGINEERING — FETAL BIRTH WEIGHT (32 features)
# ─────────────────────────────────────────────
def engineer_fbw_features(gestational_age, maternal_age, maternal_weight,
                           maternal_height, pre_pregnancy_bmi, weight_gain,
                           fundal_height, biparietal_diameter, head_circumference,
                           abdominal_circumference, femur_length,
                           parity, gravida, glucose_level, amniotic_fluid):
    d = {}
    # Raw inputs
    d['GestationalAge']         = gestational_age
    d['MaternalAge']            = maternal_age
    d['MaternalWeight']         = maternal_weight      # kg
    d['MaternalHeight']         = maternal_height      # cm
    d['PrePregnancyBMI']        = pre_pregnancy_bmi
    d['GestationalWeightGain']  = weight_gain          # kg
    d['FundalHeight']           = fundal_height        # cm
    d['BPD']                    = biparietal_diameter  # cm
    d['HC']                     = head_circumference   # cm
    d['AC']                     = abdominal_circumference  # cm
    d['FL']                     = femur_length         # cm
    d['Parity']                 = parity
    d['Gravida']                = gravida
    d['MaternalGlucose']        = glucose_level
    d['AFI']                    = amniotic_fluid

    # Ultrasound-derived (Hadlock-inspired features)
    d['HC_AC_Ratio']            = head_circumference / (abdominal_circumference + 1e-6)
    d['FL_HC_Ratio']            = femur_length / (head_circumference + 1e-6)
    d['FL_AC_Ratio']            = femur_length / (abdominal_circumference + 1e-6)
    d['BPD_AC_Product']         = biparietal_diameter * abdominal_circumference
    d['HadlockScore']           = (0.5336 * abdominal_circumference +
                                   0.1714 * femur_length +
                                   0.0664 * head_circumference) / 10  # simplified

    # Maternal features
    d['CurrentBMI']             = maternal_weight / ((maternal_height / 100) ** 2 + 1e-6)
    d['BMIChange']              = d['CurrentBMI'] - pre_pregnancy_bmi
    d['WeightGainPerWeek']      = weight_gain / (gestational_age + 1e-6)
    d['FH_GA_Ratio']            = fundal_height / (gestational_age + 1e-6)

    # Clinical flags
    d['ExcessWeightGain']       = int(weight_gain > 16)
    d['LowWeightGain']          = int(weight_gain < 7)
    d['ObeseMother']            = int(pre_pregnancy_bmi >= 30)
    d['HighGlucose']            = int(glucose_level > 130)
    d['Macrosomia_Risk']        = int(abdominal_circumference > 35 and gestational_age > 36)
    d['IUGR_Risk']              = int(abdominal_circumference < 28 and gestational_age > 36)
    d['Polyhydramnios']         = int(amniotic_fluid > 24)
    d['Oligohydramnios']        = int(amniotic_fluid < 5)

    # Gestational age category
    d['TermCategory']           = 0 if gestational_age < 34 else (1 if gestational_age < 37 else (2 if gestational_age <= 41 else 3))

    return np.array(list(d.values())).reshape(1, -1)

# ─────────────────────────────────────────────
# SHAP EXPLAINABILITY HELPERS
# ─────────────────────────────────────────────
@st.cache_resource
def get_maternal_explainer(_model):
    return shap.TreeExplainer(_model)

def compute_maternal_shap(explainer, features_scaled):
    return explainer.shap_values(features_scaled)

def compute_fetal_shap(model, features_scaled):
    background  = np.zeros((1, features_scaled.shape[1]))
    explainer   = shap.KernelExplainer(model.predict_proba, background)
    return explainer.shap_values(features_scaled, nsamples=100, silent=True)

def render_shap_chart(shap_vals, feature_names, prediction_class, title, top_n=10):
    try:
        if isinstance(shap_vals, list):
            raw = np.array(shap_vals[prediction_class])
            vals = raw.flatten()[:len(feature_names)]
        elif isinstance(shap_vals, np.ndarray):
            if shap_vals.ndim == 3:
                vals = shap_vals[0, :, prediction_class]
            elif shap_vals.ndim == 2:
                vals = shap_vals[0]
            else:
                vals = shap_vals
        else:
            vals = np.array(shap_vals).flatten()[:len(feature_names)]

        vals = np.array(vals, dtype=float).flatten()[:len(feature_names)]
        df = pd.DataFrame({"Feature": feature_names[:len(vals)], "SHAP": vals})
        df["Abs"] = df["SHAP"].abs()
        df = df.nlargest(top_n, "Abs").sort_values("SHAP")

        colors = ["#EC4899" if v > 0 else "#3B82F6" for v in df["SHAP"]]
        labels = [f"+{v:.3f}" if v > 0 else f"{v:.3f}" for v in df["SHAP"]]

        fig = go.Figure(go.Bar(
            x=df["SHAP"], y=df["Feature"], orientation="h",
            marker_color=colors, text=labels, textposition="outside",
            textfont=dict(size=11, color="#374151"),
            hovertemplate="<b>%{y}</b><br>SHAP value: %{x:.4f}<extra></extra>"
        ))
        fig.add_vline(x=0, line_color="#9CA3AF", line_width=1.5)
        fig.update_layout(
            title=dict(text=title, font=dict(size=13, color="#9D174D"), x=0),
            plot_bgcolor="white", paper_bgcolor="white",
            margin=dict(l=10, r=60, t=36, b=10), height=340,
            xaxis=dict(title="SHAP Value  (← opposes prediction  |  supports prediction →)",
                       gridcolor="#F3F4F6", zeroline=False, tickfont=dict(size=10, color="#6B7280")),
            yaxis=dict(tickfont=dict(size=11, color="#374151"), automargin=True),
        )
        return fig
    except Exception as e:
        raise ValueError(f"render_shap_chart failed: {e}")

# ─────────────────────────────────────────────
# RULE-BASED GDM PREDICTION (until model trained)
# ─────────────────────────────────────────────
def predict_gdm_rule_based(glucose, bmi, age, insulin, blood_pressure,
                            pregnancies, diabetes_pedigree):
    """
    Clinical rule-based GDM risk scoring using IADPSG + WHO criteria.
    Returns (risk_level: 0/1/2, confidence %, risk_signals list)
    Replace this with your trained model once available.
    """
    score = 0
    signals = []

    if glucose >= 200:
        score += 4
        signals.append(("🔴", "Critically High Glucose", f"{glucose} mg/dL — Diagnostic of diabetes"))
    elif glucose >= 140:
        score += 3
        signals.append(("🟠", "High Fasting Glucose", f"{glucose} mg/dL — Above GDM threshold (≥140)"))
    elif glucose >= 100:
        score += 1
        signals.append(("🟡", "Borderline Glucose", f"{glucose} mg/dL — Pre-diabetic range (100-139)"))

    if bmi >= 35:
        score += 3
        signals.append(("🔴", "Morbid Obesity", f"BMI {bmi:.1f} — High GDM risk factor"))
    elif bmi >= 30:
        score += 2
        signals.append(("🟠", "Obesity", f"BMI {bmi:.1f} — Significant GDM risk"))
    elif bmi >= 25:
        score += 1
        signals.append(("🟡", "Overweight", f"BMI {bmi:.1f} — Moderate risk factor"))

    if age > 40:
        score += 2
        signals.append(("🟠", "Advanced Maternal Age", f"Age {age} — Very high GDM risk"))
    elif age > 35:
        score += 1
        signals.append(("🟡", "Older Mother", f"Age {age} — Elevated GDM risk"))

    if insulin > 200:
        score += 2
        signals.append(("🟠", "Hyperinsulinemia", f"Insulin {insulin} µU/mL — Severe insulin resistance"))
    elif insulin > 166:
        score += 1
        signals.append(("🟡", "High Insulin", f"Insulin {insulin} µU/mL — Insulin resistance"))

    if blood_pressure > 90:
        score += 1
        signals.append(("🟡", "Elevated Diastolic BP", f"{blood_pressure} mmHg — Associated with GDM"))

    if pregnancies >= 4:
        score += 1
        signals.append(("🟡", "High Parity", f"{pregnancies} pregnancies — Increased risk"))

    if diabetes_pedigree > 0.8:
        score += 2
        signals.append(("🟠", "Strong Family History", f"Pedigree {diabetes_pedigree:.2f} — High hereditary risk"))
    elif diabetes_pedigree > 0.5:
        score += 1
        signals.append(("🟡", "Family History", f"Pedigree {diabetes_pedigree:.2f} — Moderate hereditary risk"))

    # HOMA-IR
    homa = (glucose * insulin) / 405
    if homa > 3.5:
        score += 2
        signals.append(("🟠", "High HOMA-IR", f"{homa:.1f} — Significant insulin resistance"))

    # Classification
    max_score = 18
    pct = min(score / max_score, 1.0)
    if score <= 2:
        risk, conf = 0, round(85 - pct * 20, 1)
    elif score <= 5:
        risk, conf = 1, round(60 + pct * 20, 1)
    else:
        risk, conf = 2, round(70 + pct * 15, 1)

    return risk, conf, signals

# ─────────────────────────────────────────────
# RULE-BASED C-SECTION PREDICTION
# ─────────────────────────────────────────────
def predict_csection_rule_based(age, bmi, systolic_bp, prev_csection,
                                 fetal_presentation, labor_duration,
                                 cervical_dilation, fetal_weight_est,
                                 gestational_age, amniotic_fluid, parity):
    score = 0
    signals = []

    if prev_csection >= 2:
        score += 5
        signals.append(("🔴", "Multiple Previous C-Sections", f"{prev_csection} prior CS — High repeat risk"))
    elif prev_csection == 1:
        score += 3
        signals.append(("🟠", "Previous C-Section", "Trial of Labour After CS (TOLAC) candidate"))

    if fetal_presentation == 1:
        score += 4
        signals.append(("🔴", "Breech Presentation", "Vaginal breech delivery rarely attempted"))
    elif fetal_presentation == 2:
        score += 5
        signals.append(("🔴", "Transverse Lie", "C-section mandatory for transverse presentation"))

    if fetal_weight_est >= 4.5:
        score += 3
        signals.append(("🔴", "Severe Macrosomia", f"Est. weight {fetal_weight_est:.1f} kg — Very high CS risk"))
    elif fetal_weight_est >= 4.0:
        score += 2
        signals.append(("🟠", "Macrosomia", f"Est. weight {fetal_weight_est:.1f} kg — Elevated CS risk"))

    if labor_duration > 24:
        score += 3
        signals.append(("🔴", "Arrested Labor", f"{labor_duration}h — Failure to progress"))
    elif labor_duration > 18:
        score += 2
        signals.append(("🟠", "Prolonged Labor", f"{labor_duration}h — Risk of dystocia"))

    progress = cervical_dilation / (labor_duration + 1e-6)
    if progress < 0.3 and labor_duration > 4:
        score += 2
        signals.append(("🟠", "Slow Labor Progress", f"{progress:.2f} cm/hr — Below normal (>0.5 cm/hr)"))

    if systolic_bp > 160:
        score += 2
        signals.append(("🔴", "Severe Hypertension", f"SBP {systolic_bp} mmHg — Emergency indication"))
    elif systolic_bp > 140:
        score += 1
        signals.append(("🟡", "Hypertension", f"SBP {systolic_bp} mmHg"))

    if bmi >= 40:
        score += 2
        signals.append(("🟠", "Morbid Obesity", f"BMI {bmi:.1f} — Increased operative risk"))
    elif bmi >= 35:
        score += 1
        signals.append(("🟡", "Severe Obesity", f"BMI {bmi:.1f}"))

    if age > 40:
        score += 1
        signals.append(("🟡", "Advanced Maternal Age", f"Age {age}"))

    if amniotic_fluid < 5:
        score += 2
        signals.append(("🟠", "Oligohydramnios", f"AFI {amniotic_fluid} cm — Possible fetal compromise"))
    elif amniotic_fluid > 24:
        score += 1
        signals.append(("🟡", "Polyhydramnios", f"AFI {amniotic_fluid} cm"))

    if parity == 0 and gestational_age > 41:
        score += 1
        signals.append(("🟡", "Nulliparous Post-term", "First delivery after 41 weeks"))

    max_score = 20
    pct = min(score / max_score, 1.0)
    if score <= 3:
        risk, conf = 0, round(80 - pct * 15, 1)   # Normal delivery likely
    elif score <= 7:
        risk, conf = 1, round(55 + pct * 20, 1)   # C-section possible
    else:
        risk, conf = 2, round(75 + pct * 10, 1)   # C-section very likely

    return risk, conf, signals

# ─────────────────────────────────────────────
# RULE-BASED FETAL BIRTH WEIGHT ESTIMATION
# ─────────────────────────────────────────────
def estimate_fetal_birth_weight(gestational_age, bpd, hc, ac, fl,
                                 maternal_weight, weight_gain,
                                 glucose_level, parity, amniotic_fluid):
    """
    Hadlock Formula 4 (most accurate): uses HC, AC, FL
    EFW (g) = 10^(1.3596 + 0.0064*HC + 0.0424*AC + 0.174*FL + 0.00061*BPD*AC - 0.00386*AC*FL)
    Returns estimated weight in grams.
    """
    # Hadlock Formula 4 (log10 based)
    log_efw = (1.3596
               + 0.0064 * hc
               + 0.0424 * ac
               + 0.174  * fl
               + 0.00061 * bpd * ac
               - 0.00386 * ac * fl)
    efw_g = 10 ** log_efw

    # Maternal adjustment factors
    if glucose_level > 130:
        efw_g *= 1.04   # +4% for GDM macrosomia
    if weight_gain > 20:
        efw_g *= 1.02   # +2% for excess weight gain
    if parity > 3:
        efw_g *= 1.015  # +1.5% multiparous

    # Gestational age normalization
    if gestational_age < 37:
        ga_factor = gestational_age / 40
        efw_g *= (0.7 + 0.3 * ga_factor)

    efw_kg  = efw_g / 1000
    signals = []

    if efw_g > 4500:
        category, cat_color = "Severe Macrosomia", "#E11D48"
        signals.append(("🔴", "Severe Macrosomia", f"{efw_g:.0f}g — C-section likely recommended"))
    elif efw_g > 4000:
        category, cat_color = "Macrosomia", "#F59E0B"
        signals.append(("🟠", "Macrosomia", f"{efw_g:.0f}g — Monitor closely, discuss delivery mode"))
    elif efw_g >= 2500:
        category, cat_color = "Normal Weight", "#10B981"
    elif efw_g >= 1500:
        category, cat_color = "Low Birth Weight", "#F59E0B"
        signals.append(("🟠", "Low Birth Weight", f"{efw_g:.0f}g — Neonatal care preparation advised"))
    else:
        category, cat_color = "Very Low Birth Weight", "#E11D48"
        signals.append(("🔴", "Very Low Birth Weight", f"{efw_g:.0f}g — NICU readiness required"))

    # Additional clinical signals
    if glucose_level > 130:
        signals.append(("🟡", "Maternal Hyperglycemia", f"{glucose_level} mg/dL — GDM macrosomia risk"))
    if amniotic_fluid > 24:
        signals.append(("🟡", "Polyhydramnios", f"AFI {amniotic_fluid} cm — Often associated with macrosomia"))
    if amniotic_fluid < 5:
        signals.append(("🟠", "Oligohydramnios", f"AFI {amniotic_fluid} cm — Possible IUGR"))

    # Percentile estimation (simplified Hadlock tables)
    ga_50th = {36: 2600, 37: 2950, 38: 3100, 39: 3250, 40: 3400, 41: 3500, 42: 3550}
    ref_50th = ga_50th.get(int(gestational_age), 3400)
    percentile_approx = min(max(int(50 * (efw_g / ref_50th)), 3), 97)

    return efw_g, efw_kg, category, cat_color, percentile_approx, signals

# ─────────────────────────────────────────────
# MAIN APP LOGIC
# ─────────────────────────────────────────────
if not st.session_state.logged_in:
    show_auth_page()
    st.stop()

# ── Load models only when logged in ──
try:
    maternal_model, maternal_scaler, fetal_model, fetal_scaler = load_models()
    models_loaded = True
except Exception:
    models_loaded = False

user_info = st.session_state.user_info
is_clinician = user_info.get("role") == "clinician"

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"""
    <div style='text-align:center; padding:16px 0 8px 0;'>
        <div style='font-family:"DM Serif Display",serif; font-size:26px; color:#9D174D;'>MedPredict</div>
        <div style='font-size:12px; color:#9CA3AF; margin-top:4px;'>
            Maternal Health Risk &amp; Fetal Health Monitoring System
        </div>
    </div>
    <hr style='border:none; border-top:1.5px solid #FADADD; margin:12px 0;'>
    <div style='background:#FFF5F7; border:1px solid #FBCFE8; border-radius:10px;
                padding:10px 14px; margin-bottom:12px;'>
        <div style='font-size:12px; color:#9CA3AF; margin-bottom:3px;'>Signed in as</div>
        <div style='font-size:14px; font-weight:700; color:#1F2937;'>{user_info.get("name","")}</div>
        <div style='margin-top:4px;'>
            {"<span class='role-badge-clinician'>👩‍⚕️ Clinician</span>" if is_clinician else "<span class='role-badge-user'>👤 Normal User</span>"}
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Build nav options by role
    nav_options = ["About", "Maternal Risk", "Gestational Diabetes", "C-Section Prediction"]
    nav_icons   = ["house-heart", "person-heart", "droplet-half", "hospital"]

    if is_clinician:
        nav_options += ["Fetal Health", "Fetal Birth Weight"]
        nav_icons   += ["heart-pulse", "rulers"]

    nav_options.append("System Performance Dashboard")
    nav_icons.append("bar-chart")

    selected = option_menu(
        menu_title=None,
        options=nav_options,
        icons=nav_icons,
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
    """, unsafe_allow_html=True)

    if st.button("🚪  Sign Out", key="signout_btn"):
        st.session_state.logged_in = False
        st.session_state.user_info = {}
        st.session_state.username  = ""
        st.rerun()

    st.markdown("""
    <div style='font-size:11px; color:#D1A8B8; text-align:center; line-height:1.6; margin-top:8px;'>
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
            <div style='font-size:17px; font-weight:700; color:#9D174D; margin-bottom:8px;'>Maternal Health Risk Prediction</div>
            <div style='font-size:14px; color:#4B5563; line-height:1.75;'>
                Analyzes 6 clinical parameters extended to <b>26 engineered features</b>
                to classify pregnancy risk as <b>Low</b>, <b>Mid</b>, or <b>High</b>.
                Powered by LightGBM with Optuna hyperparameter tuning.
            </div>
            <div style='margin-top:14px; display:flex; gap:8px; flex-wrap:wrap;'>
                <span style='background:#F0FDF4; color:#16A34A; font-size:11px; font-weight:600; padding:3px 10px; border-radius:20px; border:1px solid #BBF7D0;'>90.64% Accuracy</span>
                <span style='background:#FFF5F7; color:#BE185D; font-size:11px; font-weight:600; padding:3px 10px; border-radius:20px; border:1px solid #FBCFE8;'>LightGBM</span>
                <span style='background:#F5F3FF; color:#7C3AED; font-size:11px; font-weight:600; padding:3px 10px; border-radius:20px; border:1px solid #DDD6FE;'>26 Features</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="info-card">
            <div style='font-size:34px; margin-bottom:10px;'>👶</div>
            <div style='font-size:17px; font-weight:700; color:#9D174D; margin-bottom:8px;'>Fetal Health Classification</div>
            <div style='font-size:14px; color:#4B5563; line-height:1.75;'>
                Interprets 21 CTG measurements extended to <b>39 features</b> to classify
                fetal condition as <b>Normal</b>, <b>Suspect</b>, or <b>Pathological</b>.
                Stacking Ensemble with ±0.59% CV stability.
            </div>
            <div style='margin-top:14px; display:flex; gap:8px; flex-wrap:wrap;'>
                <span style='background:#F0FDF4; color:#16A34A; font-size:11px; font-weight:600; padding:3px 10px; border-radius:20px; border:1px solid #BBF7D0;'>94.13% Accuracy</span>
                <span style='background:#FFF5F7; color:#BE185D; font-size:11px; font-weight:600; padding:3px 10px; border-radius:20px; border:1px solid #FBCFE8;'>Stacking Ensemble</span>
                <span style='background:#F5F3FF; color:#7C3AED; font-size:11px; font-weight:600; padding:3px 10px; border-radius:20px; border:1px solid #DDD6FE;'>39 Features</span>
                <span style='background:#FEF3C7; color:#92400E; font-size:11px; font-weight:600; padding:3px 10px; border-radius:20px; border:1px solid #FDE68A;'>👩‍⚕️ Clinician Only</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    col3, col4 = st.columns(2, gap="large")
    with col3:
        st.markdown("""
        <div class="info-card">
            <div style='font-size:34px; margin-bottom:10px;'>🩸</div>
            <div style='font-size:17px; font-weight:700; color:#9D174D; margin-bottom:8px;'>Gestational Diabetes Risk</div>
            <div style='font-size:14px; color:#4B5563; line-height:1.75;'>
                Assesses GDM risk using 8 clinical inputs extended to <b>30 engineered features</b>
                including HOMA-IR, glucose-BMI interaction, and IADPSG criteria flags.
                Clinical rule-based scoring pending model training.
            </div>
            <div style='margin-top:14px; display:flex; gap:8px; flex-wrap:wrap;'>
                <span style='background:#FFF5F7; color:#BE185D; font-size:11px; font-weight:600; padding:3px 10px; border-radius:20px; border:1px solid #FBCFE8;'>IADPSG Criteria</span>
                <span style='background:#F5F3FF; color:#7C3AED; font-size:11px; font-weight:600; padding:3px 10px; border-radius:20px; border:1px solid #DDD6FE;'>30 Features</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown("""
        <div class="info-card">
            <div style='font-size:34px; margin-bottom:10px;'>🏥</div>
            <div style='font-size:17px; font-weight:700; color:#9D174D; margin-bottom:8px;'>C-Section vs Normal Delivery</div>
            <div style='font-size:14px; color:#4B5563; line-height:1.75;'>
                Predicts likelihood of Caesarean delivery using <b>35 engineered features</b>
                from maternal, fetal, and labor parameters. Supports obstetric planning
                and resource allocation.
            </div>
            <div style='margin-top:14px; display:flex; gap:8px; flex-wrap:wrap;'>
                <span style='background:#F0FDF4; color:#16A34A; font-size:11px; font-weight:600; padding:3px 10px; border-radius:20px; border:1px solid #BBF7D0;'>Evidence-Based</span>
                <span style='background:#F5F3FF; color:#7C3AED; font-size:11px; font-weight:600; padding:3px 10px; border-radius:20px; border:1px solid #DDD6FE;'>35 Features</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class="disclaimer">
        ⚠️ <b>Clinical Disclaimer:</b> MedPredict is a decision support tool only. All predictions must be reviewed by a qualified healthcare professional. Not approved for standalone diagnostic use.
    </div>
    """, unsafe_allow_html=True)


# ═════════════════════════════════════════════
# PAGE: MATERNAL RISK
# ═════════════════════════════════════════════
elif selected == "Maternal Risk":
    st.markdown('<div class="section-header">🤱 Maternal Health Risk Prediction</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Enter the patient\'s clinical parameters. The model classifies risk as Low, Mid, or High using LightGBM (90.64% accuracy).</div>', unsafe_allow_html=True)

    if not models_loaded:
        st.error("⚠️ Model files not found. Please ensure model files are in the `model/` folder.")
        st.stop()

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
            bs = st.number_input("Blood Glucose (mmol/L)", min_value=1.0, max_value=25.0, value=5.0, step=0.1, format="%.1f")

        with col2:
            st.markdown('<div class="input-group-label">🌡️ Vitals</div>', unsafe_allow_html=True)
            body_temp = st.number_input("Body Temperature (°C)", min_value=34.0, max_value=42.0, value=37.0, step=0.1, format="%.1f")

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
            probs           = maternal_model.predict_proba(features_scaled)
            confidence      = max(probs[0]) * 100

            st.markdown("<br>", unsafe_allow_html=True)

            if prediction == 0:
                st.markdown(f"""
                <div class="result-low">
                    <div style='font-size:35px;'>✅</div>
                    <div class="result-title" style='color:#059669;'>Low Risk</div>
                    <div style='margin-top:8px; font-size:14px; font-weight:600; color:#065F46;'>Confidence: {confidence:.2f}%</div>
                    <div class="result-desc">Parameters are within normal ranges. Continue routine prenatal care and regular scheduled monitoring. No immediate intervention required.</div>
                </div>""", unsafe_allow_html=True)
            elif prediction == 1:
                st.markdown(f"""
                <div class="result-mid">
                    <div style='font-size:35px;'>⚠️</div>
                    <div class="result-title" style='color:#D97706;'>Mid Risk</div>
                    <div style='margin-top:8px; font-size:14px; font-weight:600; color:#92400E;'>Confidence: {confidence:.2f}%</div>
                    <div class="result-desc">Some parameters indicate elevated risk. Enhanced monitoring and specialist consultation recommended. Schedule follow-up within 1–2 weeks.</div>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="result-high">
                    <div style='font-size:35px;'>🚨</div>
                    <div class="result-title" style='color:#E11D48;'>High Risk</div>
                    <div style='margin-top:8px; font-size:14px; font-weight:600; color:#9F1239;'>Confidence: {confidence:.2f}%</div>
                    <div class="result-desc">Critical parameters detected. Immediate medical attention required. Refer to a specialist or hospital without delay.</div>
                </div>""", unsafe_allow_html=True)

            st.markdown("""
            <div class="disclaimer">⚠️ This prediction is for clinical decision support only. A qualified healthcare professional must review all results before any medical action.</div>
            """, unsafe_allow_html=True)

            # SHAP
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("""
            <div class="info-card">
                <div style='font-size:14px; font-weight:700; color:#9D174D; margin-bottom:4px;'>🔍 Model Explainability — Why did the model predict this?</div>
                <div style='font-size:12px; color:#6B7280; margin-bottom:14px;'>
                    SHAP shows how each feature contributed to this prediction.
                    <b style='color:#EC4899;'>Pink bars</b> push toward this prediction.
                    <b style='color:#3B82F6;'>Blue bars</b> push away from it.
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
                    fig_shap  = render_shap_chart(shap_vals, maternal_feature_names,
                                                  int(prediction),
                                                  f"Top 10 Feature Contributions — {['Low Risk','Mid Risk','High Risk'][int(prediction)]}")
                    st.plotly_chart(fig_shap, use_container_width=True)
                except Exception as se:
                    st.info(f"SHAP explanation unavailable: {se}")
            st.markdown("</div>", unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Prediction error: {e}")


# ═════════════════════════════════════════════
# PAGE: GESTATIONAL DIABETES
# ═════════════════════════════════════════════
elif selected == "Gestational Diabetes":
    st.markdown('<div class="section-header">🩸 Gestational Diabetes Risk Assessment</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Assess gestational diabetes mellitus (GDM) risk using 8 clinical parameters extended to 30 engineered features. Based on IADPSG and WHO diagnostic criteria. Train a model on the Pima Indians / BRFSS dataset and plug it in.</div>', unsafe_allow_html=True)

    st.markdown("""
    <div style='background:#FFF5F7; border:1px solid #FBCFE8; border-radius:12px;
                padding:14px 18px; margin-bottom:18px; font-size:13px; color:#6B7280;'>
        💡 <b style='color:#BE185D;'>Dataset Recommendation:</b>
        Train on the <b>Pima Indians Diabetes Dataset</b> (UCI ML Repository) or
        <b>BRFSS 2023 Diabetes Dataset</b> for GDM-specific features.
        The 30-feature engineering code is ready — just attach your trained model.
    </div>
    """, unsafe_allow_html=True)

    with st.form("gdm_form"):
        st.markdown("**🔬 Metabolic Parameters (OGTT / Fasting)**")
        c1, c2, c3 = st.columns(3, gap="medium")
        glucose     = c1.number_input("Plasma Glucose (mg/dL)", min_value=0.0, max_value=300.0, value=120.0, step=1.0,
                                       help="Fasting plasma glucose or 1-hour OGTT value")
        insulin     = c2.number_input("Serum Insulin (µU/mL)",  min_value=0.0, max_value=900.0, value=80.0,  step=1.0)
        skin_thick  = c3.number_input("Skin Thickness (mm)",     min_value=0.0, max_value=100.0, value=20.0,  step=1.0,
                                       help="Triceps skin fold thickness")

        st.markdown("---")
        st.markdown("**👤 Maternal Characteristics**")
        c1, c2, c3 = st.columns(3, gap="medium")
        age_gdm     = c1.number_input("Age (years)",             min_value=10,  max_value=60,    value=28,    step=1)
        bmi_gdm     = c2.number_input("BMI (kg/m²)",             min_value=10.0, max_value=60.0, value=26.0,  step=0.1, format="%.1f")
        pregnancies = c3.number_input("Number of Pregnancies",   min_value=0,   max_value=20,    value=1,     step=1)

        st.markdown("---")
        st.markdown("**🩸 Clinical Vitals & Family History**")
        c1, c2, c3 = st.columns(3, gap="medium")
        bp_gdm      = c1.number_input("Diastolic BP (mmHg)",     min_value=0,   max_value=130,   value=70,    step=1)
        pedigree    = c2.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=2.5, value=0.47,  step=0.01, format="%.2f",
                                       help="Genetic risk score for diabetes (0.0 = no family history, >0.8 = strong history)")
        _ = c3  # spacer

        st.markdown("<br>", unsafe_allow_html=True)
        _, mid, _ = st.columns([2, 1, 2])
        with mid:
            submitted_gdm = st.form_submit_button("🔍 Assess GDM Risk", use_container_width=True)

    if submitted_gdm:
        features_gdm = engineer_gdm_features(pregnancies, glucose, bp_gdm, skin_thick,
                                              insulin, bmi_gdm, pedigree, age_gdm)
        risk, conf, signals = predict_gdm_rule_based(
            glucose, bmi_gdm, age_gdm, insulin, bp_gdm, pregnancies, pedigree
        )

        st.markdown("<br>", unsafe_allow_html=True)
        r1, r2 = st.columns([1, 1], gap="large")

        with r1:
            labels = ["Low Risk", "Moderate Risk", "High Risk"]
            icons  = ["✅", "⚠️", "🚨"]
            colors = ["result-low", "result-mid", "result-high"]
            title_colors = ["#059669", "#D97706", "#E11D48"]
            descs = [
                "Current parameters suggest low GDM likelihood. Maintain healthy diet and continue routine screening at 24–28 weeks.",
                "Elevated risk indicators present. Early OGTT testing recommended. Dietary counselling and glucose monitoring advised.",
                "Multiple high-risk factors detected. Immediate OGTT and specialist referral required. Strict glucose monitoring essential."
            ]
            st.markdown(f"""
            <div class="{colors[risk]}">
                <div style='font-size:35px;'>{icons[risk]}</div>
                <div class="result-title" style='color:{title_colors[risk]};'>{labels[risk]}</div>
                <div style='margin-top:8px; font-size:14px; font-weight:600; color:{title_colors[risk]};'>
                    Clinical Confidence: {conf:.1f}%
                </div>
                <div class="result-desc">{descs[risk]}</div>
            </div>""", unsafe_allow_html=True)

        with r2:
            st.markdown("""
            <div class="info-card">
                <div style='font-size:14px; font-weight:700; color:#9D174D; margin-bottom:12px;'>📋 Clinical Risk Signals</div>
            """, unsafe_allow_html=True)
            if signals:
                for icon, name, detail in signals:
                    st.markdown(f"""
                    <div style='display:flex; align-items:flex-start; gap:10px; padding:7px 0;
                                border-bottom:1px solid #FFF0F5;'>
                        <span style='font-size:15px; margin-top:1px;'>{icon}</span>
                        <div>
                            <div style='font-size:13px; font-weight:600; color:#B45309;'>{name}</div>
                            <div style='font-size:12px; color:#9CA3AF;'>{detail}</div>
                        </div>
                    </div>""", unsafe_allow_html=True)
            else:
                st.markdown("""<div style='font-size:13px; color:#059669; font-weight:600; padding:8px 0;'>✅ No significant risk signals detected</div>""", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

            # HOMA-IR display
            homa = (glucose * insulin) / 405
            st.markdown(f"""
            <div class="info-card" style='margin-top:0;'>
                <div style='font-size:13px; font-weight:700; color:#9D174D; margin-bottom:10px;'>📊 Derived Markers</div>
                <div style='display:grid; grid-template-columns:1fr 1fr; gap:10px;'>
                    <div style='text-align:center; background:#FFF5F7; border-radius:8px; padding:10px;'>
                        <div style='font-size:18px; font-weight:700; color:#BE185D;'>{homa:.2f}</div>
                        <div style='font-size:11px; color:#9CA3AF;'>HOMA-IR<br><span style='color:#6B7280;'>(&lt;2.5 normal)</span></div>
                    </div>
                    <div style='text-align:center; background:#FFF5F7; border-radius:8px; padding:10px;'>
                        <div style='font-size:18px; font-weight:700; color:#BE185D;'>{glucose * bmi_gdm / 100:.1f}</div>
                        <div style='font-size:11px; color:#9CA3AF;'>Glucose×BMI<br><span style='color:#6B7280;'>interaction</span></div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("""
        <div class="info-card" style='margin-top:12px; background:#FFFBEB; border-color:#FDE68A;'>
            <div style='font-size:13px; font-weight:700; color:#92400E; margin-bottom:8px;'>🏋️ Model Training Guide</div>
            <div style='font-size:12px; color:#78350F; line-height:1.7;'>
                <b>Recommended Dataset:</b> Pima Indians Diabetes Dataset (UCI) or BRFSS 2023 |
                <b>Target Variable:</b> GDM diagnosis (binary) |
                <b>Suggested Models:</b> LightGBM, XGBoost, Random Forest with SMOTE balancing |
                <b>Feature Function:</b> <code>engineer_gdm_features()</code> already coded — generates 30 features |
                <b>Integration:</b> Replace <code>predict_gdm_rule_based()</code> with <code>gdm_model.predict()</code>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="disclaimer">⚠️ GDM assessment requires formal OGTT testing. This tool provides risk stratification only — not a diagnostic result. Consult a qualified endocrinologist or obstetrician.</div>
        """, unsafe_allow_html=True)


# ═════════════════════════════════════════════
# PAGE: C-SECTION PREDICTION
# ═════════════════════════════════════════════
elif selected == "C-Section Prediction":
    st.markdown('<div class="section-header">🏥 C-Section vs Normal Delivery Prediction</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Predict the likelihood of Caesarean delivery using 13 clinical inputs extended to 35 engineered features. Supports obstetric planning and resource allocation. Train on a delivery outcome dataset to upgrade to ML predictions.</div>', unsafe_allow_html=True)

    with st.form("csection_form"):
        st.markdown("**👤 Maternal Characteristics**")
        c1, c2, c3 = st.columns(3, gap="medium")
        age_cs   = c1.number_input("Age (years)",             min_value=14,  max_value=55,   value=28,  step=1)
        bmi_cs   = c2.number_input("BMI (kg/m²)",             min_value=15.0, max_value=60.0, value=25.0, step=0.1, format="%.1f")
        height_cs= c3.number_input("Maternal Height (cm)",    min_value=130.0, max_value=200.0, value=160.0, step=1.0)

        st.markdown("---")
        st.markdown("**🩸 Blood Pressure & Obstetric History**")
        c1, c2, c3 = st.columns(3, gap="medium")
        sbp_cs   = c1.number_input("Systolic BP (mmHg)",      min_value=80,  max_value=200,  value=120, step=1)
        dbp_cs   = c2.number_input("Diastolic BP (mmHg)",     min_value=40,  max_value=130,  value=80,  step=1)
        parity_cs= c3.number_input("Parity (previous births)",min_value=0,   max_value=10,   value=0,   step=1)
        prev_cs  = c1.number_input("Previous C-Sections",     min_value=0,   max_value=5,    value=0,   step=1)
        ga_cs    = c2.number_input("Gestational Age (weeks)", min_value=28.0, max_value=44.0, value=39.0, step=0.5, format="%.1f")

        st.markdown("---")
        st.markdown("**👶 Fetal & Labor Parameters**")
        c1, c2, c3 = st.columns(3, gap="medium")
        presentation = c1.selectbox("Fetal Presentation",
                                     options=[0, 1, 2],
                                     format_func=lambda x: {0:"Cephalic (head-first)", 1:"Breech (feet-first)", 2:"Transverse (sideways)"}[x])
        efw_cs   = c2.number_input("Estimated Fetal Weight (kg)", min_value=0.5, max_value=6.0, value=3.2, step=0.1, format="%.1f")
        afi_cs   = c3.number_input("Amniotic Fluid Index (cm)",   min_value=0.0, max_value=40.0, value=12.0, step=0.5, format="%.1f")
        labor_h  = c1.number_input("Labor Duration (hours)",      min_value=0.0, max_value=48.0, value=6.0,  step=0.5, format="%.1f")
        cx_dil   = c2.number_input("Cervical Dilation (cm)",      min_value=0.0, max_value=10.0, value=4.0,  step=0.5, format="%.1f")

        st.markdown("<br>", unsafe_allow_html=True)
        _, mid, _ = st.columns([2, 1, 2])
        with mid:
            submitted_cs = st.form_submit_button("🔍 Predict Delivery Mode", use_container_width=True)

    if submitted_cs:
        features_cs = engineer_csection_features(
            age_cs, bmi_cs, sbp_cs, dbp_cs, ga_cs, parity_cs,
            prev_cs, presentation, labor_h, cx_dil, efw_cs, height_cs, afi_cs
        )
        risk_cs, conf_cs, signals_cs = predict_csection_rule_based(
            age_cs, bmi_cs, sbp_cs, prev_cs, presentation,
            labor_h, cx_dil, efw_cs, ga_cs, afi_cs, parity_cs
        )

        st.markdown("<br>", unsafe_allow_html=True)
        r1, r2 = st.columns([1, 1], gap="large")

        with r1:
            labels_cs = ["Normal Delivery Likely", "C-Section Possible", "C-Section Very Likely"]
            icons_cs  = ["✅", "⚠️", "🚨"]
            colors_cs = ["result-low", "result-mid", "result-high"]
            tcols_cs  = ["#059669", "#D97706", "#E11D48"]
            descs_cs  = [
                "Current indicators suggest vaginal delivery is feasible. Continue active labor management and monitoring.",
                "Mixed indicators — both vaginal delivery and C-section possible. Close monitoring and obstetric review advised.",
                "Multiple clinical indicators strongly suggest Caesarean delivery. Prepare surgical team and inform patient."
            ]
            st.markdown(f"""
            <div class="{colors_cs[risk_cs]}">
                <div style='font-size:35px;'>{icons_cs[risk_cs]}</div>
                <div class="result-title" style='color:{tcols_cs[risk_cs]};'>{labels_cs[risk_cs]}</div>
                <div style='margin-top:8px; font-size:14px; font-weight:600; color:{tcols_cs[risk_cs]};'>
                    Clinical Confidence: {conf_cs:.1f}%
                </div>
                <div class="result-desc">{descs_cs[risk_cs]}</div>
            </div>""", unsafe_allow_html=True)

        with r2:
            st.markdown("""
            <div class="info-card">
                <div style='font-size:14px; font-weight:700; color:#9D174D; margin-bottom:12px;'>📋 C-Section Risk Indicators</div>
            """, unsafe_allow_html=True)
            if signals_cs:
                for icon, name, detail in signals_cs:
                    st.markdown(f"""
                    <div style='display:flex; align-items:flex-start; gap:10px; padding:7px 0;
                                border-bottom:1px solid #FFF0F5;'>
                        <span style='font-size:15px; margin-top:1px;'>{icon}</span>
                        <div>
                            <div style='font-size:13px; font-weight:600; color:#B45309;'>{name}</div>
                            <div style='font-size:12px; color:#9CA3AF;'>{detail}</div>
                        </div>
                    </div>""", unsafe_allow_html=True)
            else:
                st.markdown("""<div style='font-size:13px; color:#059669; font-weight:600; padding:8px 0;'>✅ No significant C-section indicators</div>""", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

            # Labor progress indicator
            progress = cx_dil / (labor_h + 1e-6)
            st.markdown(f"""
            <div class="info-card" style='margin-top:0;'>
                <div style='font-size:13px; font-weight:700; color:#9D174D; margin-bottom:10px;'>📊 Labor Progress Metrics</div>
                <div style='display:grid; grid-template-columns:1fr 1fr; gap:10px;'>
                    <div style='text-align:center; background:#FFF5F7; border-radius:8px; padding:10px;'>
                        <div style='font-size:18px; font-weight:700; color:#BE185D;'>{progress:.2f}</div>
                        <div style='font-size:11px; color:#9CA3AF;'>cm/hour<br><span style='color:#6B7280;'>(normal &gt;0.5)</span></div>
                    </div>
                    <div style='text-align:center; background:#FFF5F7; border-radius:8px; padding:10px;'>
                        <div style='font-size:18px; font-weight:700; color:#BE185D;'>{efw_cs:.1f} kg</div>
                        <div style='font-size:11px; color:#9CA3AF;'>Est. fetal<br>weight</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("""
        <div class="disclaimer">⚠️ Delivery mode decisions must be made by a qualified obstetrician considering full clinical context. This tool is for decision support only.</div>
        """, unsafe_allow_html=True)


# ═════════════════════════════════════════════
# PAGE: FETAL HEALTH (Clinician only)
# ═════════════════════════════════════════════
elif selected == "Fetal Health":
    if not is_clinician:
        st.markdown("""
        <div class="access-denied">
            <div style='font-size:48px; margin-bottom:16px;'>🔒</div>
            <div style='font-size:22px; font-weight:700; color:#E11D48; margin-bottom:10px;'>Clinician Access Required</div>
            <div style='font-size:14px; color:#6B7280; line-height:1.7; max-width:500px; margin:0 auto;'>
                The Fetal Health Classification module requires interpretation of complex
                Cardiotocography (CTG) measurements. This section is restricted to registered
                clinicians and healthcare experts.<br><br>
                If you are a healthcare professional, please create a <b>Clinician account</b>
                by signing out and registering with the Clinician role.
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.stop()

    if not models_loaded:
        st.error("⚠️ Model files not found.")
        st.stop()

    st.markdown('<div class="section-header">👶 Fetal Health Classification</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Enter CTG measurements. The Stacking Ensemble classifies fetal condition as Normal, Suspect, or Pathological (94.13% accuracy).</div>', unsafe_allow_html=True)

    with st.form("fetal_form"):
        st.markdown("**📊 Fetal Heart Rate Parameters**")
        c1, c2, c3 = st.columns(3, gap="medium")
        baseline  = c1.number_input("Baseline FHR (bpm)",  min_value=50.0,  max_value=250.0, value=133.0, step=1.0)
        accels    = c2.number_input("Accelerations",        min_value=0.0,   max_value=0.1,   value=0.0015, step=0.001, format="%.3f")
        fetal_mov = c3.number_input("Fetal Movement",       min_value=0.0,   max_value=0.5,   value=0.01,   step=0.001, format="%.3f")

        st.markdown("---")
        st.markdown("**📉 Contractions & Decelerations**")
        c1, c2, c3 = st.columns(3, gap="medium")
        uterine_cont    = c1.number_input("Uterine Contractions",     min_value=0.0, max_value=0.02, value=0.004, step=0.001, format="%.3f")
        light_decel     = c2.number_input("Light Decelerations",      min_value=0.0, max_value=0.02, value=0.0,   step=0.001, format="%.3f")
        severe_decel    = c3.number_input("Severe Decelerations",     min_value=0.0, max_value=0.01, value=0.0,   step=0.0001, format="%.4f")
        c1b, _, _       = st.columns(3, gap="medium")
        prolonged_decel = c1b.number_input("Prolongued Decelerations",min_value=0.0, max_value=0.01, value=0.0,  step=0.0001, format="%.4f")

        st.markdown("---")
        st.markdown("**〰️ Short & Long Term Variability**")
        c1, c2, c3 = st.columns(3, gap="medium")
        abnormal_stv = c1.number_input("Abnormal STV (%)",    min_value=0.0, max_value=100.0, value=47.0, step=1.0)
        mean_stv     = c2.number_input("Mean STV Value",       min_value=0.0, max_value=10.0,  value=1.0,  step=0.1, format="%.1f")
        pct_altv     = c3.number_input("% Time Abnormal LTV", min_value=0.0, max_value=100.0, value=10.0, step=1.0)
        c1c, _, _    = st.columns(3, gap="medium")
        mean_ltv     = c1c.number_input("Mean LTV Value",      min_value=0.0, max_value=50.0,  value=7.0,  step=0.1, format="%.1f")

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
            submitted_fetal = st.form_submit_button("🔍 Classify Fetal Health", use_container_width=True)

    if submitted_fetal:
        try:
            features        = engineer_fetal_features(
                baseline, accels, fetal_mov, uterine_cont,
                light_decel, severe_decel, prolonged_decel,
                abnormal_stv, mean_stv, pct_altv, mean_ltv,
                hist_width, hist_min, hist_max, hist_peaks,
                hist_zeroes, hist_mode, hist_mean, hist_median,
                hist_variance, hist_tendency
            )
            features_scaled = fetal_scaler.transform(features)
            prediction      = fetal_model.predict(features_scaled)[0]
            probs           = fetal_model.predict_proba(features_scaled)
            confidence      = max(probs[0]) * 100

            st.markdown("<br>", unsafe_allow_html=True)
            if prediction == 0:
                st.markdown(f"""
                <div class="result-normal">
                    <div style='font-size:35px;'>✅</div>
                    <div class="result-title" style='color:#059669;'>Normal</div>
                    <div style='margin-top:8px; font-size:14px; font-weight:600; color:#065F46;'>Confidence: {confidence:.2f}%</div>
                    <div class="result-desc">CTG patterns within normal limits. Continue routine monitoring.</div>
                </div>""", unsafe_allow_html=True)
            elif prediction == 1:
                st.markdown(f"""
                <div class="result-suspect">
                    <div style='font-size:35px;'>⚠️</div>
                    <div class="result-title" style='color:#D97706;'>Suspect</div>
                    <div style='margin-top:8px; font-size:14px; font-weight:600; color:#92400E;'>Confidence: {confidence:.2f}%</div>
                    <div class="result-desc">Borderline CTG abnormalities. Further evaluation recommended.</div>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="result-pathological">
                    <div style='font-size:35px;'>🚨</div>
                    <div class="result-title" style='color:#E11D48;'>Pathological</div>
                    <div style='margin-top:8px; font-size:14px; font-weight:600; color:#9F1239;'>Confidence: {confidence:.2f}%</div>
                    <div class="result-desc">Abnormal CTG. Immediate obstetric review required — possible fetal distress.</div>
                </div>""", unsafe_allow_html=True)

            st.markdown("""<div class="disclaimer">⚠️ All Suspect and Pathological results must be reviewed immediately by a qualified obstetrician.</div>""", unsafe_allow_html=True)

            # SHAP
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("""
            <div class="info-card">
                <div style='font-size:14px; font-weight:700; color:#9D174D; margin-bottom:4px;'>🔍 Model Explainability</div>
                <div style='font-size:12px; color:#6B7280; margin-bottom:14px;'>
                    SHAP feature contributions for this CTG prediction. KernelExplainer may take 10–15 seconds.
                </div>
            """, unsafe_allow_html=True)
            with st.spinner("Computing SHAP values..."):
                try:
                    fetal_feature_names = [
                        'baseline value','accelerations','fetal_movement','uterine_contractions',
                        'light_decelerations','severe_decelerations','prolongued_decelerations',
                        'abnormal_short_term_variability','mean_value_of_short_term_variability',
                        'percentage_of_time_with_abnormal_long_term_variability',
                        'mean_value_of_long_term_variability','histogram_width','histogram_min',
                        'histogram_max','histogram_number_of_peaks','histogram_number_of_zeroes',
                        'histogram_mode','histogram_mean','histogram_median','histogram_variance',
                        'histogram_tendency','TotalDecelerations','SevereDecelFlag','ProlongedDecelFlag',
                        'DecelerationRatio','VariabilityRatio','LongTermVariabilityHigh','VariabilityStress',
                        'TachycardiaFlag','BradycardiaFlag','AccelDecelRatio','AccelFlag',
                        'HistogramSkewness','HistogramModeOffset','HistogramSymmetry','FetalRiskScore',
                        'ContractionAccelResponse','ContractionDecelResponse','MovementAccelRatio'
                    ]
                    shap_vals = compute_fetal_shap(fetal_model, features_scaled)
                    fig_shap  = render_shap_chart(shap_vals, fetal_feature_names, int(prediction),
                                                  f"Top 10 Feature Contributions — {['Normal','Suspect','Pathological'][prediction]}")
                    st.plotly_chart(fig_shap, use_container_width=True)
                except Exception as se:
                    st.info(f"SHAP explanation unavailable: {se}")
            st.markdown("</div>", unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Prediction error: {e}")


# ═════════════════════════════════════════════
# PAGE: FETAL BIRTH WEIGHT (Clinician only)
# ═════════════════════════════════════════════
elif selected == "Fetal Birth Weight":
    if not is_clinician:
        st.markdown("""
        <div class="access-denied">
            <div style='font-size:48px; margin-bottom:16px;'>🔒</div>
            <div style='font-size:22px; font-weight:700; color:#E11D48; margin-bottom:10px;'>Clinician Access Required</div>
            <div style='font-size:14px; color:#6B7280; line-height:1.7; max-width:500px; margin:0 auto;'>
                Fetal birth weight estimation requires ultrasound biometry data (BPD, HC, AC, FL)
                that must be interpreted by trained clinicians only.
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.stop()

    st.markdown('<div class="section-header">⚖️ Fetal Birth Weight Estimation</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Estimate fetal birth weight using the Hadlock Formula 4 (HC + AC + FL + BPD) extended with maternal and clinical adjustment factors. 32 engineered features for model training.</div>', unsafe_allow_html=True)

    with st.form("fbw_form"):
        st.markdown("**📡 Ultrasound Biometry (required for Hadlock formula)**")
        c1, c2, c3 = st.columns(3, gap="medium")
        bpd_fbw = c1.number_input("Biparietal Diameter — BPD (cm)", min_value=3.0, max_value=12.0, value=8.8, step=0.1, format="%.1f")
        hc_fbw  = c2.number_input("Head Circumference — HC (cm)",   min_value=10.0, max_value=40.0, value=31.5, step=0.1, format="%.1f")
        ac_fbw  = c3.number_input("Abdominal Circumference — AC (cm)", min_value=10.0, max_value=45.0, value=30.0, step=0.1, format="%.1f")
        fl_fbw  = c1.number_input("Femur Length — FL (cm)",         min_value=2.0, max_value=10.0, value=6.0, step=0.1, format="%.1f")

        st.markdown("---")
        st.markdown("**🤱 Gestational & Maternal Parameters**")
        c1, c2, c3 = st.columns(3, gap="medium")
        ga_fbw      = c1.number_input("Gestational Age (weeks)",      min_value=24.0, max_value=44.0, value=38.0, step=0.5, format="%.1f")
        m_age_fbw   = c2.number_input("Maternal Age (years)",         min_value=15,   max_value=55,   value=28,   step=1)
        m_weight_fbw= c3.number_input("Maternal Weight (kg)",         min_value=35.0, max_value=150.0, value=70.0, step=0.5, format="%.1f")
        m_height_fbw= c1.number_input("Maternal Height (cm)",         min_value=130.0, max_value=200.0, value=160.0, step=1.0)
        ppbmi_fbw   = c2.number_input("Pre-Pregnancy BMI (kg/m²)",    min_value=14.0, max_value=55.0, value=22.0, step=0.1, format="%.1f")
        wg_fbw      = c3.number_input("Gestational Weight Gain (kg)", min_value=0.0,  max_value=40.0, value=11.0, step=0.5, format="%.1f")

        st.markdown("---")
        st.markdown("**📏 Clinical Measurements**")
        c1, c2, c3 = st.columns(3, gap="medium")
        fh_fbw      = c1.number_input("Fundal Height (cm)",           min_value=10.0, max_value=50.0, value=36.0, step=1.0)
        glucose_fbw = c2.number_input("Maternal Glucose (mg/dL)",     min_value=60.0, max_value=300.0, value=90.0, step=1.0)
        afi_fbw     = c3.number_input("Amniotic Fluid Index (cm)",    min_value=0.0,  max_value=40.0,  value=12.0, step=0.5, format="%.1f")
        parity_fbw  = c1.number_input("Parity",                       min_value=0,    max_value=10,    value=0,    step=1)
        gravida_fbw = c2.number_input("Gravida",                      min_value=1,    max_value=12,    value=1,    step=1)

        st.markdown("<br>", unsafe_allow_html=True)
        _, mid, _ = st.columns([2, 1, 2])
        with mid:
            submitted_fbw = st.form_submit_button("⚖️ Estimate Birth Weight", use_container_width=True)

    if submitted_fbw:
        features_fbw = engineer_fbw_features(
            ga_fbw, m_age_fbw, m_weight_fbw, m_height_fbw, ppbmi_fbw, wg_fbw,
            fh_fbw, bpd_fbw, hc_fbw, ac_fbw, fl_fbw,
            parity_fbw, gravida_fbw, glucose_fbw, afi_fbw
        )
        efw_g, efw_kg, category, cat_color, percentile, signals_fbw = estimate_fetal_birth_weight(
            ga_fbw, bpd_fbw, hc_fbw, ac_fbw, fl_fbw,
            m_weight_fbw, wg_fbw, glucose_fbw, parity_fbw, afi_fbw
        )

        st.markdown("<br>", unsafe_allow_html=True)
        r1, r2 = st.columns([1, 1], gap="large")

        with r1:
            is_normal = 2500 <= efw_g <= 4000
            card_class = "result-low" if is_normal else ("result-mid" if efw_g < 2500 else "result-high")
            st.markdown(f"""
            <div class="{card_class}" style='border-color:{cat_color};'>
                <div style='font-size:35px;'>⚖️</div>
                <div style='font-size:38px; font-weight:800; color:{cat_color}; margin:8px 0;'>
                    {efw_g:.0f} g
                </div>
                <div style='font-size:16px; font-weight:700; color:{cat_color};'>{category}</div>
                <div style='font-size:14px; color:#6B7280; margin-top:6px;'>
                    ≈ {efw_kg:.2f} kg &nbsp;|&nbsp; ~{percentile}th percentile for {ga_fbw:.0f} weeks GA
                </div>
                <div class="result-desc" style='margin-top:10px;'>
                    Estimated using Hadlock Formula 4 (HC + AC + FL + BPD) with
                    maternal adjustment factors. ±15% measurement uncertainty is typical.
                </div>
            </div>""", unsafe_allow_html=True)

            # Growth gauge
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=efw_g,
                number={"suffix": " g", "font": {"size": 20, "color": "#9D174D"}},
                gauge={
                    "axis": {"range": [500, 5000], "tickcolor": "#6B7280"},
                    "bar": {"color": cat_color, "thickness": 0.3},
                    "steps": [
                        {"range": [500, 1500],  "color": "#FFF1F2"},
                        {"range": [1500, 2500], "color": "#FFF7ED"},
                        {"range": [2500, 4000], "color": "#ECFDF5"},
                        {"range": [4000, 5000], "color": "#FFF7ED"},
                    ],
                    "threshold": {"line": {"color": "#E11D48", "width": 2}, "thickness": 0.75, "value": 4000}
                },
                title={"text": "Estimated Fetal Weight", "font": {"size": 13, "color": "#9D174D"}}
            ))
            fig_gauge.update_layout(
                height=220, margin=dict(l=20, r=20, t=40, b=10),
                paper_bgcolor="white", font={"color": "#374151"}
            )
            st.plotly_chart(fig_gauge, use_container_width=True)

        with r2:
            st.markdown("""
            <div class="info-card">
                <div style='font-size:14px; font-weight:700; color:#9D174D; margin-bottom:12px;'>📋 Clinical Flags & Signals</div>
            """, unsafe_allow_html=True)
            if signals_fbw:
                for icon, name, detail in signals_fbw:
                    st.markdown(f"""
                    <div style='display:flex; align-items:flex-start; gap:10px; padding:7px 0;
                                border-bottom:1px solid #FFF0F5;'>
                        <span style='font-size:15px; margin-top:1px;'>{icon}</span>
                        <div>
                            <div style='font-size:13px; font-weight:600; color:#B45309;'>{name}</div>
                            <div style='font-size:12px; color:#9CA3AF;'>{detail}</div>
                        </div>
                    </div>""", unsafe_allow_html=True)
            else:
                st.markdown("""<div style='font-size:13px; color:#059669; font-weight:600; padding:8px 0;'>✅ No significant clinical flags</div>""", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

            hc_ac = hc_fbw / (ac_fbw + 1e-6)
            current_bmi = m_weight_fbw / ((m_height_fbw / 100) ** 2)
            st.markdown(f"""
            <div class="info-card" style='margin-top:0;'>
                <div style='font-size:13px; font-weight:700; color:#9D174D; margin-bottom:10px;'>📊 Biometry Ratios</div>
                <div style='display:grid; grid-template-columns:1fr 1fr; gap:8px;'>
                    <div style='text-align:center; background:#FFF5F7; border-radius:8px; padding:8px;'>
                        <div style='font-size:16px; font-weight:700; color:#BE185D;'>{hc_ac:.2f}</div>
                        <div style='font-size:10px; color:#9CA3AF;'>HC/AC Ratio<br>(norm 1.0–1.1 at term)</div>
                    </div>
                    <div style='text-align:center; background:#FFF5F7; border-radius:8px; padding:8px;'>
                        <div style='font-size:16px; font-weight:700; color:#BE185D;'>{fl_fbw/ac_fbw:.2f}</div>
                        <div style='font-size:10px; color:#9CA3AF;'>FL/AC Ratio<br>(norm ~0.22)</div>
                    </div>
                    <div style='text-align:center; background:#FFF5F7; border-radius:8px; padding:8px;'>
                        <div style='font-size:16px; font-weight:700; color:#BE185D;'>{current_bmi:.1f}</div>
                        <div style='font-size:10px; color:#9CA3AF;'>Current BMI<br>(kg/m²)</div>
                    </div>
                    <div style='text-align:center; background:#FFF5F7; border-radius:8px; padding:8px;'>
                        <div style='font-size:16px; font-weight:700; color:#BE185D;'>{fh_fbw/ga_fbw:.2f}</div>
                        <div style='font-size:10px; color:#9CA3AF;'>FH/GA Ratio<br>(norm ~1.0)</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("""
        <div class="info-card" style='margin-top:12px; background:#FFFBEB; border-color:#FDE68A;'>
            <div style='font-size:13px; font-weight:700; color:#92400E; margin-bottom:8px;'>🏋️ Model Training Guide</div>
            <div style='font-size:12px; color:#78350F; line-height:1.7;'>
                <b>Recommended Dataset:</b> INTERGROWTH-21st Fetal Growth Standards or WHO Fetal Growth Charts |
                <b>Target Variable:</b> Actual birth weight (regression) or weight category (classification) |
                <b>Suggested Models:</b> XGBoost Regressor, Random Forest Regressor |
                <b>Feature Function:</b> <code>engineer_fbw_features()</code> generates 32 features |
                <b>Integration:</b> Replace Hadlock formula output with <code>fbw_model.predict(features_fbw)[0]</code>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="disclaimer">⚠️ EFW has ±15% variability. Clinical decisions must not rely solely on this estimate. Always combine with serial growth scans and clinical assessment.</div>
        """, unsafe_allow_html=True)


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
            yaxis=dict(range=[70, 95], title="Accuracy (%)", gridcolor="#FFF0F5", tickfont=dict(color="#37393D")),
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
            yaxis=dict(range=[80, 98], title="Score (%)", gridcolor="#FFF0F5", tickfont=dict(color="#151616")),
            xaxis=dict(tickfont=dict(color="#1C1C1D", size=10)),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, font=dict(size=11))
        )
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("#### Fetal Stacking Ensemble — Per Class Breakdown")
    df = pd.DataFrame({
        "Class":     ["Normal (332)", "Suspect (59)", "Pathological (35)", "Macro Avg"],
        "Precision": [95.87, 86.79, 88.24, 90.30],
        "Recall":    [97.89, 77.97, 85.71, 87.19],
        "F1 Score":  [96.87, 82.14, 86.96, 88.66],
    })
    st.dataframe(
        df.style
          .background_gradient(subset=["Precision","Recall","F1 Score"], cmap="Blues")
          .format({"Precision":"{:.2f}%","Recall":"{:.2f}%","F1 Score":"{:.2f}%"})
          .set_table_styles([
              {"selector": "th", "props": [("background-color","#EAF2FF"),("color","#1E3A8A"),("font-weight","bold"),("text-align","center"),("border","1px solid #D1D5DB")]},
              {"selector": "td", "props": [("border","1px solid #E5E7EB"),("color","#111827"),("text-align","center")]}
          ])
          .set_properties(**{"background-color":"white","color":"#111827","text-align":"center"}),
        use_container_width=True, hide_index=True
    )
