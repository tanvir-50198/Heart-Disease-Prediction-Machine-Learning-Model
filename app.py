"""Heart Disease Prediction - single-file Streamlit app.

Run:  streamlit run app.py
Needs model.pkl (saved from your notebook) in the same folder.
"""
from __future__ import annotations

import html as _html
import pickle
from datetime import datetime
from pathlib import Path

import pandas as pd
import sklearn
import streamlit as st


MODEL_PATH = Path(__file__).parent / "model.pkl"

# =============================================================================
# Helpers (features, batch validation, reference ranges)
# =============================================================================
# Must match the column order the model was trained with in the notebook
FEATURES = [
    "Gender", "age", "currentSmoker", "cigsPerDay", "BPMeds", "prevalentStroke",
    "prevalentHyp", "diabetes", "totChol", "sysBP", "diaBP", "BMI",
    "heartRate", "glucose",
]
BINARY_COLS = ["currentSmoker", "BPMeds", "prevalentStroke", "prevalentHyp", "diabetes"]
NUMERIC_COLS = ["age", "cigsPerDay", "totChol", "sysBP", "diaBP", "BMI", "heartRate", "glucose"]

# Example patient taken from the notebook (the model predicts 1 = at risk)
EXAMPLE_ROW = [0, 59, 0, 0, 0, 0, 1, 0, 209, 150, 85, 20.77, 90, 88]

_BINARY_MAP = {
    "yes": 1, "y": 1, "true": 1, "1": 1, "1.0": 1,
    "no": 0, "n": 0, "false": 0, "0": 0, "0.0": 0,
}
_GENDER_MAP = {"male": 1, "m": 1, "1": 1, "1.0": 1, "female": 0, "f": 0, "0": 0, "0.0": 0}


# ----------------------------------------------------------------------------
# Prediction helpers
# ----------------------------------------------------------------------------
def predict(model, X: pd.DataFrame):
    """Return (predictions, probability_of_class_1 or None)."""
    pred = model.predict(X)
    prob = None
    if hasattr(model, "predict_proba"):
        classes = list(getattr(model, "classes_", [0, 1]))
        idx = classes.index(1) if 1 in classes else -1
        prob = model.predict_proba(X)[:, idx]
    return pred, prob


def label(pred: int) -> str:
    return "At risk" if int(pred) == 1 else "Not at risk"


# ----------------------------------------------------------------------------
# Batch (CSV) preparation
# ----------------------------------------------------------------------------
def prepare_batch(df: pd.DataFrame):
    """Validate/convert an uploaded CSV.

    Returns (clean_df, invalid_rows, missing_columns):
      clean_df       - model-ready rows (columns = FEATURES), original index kept
      invalid_rows   - 1-based row numbers that were skipped (blank / unreadable values)
      missing_columns- required columns that are absent (clean_df is None in that case)
    """
    lookup = {str(c).strip().lower(): c for c in df.columns}
    missing = [c for c in FEATURES if c.lower() not in lookup]
    if missing:
        return None, [], missing

    out = pd.DataFrame(index=df.index)
    out["Gender"] = df[lookup["gender"]].astype(str).str.strip().str.lower().map(_GENDER_MAP)
    for col in BINARY_COLS:
        out[col] = df[lookup[col.lower()]].astype(str).str.strip().str.lower().map(_BINARY_MAP)
    for col in NUMERIC_COLS:
        out[col] = pd.to_numeric(df[lookup[col.lower()]], errors="coerce")

    out = out[FEATURES]
    valid = out.notna().all(axis=1)
    invalid_rows = [int(i) + 1 for i in out.index[~valid]]
    return out[valid], invalid_rows, []


# ----------------------------------------------------------------------------
# General reference ranges (NOT model output)
# ----------------------------------------------------------------------------
def bp_category(sys_bp: float, dia_bp: float):
    if sys_bp > 180 or dia_bp > 120:
        return "Hypertensive crisis range", "bad"
    if sys_bp >= 140 or dia_bp >= 90:
        return "High (stage 2)", "bad"
    if sys_bp >= 130 or dia_bp >= 80:
        return "High (stage 1)", "warn"
    if sys_bp >= 120:
        return "Elevated", "warn"
    return "Normal", "ok"


def bmi_category(bmi: float):
    if bmi < 18.5:
        return "Underweight", "warn"
    if bmi < 25:
        return "Healthy weight", "ok"
    if bmi < 30:
        return "Overweight", "warn"
    return "Obese", "bad"


def cholesterol_category(chol: float):
    if chol < 200:
        return "Desirable", "ok"
    if chol < 240:
        return "Borderline high", "warn"
    return "High", "bad"


def glucose_category(glucose: float):
    if glucose < 100:
        return "Normal", "ok"
    if glucose < 126:
        return "Elevated", "warn"
    return "High", "bad"


def heart_rate_category(hr: float):
    if hr < 60:
        return "Below typical resting range", "warn"
    if hr <= 100:
        return "Typical resting range", "ok"
    return "Above typical resting range", "warn"


def health_checks(v: dict):
    """v uses the FEATURES keys. Returns [(name, value_text, status_text, level), ...]."""
    checks = []
    s, lvl = bp_category(v["sysBP"], v["diaBP"])
    checks.append(("Blood pressure", f'{v["sysBP"]:g}/{v["diaBP"]:g} mmHg', s, lvl))
    s, lvl = bmi_category(v["BMI"])
    checks.append(("BMI", f'{v["BMI"]:.1f}', s, lvl))
    s, lvl = cholesterol_category(v["totChol"])
    checks.append(("Total cholesterol", f'{v["totChol"]:g} mg/dL', s, lvl))
    s, lvl = glucose_category(v["glucose"])
    checks.append(("Glucose", f'{v["glucose"]:g} mg/dL', s, lvl))
    s, lvl = heart_rate_category(v["heartRate"])
    checks.append(("Heart rate", f'{v["heartRate"]:g} bpm', s, lvl))
    if v["currentSmoker"]:
        checks.append(("Smoking", f'{v["cigsPerDay"]:g} cigarettes/day', "Current smoker", "warn"))
    else:
        checks.append(("Smoking", "—", "Non-smoker", "ok"))
    return checks

# =============================================================================
# Stylesheet (lavender / purple design system)
# =============================================================================
CSS = r"""
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

:root {
  color-scheme: dark;
  --bg: #0E1120;
  --card: #171B2E;
  --surface: #1F2440;
  --input: #11152A;
  --border: #2B3150;
  --text: #EDEFFB;
  --muted: #9AA3C0;
  --accent: #635BFF;
  --accent-hover: #5147E5;
  --accent-text: #A29DFF;
  --ok-bg: rgba(18, 183, 106, .15);  --ok-fg: #6CE9A6;
  --warn-bg: rgba(247, 144, 9, .15); --warn-fg: #FEC84B;
  --bad-bg: rgba(240, 68, 56, .16);  --bad-fg: #FDA29B;
}

/* ---------- Base ---------- */
.stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"] { background: var(--bg); color: var(--text); }
:where(.stApp) :where(p, label, li, span, h1, h2, h3, h4, h5, strong, td, th, summary) { color: var(--text); }
.stApp, .stApp p, .stApp label, .stApp li, .stApp input, .stApp textarea,
.stApp button, .stApp h1, .stApp h2, .stApp h3, .stApp h4 {
  font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
}
[data-testid="stMainBlockContainer"], .block-container { max-width: 1040px; padding: 2.2rem 2rem 3rem 2rem; }
[data-testid="stHeader"] { background: transparent; }
[data-testid="stHeader"] button, [data-testid="stHeader"] svg, [data-testid="stToolbar"] svg { color: var(--muted); fill: var(--muted); }
footer { visibility: hidden; }
a { color: var(--accent-text); }
code { background: var(--surface); color: var(--accent-text); padding: .1rem .35rem; border-radius: 6px; font-size: .85em; }
[data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] p { color: var(--muted); }

/* ---------- Page header ---------- */
.page-head { margin-bottom: 1.2rem; }
.page-head h1 { font-size: 1.9rem; font-weight: 700; letter-spacing: -0.02em; color: var(--text); margin: 0 0 .35rem 0; padding: 0; line-height: 1.2; }
.page-head p { color: var(--muted); font-size: 1rem; line-height: 1.6; margin: 0; max-width: 720px; }

/* ---------- Sidebar ---------- */
[data-testid="stSidebar"] { background: var(--card); border-right: 1px solid var(--border); }
[data-testid="stSidebar"] > div:first-child { padding-top: .6rem; }
[data-testid="stSidebar"] button svg { fill: currentColor; }
.brand { display: flex; align-items: center; gap: .75rem; padding: .4rem .4rem 1.1rem .4rem; }
.brand-logo { width: 40px; height: 40px; border-radius: 12px; background: var(--accent); color: #fff; display: flex; align-items: center; justify-content: center; font-size: 1.2rem; box-shadow: 0 4px 14px rgba(99, 91, 255, .45); }
.brand-name { font-weight: 700; font-size: 1rem; color: var(--text); line-height: 1.2; }
.brand-sub  { font-size: .78rem; color: var(--muted); }
.side-label { font-size: .7rem; font-weight: 600; letter-spacing: .08em; text-transform: uppercase; color: var(--muted); padding: .4rem .5rem .3rem .5rem; }
.side-status { margin-top: 1.2rem; padding: .8rem .9rem; border: 1px solid var(--border); border-radius: 12px; background: var(--surface); font-size: .8rem; color: var(--muted); line-height: 1.5; }
.side-status b { color: var(--text); }

[data-testid="stSidebar"] [class*="st-key-nav_"] button,
[data-testid="stSidebar"] [class*="st-key-navactive_"] button {
  width: 100%; height: 44px; border-radius: 10px; border: none; box-shadow: none;
  justify-content: flex-start; padding: 0 .85rem; background: transparent;
  color: var(--muted); font-weight: 500; font-size: .95rem; transition: background .15s, color .15s;
}
[data-testid="stSidebar"] [class*="st-key-nav_"] button p,
[data-testid="stSidebar"] [class*="st-key-navactive_"] button p { color: inherit; font-weight: inherit; }
[data-testid="stSidebar"] [class*="st-key-nav_"] button:hover { background: var(--surface); color: var(--accent-text); }
[data-testid="stSidebar"] [class*="st-key-navactive_"] button { background: rgba(99, 91, 255, .22); color: var(--accent-text); font-weight: 600; }
[data-testid="stSidebar"] [class*="st-key-nav"] button:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }

/* ---------- Cards ---------- */
[class*="st-key-card_"] {
  background: var(--card); border: 1px solid var(--border); border-radius: 16px;
  padding: 1.4rem 1.5rem 1.5rem 1.5rem; box-shadow: 0 8px 24px rgba(0, 0, 0, .25); margin-bottom: .6rem;
}
.card-title { font-size: 1.08rem; font-weight: 600; color: var(--text); margin-bottom: .15rem; }
.card-desc  { font-size: .9rem; color: var(--muted); margin-bottom: .9rem; line-height: 1.5; }

/* ---------- Buttons ---------- */
.stButton > button, [data-testid="stDownloadButton"] > button, [data-testid="stFileUploaderDropzone"] button {
  border-radius: 10px; height: 44px; padding: 0 1.1rem; font-weight: 600; font-size: .95rem;
  border: 1px solid var(--border); background: var(--surface); color: var(--text);
  transition: background .15s, border-color .15s, color .15s, box-shadow .15s;
}
.stButton > button p, [data-testid="stDownloadButton"] > button p { color: inherit; }
.stButton > button:hover, [data-testid="stDownloadButton"] > button:hover, [data-testid="stFileUploaderDropzone"] button:hover {
  border-color: var(--accent); color: var(--accent-text); background: var(--surface);
}
.stButton > button[kind="primary"] { background: var(--accent); border-color: var(--accent); color: #fff; }
.stButton > button[kind="primary"] p { color: #fff; }
.stButton > button[kind="primary"]:hover { background: var(--accent-hover); border-color: var(--accent-hover); color: #fff; }
.stButton > button:focus-visible, [data-testid="stDownloadButton"] > button:focus-visible { outline: none; box-shadow: 0 0 0 3px rgba(99, 91, 255, .35); }
.stButton > button:disabled { opacity: .45; cursor: not-allowed; }
.st-key-predict_btn button { width: 100%; }

/* ---------- Inputs ---------- */
[data-baseweb="input"], [data-baseweb="select"] > div, [data-baseweb="textarea"] {
  background: var(--input) !important; border: 1px solid var(--border) !important; border-radius: 10px !important;
}
[data-baseweb="base-input"] { background: transparent !important; }
[data-baseweb="input"] input { background: transparent !important; color: var(--text) !important; -webkit-text-fill-color: var(--text); }
[data-baseweb="input"] input::placeholder { color: var(--muted); }
[data-baseweb="input"]:focus-within, [data-baseweb="select"] > div:focus-within, [data-baseweb="textarea"]:focus-within {
  border-color: var(--accent) !important; box-shadow: 0 0 0 3px rgba(99, 91, 255, .25) !important;
}
[data-testid="stNumberInputStepDown"], [data-testid="stNumberInputStepUp"] { background: var(--surface) !important; color: var(--text) !important; }
[data-testid="stNumberInputStepDown"]:hover, [data-testid="stNumberInputStepUp"]:hover { background: var(--accent) !important; }
[data-baseweb="select"] * { color: var(--text); }
[data-baseweb="select"] svg { fill: var(--muted); }
[data-testid="stWidgetLabel"] p { font-weight: 500; font-size: .88rem; color: var(--text); }
[data-testid="stTooltipIcon"] svg { fill: var(--muted); color: var(--muted); }

/* dropdown menus / popovers / tooltips (rendered outside the main tree) */
[data-baseweb="popover"], [data-baseweb="popover"] > div, [data-baseweb="menu"], ul[role="listbox"] {
  background: var(--card) !important; color: var(--text) !important; border-radius: 10px;
}
[data-baseweb="popover"] li, [data-baseweb="menu"] li, li[role="option"] { background: transparent !important; color: var(--text) !important; }
[data-baseweb="popover"] li:hover, [data-baseweb="menu"] li:hover, li[role="option"]:hover, li[role="option"][aria-selected="true"] {
  background: var(--surface) !important;
}
[data-testid="stTooltipContent"] { background: var(--card); color: var(--text); border: 1px solid var(--border); }

/* radios: purple instead of Streamlit's default red */
label[data-baseweb="radio"] > div:first-child { border-color: var(--muted) !important; background-color: transparent !important; }
label[data-baseweb="radio"]:has(input:checked) > div:first-child { background-color: var(--accent) !important; border-color: var(--accent) !important; }
input[type="radio"], input[type="checkbox"] { accent-color: var(--accent); }

/* file uploader, expander, alerts */
[data-testid="stFileUploaderDropzone"] { background: var(--surface); border: 1.5px dashed #444C7A; border-radius: 14px; }
[data-testid="stFileUploaderDropzone"]:hover { border-color: var(--accent); }
[data-testid="stFileUploaderDropzone"] small, [data-testid="stFileUploaderDropzoneInstructions"] span { color: var(--muted); }
[data-testid="stFileUploaderFile"] { background: var(--surface); border-radius: 10px; }
[data-testid="stExpander"] { border: 1px solid var(--border); border-radius: 12px; background: var(--surface); }
[data-testid="stExpander"] details, [data-testid="stExpander"] summary { background: transparent; color: var(--text); }
[data-testid="stExpander"] summary svg { fill: var(--muted); }
[data-testid="stAlert"] { background: var(--surface); border: 1px solid var(--border); border-radius: 12px; }
[data-testid="stAlert"] p, [data-testid="stAlert"] div { color: var(--text); }

/* ---------- Metrics, pills, bars ---------- */
[data-testid="stMetric"] { background: var(--surface); border: 1px solid var(--border); border-radius: 12px; padding: .9rem 1rem; }
[data-testid="stMetricLabel"] p { color: var(--muted); font-size: .8rem; font-weight: 500; }
[data-testid="stMetricValue"], [data-testid="stMetricValue"] div { color: var(--text); font-weight: 700; }

.pill { display: inline-block; padding: .22rem .65rem; border-radius: 999px; font-size: .78rem; font-weight: 600; white-space: nowrap; }
.pill-ok   { background: var(--ok-bg);   color: var(--ok-fg); }
.pill-warn { background: var(--warn-bg); color: var(--warn-fg); }
.pill-bad  { background: var(--bad-bg);  color: var(--bad-fg); }
.pill-info { background: rgba(99, 91, 255, .2); color: var(--accent-text); }

.result-title { font-size: 1.5rem; font-weight: 700; margin: .5rem 0 .25rem 0; letter-spacing: -.01em; color: var(--text); }
.result-text  { color: var(--muted); font-size: .95rem; line-height: 1.6; margin-bottom: 1rem; }
.track { width: 100%; height: 10px; border-radius: 999px; background: var(--surface); overflow: hidden; margin: .3rem 0 .2rem 0; }
.fill  { height: 100%; border-radius: 999px; background: var(--accent); }
.fill-bad { background: #F04438; } .fill-ok { background: #12B76A; }
.bar-scale { display: flex; justify-content: space-between; font-size: .72rem; color: var(--muted); }

.check-row { display: flex; align-items: center; justify-content: space-between; gap: 1rem; padding: .7rem 0; border-bottom: 1px solid var(--border); }
.check-row:last-child { border-bottom: none; }
.check-name { font-weight: 500; color: var(--text); font-size: .93rem; }
.check-val  { color: var(--muted); font-size: .82rem; }

.empty-state { text-align: center; padding: 2.2rem 1rem; color: var(--muted); }
.empty-state .big { font-size: 2rem; margin-bottom: .4rem; }
.empty-state b { color: var(--text); display: block; margin-bottom: .25rem; font-size: 1.02rem; }

.glossary-row { display: flex; gap: 1rem; padding: .6rem 0; border-bottom: 1px solid var(--border); font-size: .9rem; }
.glossary-row:last-child { border-bottom: none; }
.glossary-key { min-width: 150px; font-weight: 600; color: var(--text); }
.glossary-val { color: var(--muted); }
.disclaimer { font-size: .82rem; color: var(--muted); background: var(--surface); border: 1px solid var(--border); border-radius: 12px; padding: .8rem 1rem; line-height: 1.55; margin-top: .5rem; }

/* ---------- HTML tables (dark-friendly replacement for st.dataframe) ---------- */
.tbl-wrap { overflow: auto; max-height: 440px; border: 1px solid var(--border); border-radius: 12px; margin: .4rem 0 .8rem 0; }
table.tbl { width: 100%; border-collapse: collapse; font-size: .85rem; }
table.tbl th { position: sticky; top: 0; background: var(--surface); color: var(--muted); text-align: left; font-weight: 600; padding: .6rem .8rem; border-bottom: 1px solid var(--border); white-space: nowrap; }
table.tbl td { padding: .55rem .8rem; border-bottom: 1px solid var(--border); color: var(--text); white-space: nowrap; }
table.tbl tr:last-child td { border-bottom: none; }
table.tbl tbody tr:hover td { background: rgba(99, 91, 255, .08); }
.tbl-note { color: var(--muted); font-size: .8rem; margin: -.4rem 0 .8rem 0; }

/* ---------- Coefficient bars ---------- */
.cbar-legend { display: flex; gap: 1.2rem; align-items: center; font-size: .8rem; color: var(--muted); margin-bottom: .6rem; flex-wrap: wrap; }
.cbar-legend .dot { display: inline-block; width: 10px; height: 10px; border-radius: 50%; margin-right: .35rem; }
.dot.up, .cbar-fill.up { background: #F97066; }
.dot.down, .cbar-fill.down { background: #32D583; }
.cbar-row { display: grid; grid-template-columns: 130px 1fr 70px; gap: .8rem; align-items: center; padding: .32rem 0; }
.cbar-name { font-size: .85rem; color: var(--text); font-weight: 500; }
.cbar-track { height: 10px; background: var(--surface); border-radius: 999px; overflow: hidden; }
.cbar-fill { height: 100%; border-radius: 999px; }
.cbar-val { font-size: .8rem; color: var(--muted); text-align: right; font-variant-numeric: tabular-nums; }

/* ---------- Responsive ---------- */
@media (max-width: 768px) {
  [data-testid="stMainBlockContainer"], .block-container { padding: 1.2rem 1rem 2rem 1rem; }
  .page-head h1 { font-size: 1.5rem; }
  [class*="st-key-card_"] { padding: 1.1rem 1rem 1.2rem 1rem; }
  .glossary-row { flex-direction: column; gap: .2rem; }
  .check-row { flex-wrap: wrap; }
  .cbar-row { grid-template-columns: 100px 1fr 56px; }
}
"""

# =============================================================================
# App
# =============================================================================
st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------------------------------------------------------
# State
# ----------------------------------------------------------------------------
DEFAULTS = {
    "gender": "Male", "age": 50, "smoker": "No", "cigs": 0, "bpmeds": "No",
    "stroke": "No", "hyp": "No", "diabetes": "No", "chol": 200, "glucose": 90,
    "sys": 120.0, "dia": 80.0, "bmi": 25.0, "hr": 75,
    "h_cm": 170.0, "w_kg": 70.0,
}
# Example patient from the notebook (same values as utils.EXAMPLE_ROW)
EXAMPLE = {
    "gender": "Female", "age": 59, "smoker": "No", "cigs": 0, "bpmeds": "No",
    "stroke": "No", "hyp": "Yes", "diabetes": "No", "chol": 209, "glucose": 88,
    "sys": 150.0, "dia": 85.0, "bmi": 20.77, "hr": 90,
}

for _k, _v in DEFAULTS.items():
    st.session_state.setdefault(_k, _v)
st.session_state.setdefault("page", "Predict")
st.session_state.setdefault("history", [])
st.session_state.setdefault("last_result", None)

NAV = [
    ("Predict", ":material/favorite:"),
    ("Batch Prediction", ":material/upload_file:"),
    ("History", ":material/history:"),
    ("Model Insights", ":material/insights:"),
    ("About", ":material/info:"),
]

FIELD_HELP = [
    ("Gender", "Patient's sex (Male / Female)."),
    ("age", "Age in years."),
    ("currentSmoker", "Whether the patient currently smokes."),
    ("cigsPerDay", "Average number of cigarettes smoked per day (0 for non-smokers)."),
    ("BPMeds", "Whether the patient takes blood-pressure medication."),
    ("prevalentStroke", "Whether the patient has had a stroke before."),
    ("prevalentHyp", "Whether the patient is hypertensive."),
    ("diabetes", "Whether the patient has diabetes."),
    ("totChol", "Total cholesterol in mg/dL."),
    ("sysBP / diaBP", "Systolic / diastolic blood pressure in mmHg."),
    ("BMI", "Body Mass Index (kg/m²)."),
    ("heartRate", "Resting heart rate in beats per minute."),
    ("glucose", "Blood glucose level in mg/dL."),
]


# ----------------------------------------------------------------------------
# Styling + model
# ----------------------------------------------------------------------------
def inject_css():
    st.markdown(f"<style>{CSS}</style>", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


def final_estimator(model):
    """Return the last step if the model is an sklearn Pipeline, else the model itself."""
    return model.steps[-1][1] if hasattr(model, "steps") else model


# ----------------------------------------------------------------------------
# UI helpers
# ----------------------------------------------------------------------------
def page_header(title: str, desc: str):
    st.markdown(f'<div class="page-head"><h1>{title}</h1><p>{desc}</p></div>', unsafe_allow_html=True)


def card_header(title: str, desc: str = ""):
    html = f'<div class="card-title">{title}</div>'
    if desc:
        html += f'<div class="card-desc">{desc}</div>'
    st.markdown(html, unsafe_allow_html=True)


def pill(text: str, level: str = "info") -> str:
    return f'<span class="pill pill-{level}">{text}</span>'


def html_table(df, pill_col=None, max_rows=200):
    """Dark-theme friendly table (Streamlit's built-in table cannot follow custom colours)."""
    head = "".join(f"<th>{_html.escape(str(c))}</th>" for c in df.columns)
    body = []
    for row in df.head(max_rows).itertuples(index=False):
        cells = []
        for col, val in zip(df.columns, row):
            if val is None or (isinstance(val, float) and pd.isna(val)):
                txt = "—"
            else:
                txt = _html.escape(str(val))
            if col == pill_col and txt != "—":
                txt = pill(txt, "bad" if val == "At risk" else "ok")
            cells.append(f"<td>{txt}</td>")
        body.append("<tr>" + "".join(cells) + "</tr>")
    note = ""
    if len(df) > max_rows:
        note = f'<div class="tbl-note">Showing the first {max_rows} of {len(df)} rows — download the CSV for all rows.</div>'
    return (f'<div class="tbl-wrap"><table class="tbl"><thead><tr>{head}</tr></thead>'
            f'<tbody>{"".join(body)}</tbody></table></div>{note}')


def coef_bars(coef_df):
    """Horizontal bars for model coefficients (green = toward not-at-risk, red = toward at-risk)."""
    m = float(coef_df["Coefficient"].abs().max()) or 1.0
    rows = []
    for f, c in zip(coef_df["Feature"], coef_df["Coefficient"]):
        cls = "up" if c > 0 else "down"
        rows.append(
            f'<div class="cbar-row"><div class="cbar-name">{_html.escape(str(f))}</div>'
            f'<div class="cbar-track"><div class="cbar-fill {cls}" style="width:{abs(c) / m * 100:.1f}%"></div></div>'
            f'<div class="cbar-val">{c:+.3f}</div></div>'
        )
    legend = ('<div class="cbar-legend"><span><span class="dot up"></span>Toward “at risk”</span>'
              '<span><span class="dot down"></span>Toward “not at risk”</span></div>')
    return legend + "".join(rows)


def empty_state(icon: str, title: str, text: str):
    st.markdown(
        f'<div class="empty-state"><div class="big">{icon}</div><b>{title}</b>{text}</div>',
        unsafe_allow_html=True,
    )


def disclaimer():
    st.markdown(
        '<div class="disclaimer">⚠️ This is a student machine-learning project, not a medical device. '
        "Its output is an estimate based on a statistical model and must not be used for diagnosis or "
        "treatment decisions. Always consult a qualified healthcare professional.</div>",
        unsafe_allow_html=True,
    )


def go(page: str):
    st.session_state["page"] = page


def yn(value: str) -> int:
    return 1 if value == "Yes" else 0


# Callbacks (run before the next rerun, so they may safely edit widget state)
def fill_example():
    for k, v in EXAMPLE.items():
        st.session_state[k] = v


def reset_form():
    for k, v in DEFAULTS.items():
        st.session_state[k] = v
    st.session_state["last_result"] = None


def apply_bmi():
    h = st.session_state["h_cm"] / 100
    bmi = st.session_state["w_kg"] / (h * h)
    st.session_state["bmi"] = round(min(max(bmi, 10.0), 70.0), 1)


def clear_history():
    st.session_state["history"] = []


# ----------------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------------
def render_sidebar(model_ok: bool):
    with st.sidebar:
        st.markdown(
            '<div class="brand"><div class="brand-logo">❤</div><div>'
            '<div class="brand-name">Heart Disease</div>'
            '<div class="brand-sub">Risk Prediction</div></div></div>'
            '<div class="side-label">Menu</div>',
            unsafe_allow_html=True,
        )
        for i, (name, icon) in enumerate(NAV):
            active = st.session_state["page"] == name
            st.button(
                name, icon=icon, key=f"{'navactive' if active else 'nav'}_{i}",
                on_click=go, args=(name,),
            )
        status = "Model loaded ✓" if model_ok else "Model not loaded"
        n_hist = len(st.session_state["history"])
        st.markdown(
            f'<div class="side-status"><b>{status}</b><br>scikit-learn {sklearn.__version__}'
            f"<br>Predictions this session: <b>{n_hist}</b></div>",
            unsafe_allow_html=True,
        )


# ----------------------------------------------------------------------------
# Page: Predict
# ----------------------------------------------------------------------------
def render_result(res: dict):
    pred, prob = res["pred"], res["prob"]
    at_risk = pred == 1
    level = "bad" if at_risk else "ok"
    with st.container(key="card_result"):
        st.markdown(
            pill("Higher risk" if at_risk else "Lower risk", level)
            + f'<div class="result-title">{"The model predicts this patient is at risk" if at_risk else "The model predicts this patient is not at risk"}</div>'
            + '<div class="result-text">'
            + ("Please consider speaking with a doctor about these results." if at_risk
               else "This is an estimate only; regular check-ups are still recommended.")
            + "</div>",
            unsafe_allow_html=True,
        )
        c1, c2 = st.columns(2)
        c1.metric("Prediction", label(pred))
        if prob is not None:
            c2.metric("Estimated risk probability", f"{prob * 100:.1f}%")
            st.markdown(
                f'<div class="track"><div class="fill fill-{level}" style="width:{prob * 100:.1f}%"></div></div>'
                '<div class="bar-scale"><span>0%</span><span>50%</span><span>100%</span></div>',
                unsafe_allow_html=True,
            )

        st.markdown("&nbsp;", unsafe_allow_html=True)
        card_header("Health indicators", "Compared with general reference ranges — this is not model output.")
        rows = "".join(
            f'<div class="check-row"><div><div class="check-name">{name}</div>'
            f'<div class="check-val">{value}</div></div>{pill(status, lvl)}</div>'
            for name, value, status, lvl in res["checks"]
        )
        st.markdown(rows, unsafe_allow_html=True)

        report = res["row"].assign(Prediction=label(pred))
        if prob is not None:
            report["Risk probability (%)"] = round(prob * 100, 1)
        st.download_button(
            "Download report (CSV)", report.to_csv(index=False).encode("utf-8"),
            file_name="heart_disease_prediction.csv", mime="text/csv", icon=":material/download:",
        )


def render_predict(model):
    page_header(
        "Heart disease risk prediction",
        "Enter the patient's details to estimate the risk of heart disease using the trained "
        "Logistic Regression model.",
    )

    with st.container(key="card_profile"):
        card_header("Patient profile", "Personal details and medical history.")
        c1, c2 = st.columns(2)
        c1.selectbox("Gender", ["Male", "Female"], key="gender")
        c2.number_input("Age (years)", 1, 120, key="age")

        c1, c2 = st.columns(2)
        c1.radio("Current smoker?", ["No", "Yes"], horizontal=True, key="smoker")
        c2.number_input(
            "Cigarettes per day", 0, 100, key="cigs",
            disabled=(st.session_state["smoker"] == "No"),
            help="Only used if the patient currently smokes.",
        )

        c1, c2, c3, c4 = st.columns(4)
        c1.radio("BP medication?", ["No", "Yes"], horizontal=True, key="bpmeds")
        c2.radio("Previous stroke?", ["No", "Yes"], horizontal=True, key="stroke")
        c3.radio("Hypertensive?", ["No", "Yes"], horizontal=True, key="hyp")
        c4.radio("Diabetic?", ["No", "Yes"], horizontal=True, key="diabetes")

    with st.container(key="card_clinical"):
        card_header("Clinical measurements", "Latest lab and vital-sign values.")
        c1, c2, c3 = st.columns(3)
        c1.number_input("Total cholesterol (mg/dL)", 50, 700, key="chol")
        c2.number_input("Glucose (mg/dL)", 30, 500, key="glucose")
        c3.number_input("Heart rate (bpm)", 30, 220, key="hr")

        c1, c2, c3 = st.columns(3)
        c1.number_input("Systolic BP (mmHg)", 70.0, 300.0, step=0.5, key="sys")
        c2.number_input("Diastolic BP (mmHg)", 40.0, 200.0, step=0.5, key="dia")
        c3.number_input("BMI", 10.0, 70.0, step=0.1, key="bmi")

        if st.session_state["sys"] <= st.session_state["dia"]:
            st.warning("Systolic BP is normally higher than diastolic BP — please double-check.")

        with st.expander("Calculate BMI from height and weight"):
            c1, c2, c3 = st.columns([1, 1, 1])
            c1.number_input("Height (cm)", 100.0, 250.0, step=1.0, key="h_cm")
            c2.number_input("Weight (kg)", 20.0, 300.0, step=0.5, key="w_kg")
            calc = st.session_state["w_kg"] / ((st.session_state["h_cm"] / 100) ** 2)
            c3.metric("Calculated BMI", f"{calc:.1f}")
            st.button("Use this BMI", on_click=apply_bmi, key="use_bmi_btn")

    c1, c2, c3 = st.columns([2, 1, 1])
    with c1:
        predict_clicked = st.button("Predict risk", type="primary", key="predict_btn")
    c2.button("Fill example patient", on_click=fill_example, key="example_btn")
    c3.button("Reset form", on_click=reset_form, key="reset_btn")

    if predict_clicked:
        s = st.session_state
        smoker = yn(s["smoker"])
        values = {
            "Gender": 1 if s["gender"] == "Male" else 0,
            "age": s["age"],
            "currentSmoker": smoker,
            "cigsPerDay": s["cigs"] if smoker else 0,
            "BPMeds": yn(s["bpmeds"]),
            "prevalentStroke": yn(s["stroke"]),
            "prevalentHyp": yn(s["hyp"]),
            "diabetes": yn(s["diabetes"]),
            "totChol": s["chol"],
            "sysBP": s["sys"],
            "diaBP": s["dia"],
            "BMI": s["bmi"],
            "heartRate": s["hr"],
            "glucose": s["glucose"],
        }
        row = pd.DataFrame([[values[f] for f in FEATURES]], columns=FEATURES)
        try:
            pred, prob = predict(model, row)
        except Exception as exc:  # keep the app alive on model/feature mismatches
            st.error(f"Prediction failed: {exc}")
        else:
            pred_i = int(pred[0])
            prob_f = float(prob[0]) if prob is not None else None
            st.session_state["last_result"] = {
                "row": row, "pred": pred_i, "prob": prob_f, "checks": health_checks(values),
            }
            entry = {
                "Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "Gender": s["gender"], "Age": values["age"],
                "Smoker": s["smoker"], "Cigs/day": values["cigsPerDay"],
                "BP meds": s["bpmeds"], "Stroke": s["stroke"], "Hypertensive": s["hyp"],
                "Diabetic": s["diabetes"], "Cholesterol": values["totChol"],
                "Systolic": values["sysBP"], "Diastolic": values["diaBP"],
                "BMI": values["BMI"], "Heart rate": values["heartRate"],
                "Glucose": values["glucose"],
                "Prediction": label(pred_i),
                "Risk probability (%)": round(prob_f * 100, 1) if prob_f is not None else None,
            }
            st.session_state["history"].append(entry)

    if st.session_state["last_result"] is not None:
        render_result(st.session_state["last_result"])

    disclaimer()


# ----------------------------------------------------------------------------
# Page: Batch prediction
# ----------------------------------------------------------------------------
def render_batch(model):
    page_header(
        "Batch prediction",
        "Upload a CSV with one patient per row and get a prediction for every record at once.",
    )

    with st.container(key="card_batch_upload"):
        card_header("Upload patient records", "The CSV must contain these columns (names are case-insensitive).")
        st.markdown(" ".join(pill(f, "info") for f in FEATURES), unsafe_allow_html=True)
        st.markdown(
            '<div class="card-desc" style="margin-top:.8rem">Gender accepts <b>Male/Female</b> (or 1/0); '
            "the yes/no fields accept <b>Yes/No</b> (or 1/0).</div>",
            unsafe_allow_html=True,
        )
        template = pd.DataFrame([EXAMPLE_ROW], columns=FEATURES)
        template["Gender"] = template["Gender"].map({1: "Male", 0: "Female"})
        st.download_button(
            "Download CSV template", template.to_csv(index=False).encode("utf-8"),
            file_name="heart_disease_template.csv", mime="text/csv", icon=":material/description:",
        )
        uploaded = st.file_uploader("Choose a CSV file", type=["csv"], key="batch_file")

    if uploaded is None:
        with st.container(key="card_batch_empty"):
            empty_state("📄", "No file uploaded yet", "Upload a CSV above to see predictions here.")
        disclaimer()
        return

    try:
        raw = pd.read_csv(uploaded)
    except Exception as exc:
        st.error(f"Could not read the CSV file: {exc}")
        return

    if raw.empty:
        st.warning("The uploaded file has no rows.")
        return

    clean, invalid_rows, missing = prepare_batch(raw)
    if missing:
        st.error("Missing required column(s): " + ", ".join(missing))
        return

    if clean.empty:
        st.error("None of the rows contained valid values for every column.")
        return

    try:
        pred, prob = predict(model, clean)
    except Exception as exc:
        st.error(f"Prediction failed: {exc}")
        return

    result = raw.loc[clean.index].copy()
    result["Prediction"] = [label(p) for p in pred]
    if prob is not None:
        result["Risk probability (%)"] = (prob * 100).round(1)

    n_risk = int((pred == 1).sum())
    with st.container(key="card_batch_results"):
        card_header("Results", f"Predictions for {len(clean)} record(s).")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Records predicted", len(clean))
        c2.metric("At risk", n_risk)
        c3.metric("Not at risk", len(clean) - n_risk)
        c4.metric("Rows skipped", len(invalid_rows))
        if invalid_rows:
            shown = ", ".join(map(str, invalid_rows[:10])) + (" …" if len(invalid_rows) > 10 else "")
            st.warning(f"Skipped rows with blank or unreadable values (row numbers: {shown}).")
        st.markdown(html_table(result, pill_col="Prediction"), unsafe_allow_html=True)
        st.download_button(
            "Download results (CSV)", result.to_csv(index=False).encode("utf-8"),
            file_name="heart_disease_batch_results.csv", mime="text/csv", icon=":material/download:",
        )
    disclaimer()


# ----------------------------------------------------------------------------
# Page: History
# ----------------------------------------------------------------------------
def render_history():
    page_header("Prediction history", "Predictions made in this browser session (cleared when the session ends).")
    hist = st.session_state["history"]

    with st.container(key="card_history"):
        if not hist:
            empty_state("🕘", "No predictions yet", "Run a prediction on the Predict page and it will appear here.")
            st.button("Go to Predict", on_click=go, args=("Predict",), key="hist_go_predict", type="primary")
            return
        df = pd.DataFrame(hist).iloc[::-1].reset_index(drop=True)
        n_risk = int((df["Prediction"] == "At risk").sum())
        c1, c2, c3 = st.columns(3)
        c1.metric("Total predictions", len(df))
        c2.metric("At risk", n_risk)
        c3.metric("Not at risk", len(df) - n_risk)
        st.markdown(html_table(df, pill_col="Prediction"), unsafe_allow_html=True)
        c1, c2, _ = st.columns([1, 1, 2])
        c1.download_button(
            "Download history (CSV)", df.to_csv(index=False).encode("utf-8"),
            file_name="heart_disease_history.csv", mime="text/csv", icon=":material/download:",
        )
        c2.button("Clear history", on_click=clear_history, key="clear_hist_btn")


# ----------------------------------------------------------------------------
# Page: Model insights
# ----------------------------------------------------------------------------
def render_insights(model):
    page_header("Model insights", "How the model was built and which inputs push its prediction up or down.")
    est = final_estimator(model)
    is_pipeline = hasattr(model, "steps")

    with st.container(key="card_model_summary"):
        card_header("Model summary")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Algorithm", type(est).__name__)
        c2.metric("Input features", len(FEATURES))
        c3.metric("Classes", " / ".join(str(c) for c in getattr(est, "classes_", [0, 1])))
        c4.metric("Test accuracy*", "64.2%")
        st.caption("*Accuracy reported in the project notebook on the 20% hold-out set.")

    with st.container(key="card_coefs"):
        card_header(
            "Feature coefficients",
            "Positive values push the prediction toward “at risk”, negative values toward “not at risk”.",
        )
        coef = getattr(est, "coef_", None)
        if coef is None or len(FEATURES) != coef.shape[-1]:
            st.info("This model type does not expose coefficients.")
        else:
            coef_df = pd.DataFrame({"Feature": FEATURES, "Coefficient": coef[0]})
            coef_df = coef_df.sort_values("Coefficient", key=abs, ascending=False)
            st.markdown(coef_bars(coef_df), unsafe_allow_html=True)
            st.markdown(html_table(coef_df.round(4)), unsafe_allow_html=True)
            if getattr(est, "intercept_", None) is not None:
                st.caption(f"Intercept: {float(est.intercept_[0]):.4f}")
            st.caption(
                "Features were standardised inside the pipeline, so coefficient sizes are comparable."
                if is_pipeline else
                "Features were not scaled, so coefficient size also depends on each feature's units "
                "(e.g. cholesterol vs. a 0/1 flag) — treat this as direction, not a strict importance ranking."
            )

    with st.container(key="card_training"):
        card_header("Training summary", "Taken from the project notebook.")
        steps = [
            ("Dataset", "4,238 patient records with 15 columns (14 inputs + heart-disease outcome)."),
            ("Cleaning", "Dropped the <code>education</code> column and rows with missing values → 3,749 records."),
            ("Encoding", "Gender, prevalent stroke and the outcome were converted to 0/1."),
            ("Class balance", "3,177 “No” vs 572 “Yes” → balanced to 3,177 each with SMOTE (6,354 records)."),
            ("Split", "80% training / 20% testing."),
            ("Model", "Logistic Regression (scikit-learn), saved with pickle as model.pkl."),
        ]
        st.markdown(
            "".join(
                f'<div class="glossary-row"><div class="glossary-key">{k}</div><div class="glossary-val">{v}</div></div>'
                for k, v in steps
            ),
            unsafe_allow_html=True,
        )
    disclaimer()


# ----------------------------------------------------------------------------
# Page: About
# ----------------------------------------------------------------------------
def render_about():
    page_header("About this app", "A machine-learning project that estimates heart-disease risk from patient data.")

    with st.container(key="card_about"):
        card_header("What you can do")
        st.markdown(
            "- **Predict** – enter one patient's details, get a risk estimate and health indicators.\n"
            "- **Batch Prediction** – upload a CSV and predict many patients at once.\n"
            "- **History** – review and download this session's predictions.\n"
            "- **Model Insights** – see how the model was trained and what drives its output."
        )

    with st.container(key="card_glossary"):
        card_header("Input fields", "What each value means.")
        st.markdown(
            "".join(
                f'<div class="glossary-row"><div class="glossary-key">{k}</div><div class="glossary-val">{v}</div></div>'
                for k, v in FIELD_HELP
            ),
            unsafe_allow_html=True,
        )
    disclaimer()


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
inject_css()

model, model_error = None, None
if not MODEL_PATH.exists():
    model_error = (
        "`model.pkl` not found. Put the file saved from your notebook "
        "(`pk.dump(model, open('model.pkl', 'wb'))`) in the same folder as `app.py`."
    )
else:
    try:
        model = load_model()
    except Exception as exc:
        model_error = (
            f"Could not load `model.pkl`: {exc}. Make sure the scikit-learn version here "
            f"({sklearn.__version__}) matches the one used to train the model."
        )

render_sidebar(model_ok=model is not None)

page = st.session_state["page"]
if page == "About":
    render_about()
elif model_error:
    page_header("Heart disease risk prediction", "The model needs to be available before this page can be used.")
    st.error(model_error)
elif page == "Predict":
    render_predict(model)
elif page == "Batch Prediction":
    render_batch(model)
elif page == "History":
    render_history()
elif page == "Model Insights":
    render_insights(model)