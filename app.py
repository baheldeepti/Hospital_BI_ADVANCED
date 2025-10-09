# ValueCare Predict & Plan Studio — Advanced AI Predictive Engine (Rev)
import os, io, json, textwrap, warnings, traceback, builtins
from datetime import datetime
from typing import Dict, Optional, Tuple, List, Any

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

from sklearn.ensemble import RandomForestClassifier, IsolationForest, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC, SVR
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, label_binarize
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    mean_absolute_error, mean_squared_error, r2_score,
    roc_auc_score, roc_curve
)
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA
# from pages.header import add_top_bar  # keep if you use it

# ---------------- UI Setup ----------------
# add_top_bar(active_page="Predictive Insights")
warnings.filterwarnings("ignore")
st.set_page_config(page_title="ValueCare Predict & Plan Studio", layout="wide", page_icon="🏥")

hide_sidebar = """
    <style>
        section[data-testid="stSidebar"] { display: none; }
        .block-container { padding-top: 1rem !important; }
        h3 { margin-top: -2px !important; }
        .stTabs [data-baseweb="tab-list"] { gap: 60px; }
        .stTabs [data-baseweb="tab"] { padding: 8px 16px; border-radius: 8px; }
        div.streamlit-expanderHeader { font-weight: 600; }
    </style>
"""
st.markdown(hide_sidebar, unsafe_allow_html=True)
st.markdown("<h3 style='color:#1F41BB'>ValueCare Predict & Plan Studio</h3>", unsafe_allow_html=True)
st.write("Data-driven insights for better patient outcomes and operational efficiency.")

st.markdown("""
<style>
div.stButton > button { min-width: 200px; max-width: 200px; width: 200px; }

/* Cards, sections */
.data-overview { background: #f8f9fa; padding: 1rem; border-radius: 8px; margin-bottom: 1.5rem; border: 1px solid #e9ecef; }
.metric-card { background: #fff; padding: 1rem; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,.06); border: 1px solid #e9ecef; margin: .5rem 0; text-align: center; }
.analysis-section { background: #fff; padding: 1.25rem 1.5rem; border-radius: 12px; border: 1px solid #e9ecef; margin: 1rem 0; box-shadow: 0 2px 8px rgba(0,0,0,.04); }
.config-row { background: #f8f9fa; padding: 1rem; border-radius: 8px; margin: 1rem 0; border-left: 4px solid #4285f4; }

/* Screenshot-style AI Summary */
.ai-summary {
    background: #ffffff;
    color: #111827;
    border-radius: 12px;
    border: 1px solid #e5e7eb;
    padding: 1.25rem 1.5rem;
    margin: 1rem 0 0.25rem 0;
}
.ai-summary h3 {
    font-size: 1.75rem;
    line-height: 1.25;
    margin: 0 0 .75rem 0;
    font-weight: 700;
}
.ai-summary p { margin: 0 0 1rem 0; font-size: 1rem; }
.ai-summary p:last-child { margin-bottom: 0; }

/* AI assistant box */
.ai-assistant-box {
    background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
    padding: 1.2rem 1.4rem; border-radius: 12px; margin: 1.2rem 0; color: #fff;
}
</style>
""", unsafe_allow_html=True)

# ---------- Optional libraries ----------
try:
    from prophet import Prophet
    HAS_PROPHET = True
except Exception:
    HAS_PROPHET = False

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except Exception:
    HAS_XGBOOST = False

# ---------- OpenAI (optional) – default GPT-3.5-turbo ----------
OPENAI_AVAILABLE = False
OPENAI_CLIENT = None
OPENAI_MODEL_MAP = {"GPT-3.5": "gpt-3.5-turbo", "GPT-4.0": "gpt-4"}

def get_openai_key():
    key = None
    try:
        if hasattr(st, "secrets") and "OPENAI_API_KEY" in st.secrets:
            key = st.secrets["OPENAI_API_KEY"]
    except Exception:
        pass
    if not key:
        key = os.getenv("OPENAI_API_KEY")
    return key

try:
    from openai import OpenAI   # new SDK
    _api_key = get_openai_key()
    if _api_key:
        OPENAI_CLIENT = OpenAI(api_key=_api_key)
        OPENAI_AVAILABLE = True
except Exception:
    try:
        import openai            # legacy
        _api_key = get_openai_key()
        if _api_key:
            openai.api_key = _api_key
            OPENAI_CLIENT = openai
            OPENAI_AVAILABLE = True
    except Exception:
        OPENAI_AVAILABLE = False
        OPENAI_CLIENT = None

# ---------- Session state ----------
st.session_state.setdefault("model_results", {})
st.session_state.setdefault("custom_results", {})
st.session_state.setdefault("selected_features", {})
st.session_state.setdefault("decision_log", [])
st.session_state.setdefault("USER_ROLE", "Executive")
st.session_state.setdefault("INSIGHTS_USE_LLM", False)
st.session_state.setdefault("INSIGHTS_MODEL", "GPT-3.5")
st.session_state.setdefault("INSIGHTS_TEMP", 0.2)
st.session_state.setdefault("los_is_classification", None)
st.session_state.setdefault("los_target_type", None)
st.session_state.setdefault("los_avg_los", None)
st.session_state.setdefault("trained_models", {})
st.session_state.setdefault("chat_history", {})

# ---------- Global controls ----------
col1, col2, col3, col4 = st.columns([1, 1.2, 1.3, 1.4])
with col1:
    st.session_state.USER_ROLE = st.radio(
        "Audience", ["Executive", "Analyst"], horizontal=True,
        help="Changes tone, depth, and KPIs in generated insights.",
        index=0 if st.session_state.USER_ROLE == "Executive" else 1
    )
with col2:
    st.session_state.INSIGHTS_USE_LLM = st.checkbox(
        "Use ChatGPT for section insights",
        value=st.session_state.INSIGHTS_USE_LLM,
        help="If off or unavailable, we use a built-in, data-driven summary."
    )
with col3:
    st.session_state.INSIGHTS_MODEL = st.selectbox(
        "ChatGPT Model", ["GPT-3.5", "GPT-4.0"],
        index=0 if st.session_state.INSIGHTS_MODEL == "GPT-3.5" else 1
    )
with col4:
    st.session_state.INSIGHTS_TEMP = st.slider("Creativity", 0.0, 1.0, float(st.session_state.INSIGHTS_TEMP), 0.1)

if st.session_state.INSIGHTS_USE_LLM and not OPENAI_AVAILABLE:
    st.warning("⚠️ OpenAI API key not found. Set it via secrets or env. Falling back to local insights.")

# ---------- Data loading ----------
@st.cache_data(ttl=3600, show_spinner="Loading hospital data...")
def load_hospital_data() -> pd.DataFrame:
    local_path = "/mnt/data/modified_healthcare_dataset.csv"
    try:
        if os.path.exists(local_path):
            df = pd.read_csv(local_path)
        else:
            url = "https://raw.githubusercontent.com/baheldeepti/hospital-streamlit-app/main/modified_healthcare_dataset.csv"
            df = pd.read_csv(url)
    except Exception as e:
        st.warning(f"Could not load data from file/url: {e}. Using synthetic data.")
        np.random.seed(42)
        n_records = 6000
        dates = pd.date_range(start="2023-01-01", end="2024-01-01", periods=n_records)
        df = pd.DataFrame(
            {
                "Date of Admission": dates,
                "Discharge Date": dates + pd.to_timedelta(np.random.exponential(5, n_records), unit="D"),
                "Age": np.random.normal(55, 20, n_records).clip(0, 100),
                "Medical Condition": np.random.choice(["Diabetes","Hypertension","Heart Disease","Cancer","Asthma"], n_records),
                "Admission Type": np.random.choice(["Emergency","Elective","Urgent"], n_records, p=[0.4,0.4,0.2]),
                "Hospital": np.random.choice(["General Hospital","Children Hospital","University Hospital"], n_records),
                "Insurance Provider": np.random.choice(["Medicare","Medicaid","Blue Cross","Aetna","UnitedHealth"], n_records),
                "Billing Amount": np.random.lognormal(8, 1, n_records),
                "Doctor": [f"Dr. {n}" for n in np.random.choice(["Smith","Johnson","Williams","Brown","Jones"], n_records)],
                "Test Results": np.random.choice(["Normal","Abnormal","Inconclusive"], n_records, p=[0.6,0.3,0.1]),
            }
        )

    # Normalize & engineer
    df.columns = df.columns.str.replace(" ", "_").str.lower()
    for c in ["date_of_admission", "discharge_date"]:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")

    if {"date_of_admission","discharge_date"}.issubset(df.columns):
        los = (df["discharge_date"] - df["date_of_admission"]).dt.days
        df["length_of_stay"] = np.clip(los.fillna(0), 0, 365).astype(int)

    if "date_of_admission" in df.columns:
        df["admission_month"] = df["date_of_admission"].dt.month
        df["admission_day_of_week"] = df["date_of_admission"].dt.dayofweek
        df["admission_quarter"] = df["date_of_admission"].dt.quarter

    if "billing_amount" in df.columns:
        df["billing_amount"] = pd.to_numeric(df["billing_amount"], errors="coerce").fillna(0).clip(0)

    if "age" in df.columns:
        df["age"] = pd.to_numeric(df["age"], errors="coerce").fillna(df["age"].median())
        df["age_group"] = pd.cut(
            df["age"], bins=[0,18,35,50,65,100],
            labels=["Child","Young Adult","Adult","Senior","Elderly"], include_lowest=True
        )

    if "length_of_stay" in df.columns:
        df["los_category"] = pd.cut(
            df["length_of_stay"], bins=[-0.1,3,7,14,float("inf")],
            labels=["Short","Medium","Long","Extended"], include_lowest=True
        )
    return df.dropna(subset=["date_of_admission"])

df = load_hospital_data()

# ---------- Overview ----------
st.markdown('<div class="data-overview"><h3>📊 Dataset Overview</h3></div>', unsafe_allow_html=True)
base_card_style = "padding:5px 8px 8px 10px;border-radius:14px; text-align:center; font-size:24px; margin-bottom:28px;"
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown(f"<div style='background-color:#ECF1FF94;{base_card_style}'><h5>Total Records</h5><p>{len(df):,}</p></div>", unsafe_allow_html=True)
with c2:
    days = (df["date_of_admission"].max() - df["date_of_admission"].min()).days
    st.markdown(f"<div style='background-color:#FCBB021C;{base_card_style}'><h5>Date Range</h5><p>{days} days</p></div>", unsafe_allow_html=True)
with c3:
    hospitals_count = df["hospital"].nunique() if "hospital" in df.columns else "N/A"
    st.markdown(f"<div style='background-color:#B820EB14;{base_card_style}'><h5>Hospitals</h5><p>{hospitals_count}</p></div>", unsafe_allow_html=True)
with c4:
    conditions_count = df["medical_condition"].nunique() if "medical_condition" in df.columns else "N/A"
    st.markdown(f"<div style='background-color:#28A74514;{base_card_style}'><h5>Conditions</h5><p>{conditions_count}</p></div>", unsafe_allow_html=True)

# ---------- Filters ----------
with st.expander("🔍 Filter Data", expanded=False):
    f1, f2, f3 = st.columns(3)
    with f1:
        hospitals = st.multiselect("Select Hospitals", df["hospital"].dropna().unique() if "hospital" in df.columns else [])
    with f2:
        conditions = st.multiselect("Medical Conditions", df["medical_condition"].dropna().unique() if "medical_condition" in df.columns else [])
    with f3:
        admission_types = st.multiselect("Admission Types", df["admission_type"].dropna().unique() if "admission_type" in df.columns else [])

filtered_df = df.copy()
if hospitals: filtered_df = filtered_df[filtered_df["hospital"].isin(hospitals)]
if conditions: filtered_df = filtered_df[filtered_df["medical_condition"].isin(conditions)]
if admission_types: filtered_df = filtered_df[filtered_df["admission_type"].isin(admission_types)]
if len(filtered_df) != len(df):
    st.info(f"Applied filters: {len(filtered_df):,} of {len(df):,} records selected")
st.session_state["_filtered_df_for_summary"] = filtered_df

# ---------- Helpers ----------
def _fmt_pct(x, d=1):
    try: return f"{100*float(x):.{d}f}%"
    except: return "N/A"

def _fmt_num(x, d=1, money=False):
    try:
        f = float(x)
        return f"${f:,.0f}" if money else f"{f:.{d}f}"
    except: return "N/A"

import html

def render_ai_summary(text: str, title: str = "AI Summary"):
    # Escape first, then format line breaks
    escaped = html.escape(text or "")
    safe_text = escaped.replace("\n\n", "</p><p>").replace("\n", "<br/>")

    st.markdown(
        f"""
        <div class="ai-summary">
            <h3>{title}</h3>
            <p>{safe_text}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---- secure custom code executor (allow-listed imports) ----
def run_user_code(user_code: str, context: Dict[str, Any]) -> Tuple[bool, Any, str]:
    """
    Executes user Python code in a constrained namespace with allow-listed imports.
    Makes 'np', 'pd', 'go', 'px', statsmodels classes, sklearn available by default.
    """
    allowed_prefixes = ("numpy", "pandas", "sklearn", "statsmodels", "prophet")

    real_import = builtins.__import__
    def safe_import(name, globals=None, locals=None, fromlist=(), level=0):
        root = name.split(".")[0]
        if name.startswith(allowed_prefixes) or root in [p.split(".")[0] for p in allowed_prefixes]:
            return real_import(name, globals, locals, fromlist, level)
        raise ImportError("import not allowed")

    safe_builtins = {
        "abs": abs, "min": min, "max": max, "sum": sum, "len": len, "range": range, "round": round,
        "float": float, "int": int, "str": str, "dict": dict, "list": list, "tuple": tuple,
        "enumerate": enumerate, "zip": zip, "sorted": sorted, "print": print, "bool": bool,
        "__import__": safe_import,
    }
    safe_globals = {"__builtins__": safe_builtins, "pd": pd, "np": np, "go": go, "px": px,
                    "ARIMA": ARIMA, "ExponentialSmoothing": ExponentialSmoothing}
    safe_locals = context.copy()

    try:
        compiled = compile(user_code, "<user_code>", "exec")
        exec(compiled, safe_globals, safe_locals)
        if "result" in safe_locals: return True, safe_locals["result"], ""
        if "output" in safe_locals: return True, safe_locals["output"], ""
        return True, safe_locals, ""
    except Exception as e:
        return False, None, f"Error: {e}\n{traceback.format_exc()}"

# ---- local (non-LLM) insight ----
def local_insight(section_title: str, data_summary: dict, model_results: Optional[dict]) -> str:
    section = section_title.lower()
    gdf = st.session_state.get("_filtered_df_for_summary")

    if "admission" in section:
        if gdf is None or len(gdf) == 0: return "Insufficient admissions data."
        adm_by_dow = gdf.groupby(gdf["date_of_admission"].dt.dayofweek).size().reindex(range(7), fill_value=0)
        weekday_uplift = (adm_by_dow[:5].mean() - adm_by_dow[5:].mean()) / max(adm_by_dow.mean(), 1e-9)
        body = [
            f"Weekday volumes run {_fmt_pct(weekday_uplift)} higher than weekends.",
            f"Best model: {data_summary.get('best_model','?')} (MAPE {_fmt_num(data_summary.get('best_model_mape', np.nan),1)}%).",
            "Align roster to weekday peaks; keep a flex pool for 2σ days; refresh forecast weekly."
        ]
        return "\n".join(body)

    if "revenue" in section:
        if gdf is None or "billing_amount" not in gdf.columns or len(gdf) == 0:
            return "Insufficient revenue data."
        bills = gdf["billing_amount"].sort_values(ascending=False)
        top_decile = bills.iloc[: max(1,int(.1*len(bills)))].sum() / max(bills.sum(), 1e-9)
        best = None
        if model_results:
            best = min(model_results.keys(), key=lambda x: abs(model_results[x].get("anomaly_rate", 1.0) - 0.05))
        anomaly_line = "" if best is None else f"{best} flags {int(model_results[best].get('anomalies_detected',0))} cases ({_fmt_pct(model_results[best].get('anomaly_rate',0))})."
        body = [f"Top 10% encounters drive {_fmt_pct(top_decile)} of revenue.", anomaly_line, "Audit top-decile encounters; add pre-bill rules for spike days."]
        return "\n".join([b for b in body if b])

    if "length of stay" in section or "los" in section:
        if gdf is None or "length_of_stay" not in gdf.columns or len(gdf)==0:
            return "Insufficient LOS data."
        los = gdf["length_of_stay"].astype(float)
        iqr = np.percentile(los,75) - np.percentile(los,25)
        return f"Average LOS {los.mean():.1f} days (median {los.median():.1f}; IQR {iqr:.1f}).\nStandardize discharge pathways for highest-LOS cohorts."

    return "No summary available."

# ---- LLM insight (role-aware) ----
def llm_insight(section_title: str, data_summary: dict, model_results: Optional[dict], prompt_overrides: str="") -> str:
    use_llm = bool(st.session_state.get("INSIGHTS_USE_LLM", False))
    if not use_llm or not OPENAI_AVAILABLE:
        return local_insight(section_title, data_summary, model_results)

    model_label = st.session_state.get("INSIGHTS_MODEL", "GPT-3.5")
    model_name = OPENAI_MODEL_MAP.get(model_label, "gpt-3.5-turbo")
    temperature = float(st.session_state.get("INSIGHTS_TEMP", 0.2))
    audience = st.session_state.get("USER_ROLE", "Executive")

    base_prompt = f"""
You are a hospital operations advisor. Audience: {audience}.
Write crisp, action-oriented insights for "{section_title}".

Return EXACTLY:
- 1 headline (max 12 words)
- 4 bullets (business language, no formulas)
- "Next 2 Weeks: <3 short actions>"

Context:
data_summary = {json.dumps(data_summary, default=str)}
model_results = {json.dumps(model_results or {}, default=str)}
"""
    if prompt_overrides.strip():
        base_prompt += f"\nAdditional Guidance:\n{prompt_overrides.strip()}\n"

    try:
        if hasattr(OPENAI_CLIENT, "chat") and hasattr(OPENAI_CLIENT.chat, "completions"):
            resp = OPENAI_CLIENT.chat.completions.create(
                model=model_name, temperature=temperature,
                messages=[{"role":"system","content":"You turn analytics into clear, executive/analyst insights."},
                          {"role":"user","content":base_prompt}]
            )
            text = resp.choices[0].message.content.strip()
        else:
            resp = OPENAI_CLIENT.ChatCompletion.create(
                model=model_name, temperature=temperature,
                messages=[{"role":"system","content":"You turn analytics into clear, executive/analyst insights."},
                          {"role":"user","content":base_prompt}]
            )
            text = resp.choices[0].message["content"].strip()
        return text
    except Exception as e:
        st.info(f"LLM insights unavailable ({e}). Showing local summary.")
        return local_insight(section_title, data_summary, model_results)

# ---- result registries ----
def register_custom_result(section_key: str, payload: dict):
    st.session_state.custom_results[section_key] = payload or {}

def get_all_results(section_key: str) -> dict:
    core = st.session_state.model_results.get(section_key, {})
    custom = st.session_state.custom_results.get(section_key, {})
    if custom:
        core = dict(core)
        core["Custom"] = custom
    return core

def get_custom_only(section_key: str) -> dict:
    c = st.session_state.custom_results.get(section_key, {})
    return {"Custom": c} if c else {}

def norm_forecast_output(obj, future_days: int):
    if obj is None: return []
    if isinstance(obj, dict) and "forecast" in obj: obj = obj["forecast"]
    try:
        arr = np.array(obj, dtype=float).ravel().tolist()
        return arr[:future_days]
    except: return []

def compute_forecast_metrics(test_values: np.ndarray, pred_values: np.ndarray):
    if test_values is None or len(test_values)==0 or len(pred_values)==0: return {}
    n = min(len(test_values), len(pred_values))
    y_true = np.array(test_values[:n], dtype=float)
    y_pred = np.array(pred_values[:n], dtype=float)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mape = float(np.mean(np.abs((y_true - y_pred)/np.maximum(y_true,1e-9)))*100)
    return {"MAE": mae, "RMSE": rmse, "MAPE": mape}

def norm_anomaly_output(obj):
    res = {"anomalies_detected":0, "anomaly_rate":0.0, "avg_anomaly_amount":0.0}
    if obj is None: return res
    if isinstance(obj, dict):
        n = int(obj.get("n_anomalies", obj.get("anomalies_detected", 0)))
        idx = obj.get("anomaly_indices", [])
    else:
        idx = list(obj); n = len(idx)
    res["anomalies_detected"] = n; res["_indices"] = idx
    return res

def norm_los_output(obj, is_classification: bool):
    if not isinstance(obj, dict): return {}
    return (
        {"accuracy": float(obj.get("accuracy", 0.0)),
         "precision": float(obj.get("precision", 0.0)),
         "recall": float(obj.get("recall", 0.0)),
         "f1_score": float(obj.get("f1", obj.get("f1_score", 0.0)))}
        if is_classification else
        {"mae": float(obj.get("mae", 0.0)),
         "rmse": float(obj.get("rmse", 0.0)),
         "r2_score": float(obj.get("r2", obj.get("r2_score", 0.0)))}
    )

def explain_in_30_seconds(title: str, bullets: list[str]):
    with st.expander("📘 Modeling explained in 30 seconds", expanded=False):
        st.markdown(f"**{title}**")
        for b in bullets: st.markdown(f"- {b}")

def expander_title(label: str): return f"🧪 {label}"

# ---- AI Assistant ----
def ai_assistant_section(tab_name: str, context: str, expanded: bool = False, help_text: str = ""):
    """AI Assistant for each tab, wrapped in an expander."""
    title = f"🤖 AI Model Assistant — {tab_name}"
    with st.expander(title, expanded=expanded):
        if help_text:
            st.caption(help_text)

        # (Optional) keep the gradient box inside the expander for flair
        st.markdown(
            '<div class="ai-assistant-box"><h4>Ask about your data & models</h4>'
            '<p>Tips: request feature ideas, debugging help, or insight summaries.</p></div>',
            unsafe_allow_html=True
        )

        chat_key = f"chat_{tab_name}"
        if chat_key not in st.session_state.chat_history:
            st.session_state.chat_history[chat_key] = []

        user_question = st.text_input(
            f"Ask about {tab_name}:",
            key=f"user_q_{tab_name}",
            placeholder="e.g., How can I improve model accuracy? What features should I add?"
        )

        if st.button("Ask AI", key=f"ask_{tab_name}"):
            if user_question and OPENAI_AVAILABLE:
                with st.spinner("Thinking..."):
                    try:
                        model_label = st.session_state.get("INSIGHTS_MODEL", "GPT-4.0")
                        model_name = OPENAI_MODEL_MAP.get(model_label, "gpt-4")

                        prompt = f"""You are an AI assistant helping with hospital analytics and machine learning.
Context: {context}
Tab: {tab_name}
User role: {st.session_state.USER_ROLE}

User question: {user_question}

Provide a helpful, actionable answer focused on improving the analysis or model."""

                        if hasattr(OPENAI_CLIENT, "chat"):
                            resp = OPENAI_CLIENT.chat.completions.create(
                                model=model_name,
                                temperature=0.7,
                                messages=[
                                    {"role": "system",
                                     "content": "You are a helpful ML and analytics assistant for hospital operations."},
                                    {"role": "user", "content": prompt},
                                ],
                            )
                            answer = resp.choices[0].message.content.strip()
                        else:
                            resp = OPENAI_CLIENT.ChatCompletion.create(
                                model=model_name,
                                temperature=0.7,
                                messages=[
                                    {"role": "system",
                                     "content": "You are a helpful ML and analytics assistant for hospital operations."},
                                    {"role": "user", "content": prompt},
                                ],
                            )
                            answer = resp.choices[0].message["content"].strip()

                        st.session_state.chat_history[chat_key].append({"q": user_question, "a": answer})
                    except Exception as e:
                        st.error(f"AI Assistant error: {e}")
            elif not OPENAI_AVAILABLE:
                st.warning("AI Assistant requires OpenAI API key. Please configure it in Streamlit secrets.")

        # Display chat history
        if st.session_state.chat_history[chat_key]:
            st.markdown("### Recent Conversations")
            for i, chat in enumerate(reversed(st.session_state.chat_history[chat_key][-3:])):
                with st.expander(f"Q: {chat['q'][:50]}...", expanded=(i == 0)):
                    st.markdown(f"**Question:** {chat['q']}")
                    st.markdown(f"**Answer:** {chat['a']}")


# ---------- Tabs ----------
def _build_anomaly_matrix(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    if not features:
        return pd.DataFrame(index=df.index)
    # Keep index to map anomalies back to filtered_df rows
    return df[features].dropna()

tabs = st.tabs(["📈 Admissions Forecasting", "💰 Revenue Analytics", "🛏️ Length of Stay Prediction", "📊 Operational KPIs & Simulator"])

# ===================== TAB 1: Admissions Forecasting =====================
with tabs[0]:
    st.markdown("""<div class="analysis-section"><h3>📈 Patient Admission Forecasting</h3><p>Predict future admission patterns to optimize staffing and resource allocation.</p></div>""", unsafe_allow_html=True)
    with st.expander("ℹ️ Model Primer: Forecasting", expanded=False):
        st.markdown("""- **Linear Trend**: Fast baseline for steady growth/decline.\n- **ARIMA(1,1,1)**: Handles autocorrelation.\n- **Exponential Smoothing (HW, weekly)**: Captures 7-day seasonality.\n- **Prophet** *(optional)*: Flexible seasonality/holidays.""")

    explain_in_30_seconds("Forecasting admissions",
        ["Turn daily counts into a time series.",
         "Fit trend/ARIMA/Holt-Winters/Prophet.",
         "Pick the lowest error on a hold-out window.",
         "Project next N days for staffing/beds."])

    daily_adm = filtered_df.groupby(filtered_df["date_of_admission"].dt.date).size().reset_index()
    daily_adm.columns = ["date", "admissions"]
    st.session_state["_daily_adm_for_summary"] = daily_adm

    c1, c2 = st.columns([2, 1])
    with c1:
        st.subheader("📊 Current Admission Trends")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=daily_adm["date"], y=daily_adm["admissions"], mode="lines+markers", name="Daily Admissions", line=dict(width=2)))
        xnum = np.arange(len(daily_adm))
        if len(daily_adm) >= 2:
            z = np.polyfit(xnum, daily_adm["admissions"], 1)
            p = np.poly1d(z)
            fig.add_trace(go.Scatter(x=daily_adm["date"], y=p(xnum), mode="lines", name="Trend", line=dict(dash="dash")))
        fig.update_layout(title="Daily Admissions Over Time", xaxis_title="Date", yaxis_title="Admissions", height=400)
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        st.subheader("📈 Key Metrics")
        avg_daily = daily_adm["admissions"].mean() if len(daily_adm) else 0.0
        max_daily = daily_adm["admissions"].max() if len(daily_adm) else 0
        std_daily = daily_adm["admissions"].std() if len(daily_adm) else 0.0
        st.markdown(f"""
        <div class="metric-card"><h4>Average Daily</h4><h2 style="color:#4285f4;">{avg_daily:.1f}</h2></div>
        <div class="metric-card"><h4>Peak Daily</h4><h2 style="color:#34a853;">{max_daily}</h2></div>
        <div class="metric-card"><h4>Variability (σ)</h4><h2 style="color:#fbbc04;">{std_daily:.1f}</h2></div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="config-row"><h4>🔧 Forecasting Configuration</h4></div>', unsafe_allow_html=True)
    c3, c4, c5 = st.columns(3)
    with c3: forecast_days = st.slider("Forecast Period (days)", 7, 30, 14)
    with c4: _ = st.selectbox("Seasonality (informational)", ["Auto-detect", "Weekly", "Monthly", "None"])
    with c5: _ = st.slider("Confidence Level (visual only)", 0.8, 0.99, 0.95)

    col_models, col_btn = st.columns([17, 3])
    with col_models:
        selected_models = st.multiselect("Select Forecasting Models",
            ["Linear Trend","ARIMA","Exponential Smoothing","Prophet"],
            default=["Linear Trend","ARIMA","Exponential Smoothing"])
        if "Prophet" in selected_models and not HAS_PROPHET:
            st.info("Prophet not installed; it will be skipped.")
    with col_btn:
        st.markdown("""
            <style>
            div.stButton > button:first-child { background-color:#00AC33; color:white; margin-top:12px; border:none; }
            div.stButton > button:first-child:hover { background-color:#009a2a; }
            </style>
        """, unsafe_allow_html=True)
        generate = st.button("Generate Forecasts", key="train_forecast", type="primary")

    if generate:
        with st.spinner("Training forecasting models..."):
            results: Dict[str, Dict[str, float]] = {}
            forecasts: Dict[str, np.ndarray] = {}
            ts = daily_adm.set_index("date")["admissions"].astype(float)
            if len(ts) < 7:
                st.error("Need at least 7 days of data to forecast.")
            else:
                train_size = max(1, len(ts) - forecast_days)
                train_data = ts.iloc[:train_size]
                test_data = ts.iloc[train_size:]

                # Linear Trend
                if "Linear Trend" in selected_models and len(train_data) >= 2:
                    try:
                        x = np.arange(len(train_data))
                        coeffs = np.polyfit(x, train_data.values, 1)
                        future_x = np.arange(len(train_data), len(train_data) + forecast_days)
                        forecast = np.polyval(coeffs, future_x)
                        forecasts["Linear Trend"] = forecast
                        if len(test_data) > 0:
                            test_fx = np.polyval(coeffs, np.arange(len(train_data), len(ts)))
                            mae = mean_absolute_error(test_data.values, test_fx)
                            rmse = np.sqrt(mean_squared_error(test_data.values, test_fx))
                            mape = float(np.mean(np.abs((test_data.values - test_fx) / np.maximum(test_data.values, 1e-9))) * 100)
                            results["Linear Trend"] = {"MAE": mae, "RMSE": rmse, "MAPE": mape}
                    except Exception as e:
                        st.warning(f"Linear Trend failed: {e}")

                # ARIMA
                if "ARIMA" in selected_models and len(train_data) >= 8:
                    try:
                        model = ARIMA(train_data, order=(1, 1, 1))
                        fit = model.fit()
                        forecast = fit.forecast(steps=forecast_days).values
                        forecasts["ARIMA"] = forecast
                        if len(test_data) > 0:
                            test_fx = fit.forecast(steps=len(test_data)).values
                            mae = mean_absolute_error(test_data.values, test_fx)
                            rmse = np.sqrt(mean_squared_error(test_data.values, test_fx))
                            mape = float(np.mean(np.abs((test_data.values - test_fx) / np.maximum(test_data.values, 1e-9))) * 100)
                            results["ARIMA"] = {"MAE": mae, "RMSE": rmse, "MAPE": mape}
                    except Exception as e:
                        st.warning(f"ARIMA failed: {e}")

                # Exponential Smoothing
                if "Exponential Smoothing" in selected_models and len(train_data) >= 14:
                    try:
                        model = ExponentialSmoothing(train_data, trend="add", seasonal="add", seasonal_periods=7)
                        fit = model.fit()
                        forecast = fit.forecast(steps=forecast_days).values
                        forecasts["Exponential Smoothing"] = forecast
                        if len(test_data) > 0:
                            test_fx = fit.forecast(steps=len(test_data)).values
                            mae = mean_absolute_error(test_data.values, test_fx)
                            rmse = np.sqrt(mean_squared_error(test_data.values, test_fx))
                            mape = float(np.mean(np.abs((test_data.values - test_fx) / np.maximum(test_data.values, 1e-9))) * 100)
                            results["Exponential Smoothing"] = {"MAE": mae, "RMSE": rmse, "MAPE": mape}
                    except Exception as e:
                        st.warning(f"Exponential Smoothing failed: {e}")

                # Prophet
                if "Prophet" in selected_models and HAS_PROPHET and len(train_data) >= 7:
                    try:
                        p_df = train_data.reset_index(); p_df.columns = ["ds","y"]
                        m = Prophet(daily_seasonality=False, weekly_seasonality=True, yearly_seasonality=False)
                        m.fit(p_df)
                        future = m.make_future_dataframe(periods=forecast_days)
                        fc = m.predict(future)
                        forecasts["Prophet"] = fc["yhat"].tail(forecast_days).values
                        if len(test_data) > 0:
                            tf = m.make_future_dataframe(periods=len(test_data))
                            tfc = m.predict(tf)
                            test_fx = tfc["yhat"].tail(len(test_data)).values
                            mae = mean_absolute_error(test_data.values, test_fx)
                            rmse = np.sqrt(mean_squared_error(test_data.values, test_fx))
                            mape = float(np.mean(np.abs((test_data.values - test_fx) / np.maximum(test_data.values, 1e-9))) * 100)
                            results["Prophet"] = {"MAE": mae, "RMSE": rmse, "MAPE": mape}
                    except Exception as e:
                        st.warning(f"Prophet failed: {e}")

                st.session_state.model_results["forecasting"] = results
                st.session_state.trained_models["forecasting"] = {"forecasts": forecasts, "train_data": train_data, "test_data": test_data}

                if results:
                    st.success("✅ Forecasting models trained successfully!")
                    # Performance table (sorted by MAPE)
                    st.subheader("📊 Model Performance Comparison (Sorted by MAPE)")
                    res_df = pd.DataFrame(results).T.sort_values("MAPE", ascending=True)
                    st.dataframe(res_df.style.format("{:.2f}"), use_container_width=True)

                    # Forecast plot
                    st.subheader("🔮 Admission Forecasts")
                    future_dates = pd.date_range(start=pd.to_datetime(ts.index[-1]) + pd.Timedelta(days=1), periods=forecast_days, freq="D")
                    figf = go.Figure()
                    figf.add_trace(go.Scatter(x=ts.index, y=ts.values, mode="lines", name="Historical"))
                    palette = ["#34a853","#ea4335","#fbbc04","#9aa0a6"]
                    for i, (name, fc) in enumerate(forecasts.items()):
                        figf.add_trace(go.Scatter(x=future_dates, y=fc, mode="lines+markers", name=f"{name} Forecast", line=dict(dash="dash", width=2, color=palette[i%len(palette)])))
                    figf.update_layout(title="Admission Forecasts by Model", height=500, xaxis_title="Date", yaxis_title="Predicted Admissions")
                    st.plotly_chart(figf, use_container_width=True)

                    # Real-time AI Summary (screenshot style)
                    best_model = min(results.keys(), key=lambda x: results[x]["MAPE"])
                    best_mape = float(results[best_model]["MAPE"])
                    data_summary_rt = {
                        "total_records": len(filtered_df),
                        "forecast_horizon": int(forecast_days),
                        "average_daily_admissions": float(daily_adm["admissions"].mean()) if len(daily_adm) else 0.0,
                        "best_model": best_model, "best_model_mape": best_mape,
                        "custom_present": "Custom" in st.session_state.custom_results.get("forecasting", {})
                    }
                    ai_text = llm_insight("Admissions Forecasting", data_summary_rt, results, "")
                    render_ai_summary(ai_text, "AI Summary")

                    # Download best
                    best_name = best_model
                    st.download_button("📥 Download Best Forecast (CSV)",
                        data=pd.DataFrame({"date": future_dates, "forecast": forecasts[best_name]}).to_csv(index=False),
                        file_name=f"admissions_forecast_{best_name}.csv", mime="text/csv")
                else:
                    st.error("No models produced results. Try a shorter horizon or ensure enough history.")

    # --- Custom forecast section (works reliably) ---
    st.session_state["forecast_days"] = forecast_days
    forecast_days_session = st.session_state.get("forecast_days", 14)

    def _ensure_split(daily_adm_df: pd.DataFrame, horizon: int):
        ts_local = daily_adm_df.set_index("date")["admissions"].astype(float)
        if len(ts_local) < 2: return ts_local, ts_local.iloc[0:0], ts_local
        horizon = int(max(1, min(horizon, max(1, len(ts_local)-1))))
        train_sz = max(1, len(ts_local)-horizon)
        return ts_local.iloc[:train_sz], ts_local.iloc[train_sz:], ts_local

    trained_models = st.session_state.trained_models.get("forecasting", {})
    train_data = trained_models.get("train_data")
    test_data = trained_models.get("test_data")
    ts = daily_adm.set_index("date")["admissions"].astype(float) if train_data is not None else None
    if train_data is None or test_data is None:
        train_data, test_data, ts = _ensure_split(daily_adm, forecast_days_session)

    st.markdown("---")
    with st.expander("💻 Editable Model Implementation Code", expanded=False):
        st.caption("Modify and run the code below to experiment. You may import from numpy/sklearn/statsmodels/prophet. Return either `result = forecast_array` or `result = {'forecast': [...]}`.")
        st.checkbox("I understand the risks and want to run custom code", key="ok_custom_fc")
        default_code = f"""
# Linear Trend Forecast (safe example)
import numpy as np
x = np.arange(len(train_data))
coeffs = np.polyfit(x, train_data.values, 1)
future_x = np.arange(len(train_data), len(train_data) + forecast_days)
forecast = np.polyval(coeffs, future_x)
result = forecast
"""
        user_code = st.text_area(f"Your function: custom_forecast({forecast_days_session} days)",
                                 value=default_code, height=260, key="forecast_code_edit_v3")
        run_custom = st.button("Run Custom Forecast", key="run_custom_forecast_v3", type="primary")
        if run_custom:
            if not st.session_state.ok_custom_fc:
                st.warning("Please confirm the checkbox above to run custom code.")
            else:
                with st.spinner("Executing custom forecast..."):
                    context = {'np': np, 'pd': pd, 'ARIMA': ARIMA, 'ExponentialSmoothing': ExponentialSmoothing,
                               'Prophet': Prophet if HAS_PROPHET else None,
                               'mean_absolute_error': mean_absolute_error, 'mean_squared_error': mean_squared_error,
                               'train_data': train_data, 'test_data': test_data,
                               'forecast_days': forecast_days_session, 'filtered_df': filtered_df, 'daily_adm': daily_adm}
                    success, result, error = run_user_code(user_code, context)
                    if not success:
                        st.error(f"❌ Execution failed:\n{error}")
                    else:
                        fc = norm_forecast_output(result, forecast_days_session)
                        if len(fc) == 0:
                            st.warning("Custom code ran, but no forecast values were returned.")
                        else:
                            metrics = compute_forecast_metrics(np.array(test_data.values if test_data is not None else []), np.array(fc))
                            register_custom_result("forecasting", metrics)
                            future_dates = pd.date_range(start=pd.to_datetime(ts.index[-1]) + pd.Timedelta(days=1), periods=len(fc), freq="D")
                            figc = go.Figure()
                            figc.add_trace(go.Scatter(x=ts.index, y=ts.values, name="Historical", mode="lines"))
                            figc.add_trace(go.Scatter(x=future_dates, y=fc, name="Custom Forecast", mode="lines+markers"))
                            figc.update_layout(title="Custom Forecast", height=420, xaxis_title="Date", yaxis_title="Admissions")
                            st.plotly_chart(figc, use_container_width=True)
                            if metrics:
                                st.success(f"MAE {metrics.get('MAE', float('nan')):.2f} • RMSE {metrics.get('RMSE', float('nan')):.2f} • MAPE {metrics.get('MAPE', float('nan')):.2f}%")
                            # AI summary for custom
                            ai_text_c = llm_insight("Admissions Forecasting — Custom Model",
                                {"total_records": len(filtered_df), "forecast_horizon": int(forecast_days_session),
                                 "average_daily_admissions": float(daily_adm["admissions"].mean()) if len(daily_adm) else 0.0,
                                 "best_model": "Custom", "best_model_mape": float(metrics.get("MAPE", 0.0)) if metrics else 0.0,
                                 "custom_present": True},
                                {"Custom": metrics})
                            render_ai_summary(ai_text_c, "AI Summary (Custom)")
                            st.download_button("📥 Download Custom Forecast (CSV)",
                                data=pd.DataFrame({"date": future_dates, "forecast": fc}).to_csv(index=False),
                                file_name="admissions_forecast_custom.csv", mime="text/csv")

    # AI Assistant for this tab
    st.markdown("---")
    context = f"Admissions forecasting with {len(daily_adm)} days. Models: {', '.join(['Linear Trend','ARIMA','Exponential Smoothing'] + (['Prophet'] if HAS_PROPHET else []))}. Role={st.session_state.USER_ROLE}"
    ai_assistant_section("Admissions Forecasting", context, expanded=False)

# ===================== TAB 2: Revenue Analytics =====================
# ===================== TAB 2: Revenue Analytics =====================
with tabs[1]:
    st.markdown("""<div class="analysis-section"><h3>💰 Revenue Pattern Analysis</h3><p>Detect unusual billing patterns and identify potential revenue optimization opportunities.</p></div>""", unsafe_allow_html=True)
    with st.expander("ℹ️ Model Primer: Revenue Anomaly Detection", expanded=False):
        st.markdown("""- **Isolation Forest**: Unsupervised outlier detection.\n- **Statistical Outliers (z>3)**: Transparent rule.\n- **Ensemble**: Cross-flag both to reduce false positives.""")

    explain_in_30_seconds("Finding odd revenue days/encounters",
        ["Build a feature table (billing, LOS, weekday, etc.).",
         "Run unsupervised detectors to flag unlikely points.",
         "Compare methods by % flagged and plausibility.",
         "Export anomalies for human audit."])

    # ----- Trends + headline metrics
    c1, c2 = st.columns([2, 1])
    with c1:
        st.subheader("💳 Revenue Trends")
        daily_rev = filtered_df.groupby(filtered_df["date_of_admission"].dt.date)["billing_amount"].sum().reset_index()
        daily_rev.columns = ["date","revenue"]
        st.session_state["_daily_rev_for_summary"] = daily_rev
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=daily_rev["date"], y=daily_rev["revenue"], mode="lines+markers", name="Daily Revenue", line=dict(width=2)))
        fig.update_layout(title="Daily Hospital Revenue", xaxis_title="Date", yaxis_title="Revenue ($)", height=400)
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        st.subheader("💰 Revenue Metrics")
        total_revenue = float(filtered_df["billing_amount"].sum()) if "billing_amount" in filtered_df.columns else 0.0
        avg_daily_revenue = float(daily_rev["revenue"].mean()) if len(daily_rev) else 0.0
        avg_bill = float(filtered_df["billing_amount"].mean()) if "billing_amount" in filtered_df.columns else 0.0
        st.markdown(f"""
        <div class="metric-card"><h4>Total Revenue</h4><h2 style="color:#34a853;">${total_revenue:,.0f}</h2></div>
        <div class="metric-card"><h4>Daily Average</h4><h2 style="color:#4285f4;">${avg_daily_revenue:,.0f}</h2></div>
        <div class="metric-card"><h4>Avg Per Patient</h4><h2 style="color:#fbbc04;">${avg_bill:,.0f}</h2></div>
        """, unsafe_allow_html=True)

    # ----- Config row (shared by baseline + custom)
    st.markdown('<div class="config-row"><h4>🔧 Anomaly Detection Settings</h4></div>', unsafe_allow_html=True)
    a1, a2, a3 = st.columns(3)
    with a1:
        detection_methods = st.multiselect("Detection Methods", ["Isolation Forest","Statistical Outliers","Ensemble"], default=["Isolation Forest"])
    with a2:
        sensitivity = st.slider("Sensitivity (contamination)", 0.01, 0.10, 0.05, 0.01)
    with a3:
        features_for_anomaly = st.multiselect("Analysis Features",
            ["billing_amount","age","length_of_stay","admission_month","admission_day_of_week"],
            default=["billing_amount","length_of_stay"])

    # Build X once so both sections can use it
    X_current = _build_anomaly_matrix(filtered_df, features_for_anomaly)

    # ===== A) ALWAYS-ON CUSTOM MODEL EXPANDER (appears even before baseline) =====
    st.markdown("---")
    with st.expander(expander_title("Run Custom Revenue Anomaly Code (unsafe)"), expanded=False):
        st.caption("Return `{'n_anomalies': int, 'anomaly_indices': [...]} ` OR set `result` to a list of anomaly row indices. You may import numpy/sklearn. Uses your selections above for features and sensitivity.")
        st.checkbox("I understand the risks and want to run custom code", key="ok_custom_rev_always")

        default_custom_rev = f"""
# Isolation Forest custom example (uses the X provided in context)
from sklearn.ensemble import IsolationForest
if X.shape[0] == 0 or X.shape[1] == 0:
    # Nothing to analyze
    result = {{'n_anomalies': 0, 'anomaly_indices': []}}
else:
    iso = IsolationForest(contamination={float(sensitivity)}, random_state=42)
    pred = iso.fit_predict(X)  # X comes from context
    anomalies = np.where(pred == -1)[0]
    result = {{'n_anomalies': int(len(anomalies)), 'anomaly_indices': anomalies.tolist()}}
"""
        user_code_rev_always = st.text_area("Your function: custom_revenue_anomaly(X)", value=default_custom_rev, height=260, key="revenue_code_edit_always")
        run_code_rev_always = st.button("Run Custom Revenue Code", key="run_revenue_code_always", type="primary")

        if run_code_rev_always:
            if not st.session_state.ok_custom_rev_always:
                st.warning("Please confirm the checkbox above to run custom code.")
            elif X_current.shape[0] == 0 or X_current.shape[1] == 0:
                st.warning("Select at least one feature and ensure the data has rows after dropping NAs.")
            else:
                with st.spinner("Executing custom anomaly code..."):
                    # Preserve row index to map back to filtered_df
                    X = X_current.copy()
                    context = {'feature_data': filtered_df, 'filtered_df': filtered_df, 'X': X, 'np': np, 'pd': pd}
                    success, result, error = run_user_code(user_code_rev_always, context)
                    if not success:
                        st.error(f"❌ Execution failed:\n{error}")
                    else:
                        norm = norm_anomaly_output(result)
                        idx_local = norm.pop("_indices", [])
                        if len(idx_local) == 0:
                            st.info("No anomalies returned by custom code.")
                        else:
                            # idx_local are *positional* indices relative to X
                            X_indices = X.index.to_numpy()
                            selected_idx = X_indices[np.array(idx_local, dtype=int)]
                            n = len(selected_idx)
                            rate = n / max(1, len(X))
                            avg_amt = float(filtered_df.loc[selected_idx, "billing_amount"].mean()) if "billing_amount" in filtered_df.columns else 0.0
                            norm.update({"anomalies_detected": n, "anomaly_rate": rate, "avg_anomaly_amount": avg_amt})
                            register_custom_result("anomaly", norm)

                            st.success(f"Custom anomalies detected: {n} ({rate:.2%}) • Avg anomaly ${avg_amt:,.0f}")

                            # Quick scatter if at least 2 features
                            if len(features_for_anomaly) >= 2:
                                flag = np.zeros(len(X), dtype=int)
                                # Map selected positional ints back to 0..len(X)-1 for coloring
                                flag[np.array(idx_local, dtype=int)] = 1
                                fig_custom = px.scatter(x=X[features_for_anomaly[0]], y=X[features_for_anomaly[1]],
                                                        color=flag, title="Revenue Pattern Analysis — Custom",
                                                        labels={"color":"Anomaly (1=yes)"})
                                fig_custom.update_layout(height=480)
                                st.plotly_chart(fig_custom, use_container_width=True)

                            # AI Summary (Custom) – independent of baseline
                            ai_custom_rev = llm_insight(
                                "Revenue Pattern Analysis — Custom Model",
                                {
                                    "best_method":"Custom",
                                    "anomalies_detected": n,
                                    "anomaly_rate": rate,
                                    "avg_anomaly_amount": avg_amt,
                                    "total_revenue": total_revenue,
                                    "avg_daily_revenue": avg_daily_revenue
                                },
                                {"Custom": norm},
                                ""
                            )
                            render_ai_summary(ai_custom_rev, "AI Summary (Custom)")

    # ===== B) Baseline detectors (optional to run) =====
    if st.button("Analyze Revenue Patterns", key="detect_anomalies", type="primary"):
        if features_for_anomaly and detection_methods:
            with st.spinner("Analyzing revenue patterns..."):
                X = X_current
                results, anomaly_predictions = {}, {}

                for method in detection_methods:
                    try:
                        if method == "Isolation Forest":
                            iso = IsolationForest(contamination=float(sensitivity), random_state=42)
                            pred = iso.fit_predict(X) if X.shape[0] and X.shape[1] else np.array([])
                            anomaly_predictions["Isolation Forest"] = pred
                            n = int((pred == -1).sum()) if pred.size else 0
                            rate = n/len(X) if len(X) else 0
                            results["Isolation Forest"] = {"anomalies_detected": n, "anomaly_rate": rate,
                                "avg_anomaly_amount": float(filtered_df.loc[X.index[pred==-1],"billing_amount"].mean()) if n>0 and "billing_amount" in filtered_df.columns else 0.0}

                        elif method == "Statistical Outliers":
                            if X.shape[0] and X.shape[1]:
                                z = np.abs((X - X.mean())/X.std(ddof=0)).fillna(0.0)
                                stat_mask = (z>3).any(axis=1)
                                n = int(stat_mask.sum()); rate = n/len(X) if len(X) else 0
                                results["Statistical Outliers"] = {"anomalies_detected": n, "anomaly_rate": rate,
                                    "avg_anomaly_amount": float(filtered_df.loc[X.index[stat_mask],"billing_amount"].mean()) if n>0 and "billing_amount" in filtered_df.columns else 0.0}
                                anomaly_predictions["Statistical Outliers"] = np.where(stat_mask.values,-1,1)
                            else:
                                results["Statistical Outliers"] = {"anomalies_detected": 0, "anomaly_rate": 0.0, "avg_anomaly_amount": 0.0}
                                anomaly_predictions["Statistical Outliers"] = np.array([])

                        elif method == "Ensemble":
                            if X.shape[0] and X.shape[1]:
                                iso = IsolationForest(contamination=float(sensitivity), random_state=42)
                                pred_iso = iso.fit_predict(X)
                                z = np.abs((X - X.mean())/X.std(ddof=0)).fillna(0.0)
                                stat_mask = (z>3).any(axis=1)
                                pred_stat = np.where(stat_mask.values,-1,1)
                                pred_ens = np.where((pred_iso==-1) & (pred_stat==-1), -1, 1)
                                anomaly_predictions["Ensemble"] = pred_ens
                                n = int((pred_ens==-1).sum()); rate = n/len(X) if len(X) else 0
                                results["Ensemble"] = {"anomalies_detected": n, "anomaly_rate": rate,
                                    "avg_anomaly_amount": float(filtered_df.loc[X.index[pred_ens==-1],"billing_amount"].mean()) if n>0 and "billing_amount" in filtered_df.columns else 0.0}
                            else:
                                results["Ensemble"] = {"anomalies_detected": 0, "anomaly_rate": 0.0, "avg_anomaly_amount": 0.0}
                                anomaly_predictions["Ensemble"] = np.array([])
                    except Exception as e:
                        st.warning(f"{method} failed: {e}")

                st.session_state.model_results["anomaly"] = results
                st.session_state.trained_models["anomaly"] = {"predictions": anomaly_predictions, "X": X}

                if results:
                    st.success("✅ Revenue analysis completed!")
                    res_df = pd.DataFrame(results).T
                    res_df["distance_from_target"] = np.abs(res_df["anomaly_rate"] - 0.05)
                    res_df = res_df.sort_values("distance_from_target").drop("distance_from_target", axis=1)
                    st.dataframe(res_df.style.format({"anomalies_detected":"{:.0f}","anomaly_rate":"{:.2%}","avg_anomaly_amount":"${:,.0f}"}), use_container_width=True)

                    # Immediate AI Summary (baseline)
                    best_method = min(results.keys(), key=lambda x: abs(results[x]["anomaly_rate"] - 0.05))
                    det = results[best_method]
                    data_summary_rev = {
                        "total_revenue": total_revenue,
                        "avg_daily_revenue": avg_daily_revenue,
                        "best_method": best_method,
                        "anomalies_detected": int(det.get("anomalies_detected",0)),
                        "anomaly_rate": float(det.get("anomaly_rate",0.0)),
                        "custom_present": "Custom" in st.session_state.custom_results.get("anomaly",{})
                    }
                    render_ai_summary(llm_insight("Revenue Pattern Analysis", data_summary_rev, results, ""), "AI Summary")
        else:
            st.warning("Please select at least one feature and one method.")

    # AI Assistant
    st.markdown("---")
    context = f"Revenue analytics: ${float(filtered_df['billing_amount'].sum()) if 'billing_amount' in filtered_df.columns else 0:,.0f} total across {len(filtered_df)} records. Role={st.session_state.USER_ROLE}"
    ai_assistant_section("Revenue Analytics", context,  expanded=False)

# ===================== TAB 3: Length of Stay Prediction =====================
with tabs[2]:
    st.markdown("""<div class="analysis-section"><h3>🛏️ Length of Stay Prediction</h3><p>Predict stay duration to optimize bed management and discharge planning.</p></div>""", unsafe_allow_html=True)
    st.markdown("""
    ### 📋 How Length of Stay (LOS) is Calculated
    **LOS in Days:** `Discharge Date - Admission Date` (days)
    **Categories:** Short (0–3), Medium (4–7), Long (8–14), Extended (15+)
    """)

    with st.expander("ℹ️ Model Primer: LOS", expanded=False):
        st.markdown("""- **Classification:** Random Forest, Logistic Regression, SVM, XGBoost (if installed).\n- **Regression:** Random Forest Regressor, Linear Regression, SVR, XGBoost Regressor (if installed).""")

    explain_in_30_seconds("Predicting length of stay",
        ["Create labels: days or categories.",
         "Train pipelines with scaling + encoding.",
         "Compare by Accuracy (cls) or R² (reg).",
         "Tune discharge pathways from top errors."])

    if "length_of_stay" not in filtered_df.columns:
        st.error("Length of stay data not available.")
    else:
        c1, c2 = st.columns([2, 1])
        with c1:
            st.subheader("📊 LOS Distribution")
            fig = px.histogram(filtered_df, x="length_of_stay", nbins=30, title="Distribution of Length of Stay")
            fig.update_layout(height=400, xaxis_title="Length of Stay (days)", yaxis_title="Patients"); st.plotly_chart(fig, use_container_width=True)
        with c2:
            st.subheader("🏥 LOS Metrics")
            avg_los = float(filtered_df["length_of_stay"].mean()); median_los = float(filtered_df["length_of_stay"].median()); max_los = float(filtered_df["length_of_stay"].max())
            st.session_state["los_avg_los"] = avg_los
            st.markdown(f"""
            <div class="metric-card"><h4>Average LOS</h4><h2 style="color:#4285f4;">{avg_los:.1f} days</h2></div>
            <div class="metric-card"><h4>Median LOS</h4><h2 style="color:#34a853;">{median_los:.1f} days</h2></div>
            <div class="metric-card"><h4>Maximum LOS</h4><h2 style="color:#ea4335;">{max_los:.0f} days</h2></div>
            """, unsafe_allow_html=True)

        st.markdown('<div class="config-row"><h4>🔧 Prediction Model Setup</h4></div>', unsafe_allow_html=True)
        available_features = [c for c in filtered_df.columns if c not in ["length_of_stay","los_category","date_of_admission","discharge_date"]]
        s1, s2 = st.columns(2)
        with s1:
            selected_features = st.multiselect("Select Prediction Features", available_features,
                                               default=[f for f in ["age","medical_condition","admission_type","hospital"] if f in available_features])
        with s2:
            target_type = st.selectbox("Prediction Target", ["Length of Stay (Days)", "LOS Category (Short/Medium/Long)"])
            selected_models = st.multiselect("Select Models", ["Random Forest","Logistic Regression","XGBoost","SVM","Linear Regression"], default=["Random Forest","Logistic Regression"])

        if st.button("Train Prediction Models", key="train_los", type="primary"):
            if selected_features:
                with st.spinner("Training LOS prediction models..."):
                    feature_data = filtered_df[selected_features + ["length_of_stay","los_category"]].dropna()
                    if target_type == "Length of Stay (Days)":
                        target = "length_of_stay"; is_classification = False
                    else:
                        target = "los_category"; is_classification = True
                    st.session_state["los_is_classification"] = is_classification
                    st.session_state["los_target_type"] = target_type

                    X = feature_data[selected_features]; y = feature_data[target]
                    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
                    categorical_features = X.select_dtypes(include=["object","category"]).columns.tolist()
                    preprocessor = ColumnTransformer(
                        transformers=[("num", StandardScaler(), numeric_features), ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), categorical_features)]
                    )

                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                    results, models_trained = {}, {}

                    for model_name in selected_models:
                        try:
                            if is_classification:
                                if model_name == "Random Forest": model = RandomForestClassifier(n_estimators=200, random_state=42)
                                elif model_name == "Logistic Regression": model = LogisticRegression(max_iter=1000, random_state=42)
                                elif model_name == "XGBoost" and HAS_XGBOOST: model = xgb.XGBClassifier(random_state=42)
                                elif model_name == "SVM": model = SVC(probability=True, random_state=42)
                                else: continue
                                step = "classifier"
                            else:
                                if model_name == "Random Forest": model = RandomForestRegressor(n_estimators=300, random_state=42)
                                elif model_name == "Linear Regression": model = LinearRegression()
                                elif model_name == "XGBoost" and HAS_XGBOOST: model = xgb.XGBRegressor(random_state=42)
                                elif model_name == "SVM": model = SVR()
                                else: continue
                                step = "regressor"

                            pipe = Pipeline([("preprocessor", preprocessor), (step, model)])
                            pipe.fit(X_train, y_train)
                            models_trained[model_name] = pipe
                            y_pred = pipe.predict(X_test)

                            if is_classification:
                                acc = accuracy_score(y_test, y_pred)
                                precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="weighted", zero_division=0)
                                results[model_name] = {"accuracy": float(acc), "precision": float(precision), "recall": float(recall), "f1_score": float(f1)}
                            else:
                                mae = mean_absolute_error(y_test, y_pred); rmse = float(np.sqrt(mean_squared_error(y_test, y_pred))); r2 = r2_score(y_test, y_pred)
                                results[model_name] = {"mae": float(mae), "rmse": rmse, "r2_score": float(r2)}
                        except Exception as e:
                            st.warning(f"Failed to train {model_name}: {e}")

                    if results:
                        st.session_state.model_results["los"] = results
                        st.session_state.trained_models["los"] = {"models": models_trained, "X_test": X_test, "y_test": y_test, "is_classification": is_classification}
                        st.success("✅ LOS prediction models trained successfully!")

                        res_df = pd.DataFrame(results).T
                        metric_to_plot = "accuracy" if is_classification else "r2_score"
                        res_df = res_df.sort_values(metric_to_plot, ascending=not is_classification)
                        st.dataframe(res_df.style.format("{:.3f}"), use_container_width=True)

                        figm = go.Figure(data=[go.Bar(x=list(results.keys()),
                                                      y=[results[m].get(metric_to_plot, 0) for m in results.keys()],
                                                      text=[f"{results[m].get(metric_to_plot, 0):.3f}" for m in results.keys()], textposition="auto")])
                        figm.update_layout(title=f"Model Comparison — {metric_to_plot.replace('_',' ').title()}", height=400)
                        st.plotly_chart(figm, use_container_width=True)

                        # Immediate AI Summary
                        best_model_name = max(results.keys(), key=lambda x: results[x].get("accuracy", 0)) if is_classification else max(results.keys(), key=lambda x: results[x].get("r2_score", -np.inf))
                        perf = results[best_model_name]["accuracy"] if is_classification else results[best_model_name]["r2_score"]
                        data_summary_los = {"prediction_type": target_type, "best_model": best_model_name, "performance_metric": float(perf),
                                            "avg_los": float(st.session_state.get("los_avg_los") or 0.0), "total_patients": len(filtered_df)}
                        render_ai_summary(llm_insight("Length of Stay Prediction", data_summary_los, results, ""), "AI Summary")

                        # ---- Custom LOS Code (persistent & runnable) ----
                        st.markdown("---")
                        with st.expander(expander_title("Run Custom LOS Code (unsafe)"), expanded=False):
                            st.caption("Classification: return {'accuracy': float, ...}. Regression: return {'mae': float, 'r2': float, ...}. You may import numpy/sklearn.")
                            st.checkbox("I understand the risks and want to run custom code", key="ok_custom_los")

                            if is_classification:
                                generated_code = f"""
# Random Forest Classifier for LOS Category (custom)
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
X = feature_data[{selected_features}]
y = feature_data['los_category']
num = X.select_dtypes(include=[np.number]).columns.tolist()
cat = X.select_dtypes(include=['object','category']).columns.tolist()
pre = ColumnTransformer([('num', StandardScaler(), num), ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), cat)])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=250, random_state=42)
pipe = Pipeline([('pre', pre), ('clf', model)])
pipe.fit(X_train, y_train)
y_pred = pipe.predict(X_test)
result = {{'accuracy': float(accuracy_score(y_test, y_pred))}}
"""
                            else:
                                generated_code = f"""
# Random Forest Regressor for LOS Days (custom)
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, r2_score
X = feature_data[{selected_features}]
y = feature_data['length_of_stay']
num = X.select_dtypes(include=[np.number]).columns.tolist()
cat = X.select_dtypes(include=['object','category']).columns.tolist()
pre = ColumnTransformer([('num', StandardScaler(), num), ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), cat)])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestRegressor(n_estimators=320, random_state=42)
pipe = Pipeline([('pre', pre), ('reg', model)])
pipe.fit(X_train, y_train)
y_pred = pipe.predict(X_test)
result = {{'mae': float(mean_absolute_error(y_test, y_pred)), 'r2': float(r2_score(y_test, y_pred))}}
"""
                            user_code_los = st.text_area("Your function: custom_los_model(feature_data)", value=generated_code, height=340, key="los_code_edit_v3")
                            run_code_los = st.button("Run Custom LOS Code", key="run_los_code_v3", type="primary")
                            if run_code_los:
                                if not st.session_state.ok_custom_los:
                                    st.warning("Please confirm the checkbox above to run custom code.")
                                else:
                                    with st.spinner("Executing custom LOS code..."):
                                        context = {'feature_data': feature_data, 'filtered_df': filtered_df, 'np': np, 'pd': pd}
                                        success, result, error = run_user_code(user_code_los, context)
                                        if not success:
                                            st.error(f"❌ Execution failed:\n{error}")
                                        else:
                                            norm = norm_los_output(result, is_classification)
                                            if not norm:
                                                st.warning("Custom code ran, but expected metrics not found.")
                                            else:
                                                register_custom_result("los", norm)
                                                if is_classification:
                                                    st.success(f"Custom accuracy: {norm.get('accuracy', 0.0):.3f}")
                                                else:
                                                    st.success(f"Custom R²: {norm.get('r2_score', norm.get('r2', 0.0)):.3f} • MAE: {norm.get('mae', 0.0):.2f}")
                                                ai_custom_los = llm_insight("Length of Stay Prediction — Custom Model",
                                                    {"prediction_type": target_type, "best_model": "Custom",
                                                     "performance_metric": norm.get("accuracy", norm.get("r2_score", 0.0)),
                                                     "avg_los": float(st.session_state.get("los_avg_los") or 0.0),
                                                     "total_patients": len(filtered_df)},
                                                    {"Custom": norm}, "")
                                                render_ai_summary(ai_custom_los, "AI Summary (Custom)")
                    else:
                        st.error("All models failed to train. Check features & data quality.")
            else:
                st.warning("Please select at least one feature for model training.")

    # AI Assistant
    st.markdown("---")
    context = f"LOS prediction with avg LOS {float(filtered_df['length_of_stay'].mean()) if 'length_of_stay' in filtered_df.columns else 0:.1f} over {len(filtered_df)} patients. Role={st.session_state.USER_ROLE}"
    ai_assistant_section("Length of Stay", context, expanded=False)

# ===================== TAB 4: Operational KPIs & Simulator =====================
with tabs[3]:
    st.markdown("""<div class="analysis-section"><h3>📊 Operational KPIs & What-If Staffing</h3><p>Concise leadership metrics + a quick scenario sandbox.</p></div>""", unsafe_allow_html=True)
    with st.expander("ℹ️ Model Primer: Simulator", expanded=False):
        st.markdown("""- Convert admissions targets into day/night nurse counts under configurable ratios.\n- Stress-test with a surge % and inspect daily budget impact.""")

    explain_in_30_seconds("What-if staffing calculator",
        ["Estimate near-term admissions from recent averages ± surge.",
         "Divide load by patient-per-nurse ratios (day/night).",
         "Compute staffing counts and daily cost.",
         "Tweak ratios and surge to plan budgets."])

    k1, k2, k3, k4 = st.columns(4)
    with k1: st.metric("Encounters (filtered)", f"{len(filtered_df):,}")
    with k2:
        if "billing_amount" in filtered_df.columns: st.metric("Revenue (filtered)", f"${float(filtered_df['billing_amount'].sum()):,.0f}")
    with k3:
        if "length_of_stay" in filtered_df.columns: st.metric("Avg LOS", f"{filtered_df['length_of_stay'].mean():.1f} days")
    with k4:
        if "medical_condition" in filtered_df.columns and len(filtered_df): st.metric("Top Condition", str(filtered_df["medical_condition"].value_counts().idxmax()))

    if {"insurance_provider","billing_amount"}.issubset(filtered_df.columns) and len(filtered_df):
        payer_rev = filtered_df.groupby("insurance_provider")["billing_amount"].sum().sort_values(ascending=False).reset_index()
        figp = px.bar(payer_rev, x="insurance_provider", y="billing_amount", title="Revenue by Payer (Pareto)")
        figp.update_layout(height=380, xaxis_title="Payer", yaxis_title="Revenue ($)"); st.plotly_chart(figp, use_container_width=True)

    st.markdown('<div class="config-row"><h4>🧪 What-If: Staffing vs. Admissions</h4></div>', unsafe_allow_html=True)
    w1, w2, w3 = st.columns(3)
    with w1: pts_per_day_nurse = st.number_input("Patients per Day Nurse", 4, 12, 8)
    with w2: pts_per_night_nurse = st.number_input("Patients per Night Nurse", 6, 18, 12)
    with w3: surge_pct = st.slider("Potential Surge (%)", 0, 50, 15, 5)

    daily_adm_local = filtered_df.groupby(filtered_df["date_of_admission"].dt.date).size().reset_index()
    daily_adm_local.columns = ["date","admissions"]
    recent_mean = float(daily_adm_local["admissions"].tail(28).mean()) if len(daily_adm_local) else 0.0
    target_load = recent_mean * (1 + surge_pct/100.0)

    d1, d2, d3 = st.columns(3)
    with d1:
        dn = int(np.ceil(target_load / max(1, pts_per_day_nurse))); st.metric("Needed Day Nurses", f"{dn}")
    with d2:
        nn = int(np.ceil(target_load / max(1, pts_per_night_nurse))); st.metric("Needed Night Nurses", f"{nn}")
    with d3:
        est_cost = dn*350 + nn*400; st.metric("Est. Daily Cost", f"${est_cost:,.0f}")

    st.markdown("---")
    with st.expander(expander_title("Run Custom Simulator Code (unsafe)"), expanded=False):
        st.caption("Return `result` as a dict of KPIs (key → value). You may import numpy.")
        st.checkbox("I understand the risks and want to run custom code", key="ok_custom_sim")
        sim_code = f"""
# What-If Staffing Calculator (custom)
import numpy as np
recent_mean = {recent_mean:.2f}
surge = {surge_pct}/100.0
ppd = {pts_per_day_nurse}
ppn = {pts_per_night_nurse}
target_load = recent_mean * (1 + surge)
day_nurses = int(np.ceil(target_load / max(1, ppd)))
night_nurses = int(np.ceil(target_load / max(1, ppn)))
daily_cost = day_nurses*350 + night_nurses*400
result = {{
    "Target Admissions": round(target_load, 1),
    "Day Nurses": day_nurses,
    "Night Nurses": night_nurses,
    "Daily Cost ($)": int(daily_cost)
}}
"""
        user_sim_code = st.text_area("Edit simulator code:", value=sim_code, height=260, key="sim_code_edit_v3")
        run_sim = st.button("Run Custom Simulator", key="run_sim_code_v3", type="primary")
        if run_sim:
            if not st.session_state.ok_custom_sim:
                st.warning("Please confirm the checkbox above to run custom code.")
            else:
                with st.spinner("Executing simulator..."):
                    context = {'filtered_df': filtered_df, 'daily_adm_local': daily_adm_local, 'recent_mean': recent_mean, 'np': np, 'pd': pd}
                    success, result, error = run_user_code(user_sim_code, context)
                    if not success:
                        st.error(f"❌ Execution failed:\n{error}")
                    else:
                        if isinstance(result, dict) and result:
                            register_custom_result("sim", {"kpis": result})
                            cols = st.columns(len(result))
                            for i, (k, v) in enumerate(result.items()):
                                with cols[i]: st.metric(k, str(v))
                            insights_custom = llm_insight("Operational Simulator — Custom Model",
                                {"kpis": result, "surge_pct": surge_pct, "patients_per_day_nurse": pts_per_day_nurse, "patients_per_night_nurse": pts_per_night_nurse},
                                {"Custom": result}, "")
                            render_ai_summary(insights_custom, "AI Summary (Custom)")
                        else:
                            st.warning("Custom code ran, but did not return a dict of KPI results.")

    # AI Assistant
    st.markdown("---")
    context = f"Operational KPIs + staffing what-if. Recent mean admissions={recent_mean:.1f}, ratios D/N={pts_per_day_nurse}/{pts_per_night_nurse}. Role={st.session_state.USER_ROLE}"
    ai_assistant_section("Operational KPIs", context,  expanded=False

# ===================== Decision Log =====================
st.markdown("---")
st.markdown("## 📋 Decision Tracking")
with st.expander("📝 Add New Decision", expanded=False):
    d1, d2 = st.columns(2)
    with d1:
        decision_section = st.selectbox("Analysis Section", ["Admissions Forecasting","Revenue Analytics","Length of Stay Prediction","Operational KPIs & Simulator"])
        decision_action = st.selectbox("Decision", ["Approve for Production","Needs Review","Requires Additional Data","Reject"])
    with d2:
        decision_owner = st.text_input("Responsible Person")
        decision_date = st.date_input("Target Date")
    decision_notes = st.text_area("Notes and Comments")
    if st.button("Add Decision", type="secondary"):
        new_decision = {"timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "section": decision_section, "action": decision_action, "owner": decision_owner,
                        "target_date": decision_date.strftime("%Y-%m-%d"), "notes": decision_notes}
        st.session_state.decision_log.append(new_decision)
        st.success("✅ Decision added to tracking log!")

if st.session_state.decision_log:
    st.subheader("Decision History")
    decisions_df = pd.DataFrame(st.session_state.decision_log)
    c1, c2, c3 = st.columns([2,1,1])
    with c1: st.dataframe(decisions_df, use_container_width=True)
    with c2:
        if "action" in decisions_df.columns:
            approved = int((decisions_df["action"] == "Approve for Production").sum())
            pending = int((decisions_df["action"] == "Needs Review").sum())
            st.metric("Approved", approved); st.metric("Pending Review", pending)
    with c3:
        csv = decisions_df.to_csv(index=False)
        st.download_button("📥 Download Log", data=csv, file_name=f"decisions_{datetime.now().strftime('%Y%m%d')}.csv", mime="text/csv", type="secondary")
