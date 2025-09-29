# app.py — Hospital Operations Analytics Platform (Interactive, Role-Aware, ROC, Editable Implications)
# ✅ Executive/Analyst role toggle changes tone + content
# ✅ ChatGPT model picker (3.5 / 4.0) actually used for insights (no hard-coding)
# ✅ Editable “Implications” code blocks (safe exec sandbox)
# ✅ ROC curve + AUC for LOS classification
# ✅ Per-tab “Model Primer” explaining model choices
# ✅ Portable: Prophet/XGBoost optional; OpenAI optional with graceful fallback

import os
import io
import json
import textwrap
import warnings
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
    roc_auc_score, roc_curve, confusion_matrix, classification_report
)
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA

warnings.filterwarnings("ignore")

# ---------- Optional libraries with graceful fallback ----------
try:
    from prophet import Prophet  # noqa: F401
    HAS_PROPHET = True
except Exception:
    HAS_PROPHET = False

try:
    import xgboost as xgb  # noqa: F401
    HAS_XGBOOST = True
except Exception:
    HAS_XGBOOST = False

# ---------- OpenAI (optional) – supports classic and new SDKs ----------
OPENAI_AVAILABLE = False
OPENAI_CLIENT = None
OPENAI_MODEL_MAP = {
    "GPT-3.5": "gpt-3.5-turbo",
    "GPT-4.0": "gpt-4"
}
try:
    # New SDK
    from openai import OpenAI  # type: ignore
    _api_key = None
    try:
        _api_key = st.secrets.get("OPENAI_API_KEY")  # type: ignore[attr-defined]
    except Exception:
        _api_key = os.getenv("OPENAI_API_KEY")
    if _api_key:
        OPENAI_CLIENT = OpenAI(api_key=_api_key)
        OPENAI_AVAILABLE = True
except Exception:
    try:
        # Legacy SDK
        import openai  # type: ignore
        _api_key = None
        try:
            _api_key = st.secrets.get("OPENAI_API_KEY")  # type: ignore[attr-defined]
        except Exception:
            _api_key = os.getenv("OPENAI_API_KEY")
        if _api_key:
            openai.api_key = _api_key
            OPENAI_CLIENT = openai
            OPENAI_AVAILABLE = True
    except Exception:
        OPENAI_AVAILABLE = False
        OPENAI_CLIENT = None

# ---------- Streamlit page config ----------
st.set_page_config(
    page_title="Hospital Operations Analytics",
    layout="wide",
    page_icon="🏥",
    initial_sidebar_state="collapsed",
)

# ---------- CSS ----------
st.markdown(
    """
<style>
    .main-header {
        background: linear-gradient(135deg, #4285f4 0%, #34a853 100%);
        padding: 1.5rem;
        border-radius: 12px;
        color: white;
        margin-bottom: 2rem;
        text-align: center;
    }
    .data-overview {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        margin-bottom: 1.5rem;
        border: 1px solid #e9ecef;
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border: 1px solid #e9ecef;
        margin: 0.5rem 0;
        text-align: center;
    }
    .analysis-section {
        background: #fff;
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid #e9ecef;
        margin: 1rem 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    }
    .insights-panel {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        white-space: pre-wrap;
    }
    .config-row {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
        border-left: 4px solid #4285f4;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        justify-content: center;
    }
    .stTabs [data-baseweb="tab"] {
        height: 60px;
        padding: 0 2rem;
        background-color: #f8f9fa;
        border-radius: 12px 12px 0 0;
        font-weight: 500;
    }
    .stTabs [aria-selected="true"] {
        background-color: #4285f4;
        color: white;
    }
    code, pre { white-space: pre-wrap; }
</style>
""",
    unsafe_allow_html=True,
)

# ---------- Session state ----------
st.session_state.setdefault("model_results", {})
st.session_state.setdefault("selected_features", {})
st.session_state.setdefault("decision_log", [])
st.session_state.setdefault("USER_ROLE", "Executive")
st.session_state.setdefault("los_is_classification", None)
st.session_state.setdefault("los_target_type", None)
st.session_state.setdefault("los_avg_los", None)

# NEW: LLM insight settings (defaults)
st.session_state.setdefault("INSIGHTS_USE_LLM", False)
st.session_state.setdefault("INSIGHTS_MODEL", "GPT-4.0")
st.session_state.setdefault("INSIGHTS_TEMP", 0.2)

# ---------- Header ----------
st.markdown(
    """
<div class="main-header">
    <h1>🏥 Hospital Operations Analytics</h1>
    <p>Data-driven insights for better patient outcomes and operational efficiency</p>
</div>
""",
    unsafe_allow_html=True,
)

# ---------- GLOBAL CONTROLS: Role + LLM ----------
c_role, c_llm, c_temp = st.columns([1, 2, 1])
with c_role:
    st.session_state.USER_ROLE = st.radio(
        "Audience",
        options=["Executive", "Analyst"],
        horizontal=True,
        help="Changes the tone, depth, and KPIs emphasized across generated insights."
    )
with c_llm:
    st.session_state.INSIGHTS_USE_LLM = st.checkbox(
        "Use ChatGPT for section insights",
        value=st.session_state.INSIGHTS_USE_LLM,
        help="If disabled or unavailable, we use a built-in, data-driven summary."
    )
    st.session_state.INSIGHTS_MODEL = st.selectbox(
        "ChatGPT Model",
        options=["GPT-3.5", "GPT-4.0"],
        index=1 if st.session_state.INSIGHTS_MODEL == "GPT-4.0" else 0,
        help="Choice actually used at generation time (no hard-coding)."
    )
with c_temp:
    st.session_state.INSIGHTS_TEMP = st.slider(
        "Creativity",
        min_value=0.0, max_value=1.0, value=float(st.session_state.INSIGHTS_TEMP), step=0.1
    )
if st.session_state.INSIGHTS_USE_LLM and not OPENAI_AVAILABLE:
    st.warning("OpenAI API key not found. Set `st.secrets['OPENAI_API_KEY']` or `OPENAI_API_KEY`. Falling back to local insights.")

# ---------- Data loading ----------
@st.cache_data(ttl=3600, show_spinner="Loading hospital data...")
def load_hospital_data() -> pd.DataFrame:
    local_path = "/mnt/data/modified_healthcare_dataset.csv"
    df = None
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
                "Medical Condition": np.random.choice(
                    ["Diabetes", "Hypertension", "Heart Disease", "Cancer", "Asthma"], n_records
                ),
                "Admission Type": np.random.choice(["Emergency", "Elective", "Urgent"], n_records, p=[0.4, 0.4, 0.2]),
                "Hospital": np.random.choice(["General Hospital", "Children Hospital", "University Hospital"], n_records),
                "Insurance Provider": np.random.choice(["Medicare", "Medicaid", "Blue Cross", "Aetna", "UnitedHealth"], n_records),
                "Billing Amount": np.random.lognormal(8, 1, n_records),
                "Doctor": [f"Dr. {name}" for name in np.random.choice(["Smith", "Johnson", "Williams", "Brown", "Jones"], n_records)],
                "Test Results": np.random.choice(["Normal", "Abnormal", "Inconclusive"], n_records, p=[0.6, 0.3, 0.1]),
            }
        )

    # normalize & engineer
    df.columns = df.columns.str.replace(" ", "_").str.lower()
    for col in ["date_of_admission", "discharge_date"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    if {"date_of_admission", "discharge_date"}.issubset(df.columns):
        los = (df["discharge_date"] - df["date_of_admission"]).dt.days
        # clip negatives to 0, but keep 0 as 0; replace zeros with 1 to avoid weirdness only if desired for ops
        los = np.clip(los.fillna(0), 0, 365).astype(int)
        df["length_of_stay"] = los

    if "date_of_admission" in df.columns:
        df["admission_month"] = df["date_of_admission"].dt.month
        df["admission_day_of_week"] = df["date_of_admission"].dt.dayofweek
        df["admission_quarter"] = df["date_of_admission"].dt.quarter

    if "billing_amount" in df.columns:
        df["billing_amount"] = pd.to_numeric(df["billing_amount"], errors="coerce").fillna(0).clip(0)

    if "age" in df.columns:
        df["age"] = pd.to_numeric(df["age"], errors="coerce").fillna(df["age"].median())
        df["age_group"] = pd.cut(
            df["age"], bins=[0, 18, 35, 50, 65, 100],
            labels=["Child", "Young Adult", "Adult", "Senior", "Elderly"], include_lowest=True,
        )

    if "length_of_stay" in df.columns:
        df["los_category"] = pd.cut(
            df["length_of_stay"], bins=[-0.1, 3, 7, 14, float("inf")],
            labels=["Short", "Medium", "Long", "Extended"], include_lowest=True,
        )

    return df.dropna(subset=["date_of_admission"])


df = load_hospital_data()

# ---------- Dataset Overview ----------
st.markdown(
    """
<div class="data-overview">
    <h3>📊 Dataset Overview</h3>
</div>
""",
    unsafe_allow_html=True,
)

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.metric("Total Records", f"{len(df):,}")
with c2:
    days = (df["date_of_admission"].max() - df["date_of_admission"].min()).days
    st.metric("Date Range", f"{days} days")
with c3:
    st.metric("Hospitals", f"{df['hospital'].nunique()}" if "hospital" in df.columns else "N/A")
with c4:
    st.metric("Conditions", f"{df['medical_condition'].nunique()}" if "medical_condition" in df.columns else "N/A")

# ---------- Global Filters ----------
with st.expander("🔍 Filter Data", expanded=False):
    f1, f2, f3 = st.columns(3)
    with f1:
        hospitals = st.multiselect(
            "Select Hospitals", df["hospital"].dropna().unique() if "hospital" in df.columns else []
        )
    with f2:
        conditions = st.multiselect(
            "Medical Conditions",
            df["medical_condition"].dropna().unique() if "medical_condition" in df.columns else [],
        )
    with f3:
        admission_types = st.multiselect(
            "Admission Types",
            df["admission_type"].dropna().unique() if "admission_type" in df.columns else [],
        )

filtered_df = df.copy()
if hospitals:
    filtered_df = filtered_df[filtered_df["hospital"].isin(hospitals)]
if conditions:
    filtered_df = filtered_df[filtered_df["medical_condition"].isin(conditions)]
if admission_types:
    filtered_df = filtered_df[filtered_df["admission_type"].isin(admission_types)]

if len(filtered_df) != len(df):
    st.info(f"Applied filters: {len(filtered_df):,} of {len(df):,} records selected")

# expose for summaries
st.session_state["_filtered_df_for_summary"] = filtered_df

# ---------- Local (non-LLM) insight generator ----------
def _fmt_pct(x, d=1):
    try: return f"{100*float(x):.{d}f}%"
    except Exception: return "N/A"

def _fmt_num(x, d=1, money=False):
    try:
        f = float(x)
        if money: return f"${f:,.0f}"
        return f"{f:.{d}f}"
    except Exception:
        return "N/A"

def local_insight(section_title: str, data_summary: dict, model_results: Optional[dict]) -> str:
    role = st.session_state.get("USER_ROLE", "Executive")
    section = section_title.lower()
    gdf = st.session_state.get("_filtered_df_for_summary")

    if "admission" in section:
        if gdf is None or len(gdf) == 0:
            return "Insufficient admissions data."
        adm_by_dow = gdf.groupby(gdf["date_of_admission"].dt.dayofweek).size().reindex(range(7), fill_value=0)
        weekday_uplift = (adm_by_dow[:5].mean() - adm_by_dow[5:].mean()) / max(adm_by_dow.mean(), 1e-9)
        body = [
            f"Weekday volumes run {_fmt_pct(weekday_uplift)} higher than weekends.",
            f"Best model: {data_summary.get('best_model','?')} (MAPE {_fmt_num(data_summary.get('best_model_mape',np.nan),1)}%).",
            "Align roster to weekday peaks; keep a flex pool for 2σ days; refresh forecast weekly."
        ]
        return "\n".join(body)

    if "revenue" in section:
        if gdf is None or "billing_amount" not in gdf.columns or len(gdf) == 0:
            return "Insufficient revenue data."
        bills = gdf["billing_amount"].sort_values(ascending=False)
        top_decile_cut = int(max(1, np.floor(0.1 * len(bills))))
        top_decile_share = bills.iloc[:top_decile_cut].sum() / max(bills.sum(), 1e-9)
        payer_mix = gdf.groupby("insurance_provider")["billing_amount"].sum().sort_values(ascending=False)
        top_payer = payer_mix.index[0] if len(payer_mix) else "N/A"
        top_payer_share = (payer_mix.iloc[0] / payer_mix.sum()) if len(payer_mix) else np.nan
        anomaly_line = ""
        if model_results:
            best = min(model_results.keys(), key=lambda x: abs(model_results[x].get("anomaly_rate", 1.0) - 0.05))
            det = model_results[best]
            anomaly_line = f"{best} flags {int(det.get('anomalies_detected',0))} cases ({_fmt_pct(det.get('anomaly_rate',0))})."
        body = [
            f"Top 10% encounters drive {_fmt_pct(top_decile_share)} of revenue.",
            f"{top_payer} accounts for {_fmt_pct(top_payer_share)} of payer mix.",
            anomaly_line,
            "Audit top-decile encounters; negotiate with top payer; add pre-bill rules for spike days."
        ]
        return "\n".join([b for b in body if b])

    if "length of stay" in section or "los" in section:
        if gdf is None or "length_of_stay" not in gdf.columns or len(gdf) == 0:
            return "Insufficient LOS data."
        los = gdf["length_of_stay"].astype(float)
        iqr = np.percentile(los, 75) - np.percentile(los, 25)
        body = [
            f"Average LOS {los.mean():.1f} days (median {los.median():.1f}; IQR {iqr:.1f}).",
            "Standardize discharge pathways for highest-LOS cohorts; focus variance-reduction on IQR tail."
        ]
        return "\n".join(body)

    return "No summary available."

# ---------- LLM insight generator (role-aware) ----------
def llm_insight(section_title: str, data_summary: dict, model_results: Optional[dict], prompt_overrides: str = "") -> str:
    use_llm = bool(st.session_state.get("INSIGHTS_USE_LLM", False))
    if not use_llm or not OPENAI_AVAILABLE:
        return local_insight(section_title, data_summary, model_results)

    model_label = st.session_state.get("INSIGHTS_MODEL", "GPT-4.0")
    model_name = OPENAI_MODEL_MAP.get(model_label, "gpt-4")
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
        base_prompt += f"\nAdditional Guidance from User:\n{prompt_overrides.strip()}\n"

    try:
        if hasattr(OPENAI_CLIENT, "chat") and hasattr(OPENAI_CLIENT.chat, "completions"):
            resp = OPENAI_CLIENT.chat.completions.create(
                model=model_name,
                temperature=temperature,
                messages=[
                    {"role": "system", "content": "You turn analytics into clear, executive/analyst insights."},
                    {"role": "user", "content": base_prompt},
                ],
            )
            text = resp.choices[0].message.content.strip()
        else:
            resp = OPENAI_CLIENT.ChatCompletion.create(
                model=model_name,
                temperature=temperature,
                messages=[
                    {"role": "system", "content": "You turn analytics into clear, executive/analyst insights."},
                    {"role": "user", "content": base_prompt},
                ],
            )
            text = resp.choices[0].message["content"].strip()
        return text
    except Exception as e:
        st.info(f"LLM insights unavailable ({e}). Showing local summary.")
        return local_insight(section_title, data_summary, model_results)

# ---------- Safe “Implications” code executor ----------
def run_implications_code(user_code: str, context: Dict[str, Any]) -> Tuple[bool, Any, str]:
    """
    Executes user Python code in a constrained namespace.
    Expected: user defines a function `compute_implications(ctx)` that returns dict or DataFrame.
    """
    safe_globals = {
        "__builtins__": {
            "abs": abs, "min": min, "max": max, "sum": sum, "len": len, "range": range, "round": round,
            "float": float, "int": int, "str": str, "dict": dict, "list": list, "tuple": tuple,
            "enumerate": enumerate, "zip": zip, "sorted": sorted
        }
    }
    safe_locals = {"pd": pd, "np": np}

    try:
        compiled = compile(user_code, "<implications>", "exec")
        exec(compiled, safe_globals, safe_locals)  # noqa: S102 (intended sandbox with restricted builtins)
        if "compute_implications" not in safe_locals:
            return False, None, "Function `compute_implications(ctx)` not found."
        result = safe_locals["compute_implications"](context)
        return True, result, ""
    except Exception as e:
        return False, None, f"Error while executing implications code: {e}"

# ---------- Tabs ----------
tabs = st.tabs([
    "📈 Admissions Forecasting",
    "💰 Revenue Analytics",
    "🛏️ Length of Stay Prediction",
    "📊 Operational KPIs & Simulator"
])

# ===================== TAB 1: Admissions Forecasting =====================
with tabs[0]:
    st.markdown(
        """
    <div class="analysis-section">
        <h3>📈 Patient Admission Forecasting</h3>
        <p>Predict future admission patterns to optimize staffing and resource allocation.</p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    with st.expander("ℹ️ Model Primer: Forecasting", expanded=False):
        st.markdown(
            """
- **Linear Trend**: fast baseline for steady growth/decline; ignores seasonality.
- **ARIMA(1,1,1)**: handles autocorrelation; good short-term signal when series is stationary-ish.
- **Exponential Smoothing (HW add-add, 7-day)**: captures weekly seasonality; robust for staffing.
- **Prophet** *(optional)*: flexible seasonality/holidays; heavier dependency.
            """
        )

    daily_adm = filtered_df.groupby(filtered_df["date_of_admission"].dt.date).size().reset_index()
    daily_adm.columns = ["date", "admissions"]

    st.session_state["_daily_adm_for_summary"] = daily_adm
    st.session_state["_filtered_df_for_summary"] = filtered_df

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
        st.markdown(
            f"""
        <div class="metric-card"><h4>Average Daily</h4><h2 style="color:#4285f4;">{avg_daily:.1f}</h2></div>
        <div class="metric-card"><h4>Peak Daily</h4><h2 style="color:#34a853;">{max_daily}</h2></div>
        <div class="metric-card"><h4>Variability (σ)</h4><h2 style="color:#fbbc04;">{std_daily:.1f}</h2></div>
        """,
            unsafe_allow_html=True,
        )

    st.markdown('<div class="config-row"><h4>🔧 Forecasting Configuration</h4></div>', unsafe_allow_html=True)
    c3, c4, c5 = st.columns(3)
    with c3:
        forecast_days = st.slider("Forecast Period (days)", 7, 30, 14)
    with c4:
        _ = st.selectbox("Seasonality (informational)", ["Auto-detect", "Weekly", "Monthly", "None"])
    with c5:
        _ = st.slider("Confidence Level (visual only)", 0.8, 0.99, 0.95)

    selected_models = st.multiselect(
        "Select Forecasting Models",
        ["Linear Trend", "ARIMA", "Exponential Smoothing", "Prophet"],
        default=["Linear Trend", "ARIMA", "Exponential Smoothing"],
    )
    if "Prophet" in selected_models and not HAS_PROPHET:
        st.info("Prophet not installed; it will be skipped.")

    # Editable Implications code for staffing from forecast
    st.markdown("### 🧪 Editable Implications (Python)")
    default_implications_code_fc = textwrap.dedent("""
        # Define a function compute_implications(ctx) -> dict or DataFrame
        # ctx contains:
        #   ctx['future_dates']: pd.DatetimeIndex
        #   ctx['best_forecast']: np.ndarray
        #   ctx['avg_daily']: float
        # Return a dictionary (shown as key/value) or DataFrame
        def compute_implications(ctx):
            import numpy as np
            import pandas as pd
            fc = np.array(ctx['best_forecast']).astype(float)
            peak = float(fc.max()) if fc.size else 0.0
            nurses_day = int(np.ceil(peak / 8)) if peak else 0
            nurses_night = int(np.ceil(peak / 12)) if peak else 0
            budget = nurses_day*350 + nurses_night*400
            return {
                "Peak Admissions (next horizon)": round(peak, 1),
                "Peak Day Nurses": nurses_day,
                "Peak Night Nurses": nurses_night,
                "Estimated Peak-Day Cost ($)": int(budget),
                "Comment": "Use flex pool for 2σ days and re-evaluate weekly."
            }
    """).strip()
    user_code_fc = st.text_area(
        "Edit & run to customize implications:",
        value=default_implications_code_fc,
        height=260,
        key="code_fc"
    )

    # Forecast button
    if st.button("🚀 Generate Forecasts", key="train_forecast", type="primary"):
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
                        p_df = train_data.reset_index()
                        p_df.columns = ["ds", "y"]
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

                if results:
                    st.success("✅ Forecasting models trained successfully!")
                    st.subheader("📊 Model Performance Comparison")
                    res_df = pd.DataFrame(results).T

                    def highlight_best(s):
                        if s.name == "MAPE":
                            is_best = s == s.min()
                        else:
                            is_best = s == s.min()  # MAE/RMSE also lower is better
                        return ["background-color:#d4edda;color:#155724" if v else "" for v in is_best]

                    st.dataframe(res_df.style.apply(highlight_best, axis=0).format("{:.2f}"), use_container_width=True)

                    st.subheader("🔮 Admission Forecasts")
                    future_dates = pd.date_range(
                        start=pd.to_datetime(ts.index[-1]) + pd.Timedelta(days=1),
                        periods=forecast_days, freq="D"
                    )
                    figf = go.Figure()
                    figf.add_trace(go.Scatter(x=ts.index, y=ts.values, mode="lines", name="Historical"))
                    palette = ["#34a853", "#ea4335", "#fbbc04", "#9aa0a6"]
                    for i, (name, fc) in enumerate(forecasts.items()):
                        figf.add_trace(
                            go.Scatter(
                                x=future_dates, y=fc, mode="lines+markers",
                                name=f"{name} Forecast", line=dict(dash="dash", width=2, color=palette[i % len(palette)])
                            )
                        )
                    figf.update_layout(title="Admission Forecasts by Model", height=500,
                                       xaxis_title="Date", yaxis_title="Predicted Admissions")
                    st.plotly_chart(figf, use_container_width=True)

                    # Best forecast + implications editor
                    best_model = min(results.keys(), key=lambda x: results[x]["MAPE"])
                    best_forecast = forecasts[best_model]
                    context = {
                        "future_dates": future_dates,
                        "best_forecast": best_forecast,
                        "avg_daily": float(ts.mean())
                    }
                    ok, impl, err = run_implications_code(user_code_fc, context)
                    st.subheader("💼 Business Implications")
                    if ok:
                        if isinstance(impl, pd.DataFrame):
                            st.dataframe(impl, use_container_width=True)
                        elif isinstance(impl, dict):
                            cols = st.columns(min(4, len(impl) if impl else 1))
                            for i, (k, v) in enumerate(impl.items()):
                                with cols[i % len(cols)]:
                                    st.metric(k, str(v))
                        else:
                            st.write(impl)
                    else:
                        st.error(err)

                    st.download_button(
                        "📥 Download Best Forecast (CSV)",
                        data=pd.DataFrame({"date": future_dates, "forecast": best_forecast}).to_csv(index=False),
                        file_name=f"admissions_forecast_{best_model}.csv",
                        mime="text/csv",
                    )

                else:
                    st.error("No models produced results. Try a shorter horizon or ensure enough history.")

    # Insights (LLM-aware, role-aware, with optional custom guidance)
    st.markdown("---")
    st.markdown("## 📋 Insights")
    extra_prompt = st.text_area("Optional: Add guidance to steer the LLM/local summary (won’t be ignored).", value="", key="adm_extra")
    if "forecasting" in st.session_state.model_results and st.session_state.model_results["forecasting"]:
        forecasting_results = st.session_state.model_results["forecasting"]
        best_model = min(forecasting_results.keys(), key=lambda x: forecasting_results[x]["MAPE"])
        data_summary = {
            "total_records": len(filtered_df),
            "forecast_horizon": locals().get("forecast_days", 14),
            "average_daily_admissions": float(daily_adm["admissions"].mean()) if len(daily_adm) else 0.0,
            "best_model": best_model,
            "best_model_mape": float(forecasting_results[best_model]["MAPE"]),
        }
        insights = llm_insight("Admissions Forecasting", data_summary, forecasting_results, extra_prompt)
        st.markdown(f'<div class="insights-panel">{insights}</div>', unsafe_allow_html=True)
    else:
        st.info("Train a forecast to generate insights for this section.")

# ===================== TAB 2: Revenue Analytics =====================
with tabs[1]:
    st.markdown(
        """
    <div class="analysis-section">
        <h3>💰 Revenue Pattern Analysis</h3>
        <p>Detect unusual billing patterns and identify potential revenue optimization opportunities.</p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    with st.expander("ℹ️ Model Primer: Revenue Anomaly Detection", expanded=False):
        st.markdown(
            """
- **Isolation Forest**: unsupervised outlier detection; good default for mixed features.
- **Statistical Outliers (z-score>3)**: simple rule-based; transparent but brittle.
- **Ensemble**: run both and cross-flag; reduces false positives at small extra cost.
            """
        )

    c1, c2 = st.columns([2, 1])
    with c1:
        st.subheader("💳 Revenue Trends")
        daily_rev = (
            filtered_df.groupby(filtered_df["date_of_admission"].dt.date)["billing_amount"].sum().reset_index()
        )
        daily_rev.columns = ["date", "revenue"]

        st.session_state["_daily_rev_for_summary"] = daily_rev
        st.session_state["_filtered_df_for_summary"] = filtered_df

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=daily_rev["date"], y=daily_rev["revenue"], mode="lines+markers", name="Daily Revenue", line=dict(width=2)))
        fig.update_layout(title="Daily Hospital Revenue", xaxis_title="Date", yaxis_title="Revenue ($)", height=400)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.subheader("💰 Revenue Metrics")
        total_revenue = float(filtered_df["billing_amount"].sum()) if "billing_amount" in filtered_df.columns else 0.0
        avg_daily_revenue = float(daily_rev["revenue"].mean()) if len(daily_rev) else 0.0
        avg_bill = float(filtered_df["billing_amount"].mean()) if "billing_amount" in filtered_df.columns else 0.0
        st.markdown(
            f"""
        <div class="metric-card"><h4>Total Revenue</h4><h2 style="color:#34a853;">${total_revenue:,.0f}</h2></div>
        <div class="metric-card"><h4>Daily Average</h4><h2 style="color:#4285f4;">${avg_daily_revenue:,.0f}</h2></div>
        <div class="metric-card"><h4>Avg Per Patient</h4><h2 style="color:#fbbc04;">${avg_bill:,.0f}</h2></div>
        """,
            unsafe_allow_html=True,
        )

    st.markdown('<div class="config-row"><h4>🔧 Anomaly Detection Settings</h4></div>', unsafe_allow_html=True)
    a1, a2, a3 = st.columns(3)
    with a1:
        detection_method = st.selectbox("Detection Method", ["Isolation Forest", "Statistical Outliers", "Ensemble"])
    with a2:
        sensitivity = st.slider("Sensitivity (contamination)", 0.01, 0.10, 0.05, 0.01)
    with a3:
        features_for_anomaly = st.multiselect(
            "Analysis Features",
            ["billing_amount", "age", "length_of_stay", "admission_month", "admission_day_of_week"],
            default=["billing_amount", "length_of_stay"],
        )

    # Editable implications for revenue anomalies
    st.markdown("### 🧪 Editable Implications (Python)")
    default_implications_code_rev = textwrap.dedent("""
        # compute_implications(ctx) for revenue anomalies
        # ctx:
        #   ctx['results']: dict with method metrics
        #   ctx['best_method']: str
        #   ctx['flagged_total_amount']: float
        def compute_implications(ctx):
            best = ctx['best_method']
            flagged = ctx['flagged_total_amount']
            return {
                "Best Detector": best,
                "Flagged Amount ($)": int(flagged),
                "Recommendation": "Audit top 10 flagged claims and tighten pre-bill rules.",
            }
    """).strip()
    user_code_rev = st.text_area(
        "Edit & run to customize implications:",
        value=default_implications_code_rev,
        height=200,
        key="code_rev"
    )

    if st.button("🚀 Analyze Revenue Patterns", key="detect_anomalies", type="primary"):
        if features_for_anomaly:
            with st.spinner("Analyzing revenue patterns..."):
                X = filtered_df[features_for_anomaly].dropna()
                results = {}
                anomaly_predictions = {}

                if detection_method in ["Isolation Forest", "Ensemble"]:
                    iso = IsolationForest(contamination=float(sensitivity), random_state=42)
                    pred = iso.fit_predict(X)
                    anomaly_predictions["Isolation Forest"] = pred
                    n_anom = int((pred == -1).sum())
                    rate = n_anom / len(X) if len(X) else 0
                    results["Isolation Forest"] = {
                        "anomalies_detected": n_anom,
                        "anomaly_rate": rate,
                        "avg_anomaly_amount": float(
                            filtered_df.loc[X.index[pred == -1], "billing_amount"].mean()
                        )
                        if n_anom > 0 and "billing_amount" in filtered_df.columns
                        else 0.0,
                    }

                if detection_method in ["Statistical Outliers", "Ensemble"]:
                    z = np.abs((X - X.mean()) / X.std(ddof=0)).fillna(0.0)
                    stat_mask = (z > 3).any(axis=1)
                    n_anom = int(stat_mask.sum())
                    rate = n_anom / len(X) if len(X) else 0
                    results["Statistical Outliers"] = {
                        "anomalies_detected": n_anom,
                        "anomaly_rate": rate,
                        "avg_anomaly_amount": float(
                            filtered_df.loc[X.index[stat_mask], "billing_amount"].mean()
                        )
                        if n_anom > 0 and "billing_amount" in filtered_df.columns
                        else 0.0,
                    }
                    anomaly_predictions["Statistical Outliers"] = np.where(stat_mask.values, -1, 1)

                st.session_state.model_results["anomaly"] = results
                st.success("✅ Revenue analysis completed!")

                st.subheader("🎯 Detection Results")
                res_df = pd.DataFrame(results).T
                st.dataframe(
                    res_df.style.format(
                        {"anomalies_detected": "{:.0f}", "anomaly_rate": "{:.2%}", "avg_anomaly_amount": "${:,.0f}"}
                    ),
                    use_container_width=True,
                )

                st.subheader("📊 Pattern Visualization")
                if results:
                    best_method = min(results.keys(), key=lambda x: abs(results[x]["anomaly_rate"] - 0.05))
                    best_pred = anomaly_predictions[best_method]
                    if len(features_for_anomaly) >= 2 and len(X) == len(best_pred):
                        figp = px.scatter(
                            x=X[features_for_anomaly[0]],
                            y=X[features_for_anomaly[1]],
                            color=(best_pred == -1).astype(int),
                            title=f"Revenue Pattern Analysis — {best_method}",
                            labels={"color": "Anomaly (1=yes)"},
                        )
                        figp.update_layout(height=500)
                        st.plotly_chart(figp, use_container_width=True)

                    anomaly_idx = X.index[np.where(best_pred == -1)[0]]
                    if len(anomaly_idx) > 0:
                        st.subheader("🚨 Unusual Cases Identified")
                        cols = [
                            c
                            for c in [
                                "date_of_admission",
                                "billing_amount",
                                "medical_condition",
                                "hospital",
                                "insurance_provider",
                                "doctor",
                            ]
                            if c in filtered_df.columns
                        ]
                        details = filtered_df.loc[anomaly_idx, cols].head(100)
                        st.dataframe(details, use_container_width=True)

                        st.download_button(
                            "📥 Download Anomaly Cases (CSV)",
                            data=details.to_csv(index=False),
                            file_name="revenue_anomalies.csv",
                            mime="text/csv",
                        )

                        total_flagged = float(filtered_df.loc[anomaly_idx, "billing_amount"].sum()) if "billing_amount" in filtered_df.columns else 0.0
                    else:
                        total_flagged = 0.0

                    # run implications code
                    context = {"results": results, "best_method": best_method, "flagged_total_amount": total_flagged}
                    ok, impl, err = run_implications_code(user_code_rev, context)
                    st.subheader("💼 Business Implications")
                    if ok:
                        if isinstance(impl, pd.DataFrame):
                            st.dataframe(impl, use_container_width=True)
                        elif isinstance(impl, dict):
                            cols = st.columns(min(4, len(impl) if impl else 1))
                            for i, (k, v) in enumerate(impl.items()):
                                with cols[i % len(cols)]:
                                    st.metric(k, str(v))
                        else:
                            st.write(impl)
                    else:
                        st.error(err)

        else:
            st.warning("Please select at least one feature for analysis.")

    st.markdown("---")
    st.markdown("## 📋 Insights")
    extra_prompt_rev = st.text_area("Optional: guidance for insights", value="", key="rev_extra")
    if "anomaly" in st.session_state.model_results and st.session_state.model_results["anomaly"]:
        anomaly_results = st.session_state.model_results["anomaly"]
        best_method = min(anomaly_results.keys(), key=lambda x: abs(anomaly_results[x]["anomaly_rate"] - 0.05))
        data_summary = {
            "total_revenue": float(filtered_df["billing_amount"].sum()) if "billing_amount" in filtered_df.columns else 0.0,
            "avg_daily_revenue": float(daily_rev["revenue"].mean()) if len(daily_rev) else 0.0,
            "best_method": best_method,
            "anomalies_detected": int(anomaly_results[best_method]["anomalies_detected"]),
            "anomaly_rate": float(anomaly_results[best_method]["anomaly_rate"]),
        }
        insights = llm_insight("Revenue Pattern Analysis", data_summary, anomaly_results, extra_prompt_rev)
        st.markdown(f'<div class="insights-panel">{insights}</div>', unsafe_allow_html=True)
    else:
        st.info("Run anomaly detection to generate insights for this section.")

# ===================== TAB 3: Length of Stay Prediction =====================
with tabs[2]:
    st.markdown(
        """
    <div class="analysis-section">
        <h3>🛏️ Length of Stay Prediction</h3>
        <p>Predict stay duration to optimize bed management and discharge planning. ROC/AUC appears for classification.</p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    with st.expander("ℹ️ Model Primer: LOS", expanded=False):
        st.markdown(
            """
- **Classification (LOS Category)**: *Random Forest* (nonlinear + interactions), *Logistic Regression* (baseline, interpretable), *SVM* (margins for complex boundaries), *XGBoost* (if installed).
- **Regression (Days)**: *Random Forest Regressor* (nonlinear), *Linear Regression* (fast baseline), *SVR* (when margins matter), *XGBoost Regressor* (if installed).
            """
        )

    if "length_of_stay" not in filtered_df.columns:
        st.error("Length of stay data not available. Please check your dataset.")
    else:
        c1, c2 = st.columns([2, 1])
        with c1:
            st.subheader("📊 LOS Distribution")
            fig = px.histogram(filtered_df, x="length_of_stay", nbins=30, title="Distribution of Length of Stay")
            fig.update_layout(height=400, xaxis_title="Length of Stay (days)", yaxis_title="Patients")
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            st.subheader("🏥 LOS Metrics")
            avg_los = float(filtered_df["length_of_stay"].mean())
            median_los = float(filtered_df["length_of_stay"].median())
            max_los = float(filtered_df["length_of_stay"].max())
            st.session_state["los_avg_los"] = avg_los
            st.markdown(
                f"""
            <div class="metric-card"><h4>Average LOS</h4><h2 style="color:#4285f4;">{avg_los:.1f} days</h2></div>
            <div class="metric-card"><h4>Median LOS</h4><h2 style="color:#34a853;">{median_los:.1f} days</h2></div>
            <div class="metric-card"><h4>Maximum LOS</h4><h2 style="color:#ea4335;">{max_los:.0f} days</h2></div>
            """,
                unsafe_allow_html=True,
            )

        st.markdown('<div class="config-row"><h4>🔧 Prediction Model Setup</h4></div>', unsafe_allow_html=True)

        available_features = [
            c for c in filtered_df.columns
            if c not in ["length_of_stay", "los_category", "date_of_admission", "discharge_date"]
        ]
        s1, s2 = st.columns(2)
        with s1:
            selected_features = st.multiselect(
                "Select Prediction Features",
                available_features,
                default=[f for f in ["age", "medical_condition", "admission_type", "hospital"] if f in available_features],
            )
        with s2:
            target_type = st.selectbox("Prediction Target", ["Length of Stay (Days)", "LOS Category (Short/Medium/Long)"])
            selected_models = st.multiselect(
                "Select Models", ["Random Forest", "Logistic Regression", "XGBoost", "SVM", "Linear Regression"],
                default=["Random Forest", "Logistic Regression"],
            )

        # Editable implications for LOS
        st.markdown("### 🧪 Editable Implications (Python)")
        default_implications_code_los = textwrap.dedent("""
            # compute_implications(ctx) for LOS
            # ctx:
            #   ctx['is_classification']: bool
            #   ctx['best_model_name']: str
            #   ctx['metric']: float (accuracy or r2)
            #   ctx['avg_los']: float
            def compute_implications(ctx):
                if ctx['is_classification']:
                    perf = f"Accuracy: {ctx['metric']:.2%}"
                    rec = "Use predicted categories to trigger early discharge planning."
                else:
                    perf = f"R2: {ctx['metric']:.3f}"
                    rec = "Target cohorts where predicted LOS exceeds facility average."
                return {"Best Model": ctx['best_model_name'], "Performance": perf, "Recommendation": rec}
        """).strip()
        user_code_los = st.text_area(
            "Edit & run to customize implications:",
            value=default_implications_code_los,
            height=220,
            key="code_los"
        )

        if st.button("🚀 Train Prediction Models", key="train_los", type="primary"):
            if selected_features:
                with st.spinner("Training LOS prediction models..."):
                    feature_data = filtered_df[selected_features + ["length_of_stay", "los_category"]].dropna()
                    if target_type == "Length of Stay (Days)":
                        target = "length_of_stay"; is_classification = False
                    else:
                        target = "los_category"; is_classification = True

                    st.session_state["los_is_classification"] = is_classification
                    st.session_state["los_target_type"] = target_type

                    X = feature_data[selected_features]
                    y = feature_data[target]

                    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
                    categorical_features = X.select_dtypes(include=["object", "category"]).columns.tolist()
                    preprocessor = ColumnTransformer(
                        transformers=[
                            ("num", StandardScaler(), numeric_features),
                            ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), categorical_features),
                        ]
                    )

                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                    results = {}
                    models_trained = {}

                    for model_name in selected_models:
                        try:
                            if is_classification:
                                if model_name == "Random Forest":
                                    model = RandomForestClassifier(n_estimators=200, random_state=42)
                                elif model_name == "Logistic Regression":
                                    model = LogisticRegression(max_iter=1000, random_state=42)
                                elif model_name == "XGBoost" and HAS_XGBOOST:
                                    model = xgb.XGBClassifier(random_state=42)
                                elif model_name == "SVM":
                                    model = SVC(probability=True, random_state=42)
                                else:
                                    continue
                                final_step_name = "classifier"
                            else:
                                if model_name == "Random Forest":
                                    model = RandomForestRegressor(n_estimators=300, random_state=42)
                                elif model_name == "Linear Regression":
                                    model = LinearRegression()
                                elif model_name == "XGBoost" and HAS_XGBOOST:
                                    model = xgb.XGBRegressor(random_state=42)
                                elif model_name == "SVM":
                                    model = SVR()
                                else:
                                    continue
                                final_step_name = "regressor"

                            pipe = Pipeline([("preprocessor", preprocessor), (final_step_name, model)])
                            pipe.fit(X_train, y_train)
                            models_trained[model_name] = pipe
                            y_pred = pipe.predict(X_test)

                            if is_classification:
                                accuracy = accuracy_score(y_test, y_pred)
                                precision, recall, f1, _ = precision_recall_fscore_support(
                                    y_test, y_pred, average="weighted", zero_division=0
                                )
                                results[model_name] = {
                                    "accuracy": float(accuracy),
                                    "precision": float(precision),
                                    "recall": float(recall),
                                    "f1_score": float(f1),
                                }
                            else:
                                mae = mean_absolute_error(y_test, y_pred)
                                rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
                                r2 = r2_score(y_test, y_pred)
                                results[model_name] = {"mae": float(mae), "rmse": rmse, "r2_score": float(r2)}

                        except Exception as e:
                            st.warning(f"Failed to train {model_name}: {e}")

                    if results:
                        st.session_state.model_results["los"] = results
                        st.success("✅ LOS prediction models trained successfully!")

                        st.subheader("📊 Model Performance Comparison")
                        res_df = pd.DataFrame(results).T
                        if is_classification:
                            def highlight_max(s):
                                is_max = s == s.max()
                                return ["background-color:#d4edda;color:#155724" if v else "" for v in is_max]
                            st.dataframe(res_df.style.apply(highlight_max, axis=0).format("{:.3f}"), use_container_width=True)
                            metric_to_plot = "accuracy"
                        else:
                            def highlight_best(s):
                                if s.name == "r2_score":
                                    is_best = s == s.max()
                                else:
                                    is_best = s == s.min()
                                return ["background-color:#d4edda;color:#155724" if v else "" for v in is_best]
                            st.dataframe(res_df.style.apply(highlight_best, axis=0).format("{:.3f}"), use_container_width=True)
                            metric_to_plot = "r2_score"

                        figm = go.Figure(data=[go.Bar(
                            x=list(results.keys()),
                            y=[results[m].get(metric_to_plot, 0) for m in results.keys()],
                            text=[f"{results[m].get(metric_to_plot, 0):.3f}" for m in results.keys()],
                            textposition="auto"
                        )])
                        figm.update_layout(title=f"Model Comparison — {metric_to_plot.title()}", height=400)
                        st.plotly_chart(figm, use_container_width=True)

                        # Best model + ROC if classification
                        if is_classification:
                            best_model_name = max(results.keys(), key=lambda x: results[x].get("accuracy", -np.inf))
                            best_model = models_trained[best_model_name]
                            st.subheader("🧪 ROC Curve & AUC (if applicable)")

                            # Try to get probabilities for ROC (binary or micro-average multiclass)
                            y_test_series = pd.Series(y_test).reset_index(drop=True)
                            classes = sorted(y_test_series.unique().tolist())
                            try:
                                # proba preferred
                                if hasattr(best_model, "predict_proba"):
                                    probs = best_model.predict_proba(X_test)
                                else:
                                    # decision_function fallback
                                    if hasattr(best_model, "decision_function"):
                                        dfun = best_model.decision_function(X_test)
                                        # convert to pseudo-probs via min-max
                                        if dfun.ndim == 1:
                                            dfun = dfun.reshape(-1, 1)
                                        df_min = dfun.min()
                                        df_max = dfun.max()
                                        probs = (dfun - df_min) / (df_max - df_min + 1e-9)
                                    else:
                                        probs = None

                                if probs is not None:
                                    if len(classes) == 2:
                                        # binary AUC
                                        y_bin = (y_test_series == classes[-1]).astype(int).values
                                        # pick positive class probability
                                        if probs.ndim == 2 and probs.shape[1] >= 2:
                                            pos = 1
                                            fpr, tpr, _ = roc_curve(y_bin, probs[:, pos])
                                            auc_val = roc_auc_score(y_bin, probs[:, pos])
                                        else:
                                            # single column: treat as positive prob
                                            fpr, tpr, _ = roc_curve(y_bin, probs.ravel())
                                            auc_val = roc_auc_score(y_bin, probs.ravel())
                                        figroc = go.Figure()
                                        figroc.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"ROC (AUC={auc_val:.3f})"))
                                        figroc.add_trace(go.Scatter(x=[0,1], y=[0,1], mode="lines", name="Chance", line=dict(dash="dash")))
                                        figroc.update_layout(xaxis_title="FPR", yaxis_title="TPR", height=400)
                                        st.plotly_chart(figroc, use_container_width=True)
                                    else:
                                        # multiclass micro-average
                                        y_binarized = label_binarize(y_test_series, classes=classes)
                                        if probs.shape[1] != y_binarized.shape[1]:
                                            # try to align columns
                                            # if model class order is different, we best-effort align by label order
                                            pass
                                        auc_val = roc_auc_score(y_binarized, probs, average="micro")
                                        st.info(f"Multiclass micro-average AUC: {auc_val:.3f} (ROC per-class omitted for brevity)")
                                else:
                                    st.info("ROC not available (model lacks predict_proba / decision_function).")
                            except Exception as e:
                                st.info(f"ROC/AUC computation skipped: {e}")

                        # Implications run
                        if is_classification:
                            best_model_name = max(results.keys(), key=lambda x: results[x].get("accuracy", -np.inf))
                            perf = results[best_model_name]["accuracy"]
                        else:
                            best_model_name = max(results.keys(), key=lambda x: results[x].get("r2_score", -np.inf))
                            perf = results[best_model_name]["r2_score"]

                        ctx = {
                            "is_classification": is_classification,
                            "best_model_name": best_model_name,
                            "metric": float(perf),
                            "avg_los": float(st.session_state.get("los_avg_los") or 0.0)
                        }
                        ok, impl, err = run_implications_code(user_code_los, ctx)
                        st.subheader("💼 Business Implications")
                        if ok:
                            if isinstance(impl, pd.DataFrame):
                                st.dataframe(impl, use_container_width=True)
                            elif isinstance(impl, dict):
                                cols = st.columns(min(4, len(impl) if impl else 1))
                                for i, (k, v) in enumerate(impl.items()):
                                    with cols[i % len(cols)]:
                                        st.metric(k, str(v))
                            else:
                                st.write(impl)
                        else:
                            st.error(err)

                        # Sample Predictions
                        st.subheader("🔮 Sample Predictions")
                        best_model = models_trained[best_model_name]
                        nshow = min(10, len(X_test))
                        samples = []
                        for i in range(nshow):
                            sample_input = X_test.iloc[[i]]
                            pred = best_model.predict(sample_input)[0]
                            if is_classification:
                                samples.append(
                                    {
                                        "Case": f"Patient {i+1}",
                                        "Predicted Category": str(pred),
                                        "Actual Category": str(y_test.iloc[i]),
                                        "Match": "✅" if pred == y_test.iloc[i] else "❌",
                                    }
                                )
                            else:
                                samples.append(
                                    {
                                        "Case": f"Patient {i+1}",
                                        "Predicted LOS": f"{float(pred):.1f} days",
                                        "Actual LOS": f"{float(y_test.iloc[i]):.1f} days",
                                        "Difference": f"{abs(float(pred) - float(y_test.iloc[i])):.1f} days",
                                    }
                                )
                        st.dataframe(pd.DataFrame(samples), use_container_width=True)
                    else:
                        st.error("All models failed to train. Check feature selection and data quality.")
            else:
                st.warning("Please select at least one feature for model training.")

        # Insights (LLM-aware)
        st.markdown("---")
        st.markdown("## 📋 Insights")
        extra_prompt_los = st.text_area("Optional: guidance for insights", value="", key="los_extra")
        if "los" in st.session_state.model_results and st.session_state.model_results["los"]:
            los_results = st.session_state.model_results["los"]
            is_cls = bool(st.session_state.get("los_is_classification", False))
            if is_cls:
                best_model = max(los_results.keys(), key=lambda x: los_results[x].get("accuracy", 0))
                perf = los_results[best_model]["accuracy"]
            else:
                best_model = max(los_results.keys(), key=lambda x: los_results[x].get("r2_score", -np.inf))
                perf = los_results[best_model]["r2_score"]
            data_summary = {
                "prediction_type": st.session_state.get("los_target_type", "Unknown"),
                "best_model": best_model,
                "performance_metric": float(perf),
                "avg_los": float(st.session_state.get("los_avg_los") or 0.0),
                "total_patients": len(filtered_df),
            }
            insights = llm_insight("Length of Stay Prediction", data_summary, los_results, extra_prompt_los)
            st.markdown(f'<div class="insights-panel">{insights}</div>', unsafe_allow_html=True)
        else:
            st.info("Train at least one LOS model to generate insights for this section.")

# ===================== TAB 4: Operational KPIs & Simulator =====================
with tabs[3]:
    st.markdown("""
    <div class="analysis-section">
        <h3>📊 Operational KPIs & What-If Staffing</h3>
        <p>Concise leadership metrics + a quick scenario sandbox.</p>
    </div>
    """, unsafe_allow_html=True)

    with st.expander("ℹ️ Model Primer: Simulator", expanded=False):
        st.markdown(
            """
- **What-If**: converts admissions targets into day/night nurse counts under configurable patient-per-nurse ratios.
- Use with a surge % to stress-test coverage and daily budget.
            """
        )

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        total_enc = len(filtered_df)
        st.metric("Encounters (filtered)", f"{total_enc:,}")
    with k2:
        if "billing_amount" in filtered_df.columns:
            rev = float(filtered_df["billing_amount"].sum())
            st.metric("Revenue (filtered)", f"${rev:,.0f}")
    with k3:
        if "length_of_stay" in filtered_df.columns:
            st.metric("Avg LOS", f"{filtered_df['length_of_stay'].mean():.1f} days")
    with k4:
        if "medical_condition" in filtered_df.columns and len(filtered_df):
            top_cond = filtered_df["medical_condition"].value_counts().idxmax()
            st.metric("Top Condition", str(top_cond))

    if {"insurance_provider","billing_amount"}.issubset(filtered_df.columns) and len(filtered_df):
        payer_rev = (
            filtered_df.groupby("insurance_provider")["billing_amount"]
            .sum().sort_values(ascending=False).reset_index()
        )
        figp = px.bar(payer_rev, x="insurance_provider", y="billing_amount", title="Revenue by Payer (Pareto)")
        figp.update_layout(height=380, xaxis_title="Payer", yaxis_title="Revenue ($)")
        st.plotly_chart(figp, use_container_width=True)

    st.markdown('<div class="config-row"><h4>🧪 What-If: Staffing vs. Admissions</h4></div>', unsafe_allow_html=True)
    w1, w2, w3 = st.columns(3)
    with w1:
        pts_per_day_nurse = st.number_input("Patients per Day Nurse", min_value=4, max_value=12, value=8)
    with w2:
        pts_per_night_nurse = st.number_input("Patients per Night Nurse", min_value=6, max_value=18, value=12)
    with w3:
        surge_pct = st.slider("Potential Surge (%)", 0, 50, 15, 5)

    daily_adm_local = filtered_df.groupby(filtered_df["date_of_admission"].dt.date).size().reset_index()
    daily_adm_local.columns = ["date","admissions"]
    recent_mean = float(daily_adm_local["admissions"].tail(28).mean()) if len(daily_adm_local) else 0.0
    target_load = recent_mean * (1 + surge_pct/100.0)

    d1, d2, d3 = st.columns(3)
    with d1:
        dn = int(np.ceil(target_load / max(1, pts_per_day_nurse)))
        st.metric("Needed Day Nurses", f"{dn}")
    with d2:
        nn = int(np.ceil(target_load / max(1, pts_per_night_nurse)))
        st.metric("Needed Night Nurses", f"{nn}")
    with d3:
        est_cost = dn*350 + nn*400
        st.metric("Est. Daily Cost", f"${est_cost:,.0f}")

    with st.expander("💻 View Implementation Code (Simulator)"):
        sim_code = f'''
import numpy as np
recent_mean = {recent_mean:.2f}
surge = {surge_pct}/100.0
pts_per_day_nurse = {pts_per_day_nurse}
pts_per_night_nurse = {pts_per_night_nurse}
target_load = recent_mean * (1 + surge)
day_nurses = int(np.ceil(target_load / max(1, pts_per_day_nurse)))
night_nurses = int(np.ceil(target_load / max(1, pts_per_night_nurse)))
daily_cost = day_nurses*350 + night_nurses*400
print("Target Admissions:", round(target_load,1))
print("Day Nurses:", day_nurses, "Night Nurses:", night_nurses, "Daily Cost: $", daily_cost)
'''
        st.code(sim_code, language="python")

# ===================== Decision Log =====================
st.markdown("---")
st.markdown("## 📋 Decision Tracking")
with st.expander("📝 Add New Decision", expanded=False):
    d1, d2 = st.columns(2)
    with d1:
        decision_section = st.selectbox(
            "Analysis Section",
            ["Admissions Forecasting", "Revenue Analytics", "Length of Stay Prediction", "Operational KPIs & Simulator"]
        )
        decision_action = st.selectbox(
            "Decision",
            ["Approve for Production", "Needs Review", "Requires Additional Data", "Reject"]
        )
    with d2:
        decision_owner = st.text_input("Responsible Person")
        decision_date = st.date_input("Target Date")
    decision_notes = st.text_area("Notes and Comments")
    if st.button("Add Decision", type="secondary"):
        new_decision = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "section": decision_section,
            "action": decision_action,
            "owner": decision_owner,
            "target_date": decision_date.strftime("%Y-%m-%d"),
            "notes": decision_notes,
        }
        st.session_state.decision_log.append(new_decision)
        st.success("✅ Decision added to tracking log!")

if st.session_state.decision_log:
    st.subheader("Decision History")
    decisions_df = pd.DataFrame(st.session_state.decision_log)
    c1, c2, c3 = st.columns([2, 1, 1])
    with c1:
        st.dataframe(decisions_df, use_container_width=True)
    with c2:
        if "action" in decisions_df.columns:
            approved = int((decisions_df["action"] == "Approve for Production").sum())
            pending = int((decisions_df["action"] == "Needs Review").sum())
            st.metric("Approved", approved)
            st.metric("Pending Review", pending)
    with c3:
        csv = decisions_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Log",
            data=csv,
            file_name=f"decisions_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv",
            type="secondary",
        )

# ---------- Footer ----------
st.markdown("---")
st.markdown(
    """
<div style="text-align:center;color:#666;padding:2rem;">
    <p>Hospital Operations Analytics Platform • Built with Streamlit & Python ML Libraries</p>
    <p>Audience-aware insights • Editable implications • ROC for classification</p>
</div>
""",
    unsafe_allow_html=True,
)
