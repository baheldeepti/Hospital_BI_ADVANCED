# app.py — Hospital Operations Analytics Platform
# Redesigned for optimal user experience without sidebars

import os
import warnings
from datetime import datetime
from typing import Dict, Optional

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

from sklearn.ensemble import RandomForestClassifier, IsolationForest, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC, SVR
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    mean_absolute_error, mean_squared_error, r2_score
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

# OpenAI (optional) – supports classic and new SDKs
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

# ---------- Executive Insights (LLM) controls ----------
with st.expander("🤖 Executive Insights (LLM)", expanded=False):
    st.session_state.INSIGHTS_USE_LLM = st.checkbox(
        "Use ChatGPT to write executive-friendly Business Insights",
        value=st.session_state.INSIGHTS_USE_LLM,
        help="If disabled or unavailable, the app uses a built-in (non-LLM) summary."
    )
    st.session_state.INSIGHTS_MODEL = st.radio(
        "Model",
        options=["GPT-3.5", "GPT-4.0"],
        index=1 if st.session_state.INSIGHTS_MODEL == "GPT-4.0" else 0,
        horizontal=True,
        help="Pick GPT-3.5 (faster/cheaper) or GPT-4.0 (higher quality)."
    )
    st.session_state.INSIGHTS_TEMP = st.slider(
        "Creativity (temperature)",
        min_value=0.0, max_value=1.0, value=float(st.session_state.INSIGHTS_TEMP), step=0.1
    )
    if st.session_state.INSIGHTS_USE_LLM and not OPENAI_AVAILABLE:
        st.warning("OpenAI API key not found. Set `st.secrets['OPENAI_API_KEY']` or `OPENAI_API_KEY` env var. Falling back to local insights.")

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
        n_records = 10000
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
        df["length_of_stay"] = np.clip(los.fillna(0), 0, 365).astype(int).replace(0, 1)

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
            df["length_of_stay"], bins=[0, 3, 7, 14, float("inf")],
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
with st.expander("🔍 Filter Data (Optional)", expanded=False):
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

# Expose default frame to insight generator
st.session_state["_filtered_df_for_summary"] = filtered_df

# ---------- Built-in (non-LLM) Business Insights ----------
def generate_business_summary(section_title: str, data_summary: dict, model_results: Optional[dict] = None) -> str:
    role = st.session_state.get("USER_ROLE", "Executive")
    _ = int(data_summary.get("total_records", 0))
    section = section_title.lower()

    def pct(x, d=1):
        try: return f"{100*float(x):.{d}f}%"
        except Exception: return "N/A"

    def num(x, d=1, money=False):
        try:
            f = float(x)
            if money: return f"${f:,.0f}"
            return f"{f:.{d}f}"
        except Exception: return "N/A"

    gdf = st.session_state.get("_filtered_df_for_summary")
    daily_adm = st.session_state.get("_daily_adm_for_summary")
    daily_rev = st.session_state.get("_daily_rev_for_summary")

    if "admission" in section:
        if gdf is None or daily_adm is None or len(daily_adm) == 0:
            return "**Admissions – Insights**\nInsufficient data to compute patterns."
        adm_by_dow = gdf.groupby(gdf["date_of_admission"].dt.dayofweek).size().reindex(range(7), fill_value=0)
        dow_vals = adm_by_dow.values.astype(float)
        weekday_uplift = (dow_vals[:5].mean() - dow_vals[5:].mean()) / max(dow_vals.mean(), 1e-9)
        x = np.arange(len(daily_adm))
        slope = np.polyfit(x, daily_adm["admissions"].values, 1)[0]
        vol = daily_adm["admissions"].std()
        peak = daily_adm["admissions"].max()
        mean_adm = daily_adm["admissions"].mean()
        peak_to_mean = peak / max(mean_adm, 1e-9)
        metrics = model_results or {}
        mape_line = ""
        if metrics:
            best = min(metrics, key=lambda k: metrics[k].get("MAPE", np.inf))
            mape_line = f" • **Best model**: {best} (MAPE {num(metrics[best].get('MAPE', np.nan),1)}%)"
        body = [
            f"• **Workload pattern:** weekday volumes are {pct(weekday_uplift)} higher than weekends.",
            f"• **Trend:** {num(slope,2)} admissions/day change over time (linear slope).",
            f"• **Volatility:** σ={num(vol,1)}; **peak-to-mean** {num(peak_to_mean,2)} (peak={int(peak)}, mean={num(mean_adm,1)}).",
            mape_line,
            "• **Action:** align rosters to weekday peaks; keep a flex pool for 2σ days; revisit forecast weekly.",
        ]
        return "**Admissions – Data-Driven Insights**\n" + "\n".join([b for b in body if b])

    if "revenue" in section:
        if gdf is None or "billing_amount" not in gdf.columns or len(gdf) == 0:
            return "**Revenue – Insights**\nInsufficient data."
        bills = gdf["billing_amount"].sort_values(ascending=False).reset_index(drop=True)
        top_decile_cut = int(max(1, np.floor(0.1 * len(bills))))
        top_decile_share = bills.iloc[:top_decile_cut].sum() / max(bills.sum(), 1e-9)
        payer_mix = gdf.groupby("insurance_provider")["billing_amount"].sum().sort_values(ascending=False)
        top_payer = payer_mix.index[0] if len(payer_mix) else "N/A"
        top_payer_share = (payer_mix.iloc[0] / payer_mix.sum()) if len(payer_mix) else np.nan
        if (daily_rev := st.session_state.get("_daily_rev_for_summary")) is None or len(daily_rev) == 0:
            dr_vol, spike_days = np.nan, 0
        else:
            dr = daily_rev["revenue"].astype(float)
            dr_vol = dr.std()
            roll = dr.rolling(7, min_periods=3).mean()
            spike_days = int((dr > (roll * 1.25)).sum())
        metrics = model_results or {}
        anomaly_line = ""
        if metrics:
            best = min(metrics.keys(), key=lambda k: abs(metrics[k].get("anomaly_rate", 1.0) - 0.05))
            det = metrics[best]
            anomaly_line = f"• **{best}** flags {int(det.get('anomalies_detected',0))} cases ({pct(det.get('anomaly_rate',0))})."
        body = [
            f"• **Revenue concentration:** top 10% of encounters drive **{pct(top_decile_share)}** of revenue.",
            f"• **Payer mix:** **{top_payer}** accounts for **{pct(top_payer_share)}**.",
            f"• **Volatility:** σ={num(dr_vol,0, money=True)}; **spike days (7-day mean +25%)**: {spike_days}.",
            anomaly_line,
            "• **Action:** audit top-decile encounters; negotiate with top payer; add pre-bill rules for spike patterns.",
        ]
        return "**Revenue – Data-Driven Insights**\n" + "\n".join([b for b in body if b])

    if "length of stay" in section or "los" in section:
        if gdf is None or "length_of_stay" not in gdf.columns or len(gdf) == 0:
            return "**LOS – Insights**\nInsufficient data."
        los = gdf["length_of_stay"].astype(float)
        iqr = np.percentile(los, 75) - np.percentile(los, 25)
        lines = []
        for col in ["medical_condition", "admission_type", "hospital"]:
            if col in gdf.columns:
                m = gdf.groupby(col)["length_of_stay"].mean().sort_values(ascending=False)
                if len(m) >= 2:
                    head = m.head(1); tail = m.tail(1)
                    lines.append(
                        f"• **{col.replace('_',' ').title()} driver:** {head.index[0]} longer than {tail.index[0]} by {head.iloc[0]-tail.iloc[0]:.1f} days."
                    )
        body = [
            f"• **Tendency & spread:** mean={los.mean():.1f} days; median={los.median():.1f}; IQR={iqr:.1f}.",
            *lines[:3],
            "• **Action:** standardize discharge pathways for highest-LOS cohorts; run variance-reduction on IQR tail.",
        ]
        return "**LOS – Data-Driven Insights**\n" + "\n".join(body)

    return "**Insights**\nNo matching section."

# ---------- LLM-powered Executive Insights (uses chosen model) ----------
def generate_executive_llm_insights(section_title: str, data_summary: dict, model_results: Optional[dict]) -> str:
    """
    Tries to use the selected OpenAI model to produce succinct, executive-friendly insights.
    Falls back to the built-in summarizer on any error or if OpenAI is unavailable.
    """
    use_llm = bool(st.session_state.get("INSIGHTS_USE_LLM", False))
    if not use_llm or not OPENAI_AVAILABLE:
        return generate_business_summary(section_title, data_summary, model_results)

    model_label = st.session_state.get("INSIGHTS_MODEL", "GPT-4.0")
    model_name = OPENAI_MODEL_MAP.get(model_label, "gpt-4")
    temperature = float(st.session_state.get("INSIGHTS_TEMP", 0.2))

    # Build a compact, structured prompt
    prompt = f"""
You are a hospital operations advisor. Write an executive-friendly, concise one-pager insight for the section "{section_title}".
Tone: crisp, action-oriented, non-technical. Output 5 bullet points and a short 'Next 2 Weeks' action line.

Context (JSON-like, concise):
data_summary = {data_summary}
model_results = {model_results if model_results else {}}

Rules:
- No jargon, no formulas. Avoid caveats beyond what's necessary.
- Use plain business language (staffing, throughput, payer mix, risk).
- Start with the headline takeaway, then 4 supporting bullets.
- End with: "Next 2 Weeks: <three short actions>".
"""

    try:
        # New SDK path
        if hasattr(OPENAI_CLIENT, "chat") and hasattr(OPENAI_CLIENT.chat, "completions"):
            resp = OPENAI_CLIENT.chat.completions.create(
                model=model_name,
                temperature=temperature,
                messages=[
                    {"role": "system", "content": "You turn analytics into clear, executive insights for hospital leadership."},
                    {"role": "user", "content": prompt},
                ],
            )
            text = resp.choices[0].message.content.strip()
        else:
            # Legacy SDK path
            resp = OPENAI_CLIENT.ChatCompletion.create(
                model=model_name,
                temperature=temperature,
                messages=[
                    {"role": "system", "content": "You turn analytics into clear, executive insights for hospital leadership."},
                    {"role": "user", "content": prompt},
                ],
            )
            text = resp.choices[0].message["content"].strip()

        # Wrap for app styling
        return f"**{section_title} — Executive Insights ({model_label})**\n{text}"
    except Exception as e:
        st.info(f"LLM insights unavailable ({e}). Showing local summary.")
        return generate_business_summary(section_title, data_summary, model_results)

# ---------- Viz helper ----------
def plot_model_performance(results: dict, metric: str = "accuracy"):
    if not results:
        return None
    models = list(results.keys())
    scores = [results[m].get(metric, 0) for m in models]
    fig = go.Figure(data=[go.Bar(x=models, y=scores, text=[f"{s:.3f}" for s in scores], textposition="auto")])
    fig.update_layout(
        title=f"Model Performance Comparison — {metric.title()}",
        xaxis_title="Models",
        yaxis_title=metric.title(),
        height=400,
        showlegend=False,
    )
    return fig

# ---------- Code generator ----------
def generate_python_code(model_type: str, features: list, target: str) -> str:
    code = f"""
# Hospital Operations Predictive Model - {model_type}
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

df = pd.read_csv('hospital_data.csv')
features = {features}
target = '{target}'
X = df[features]
y = df[target]

numeric_features = X.select_dtypes(include=['int64','float64','int32','float32']).columns.tolist()
categorical_features = X.select_dtypes(include=['object']).columns.tolist()

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_features)
    ]
)
"""
    if model_type == "Random Forest":
        code += """
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
"""
    elif model_type == "Logistic Regression":
        code += """
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(max_iter=1000, random_state=42)
"""
    elif model_type == "XGBoost":
        code += """
from xgboost import XGBClassifier
model = XGBClassifier(random_state=42)
"""
    elif model_type == "SVM":
        code += """
from sklearn.svm import SVC
model = SVC(probability=True, random_state=42)
"""

    code += """
pipeline = Pipeline([('preprocessor', preprocessor), ('classifier', model)])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
pipeline.fit(X_train, y_train)

from sklearn.metrics import accuracy_score, classification_report
y_pred = pipeline.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.3f}")
print("\\nClassification Report:\\n", classification_report(y_test, y_pred))
"""
    return code

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
        <p>Predict future admission patterns to optimize staffing and resource allocation</p>
    </div>
    """,
        unsafe_allow_html=True,
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
        fig.update_layout(title="Daily Hospital Admissions Over Time", xaxis_title="Date", yaxis_title="Number of Admissions", height=400)
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
        <div class="metric-card"><h4>Variability</h4><h2 style="color:#fbbc04;">{std_daily:.1f}</h2></div>
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

    if st.button("🚀 Generate Forecasts", key="train_forecast", type="primary"):
        with st.spinner("Training forecasting models..."):
            results: Dict[str, Dict[str, float]] = {}
            forecasts: Dict[str, np.ndarray] = {}

            ts = daily_adm.set_index("date")["admissions"].astype(float)
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
            if "ARIMA" in selected_models and len(train_data) >= 3:
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
                    is_min = s == s.min()
                    return ["background-color:#d4edda;color:#155724" if v else "" for v in is_min]

                st.dataframe(res_df.style.apply(highlight_best, axis=0).format("{:.2f}"), use_container_width=True)

                st.subheader("🔮 Admission Forecasts")
                future_dates = pd.date_range(
                    start=pd.to_datetime(ts.index[-1]) + pd.Timedelta(days=1),
                    periods=forecast_days,
                    freq="D",
                )
                figf = go.Figure()
                figf.add_trace(go.Scatter(x=ts.index, y=ts.values, mode="lines", name="Historical"))
                palette = ["#34a853", "#ea4335", "#fbbc04", "#9aa0a6"]
                for i, (name, fc) in enumerate(forecasts.items()):
                    figf.add_trace(
                        go.Scatter(
                            x=future_dates,
                            y=fc,
                            mode="lines+markers",
                            name=f"{name} Forecast",
                            line=dict(dash="dash", width=2, color=palette[i % len(palette)]),
                        )
                    )
                figf.update_layout(
                    title="Admission Forecasts by Model",
                    height=500,
                    xaxis_title="Date",
                    yaxis_title="Predicted Admissions",
                )
                st.plotly_chart(figf, use_container_width=True)

                # exportable tables
                best_model = min(results.keys(), key=lambda x: results[x]["MAPE"])
                best_forecast = forecasts[best_model]
                staffing_df = pd.DataFrame(
                    {
                        "Date": future_dates,
                        "Predicted Admissions": best_forecast.astype(int),
                        "Day Shift Nurses": np.ceil(best_forecast / 8).astype(int),
                        "Night Shift Nurses": np.ceil(best_forecast / 12).astype(int),
                        "Estimated Cost ($)": (
                            np.ceil(best_forecast / 8) * 350 + np.ceil(best_forecast / 12) * 400
                        ).astype(int),
                    }
                )

                st.subheader("💼 Business Impact Analysis")
                avg_forecast = float(np.mean(best_forecast))
                peak_forecast = float(np.max(best_forecast))
                nurses_needed = int(np.ceil(peak_forecast / 8)) if peak_forecast else 0
                max_daily = daily_adm["admissions"].max() if len(daily_adm) else 0
                cbi1, cbi2, cbi3 = st.columns(3)
                with cbi1:
                    base = daily_adm["admissions"].mean() if len(daily_adm) else 0.0
                    st.metric(
                        "Avg Daily Admissions (Forecast)",
                        f"{avg_forecast:.1f}",
                        f"{((avg_forecast - base) / max(base,1e-9) * 100):+.1f}%",
                    )
                with cbi2:
                    st.metric("Peak Day Nursing Staff", f"{nurses_needed} nurses", f"{peak_forecast:.0f} admissions")
                with cbi3:
                    capacity_utilization = (avg_forecast / max_daily) * 100 if max_daily else 0.0
                    st.metric("Capacity Utilization", f"{capacity_utilization:.1f}%")

                st.subheader("👥 Staffing Recommendations")
                st.dataframe(staffing_df, use_container_width=True)

                st.download_button(
                    "📥 Download Forecast (CSV)",
                    data=pd.DataFrame({"date": future_dates, "forecast": best_forecast}).to_csv(index=False),
                    file_name=f"admissions_forecast_{best_model}.csv",
                    mime="text/csv",
                )
                st.download_button(
                    "📥 Download Staffing Plan (CSV)",
                    data=staffing_df.to_csv(index=False),
                    file_name="staffing_plan.csv",
                    mime="text/csv",
                )

                with st.expander("💻 View Implementation Code (Admissions)"):
                    code = f"""
# Admissions Forecasting — {best_model}
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA

forecast_days = {forecast_days}
# ts = daily_admissions.set_index('date')['admissions'].astype(float)
# model = ARIMA(ts, order=(1,1,1))
# fitted = model.fit()
# forecast = fitted.forecast(steps=forecast_days)

def staffing(pred):
    day_nurses = int(np.ceil(pred / 8))
    night_nurses = int(np.ceil(pred / 12))
    return day_nurses, night_nurses
"""
                    st.code(code, language="python")
            else:
                st.error("No models produced results. Try a shorter forecast horizon or ensure enough history.")

    # Business insights (now respects LLM toggle/model)
    if "forecasting" in st.session_state.model_results and st.session_state.model_results["forecasting"]:
        st.markdown("---")
        st.markdown("## 📋 Business Insights")
        forecasting_results = st.session_state.model_results["forecasting"]
        best_model = min(forecasting_results.keys(), key=lambda x: forecasting_results[x]["MAPE"])
        data_summary = {
            "total_records": len(filtered_df),
            "forecast_horizon": locals().get("forecast_days", 14),
            "average_daily_admissions": float(daily_adm["admissions"].mean()) if len(daily_adm) else 0.0,
            "best_model": best_model,
            "best_model_mape": float(forecasting_results[best_model]["MAPE"]),
        }
        insights = generate_executive_llm_insights("Admissions Forecasting", data_summary, forecasting_results)
        st.markdown(f'<div class="insights-panel">{insights}</div>', unsafe_allow_html=True)

# ===================== TAB 2: Revenue Analytics =====================
with tabs[1]:
    st.markdown(
        """
    <div class="analysis-section">
        <h3>💰 Revenue Pattern Analysis</h3>
        <p>Detect unusual billing patterns and identify potential revenue optimization opportunities</p>
    </div>
    """,
        unsafe_allow_html=True,
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
        sensitivity = st.slider("Sensitivity Level", 0.01, 0.1, 0.05, 0.01)
    with a3:
        features_for_anomaly = st.multiselect(
            "Analysis Features",
            ["billing_amount", "age", "length_of_stay", "admission_month", "admission_day_of_week"],
            default=["billing_amount", "length_of_stay"],
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

                st.subheader("🎯 Anomaly Detection Results")
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
                        investigation = total_flagged * 0.15
                        m1, m2, m3 = st.columns(3)
                        with m1:
                            st.metric("Total Flagged Amount", f"${total_flagged:,.0f}")
                        with m2:
                            st.metric("Investigation Priority", f"${investigation:,.0f}")
                        with m3:
                            st.metric("Cases for Review", f"{len(anomaly_idx)}")

                if results:
                    best_method = min(results.keys(), key=lambda x: abs(results[x]["anomaly_rate"] - 0.05))
                    with st.expander("💻 View Implementation Code (Revenue)"):
                        code = f"""
# Revenue Anomaly Detection — {best_method}
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest

df = pd.read_csv('hospital_revenue.csv')
features = {features_for_anomaly}
X = df[features].dropna()

model = IsolationForest(contamination={float(sensitivity)}, random_state=42)
pred = model.fit_predict(X)

unusual_idx = X.index[pred == -1]
flagged = df.loc[unusual_idx]

print(f"Anomalies: {{(pred == -1).sum()}} / {{len(X)}} ({{(pred == -1).mean():.2%}})")

if 'billing_amount' in flagged.columns:
    top10 = flagged.nlargest(10, 'billing_amount')[['date_of_admission','billing_amount','insurance_provider','medical_condition','hospital']]
    print(top10.to_string(index=False))
"""
                        st.code(code, language="python")
        else:
            st.warning("Please select at least one feature for analysis.")

    if "anomaly" in st.session_state.model_results and st.session_state.model_results["anomaly"]:
        st.markdown("---")
        st.markdown("## 📋 Business Insights")
        anomaly_results = st.session_state.model_results["anomaly"]
        best_method = min(anomaly_results.keys(), key=lambda x: abs(anomaly_results[x]["anomaly_rate"] - 0.05))
        data_summary = {
            "total_revenue": float(filtered_df["billing_amount"].sum()) if "billing_amount" in filtered_df.columns else 0.0,
            "avg_daily_revenue": float(daily_rev["revenue"].mean()) if len(daily_rev) else 0.0,
            "best_method": best_method,
            "anomalies_detected": int(anomaly_results[best_method]["anomalies_detected"]),
            "anomaly_rate": float(anomaly_results[best_method]["anomaly_rate"]),
        }
        insights = generate_executive_llm_insights("Revenue Pattern Analysis", data_summary, anomaly_results)
        st.markdown(f'<div class="insights-panel">{insights}</div>', unsafe_allow_html=True)

# ===================== TAB 3: Length of Stay Prediction =====================
with tabs[2]:
    st.markdown(
        """
    <div class="analysis-section">
        <h3>🛏️ Length of Stay Prediction</h3>
        <p>Predict patient stay duration to optimize bed management and discharge planning</p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    if "length_of_stay" not in filtered_df.columns:
        st.error("Length of stay data not available. Please check your dataset.")
    else:
        c1, c2 = st.columns([2, 1])
        with c1:
            st.subheader("📊 Length of Stay Distribution")
            fig = px.histogram(filtered_df, x="length_of_stay", nbins=30, title="Distribution of Length of Stay")
            fig.update_layout(height=400, xaxis_title="Length of Stay (days)", yaxis_title="Number of Patients")
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

        st.subheader("📈 LOS by Category")
        b1, b2 = st.columns(2)
        with b1:
            if "medical_condition" in filtered_df.columns:
                fig = px.box(filtered_df, x="medical_condition", y="length_of_stay", title="LOS by Medical Condition")
                fig.update_xaxes(tickangle=45)
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
        with b2:
            if "admission_type" in filtered_df.columns:
                fig = px.box(filtered_df, x="admission_type", y="length_of_stay", title="LOS by Admission Type")
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)

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

                        figm = plot_model_performance(results, metric=metric_to_plot)
                        if figm:
                            st.plotly_chart(figm, use_container_width=True)

                        st.subheader("🎯 Best Model Analysis")
                        best_model_name = max(results.keys(), key=lambda x: results[x].get(metric_to_plot, -np.inf))
                        best_model = models_trained[best_model_name]
                        m1, m2, m3 = st.columns(3)
                        if is_classification:
                            m1.metric("Best Model Accuracy", f"{results[best_model_name]['accuracy']:.1%}")
                            preds = best_model.predict(X_test)
                            short_rate = (
                                float((preds == "Short").sum()) / len(preds)
                                if hasattr(preds, "__len__") and len(preds) > 0
                                else 0.0
                            )
                            m2.metric("Predicted Short Stays", f"{short_rate:.1%}")
                            m3.metric("Classes", ", ".join(sorted(map(str, pd.Series(y_test).unique()))))
                        else:
                            ypb = best_model.predict(X_test)
                            avg_pred = float(np.mean(ypb)) if len(ypb) else 0.0
                            m1.metric("Best Model R²", f"{results[best_model_name]['r2_score']:.3f}")
                            m2.metric("Avg Predicted LOS", f"{avg_pred:.1f} days")
                            current_avg = float(filtered_df["length_of_stay"].mean())
                            m3.metric("Potential LOS Reduction", f"{max(0.0, current_avg - avg_pred):.1f} days")

                        st.subheader("🔮 Sample Predictions")
                        samples = []
                        nshow = min(10, len(X_test))
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

                        with st.expander("💻 View Implementation Code (LOS)"):
                            target_for_snippet = "los_category" if is_classification else "length_of_stay"
                            code = generate_python_code(best_model_name, selected_features, target_for_snippet)
                            st.code(code, language="python")
                    else:
                        st.error("All models failed to train. Check feature selection and data quality.")
            else:
                st.warning("Please select at least one feature for model training.")

        # Persistent business insights for LOS (LLM-aware)
        if "los" in st.session_state.model_results and st.session_state.model_results["los"]:
            st.markdown("---")
            st.markdown("## 📋 Business Insights")
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
            insights = generate_executive_llm_insights("Length of Stay Prediction", data_summary, los_results)
            st.markdown(f'<div class="insights-panel">{insights}</div>', unsafe_allow_html=True)

# ===================== TAB 4: Operational KPIs & Simulator =====================
with tabs[3]:
    st.markdown("""
    <div class="analysis-section">
        <h3>📊 Operational KPIs & What-If Staffing</h3>
        <p>Concise metrics leadership cares about + a quick scenario sandbox.</p>
    </div>
    """, unsafe_allow_html=True)

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
    <p>For questions or support, contact your analytics team</p>
</div>
""",
    unsafe_allow_html=True,
)
