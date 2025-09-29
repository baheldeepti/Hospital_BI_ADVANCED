# app.py — Hospital Operations Analytics Platform
# Redesigned with model mapping, LOS category definition, Exec vs Analyst insights, and inline code runner

import os
import io
import sys
import warnings
from datetime import datetime, date, timedelta
from typing import Dict, Optional, List, Tuple, Any

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

from sklearn.ensemble import RandomForestClassifier, IsolationForest, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge
from sklearn.svm import SVC, SVR
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, roc_auc_score,
    roc_curve, confusion_matrix, classification_report,
    mean_absolute_error, mean_squared_error, r2_score
)
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA

warnings.filterwarnings("ignore")

# ---------- Optional libs with graceful fallback ----------
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
        box-shadow: 0 2px 4px rgba(0,0,0,0.08);
        border: 1px solid #e9ecef;
        margin: 0.5rem 0;
        text-align: center;
    }
    .analysis-section {
        background: #fff;
        padding: 1.2rem 1.5rem;
        border-radius: 12px;
        border: 1px solid #e9ecef;
        margin: 1rem 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    }
    .insights-panel {
        background: #ffffff;
        padding: 1.2rem;
        border-radius: 12px;
        border: 1px solid #e9ecef;
        margin: 1rem 0;
    }
    .insight-section-title {
        font-weight: 700; font-size: 1.1rem; margin: 0.2rem 0 0.6rem 0;
    }
    .exec-box, .analyst-box {
        border-radius: 10px; padding: 1rem; height: 100%;
    }
    .exec-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: 1px solid rgba(255,255,255,0.3);
    }
    .analyst-box {
        background: #f8f9ff;
        border: 1px solid #dbe1ff;
        color: #1f1f1f;
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
    .badge {
        display:inline-block; padding: 2px 8px; font-size: 0.75rem; border-radius: 999px; margin-right: 6px;
        border: 1px solid #e1e5ef; background:#fff; color:#334155;
    }
    .badge.reg { border-color:#d1fae5; background:#ecfdf5; color:#065f46; }
    .badge.cls { border-color:#fee2e2; background:#fef2f2; color:#991b1b; }
    .mono { font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono","Courier New", monospace; }
    .small { font-size: 0.9rem; color:#444;}
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
st.session_state.setdefault("code_snippet_store", {})
st.session_state.setdefault("los_bins_info", "0–3=Short • 4–7=Medium • 8–14=Long • ≥15=Extended")

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
        # Guard: treat negatives as 0, clip extreme outliers
        df["length_of_stay"] = np.clip(los.fillna(0), 0, 365).astype(int)
        # Avoid all-zero bins by forcing 0 to 1 day minimum observable stay (business choice)
        df.loc[df["length_of_stay"] == 0, "length_of_stay"] = 1

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
        # Save bins/labels so we can show them to the user
        bins = [0, 3, 7, 14, float("inf")]
        labels = ["Short", "Medium", "Long", "Extended"]
        st.session_state["los_bins_info"] = "0–3=Short • 4–7=Medium • 8–14=Long • ≥15=Extended"
        df["los_category"] = pd.cut(
            df["length_of_stay"], bins=bins, labels=labels, include_lowest=True
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

# Make frames available to insight generator (default)
st.session_state["_filtered_df_for_summary"] = filtered_df

# ---------- Business Insights generator ----------
def _pct(x, d=1):
    try:
        return f"{100*float(x):.{d}f}%"
    except Exception:
        return "N/A"

def _num(x, d=1, money=False):
    try:
        f = float(x)
        if money:
            return f"${f:,.0f}"
        return f"{f:.{d}f}"
    except Exception:
        return "N/A"

def generate_business_summary_sections(section_title: str, data_summary: dict, metrics: Optional[dict] = None) -> Tuple[str, str]:
    """
    Returns (executive_summary_md, analyst_notes_md)
    """
    gdf = st.session_state.get("_filtered_df_for_summary")
    daily_adm = st.session_state.get("_daily_adm_for_summary")
    daily_rev = st.session_state.get("_daily_rev_for_summary")

    section = section_title.lower()
    exec_lines, analyst_lines = [], []

    if "admission" in section:
        if gdf is None or daily_adm is None or len(daily_adm) == 0:
            return "**Insufficient data**", "No daily admissions series available."
        # weekday uplift
        by_dow = gdf.groupby(gdf["date_of_admission"].dt.dayofweek).size().reindex(range(7), fill_value=0).values
        weekday_uplift = (by_dow[:5].mean() - by_dow[5:].mean()) / max(by_dow.mean(), 1e-9)
        x = np.arange(len(daily_adm))
        slope = np.polyfit(x, daily_adm["admissions"].values, 1)[0] if len(x) >= 2 else 0.0
        vol = daily_adm["admissions"].std()
        peak = daily_adm["admissions"].max()
        mean_adm = daily_adm["admissions"].mean()
        peak_to_mean = peak / max(mean_adm, 1e-9)

        best_line = ""
        if metrics:
            best = min(metrics, key=lambda k: metrics[k].get("MAPE", np.inf))
            best_line = f"Best model: **{best}** (MAPE {_num(metrics[best].get('MAPE', np.nan),1)})."

        exec_lines = [
            f"**Workload is {_pct(weekday_uplift)} higher on weekdays**; keep staffing aligned to weekday peaks.",
            f"Trend slope: {_num(slope,2)} admissions/day; peak/mean: {_num(peak_to_mean,2)}.",
            best_line,
            "Revisit forecast weekly; maintain a 2σ flex pool for spikes."
        ]
        analyst_lines = [
            f"Weekday uplift {_pct(weekday_uplift)} computed as (Mon–Fri vs Sat–Sun).",
            f"Volatility σ = {_num(vol,1)}; peak = {int(peak)}; mean = {_num(mean_adm,1)}.",
            "Check data gaps/holidays; consider robust loss if outliers dominate."
        ]

    elif "revenue" in section:
        if gdf is None or "billing_amount" not in gdf.columns or len(gdf) == 0:
            return "**Insufficient data**", "No revenue series available."
        bills = gdf["billing_amount"].sort_values(ascending=False).reset_index(drop=True)
        top_decile_cut = int(max(1, np.floor(0.1 * len(bills))))
        top_decile_share = bills.iloc[:top_decile_cut].sum() / max(bills.sum(), 1e-9)
        payer_mix = gdf.groupby("insurance_provider")["billing_amount"].sum().sort_values(ascending=False)
        top_payer = payer_mix.index[0] if len(payer_mix) else "N/A"
        top_payer_share = (payer_mix.iloc[0] / payer_mix.sum()) if len(payer_mix) else np.nan

        dr_vol, spike_days = np.nan, 0
        if daily_rev is not None and len(daily_rev) > 0:
            dr = daily_rev["revenue"].astype(float)
            dr_vol = dr.std()
            roll = dr.rolling(7, min_periods=3).mean()
            spikes = (dr > (roll * 1.25)).sum()
            spike_days = int(spikes)

        anomaly_line = ""
        if metrics:
            best = min(metrics.keys(), key=lambda k: abs(metrics[k].get("anomaly_rate", 1.0) - 0.05))
            det = metrics[best]
            anomaly_line = f"{best} flags **{int(det.get('anomalies_detected',0))}** cases ({_pct(det.get('anomaly_rate',0))})."

        exec_lines = [
            f"Top 10% encounters contribute **{_pct(top_decile_share)}** of revenue; audit high-value outliers.",
            f"Top payer **{top_payer}** drives **{_pct(top_payer_share)}**; consider negotiation strategy.",
            f"Detected {spike_days} spike days (7-day mean +25%).",
            anomaly_line
        ]
        analyst_lines = [
            f"Daily volatility σ = {_num(dr_vol,0, money=True)}; validate spikes against coding/charge lag.",
            "Tune contamination (IsolationForest) to 3–7% for stable alert volume.",
            "Cross-check anomalies by payer, service line, and attending physician."
        ]

    elif "stay" in section or "los" in section:
        if gdf is None or "length_of_stay" not in gdf.columns or len(gdf) == 0:
            return "**Insufficient data**", "LOS not present."
        los = gdf["length_of_stay"].astype(float)
        iqr = np.percentile(los, 75) - np.percentile(los, 25)
        cohort_lines = []
        for col in ["medical_condition", "admission_type", "hospital"]:
            if col in gdf.columns:
                m = gdf.groupby(col)["length_of_stay"].mean().sort_values(ascending=False)
                if len(m) >= 2:
                    head = m.head(1)
                    tail = m.tail(1)
                    cohort_lines.append(
                        f"{col.replace('_',' ').title()}: **{head.index[0]}** longer than **{tail.index[0]}** by {_num(head.iloc[0]-tail.iloc[0],1)} days."
                    )
        best_line = ""
        if metrics:
            if st.session_state.get("los_is_classification"):
                best = max(metrics.keys(), key=lambda k: metrics[k].get("accuracy", 0))
                best_line = f"Best classifier: **{best}** (accuracy {_num(metrics[best].get('accuracy',0)*100,1)}%)."
            else:
                best = max(metrics.keys(), key=lambda k: metrics[k].get("r2_score", -np.inf))
                best_line = f"Best regressor: **{best}** (R² {_num(metrics[best].get('r2_score',0),3)})."

        exec_lines = [
            f"LOS mean {_num(los.mean(),1)}d; median {_num(los.median(),1)}d; IQR {_num(iqr,1)}d.",
            best_line,
            "Standardize discharge pathways for highest-LOS cohorts; run variance-reduction on IQR tail."
        ]
        analyst_lines = cohort_lines[:3] + [
            "Confirm LOS bins are business-approved: " + (st.session_state.get("los_bins_info") or ""),
            "Check weekend discharge patterns; pilot early-AM discharge targets."
        ]
    else:
        exec_lines = ["No insights."]
        analyst_lines = ["No insights."]

    return "• " + "\n• ".join([l for l in exec_lines if l]), "• " + "\n• ".join([l for l in analyst_lines if l])


# ---------- Viz helper ----------
def plot_model_performance(results: dict, metric: str = "accuracy"):
    if not results:
        return None
    models = list(results.keys())
    scores = [results[m].get(metric, 0) for m in models]
    fig = go.Figure(
        data=[
            go.Bar(
                x=models,
                y=scores,
                text=[f"{s:.3f}" for s in scores],
                textposition="auto",
            )
        ]
    )
    fig.update_layout(
        title=f"Model Performance Comparison — {metric.title()}",
        xaxis_title="Models",
        yaxis_title=metric.title(),
        height=400,
        showlegend=False,
    )
    return fig

# ---------- Code snippet generator ----------
def generate_python_code(model_type: str, features: list, target: str, is_classification: bool) -> str:
    model_block = ""
    if is_classification:
        # classification choices
        if model_type == "Random Forest":
            model_block = "from sklearn.ensemble import RandomForestClassifier\nmodel = RandomForestClassifier(n_estimators=200, random_state=42)"
        elif model_type == "Logistic Regression":
            model_block = "from sklearn.linear_model import LogisticRegression\nmodel = LogisticRegression(max_iter=1000, random_state=42)"
        elif model_type == "XGBoost":
            model_block = "from xgboost import XGBClassifier\nmodel = XGBClassifier(random_state=42)"
        elif model_type == "SVM":
            model_block = "from sklearn.svm import SVC\nmodel = SVC(probability=True, random_state=42)"
        else:
            model_block = "from sklearn.ensemble import RandomForestClassifier\nmodel = RandomForestClassifier(n_estimators=200, random_state=42)"
    else:
        # regression choices
        if model_type == "Random Forest":
            model_block = "from sklearn.ensemble import RandomForestRegressor\nmodel = RandomForestRegressor(n_estimators=300, random_state=42)"
        elif model_type == "Linear Regression":
            model_block = "from sklearn.linear_model import LinearRegression\nmodel = LinearRegression()"
        elif model_type == "XGBoost":
            model_block = "from xgboost import XGBRegressor\nmodel = XGBRegressor(random_state=42)"
        elif model_type == "SVM":
            model_block = "from sklearn.svm import SVR\nmodel = SVR()"
        else:
            model_block = "from sklearn.linear_model import LinearRegression\nmodel = LinearRegression()"

    code = f"""# Hospital Operations Predictive Model — {model_type}
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
{"from sklearn.metrics import accuracy_score, classification_report" if is_classification else "from sklearn.metrics import mean_absolute_error, r2_score"}

# Load your data
df = pd.read_csv('hospital_data.csv')
features = {features}
target = '{target}'

X = df[features]
y = df[target]

numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
categorical_features = X.select_dtypes(include=['object','category']).columns.tolist()

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_features)
    ]
)

{model_block}

pipeline = Pipeline([('preprocessor', preprocessor), ('model', model)])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
pipeline.fit(X_train, y_train)

{"y_pred = pipeline.predict(X_test)\nprint(f'Accuracy: {accuracy_score(y_test, y_pred):.3f}')\nprint('\\nClassification Report:\\n', classification_report(y_test, y_pred))" if is_classification else "y_pred = pipeline.predict(X_test)\nmae = mean_absolute_error(y_test, y_pred)\nr2 = r2_score(y_test, y_pred)\nprint(f'MAE: {mae:.3f}  R^2: {r2:.3f}')"}

# Save model if desired:
# import joblib; joblib.dump(pipeline, 'model.joblib')
"""
    return code

def code_runner(editor_key: str, default_code: str):
    """
    Simple, sandbox-y code runner: captures stdout and shows output/errors.
    """
    st.session_state["code_snippet_store"].setdefault(editor_key, default_code)
    code = st.text_area("✏️ Editable Implementation Code (you can modify and run)", 
                        value=st.session_state["code_snippet_store"][editor_key],
                        height=260, key=f"code_editor_{editor_key}")
    colr1, colr2 = st.columns([1,3])
    with colr1:
        run_it = st.button("▶️ Run Code", key=f"run_{editor_key}", type="secondary")
    with colr2:
        st.caption("This executes the snippet in a restricted namespace with pandas/numpy/sklearn available.")
    if run_it:
        fake_stdout = io.StringIO()
        # Very restricted globals; expose only minimal safe items
        safe_globals = {
            "__builtins__": {
                "print": print, "range": range, "len": len, "min": min, "max": max, "sum": sum,
                "abs": abs, "round": round, "float": float, "int": int, "str": str, "list": list, "dict": dict, "set": set
            }
        }
        safe_locals = {}
        try:
            # capture stdout
            old_stdout = sys.stdout
            sys.stdout = fake_stdout
            exec(code, safe_globals, safe_locals)
        except Exception as e:
            sys.stdout = old_stdout
            st.error(f"Runtime error: {e}")
        else:
            sys.stdout = old_stdout
            out = fake_stdout.getvalue()
            if out.strip():
                st.text_area("🖨️ Output", value=out, height=180)
            else:
                st.info("Code ran without any printed output.")
        finally:
            fake_stdout.close()

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

    # Daily admissions
    daily_adm = filtered_df.groupby(filtered_df["date_of_admission"].dt.date).size().reset_index()
    daily_adm.columns = ["date", "admissions"]

    # Expose frames for Insights
    st.session_state["_daily_adm_for_summary"] = daily_adm
    st.session_state["_filtered_df_for_summary"] = filtered_df

    c1, c2 = st.columns([2, 1])
    with c1:
        st.subheader("📊 Current Admission Trends")
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=daily_adm["date"],
                y=daily_adm["admissions"],
                mode="lines+markers",
                name="Daily Admissions",
                line=dict(width=2),
            )
        )
        xnum = np.arange(len(daily_adm))
        if len(daily_adm) >= 2:
            z = np.polyfit(xnum, daily_adm["admissions"], 1)
            p = np.poly1d(z)
            fig.add_trace(
                go.Scatter(
                    x=daily_adm["date"],
                    y=p(xnum),
                    mode="lines",
                    name="Trend",
                    line=dict(dash="dash"),
                )
            )
        fig.update_layout(
            title="Daily Hospital Admissions Over Time",
            xaxis_title="Date",
            yaxis_title="Number of Admissions",
            height=400,
        )
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

                # downloads
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

                # Implementation code (Admissions) + runner
                with st.expander("💻 View / Edit / Run Implementation Code (Admissions)"):
                    code = f"""
# Admissions Forecasting — {best_model}
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA

# daily_admissions: dataframe with ['date','admissions']
forecast_days = {forecast_days}

# Example (ARIMA); replace with your selected model impl:
# ts = daily_admissions.set_index('date')['admissions'].astype(float)
# model = ARIMA(ts, order=(1,1,1))
# fitted = model.fit()
# forecast = fitted.forecast(steps=forecast_days)

def staffing(pred):
    day_nurses = int(np.ceil(pred / 8))
    night_nurses = int(np.ceil(pred / 12))
    return day_nurses, night_nurses
"""
                    code_runner("admissions_impl", code)
            else:
                st.error("No models produced results. Try a shorter forecast horizon or ensure enough history.")

    # Business insights (Exec vs Analyst)
    if "forecasting" in st.session_state.model_results and st.session_state.model_results["forecasting"]:
        st.markdown("---")
        st.markdown("## 📋 Business Insights")
        forecasting_results = st.session_state.model_results["forecasting"]
        data_summary = {
            "total_records": len(filtered_df),
            "forecast_horizon": locals().get("forecast_days", 14),
            "average_daily_admissions": float(daily_adm["admissions"].mean()) if len(daily_adm) else 0.0,
        }
        exec_md, analyst_md = generate_business_summary_sections("Admissions Forecasting", data_summary, forecasting_results)
        ce, ca = st.columns(2)
        with ce:
            st.markdown('<div class="insight-section-title">Executive Summary</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="exec-box">{exec_md}</div>', unsafe_allow_html=True)
        with ca:
            st.markdown('<div class="insight-section-title">Analyst Notes</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="analyst-box">{analyst_md}</div>', unsafe_allow_html=True)

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

        # Expose frames for Insights
        st.session_state["_daily_rev_for_summary"] = daily_rev
        st.session_state["_filtered_df_for_summary"] = filtered_df

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=daily_rev["date"],
                y=daily_rev["revenue"],
                mode="lines+markers",
                name="Daily Revenue",
                line=dict(width=2),
            )
        )
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
        sensitivity = st.slider("Sensitivity Level (contamination)", 0.01, 0.1, 0.05, 0.01)
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

                        # downloads
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

                # Implementation code (Revenue) + runner
                if results:
                    best_method = min(results.keys(), key=lambda x: abs(results[x]["anomaly_rate"] - 0.05))
                    with st.expander("💻 View / Edit / Run Implementation Code (Revenue)"):
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
                        code_runner("revenue_impl", code)
        else:
            st.warning("Please select at least one feature for analysis.")

    # Exec vs Analyst insights
    if "anomaly" in st.session_state.model_results and st.session_state.model_results["anomaly"]:
        st.markdown("---")
        st.markdown("## 📋 Business Insights")
        anomaly_results = st.session_state.model_results["anomaly"]
        data_summary = {
            "total_revenue": float(filtered_df["billing_amount"].sum()) if "billing_amount" in filtered_df.columns else 0.0,
            "avg_daily_revenue": float(daily_rev["revenue"].mean()) if len(daily_rev) else 0.0,
        }
        exec_md, analyst_md = generate_business_summary_sections("Revenue Pattern Analysis", data_summary, anomaly_results)
        ce, ca = st.columns(2)
        with ce:
            st.markdown('<div class="insight-section-title">Executive Summary</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="exec-box">{exec_md}</div>', unsafe_allow_html=True)
        with ca:
            st.markdown('<div class="insight-section-title">Analyst Notes</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="analyst-box">{analyst_md}</div>', unsafe_allow_html=True)

# ===================== TAB 3: Length of Stay Prediction =====================
with tabs[2]:
    st.markdown(
        """
    <div class="analysis-section">
        <h3>🛏️ Length of Stay Prediction</h3>
        <p>Predict patient stay duration (days) or category to optimize bed management and discharge planning</p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # LOS category definition block
    st.info(f"**LOS Categories:** {st.session_state.get('los_bins_info', 'Short/Medium/Long/Extended based on days')}", icon="ℹ️")

    # Model mapping badges
    st.markdown(
        """
<div class="small">
<span class="badge reg">Regression (LOS Days): Random Forest, Linear Regression, XGBoost*, SVM</span>
<span class="badge cls">Classification (LOS Category): Random Forest, Logistic Regression, XGBoost*, SVM</span>
</div>
<p class="small">*XGBoost options appear only if the library is installed.</p>
""",
        unsafe_allow_html=True
    )

    if "length_of_stay" not in filtered_df.columns:
        st.error("Length of stay data not available. Please check your dataset.")
    else:
        c1, c2 = st.columns([2, 1])
        with c1:
            st.subheader("📊 Length of Stay Distribution")
            fig = px.histogram(
                filtered_df, x="length_of_stay", nbins=30, title="Distribution of Length of Stay"
            )
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
            target_type = st.selectbox("Prediction Target", ["Length of Stay (Days)", "LOS Category (Short/Medium/Long/Extended)"])

            # Show model choices with correct mapping
            if target_type.startswith("Length of Stay (Days)"):
                default_models = ["Random Forest", "Linear Regression"]
                model_choices = ["Random Forest", "Linear Regression", "XGBoost", "SVM"] if HAS_XGBOOST else ["Random Forest", "Linear Regression", "SVM"]
            else:
                default_models = ["Random Forest", "Logistic Regression"]
                model_choices = ["Random Forest", "Logistic Regression", "XGBoost", "SVM"] if HAS_XGBOOST else ["Random Forest", "Logistic Regression", "SVM"]

            selected_models = st.multiselect("Select Models", model_choices, default=default_models)

        if st.button("🚀 Train Prediction Models", key="train_los", type="primary"):
            if selected_features:
                with st.spinner("Training LOS prediction models..."):
                    feature_data = filtered_df[selected_features + ["length_of_stay", "los_category"]].dropna()
                    if target_type.startswith("Length of Stay (Days)"):
                        target = "length_of_stay"
                        is_classification = False
                    else:
                        target = "los_category"
                        is_classification = True

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
                                    st.info(f"Skipping {model_name} (not a classifier / not available).")
                                    continue
                                final_step_name = "model"
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
                                    st.info(f"Skipping {model_name} (not a regressor / not available).")
                                    continue
                                final_step_name = "model"

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
                                float((pd.Series(preds) == "Short").sum()) / len(preds)
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

                        # Editable implementation code + runner
                        with st.expander("💻 View / Edit / Run Implementation Code (LOS)"):
                            tgt = "los_category" if is_classification else "length_of_stay"
                            code = generate_python_code(best_model_name, selected_features, tgt, is_classification)
                            code_runner("los_impl", code)
                    else:
                        st.error("All models failed to train. Check feature selection and data quality.")
            else:
                st.warning("Please select at least one feature for model training.")

        # Exec vs Analyst insights for LOS
        if "los" in st.session_state.model_results and st.session_state.model_results["los"]:
            st.markdown("---")
            st.markdown("## 📋 Business Insights")
            los_results = st.session_state.model_results["los"]
            data_summary = {
                "prediction_type": st.session_state.get("los_target_type", "Unknown"),
                "avg_los": float(st.session_state.get("los_avg_los") or 0.0),
                "total_patients": len(filtered_df),
            }
            exec_md, analyst_md = generate_business_summary_sections("Length of Stay Prediction", data_summary, los_results)
            ce, ca = st.columns(2)
            with ce:
                st.markdown('<div class="insight-section-title">Executive Summary</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="exec-box">{exec_md}</div>', unsafe_allow_html=True)
            with ca:
                st.markdown('<div class="insight-section-title">Analyst Notes</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="analyst-box">{analyst_md}</div>', unsafe_allow_html=True)

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

    # Pareto of revenue by payer
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

    # use recent admissions mean (last 28 days if possible)
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

    with st.expander("💻 View / Edit / Run Implementation Code (Simulator)"):
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
        code_runner("sim_impl", sim_code)

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
<div style="text-align:center;color:#666;padding:1rem 0 2rem 0;">
    <p>Hospital Operations Analytics Platform • Built with Streamlit & Python ML Libraries</p>
    <p class="small">Tip: In each tab, open the expander to edit & run the implementation code to fit your environment.</p>
</div>
""",
    unsafe_allow_html=True,
)
