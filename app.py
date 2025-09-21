
# app.py - Hospital Operations Cockpit (Single-Page)
# Author: Consolidated by ChatGPT
# Description: Single-page Streamlit app with horizontal navigation sections
# Features: Forecasting, anomaly detection, AI assistant, natural-language insights, recommendations

import os
import io
import json
import time
import textwrap
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# Forecasting fallbacks
try:
    from prophet import Prophet
    _HAS_PROPHET = True
except Exception:
    _HAS_PROPHET = False
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.ensemble import IsolationForest

# ========== CONFIG ==========
st.set_page_config(
    page_title="Hospital Operations Cockpit",
    layout="wide",
    page_icon="🩺",
)

APP_NAME = "Hospital Operations Cockpit"
DEFAULT_CSV_PATH = "https://raw.githubusercontent.com/baheldeepti/hospital-streamlit-app/main/modified_healthcare_dataset.csv"
 # preloaded in this environment

# ========== UTILITIES ==========

def _to_datetime_safe(s, fmt=None):
    try:
        return pd.to_datetime(s, format=fmt, errors="coerce")
    except Exception:
        return pd.to_datetime(s, errors="coerce")

@st.cache_data(show_spinner=False)
def load_data(csv_file: io.BytesIO | str | None = None) -> pd.DataFrame:
    """
    Load the healthcare CSV.
    Accepts uploaded file or default path. Ensures typed columns.
    """
    if csv_file is not None:
        df = pd.read_csv(csv_file)
    else:
        if not os.path.exists(DEFAULT_CSV_PATH):
            st.error("CSV not found. Upload a CSV to proceed.")
            st.stop()
        df = pd.read_csv(DEFAULT_CSV_PATH)

    # Normalize columns
    rename_map = {
        "Billing Amount": "billing_amount",
        "Date of Admission": "admit_date",
        "Discharge Date": "discharge_date",
        "Length of Stay": "length_of_stay",
        "Medical Condition": "condition",
        "Admission Type": "admission_type",
        "Insurance Provider": "insurer",
        "Hospital": "hospital",
        "Doctor": "doctor",
        "Test Results": "test_results",
    }
    for k,v in rename_map.items():
        if k in df.columns:
            df.rename(columns={k:v}, inplace=True)

    # Types
    if "admit_date" in df.columns:
        df["admit_date"] = _to_datetime_safe(df["admit_date"])
    if "discharge_date" in df.columns:
        df["discharge_date"] = _to_datetime_safe(df["discharge_date"])
    if "length_of_stay" in df.columns:
        df["length_of_stay"] = pd.to_numeric(df["length_of_stay"], errors="coerce")
    if "billing_amount" in df.columns:
        df["billing_amount"] = pd.to_numeric(df["billing_amount"], errors="coerce")

    # Create helpful engineered fields
    if "admit_date" in df.columns:
        df["admit_day"] = df["admit_date"].dt.date
        df["admit_week"] = df["admit_date"].dt.to_period("W").apply(lambda r: r.start_time.date())

    # Drop rows with no admission date or billing
    if "admit_date" in df.columns:
        df = df[~df["admit_date"].isna()].copy()
    return df


def smart_filters(df: pd.DataFrame) -> pd.DataFrame:
    """Render filter widgets and return filtered dataframe."""
    with st.container():
        cols = st.columns(5)
        hospital = cols[0].multiselect("Hospitals", sorted(df["hospital"].dropna().unique()) if "hospital" in df else [])
        insurer  = cols[1].multiselect("Insurers", sorted(df["insurer"].dropna().unique()) if "insurer" in df else [])
        adm_type = cols[2].multiselect("Admission Types", sorted(df["admission_type"].dropna().unique()) if "admission_type" in df else [])
        condition = cols[3].multiselect("Conditions", sorted(df["condition"].dropna().unique()) if "condition" in df else [])
        date_range = cols[4].date_input("Date range", [])

    q = pd.Series([True]*len(df))
    if hospital and "hospital" in df: q &= df["hospital"].isin(hospital)
    if insurer and "insurer" in df: q &= df["insurer"].isin(insurer)
    if adm_type and "admission_type" in df: q &= df["admission_type"].isin(adm_type)
    if condition and "condition" in df: q &= df["condition"].isin(condition)
    if date_range:
        start = pd.to_datetime(date_range[0]) if len(date_range) > 0 else None
        end   = pd.to_datetime(date_range[-1]) if len(date_range) > 0 else None
        if start is not None: q &= df["admit_date"] >= start
        if end is not None: q &= df["admit_date"] <= end + pd.Timedelta(days=1)

    return df[q].copy()


def aggregate_timeseries(df: pd.DataFrame, metric: str, freq: str = "D") -> pd.DataFrame:
    """
    Build a timeseries for a metric:
    - 'billing_amount' → sum
    - 'length_of_stay' → mean
    - 'intake' → count of admissions
    Returns columns: ['ds', 'y']
    """
    if "admit_date" not in df.columns:
        st.warning("Admission dates not found in the dataset.")
        return pd.DataFrame(columns=["ds","y"])

    if metric == "billing_amount" and "billing_amount" in df.columns:
        s = df.set_index("admit_date")["billing_amount"].resample(freq).sum().fillna(0.0)
    elif metric == "length_of_stay" and "length_of_stay" in df.columns:
        s = df.set_index("admit_date")["length_of_stay"].resample(freq).mean().fillna(method="ffill").fillna(0.0)
    elif metric == "intake":
        s = df.set_index("admit_date").assign(_one=1)["_one"].resample(freq).sum().fillna(0.0)
    else:
        st.error(f"Unknown or missing metric: {metric}")
        return pd.DataFrame(columns=["ds","y"])

    return pd.DataFrame({"ds": s.index, "y": s.values})


def forecast_series(ts: pd.DataFrame, horizon: int = 30, seasonality: str = "auto") -> dict:
    """
    Forecast the series, returning:
      dict(yhat, yhat_lower, yhat_upper, history, model_name)
    Uses Prophet if available, else Holt-Winters.
    """
    ts = ts.dropna().copy()
    if len(ts) < 10:
        return {"error": "Not enough data to forecast.", "history": ts}

    if _HAS_PROPHET:
        m = Prophet(
            daily_seasonality=True if seasonality in ("auto","daily") else False,
            weekly_seasonality=True,
            yearly_seasonality=True,
            interval_width=0.8,
        )
        m.fit(ts.rename(columns={"ds":"ds", "y":"y"}))
        future = m.make_future_dataframe(periods=horizon, freq="D")
        fcst = m.predict(future)[["ds","yhat","yhat_lower","yhat_upper"]]
        merged = ts.merge(fcst, on="ds", how="left")
        return {"forecast": fcst, "history": ts, "model_name": "Prophet"}
    else:
        # Holt-Winters fallback
        s = ts.set_index("ds")["y"].asfreq("D").fillna(method="ffill")
        try:
            model = ExponentialSmoothing(s, trend="add", seasonal="add", seasonal_periods=7).fit()
        except Exception:
            model = ExponentialSmoothing(s, trend="add").fit()
        future_index = pd.date_range(s.index.max() + pd.Timedelta(days=1), periods=horizon, freq="D")
        fc = model.forecast(horizon)
        fc.index = future_index
        fcst = pd.DataFrame({"ds": fc.index, "yhat": fc.values})
        return {"forecast": fcst, "history": ts, "model_name": "Holt-Winters"}


def detect_anomalies(ts: pd.DataFrame, sensitivity: float = 3.0) -> pd.DataFrame:
    """
    Simple robust anomaly detector using rolling median & MAD,
    plus IsolationForest vote.
    """
    if ts.empty:
        return ts.assign(anomaly=False)

    df = ts.copy()
    df = df.sort_values("ds")
    y = df["y"].values

    # Robust z-scores
    med = np.median(y)
    mad = np.median(np.abs(y - med)) + 1e-9
    rzs = 0.6745 * (y - med) / mad
    z_flags = np.abs(rzs) > sensitivity

    # IsolationForest
    try:
        iso = IsolationForest(n_estimators=200, contamination="auto", random_state=42)
        iso_flags = iso.fit_predict(y.reshape(-1,1)) == -1
    except Exception:
        iso_flags = np.zeros_like(z_flags, dtype=bool)

    df["anomaly"] = z_flags | iso_flags
    df["rzs"] = rzs
    return df


def narrative_for_metric(metric: str) -> str:
    mapping = {
        "billing_amount": "total billing (revenue capture and leakage)",
        "length_of_stay": "average length of stay (bed availability and throughput)",
        "intake": "patient intake (admissions load and staffing)",
    }
    return mapping.get(metric, metric)


def natural_language_summary(metric: str, hist: pd.DataFrame, fc_dict: dict, anomalies: pd.DataFrame) -> str:
    last7 = hist.tail(7)["y"].mean() if not hist.empty else np.nan
    last30 = hist.tail(30)["y"].mean() if len(hist) >= 30 else np.nan
    trend = "rising" if last7 > last30 else "falling" if last7 < last30 else "flat"
    anom_ct = int(anomalies["anomaly"].sum()) if "anomaly" in anomalies else 0
    model = fc_dict.get("model_name","Model")

    return (
        f"For {narrative_for_metric(metric)}, the recent trend appears **{trend}** "
        f"(7-day avg vs 30-day avg). We detected **{anom_ct}** unusual points in the history. "
        f"The current forecast uses **{model}** with an 80% confidence interval; "
        f"use it to plan staffing, discharge targets, and claims review bandwidth."
    )


def cost_saving_recommendations(metric: str, anomalies: pd.DataFrame) -> list[str]:
    tips = []
    if metric == "billing_amount":
        if anomalies["anomaly"].sum() > 0:
            tips.append("Prioritize medical-necessity documentation review for payers with recent spikes.")
        tips.append("Batch claims pre-checks before submission to reduce preventable denials.")
        tips.append("Create a weekly denials huddle with finance + coding to address root causes.")
    elif metric == "length_of_stay":
        tips.append("Add 2–3 late-day transport slots to accelerate evening discharges.")
        tips.append("Ensure weekend PT/OT coverage for high-LOS DRGs to avoid Monday pile-ups.")
        tips.append("Automate next-step checklists (imaging, meds-to-beds, family pickup).")
    elif metric == "intake":
        tips.append("Use nurse float pools to cover predictable intake surges by weekday and hour.")
        tips.append("Coordinate elective admissions to match forecast discharge windows.")
    return tips


def plot_timeseries(hist: pd.DataFrame, fcst: pd.DataFrame | None, anomalies: pd.DataFrame | None, title: str):
    fig = go.Figure()
    if not hist.empty:
        fig.add_trace(go.Scatter(x=hist["ds"], y=hist["y"], mode="lines+markers", name="History"))
    if fcst is not None and not fcst.empty and "yhat" in fcst.columns:
        fig.add_trace(go.Scatter(x=fcst["ds"], y=fcst["yhat"], mode="lines", name="Forecast"))
        if "yhat_lower" in fcst.columns and "yhat_upper" in fcst.columns:
            fig.add_traces([
                go.Scatter(
                    x=pd.concat([fcst["ds"], fcst["ds"][::-1]]),
                    y=pd.concat([fcst["yhat_upper"], fcst["yhat_lower"][::-1]]),
                    fill="toself", opacity=0.2, line=dict(width=0), name="80% CI"
                )
            ])
    if anomalies is not None and not anomalies.empty:
        an = anomalies[anomalies["anomaly"]]
        if not an.empty:
            fig.add_trace(go.Scatter(x=an["ds"], y=an["y"], mode="markers", name="Anomaly", marker=dict(size=10, symbol="x")))
    fig.update_layout(title=title, height=420, margin=dict(l=10,r=10,b=10,t=40))
    st.plotly_chart(fig, use_container_width=True)


# ========== SIDEBAR & HEADER ==========
st.title(APP_NAME)
st.caption("Single-page cockpit: forecasts, anomalies, AI explanations, and cost-saving recommendations.")

with st.sidebar:
    st.header("Data")
    uploaded = st.file_uploader("Upload CSV (optional)", type=["csv"])
    df = load_data(uploaded)
    st.success(f"Loaded {len(df):,} rows.")
    st.markdown("**Tip:** Filter first, then explore forecasts.")

    st.divider()
    st.header("Settings")
    horizon = st.slider("Forecast horizon (days)", 7, 90, 30)
    sensitivity = st.slider("Anomaly sensitivity (higher = fewer alerts)", 1.5, 5.0, 3.0, 0.1)
    freq = st.selectbox("Aggregation", ["Daily","Weekly"], index=0)
    agg_freq = "D" if freq == "Daily" else "W"

# Global filters row
st.subheader("Smart Filters")
fdf = smart_filters(df)

# Horizontal navigation using tabs
tab_labels = ["Today’s Picture", "Forecasts & Alerts", "Explore with AI", "Cost Tips"]
t_overview, t_forecast, t_ai, t_cost = st.tabs(tab_labels)

# ========== TAB 1: OVERVIEW ==========
with t_overview:
    st.markdown("### Today’s Picture")
    # KPI tiles
    c1, c2, c3 = st.columns(3)
    # Intake today
    if "admit_day" in fdf.columns:
        today = pd.Timestamp.today().date()
        intake_today = int((fdf["admit_day"] == today).sum())
    else:
        intake_today = 0
    c1.metric("Patients Admitted Today", value=intake_today)

    # Avg LOS (7d)
    if "length_of_stay" in fdf.columns:
        avg_los_7d = fdf[fdf["admit_date"] >= pd.Timestamp.today() - pd.Timedelta(days=7)]["length_of_stay"].mean()
        c2.metric("Avg LOS (7d)", value=f"{avg_los_7d:.2f}" if pd.notna(avg_los_7d) else "—")
    else:
        c2.metric("Avg LOS (7d)", value="—")

    # Billing sum (7d)
    if "billing_amount" in fdf.columns:
        bill_7d = fdf[fdf["admit_date"] >= pd.Timestamp.today() - pd.Timedelta(days=7)]["billing_amount"].sum()
        c3.metric("Billing (7d)", value=f"${bill_7d:,.0f}")
    else:
        c3.metric("Billing (7d)", value="—")

    st.markdown("#### Recent Admissions")
    if "admit_date" in fdf.columns:
        by_day = fdf.set_index("admit_date").assign(_one=1)["_one"].resample("D").sum()
        st.bar_chart(by_day)

# ========== TAB 2: FORECASTS & ALERTS ==========
with t_forecast:
    st.markdown("### Forecasts & Alerts")
    metric_map = {
        "Billing": "billing_amount",
        "Length of Stay": "length_of_stay",
        "Patient Intake": "intake"
    }
    pick = st.selectbox("Choose what to forecast", list(metric_map.keys()))
    metric = metric_map[pick]

    ts = aggregate_timeseries(fdf, metric=metric, freq=agg_freq)
    if ts.empty:
        st.info("No time series available for the selected filters/metric.")
    else:
        fc = forecast_series(ts, horizon=horizon)
        if "error" in fc:
            st.warning(fc["error"])
            fcst = None
        else:
            fcst = fc.get("forecast")

        an = detect_anomalies(ts, sensitivity=sensitivity)
        plot_timeseries(ts, fcst, an, f"{pick}: History & Forecast")

        # Alerts
        with st.expander("Realtime Alerts (last 14 periods)"):
            recent = an.tail(14)
            an_ct = int(recent["anomaly"].sum())
            if an_ct > 0:
                st.warning(f"⚠️ Detected {an_ct} unusual movement(s) recently.")
                st.dataframe(recent[recent["anomaly"]][["ds","y","rzs"]].rename(columns={"ds":"When","y":"Value","rzs":"Robust z"}))
            else:
                st.success("No anomalies flagged in the recent window.")

        # Narrative
        st.markdown("#### What this means")
        st.write(natural_language_summary(metric, ts, fc, an))

# ========== TAB 3: EXPLORE WITH AI ==========
with t_ai:
    st.markdown("### Explore with AI")
    st.caption("Ask natural-language questions. The assistant can also emit Python snippets to reproduce answers.")

    # IMPORTANT: Users should set OPENAI_API_KEY via Streamlit secrets or environment variable.
    openai_key_present = bool(st.secrets.get("OPENAI_API_KEY") or os.environ.get("OPENAI_API_KEY"))
    user_q = st.text_area("Ask a question (e.g., 'Which week looks anomalous for billing and why?')", height=100)

    if st.button("Ask the Assistant", disabled=not user_q):
        if not openai_key_present:
            st.error("OpenAI API key not found. Add OPENAI_API_KEY to Streamlit secrets or environment variables.")
        else:
            # Minimal local reasoning before LLM: create a short context slice
            context_rows = fdf.sample(min(500, len(fdf)), random_state=42) if len(fdf) > 0 else fdf
            # describe dataset
            desc = {
                "columns": list(context_rows.columns),
                "size": len(context_rows),
                "example_rows": context_rows.head(3).to_dict(orient="records"),
            }
            system_prompt = textwrap.dedent(f"""
            You are a hospital operations analytics assistant. Be concise and clear.
            Explain insights in everyday language, then provide (if useful) a short pandas code snippet
            that reproduces the answer using a dataframe named df.
            Avoid any PHI. The dataframe columns available are: {desc["columns"]}.
            """).strip()
            # Use OpenAI API if available
            try:
                from openai import OpenAI
                client = OpenAI(api_key=st.secrets.get("OPENAI_API_KEY", os.environ.get("OPENAI_API_KEY")))
                msg = [
                    {"role":"system","content": system_prompt},
                    {"role":"user","content": f"Question: {user_q}\nShort dataset description: {json.dumps(desc)[:2000]} ..."}
                ]
                rsp = client.chat.completions.create(model="gpt-4o-mini", messages=msg, temperature=0.2)
                answer = rsp.choices[0].message.content
            except Exception as e:
                answer = f"(Local fallback) Based on the data, here's how I would explore it:\n\n- Look at weekly sums for billing.\n- Flag weeks beyond ±3 robust-z.\n- Correlate with admission type & insurer mix.\n\n(Enable OpenAI to get natural-language answers)\n\nError: {e}"

            st.markdown("#### Assistant Answer")
            st.write(answer)

# ========== TAB 4: COST TIPS ==========
with t_cost:
    st.markdown("### Cost Optimization Tips")
    # Provide recommendations for each key metric
    for label, m in [("Billing","billing_amount"), ("Length of Stay","length_of_stay"), ("Patient Intake","intake")]:
        ts_m = aggregate_timeseries(fdf, metric=m, freq=agg_freq)
        an_m = detect_anomalies(ts_m, sensitivity=sensitivity) if not ts_m.empty else pd.DataFrame(columns=["anomaly"])
        tips = cost_saving_recommendations(m, an_m if not an_m.empty else pd.DataFrame({"anomaly":[False]}))
        with st.expander(f"{label}: Recommendations"):
            for t in tips:
                st.markdown(f"- {t}")

st.caption("© 2025 CareFlux demo. This app avoids PHI, provides explainable forecasts, anomaly alerts, and business-ready narratives.")
