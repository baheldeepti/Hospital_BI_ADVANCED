# app.py — Hospital Ops Studio: Control Tower (Executive-ready, no Data Chat)
# Sections:
#   1) Admissions Control  — forecasting → staffing targets (+ AI exec/analyst explainer)
#   2) Revenue Watch       — billing anomalies + root-cause (+ AI exec/analyst explainer)
#   3) LOS Planner         — LOS buckets + equity slices (+ AI exec/analyst explainer)
#
# Calibrated for Streamlit Cloud; Prophet/XGBoost are optional.

import os, json, textwrap
from datetime import datetime, date
from typing import Dict, List
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

# Optional (auto-detected). NOT required; app degrades gracefully.
try:
    from prophet import Prophet  # type: ignore
    _HAS_PROPHET = True
except Exception:
    _HAS_PROPHET = False

try:
    from xgboost import XGBClassifier  # type: ignore
    _HAS_XGB = True
except Exception:
    _HAS_XGB = False

from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX

from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, label_binarize
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, roc_auc_score, roc_curve
)

# ---------------- CONFIG / THEME ----------------
st.set_page_config(page_title="Hospital Ops Studio — Control Tower", layout="wide", page_icon="🏥")

RAW_URL = "https://raw.githubusercontent.com/baheldeepti/hospital-streamlit-app/main/modified_healthcare_dataset.csv"

st.markdown("""
<style>
:root{
  --pri:#0F4C81; --teal:#159E99; --ink:#1F2937; --sub:#6B7280;
  --ok:#14A38B; --warn:#F59E0B; --alert:#EF4444; --pin:#FF7A70;
}
.block-container{max-width:1500px;padding-top:12px}
h1,h2,h3{font-weight:700;color:var(--ink)}
a {color:var(--teal)}
.stButton>button{background:var(--pri);color:#fff;border-radius:10px}
.tag{display:inline-block;padding:.2rem .5rem;border-radius:.5rem;background:#F3F4F6;color:var(--ink);margin-right:.3rem}
.badge-ok{background:rgba(20,163,139,.12)}
.badge-warn{background:rgba(245,158,11,.12)}
.badge-alert{background:rgba(239,68,68,.12)}
.small{color:var(--sub);font-size:0.9rem}
hr{border-top:1px solid #eee}
</style>
""", unsafe_allow_html=True)

st.title("Hospital Ops Studio — Control Tower")
st.caption("Admissions Control • Revenue Watch • LOS Planner — decisions first, dashboards second")

# ---------------- DATA LOAD & FE ----------------
ICD_MAPPING = {
    'Infections': 'A49.9', 'Flu': 'J10.1', 'Cancer': 'C80.1', 'Asthma': 'J45.909',
    'Heart Disease': 'I51.9', "Alzheimer's": 'G30.9', 'Diabetes': 'E11.9', 'Obesity': 'E66.9'
}

@st.cache_data(show_spinner=False)
def load_data(url: str = RAW_URL) -> pd.DataFrame:
    df = pd.read_csv(url)
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
        "Age": "age",
    }
    for k,v in rename_map.items():
        if k in df.columns: df.rename(columns={k:v}, inplace=True)

    # Types
    if "admit_date" in df: df["admit_date"] = pd.to_datetime(df["admit_date"], errors="coerce")
    if "discharge_date" in df: df["discharge_date"] = pd.to_datetime(df["discharge_date"], errors="coerce")
    for c in ["length_of_stay","billing_amount","age"]:
        if c in df: df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=["admit_date"]).copy()

    # Feature Engineering
    df["dow"] = df["admit_date"].dt.weekday
    df["weekofyear"] = df["admit_date"].dt.isocalendar().week.astype(int)
    df["month"] = df["admit_date"].dt.month
    df["admit_day"] = df["admit_date"].dt.date
    if "admission_type" in df:
        df["is_emergency"] = df["admission_type"].str.contains("Emergency|ER|Urgent", case=False, na=False).astype(int)
    else:
        df["is_emergency"] = 0
    if "age" in df:
        bins = [-np.inf, 17, 40, 65, np.inf]
        labels = ["Child", "Adult", "Senior", "Elder"]
        df["age_group"] = pd.cut(df["age"], bins=bins, labels=labels)
    if "condition" in df:
        df["icd_code"] = df["condition"].map(ICD_MAPPING).fillna("R69")
    return df

df = load_data()

with st.expander("Data source (optional override)"):
    up = st.file_uploader("Upload CSV to override default", type=["csv"])
    if up is not None:
        try:
            df = pd.read_csv(up)
            st.success(f"Loaded {len[df):,} rows from upload.")
        except Exception as e:
            st.error(f"Failed to read uploaded CSV: {e}")

# ---------------- SHARED FILTERS ----------------
def render_filters(data: pd.DataFrame) -> pd.DataFrame:
    st.subheader("Filters")
    c1, c2, c3, c4 = st.columns(4)
    hosp = c1.multiselect("Hospitals", sorted(data["hospital"].dropna().unique()) if "hospital" in data else [])
    insurer = c2.multiselect("Insurers", sorted(data["insurer"].dropna().unique()) if "insurer" in data else [])
    adm = c3.multiselect("Admission Types", sorted(data["admission_type"].dropna().unique()) if "admission_type" in data else [])
    cond = c4.multiselect("Conditions", sorted(data["condition"].dropna().unique()) if "condition" in data else [])
    q = pd.Series(True, index=data.index)
    if hosp and "hospital" in data: q &= data["hospital"].isin(hosp)
    if insurer and "insurer" in data: q &= data["insurer"].isin(insurer)
    if adm and "admission_type" in data: q &= data["admission_type"].isin(adm)
    if cond and "condition" in data: q &= data["condition"].isin(cond)
    return data[q].copy()

fdf = render_filters(df)

# ---------------- UTIL: SERIES + METRICS ----------------
def build_timeseries(data: pd.DataFrame, metric: str, freq: str = "D") -> pd.DataFrame:
    if "admit_date" not in data: return pd.DataFrame(columns=["ds","y"])
    idx = data.set_index("admit_date")
    if metric == "intake":
        s = idx.assign(_one=1)["_one"].resample(freq).sum().fillna(0.0)
    elif metric == "billing_amount" and "billing_amount" in idx:
        s = idx["billing_amount"].resample(freq).sum().fillna(0.0)
    elif metric == "length_of_stay" and "length_of_stay" in idx:
        s = idx["length_of_stay"].resample(freq).mean().fillna(method="ffill").fillna(0.0)
    else:
        return pd.DataFrame(columns=["ds","y"])
    return pd.DataFrame({"ds": s.index, "y": s.values})

def ts_metrics(y_true, y_pred) -> dict:
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    n = min(len(y_true), len(y_pred))
    y_true = y_true[:n]; y_pred = y_pred[:n]
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred)**2)))
    denom = np.clip(np.abs(y_true), 1e-9, None)
    mape = float(np.mean(np.abs((y_true - y_pred)/denom))*100.0)
    return {"MAPE%": mape, "MAE": mae, "RMSE": rmse}

def style_lower_better(df: pd.DataFrame) -> str:
    ranks = df.rank(ascending=True, method="min")
    def bg(col, row):
        r = ranks.loc[row, col]; rmin, rmax = ranks[col].min(), ranks[col].max()
        if rmax == rmin: return "background-color: rgba(255,255,255,0.6)"
        pct = (r - rmin)/(rmax - rmin + 1e-9)
        red = int(255*pct); green = int(255*(1-pct)); blue = 200
        return f"background-color: rgba({red},{green},{blue},0.25)"
    styled = df.style.format({"MAPE%":"{:.2f}","MAE":"{:.2f}","RMSE":"{:.2f}"})
    for col in df.columns:
        styled = styled.apply(lambda s: [bg(col, s.name) for _ in s], axis=1)
    return styled.to_html()

def style_higher_better(df: pd.DataFrame) -> str:
    ranks = df.rank(ascending=False, method="min")
    def bg(col, row):
        r = ranks.loc[row, col]; rmin, rmax = ranks[col].min(), ranks[col].max()
        if rmax == rmin: return "background-color: rgba(255,255,255,0.6)"
        pct = (r - rmin)/(rmax - rmin + 1e-9)
        red = int(255*pct); green = int(255*(1-pct)); blue = 200
        return f"background-color: rgba({red},{green},{blue},0.25)"
    styled = df.style.format("{:.3f}")
    for col in df.columns:
        styled = styled.apply(lambda s: [bg(col, s.name) for _ in s], axis=1)
    return styled.to_html()

# ---------------- FORECAST MODELS (Admissions) ----------------
def _holt(train: pd.Series, horizon: int):
    try:
        m = ExponentialSmoothing(train, trend="add", seasonal="add", seasonal_periods=7).fit()
    except Exception:
        m = ExponentialSmoothing(train, trend="add").fit()
    return m.forecast(horizon)

def _sarimax(train: pd.Series, horizon: int):
    model = SARIMAX(train, order=(1,1,1), seasonal_order=(1,1,1,7),
                    enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
    return model.forecast(steps=horizon)

def backtest_series(s: pd.Series, horizon: int, folds: int, model_name: str):
    preds, trues = [], []
    if len(s) < (folds+1)*horizon + 10: return None
    for i in range(folds,0,-1):
        split = len(s) - i*horizon
        train = s.iloc[:split]; test = s.iloc[split:split+horizon]
        try:
            if model_name == "Holt-Winters":
                fc = _holt(train, horizon)
            elif model_name == "ARIMA (SARIMAX)":
                fc = _sarimax(train, horizon)
            elif model_name == "Prophet" and _HAS_PROPHET:
                dfp = train.reset_index().rename(columns={"index":"ds", train.name:"y"})
                dfp.columns = ["ds","y"]
                m = Prophet(weekly_seasonality=True, daily_seasonality=True, yearly_seasonality=True)
                m.fit(dfp)
                future = m.make_future_dataframe(periods=horizon, freq="D")
                fc = m.predict(future).tail(horizon)["yhat"].to_numpy()
            else:
                return None
        except Exception:
            return None
        preds.append(np.asarray(fc)); trues.append(test.to_numpy())
    return np.concatenate(preds), np.concatenate(trues)

def run_forecasts(ts: pd.DataFrame, horizon: int, models: List[str]) -> Dict[str, pd.DataFrame]:
    s = ts.set_index("ds")["y"].asfreq("D").fillna(method="ffill")
    out = {}
    for name in models:
        try:
            if name == "Holt-Winters":
                fc = _holt(s, horizon)
                out[name] = pd.DataFrame({"ds": pd.date_range(s.index.max()+pd.Timedelta(days=1), periods=horizon, freq="D"),
                                          "yhat": np.asarray(fc)})
            elif name == "ARIMA (SARIMAX)":
                fc = _sarimax(s, horizon)
                out[name] = pd.DataFrame({"ds": pd.date_range(s.index.max()+pd.Timedelta(days=1), periods=horizon, freq="D"),
                                          "yhat": np.asarray(fc)})
            elif name == "Prophet" and _HAS_PROPHET:
                m = Prophet(weekly_seasonality=True, daily_seasonality=True, yearly_seasonality=True)
                m.fit(ts.rename(columns={"ds":"ds","y":"y"}))
                future = m.make_future_dataframe(periods=horizon, freq="D")
                out[name] = m.predict(future).tail(horizon)[["ds","yhat"]]
        except Exception:
            pass
    return out

def compare_backtests(ts: pd.DataFrame, horizon: int, models: List[str]) -> Dict[str, Dict[str,float]]:
    s = ts.set_index("ds")["y"].asfreq("D").fillna(method="ffill")
    metrics = {}
    H = min(14, horizon)
    for name in models:
        bt = backtest_series(s, H, 3, name)
        if bt is not None:
            yhat, ytrue = bt
            metrics[name] = ts_metrics(ytrue, yhat)
    return metrics

def plot_ts(history: pd.DataFrame, forecasts: Dict[str, pd.DataFrame], title: str):
    fig = go.Figure()
    if not history.empty:
        fig.add_trace(go.Scatter(x=history["ds"], y=history["y"], name="History", mode="lines"))
    for name, dfp in forecasts.items():
        fig.add_trace(go.Scatter(x=dfp["ds"], y=dfp["yhat"], name=name, mode="lines"))
    fig.update_layout(title=title, height=420, margin=dict(l=10,r=10,b=10,t=50))
    st.plotly_chart(fig, use_container_width=True)

# ---------------- ANOMALY DETECTION (Billing) ----------------
def detect_anomalies(ts: pd.DataFrame, sensitivity: float = 3.0) -> pd.DataFrame:
    if ts.empty: return ts.assign(anomaly=False, score=0.0)
    df = ts.copy().sort_values("ds"); y = df["y"].values
    med = np.median(y); mad = np.median(np.abs(y - med)) + 1e-9
    rzs = 0.6745 * (y - med) / mad
    z_flag = np.abs(rzs) > sensitivity
    try:
        iso = IsolationForest(n_estimators=300, contamination="auto", random_state=42)
        iso_flag = (iso.fit_predict(y.reshape(-1,1)) == -1)
    except Exception:
        iso_flag = np.zeros_like(z_flag, dtype=bool)
    df["rzs"] = rzs
    df["anomaly"] = z_flag | iso_flag
    z_norm = (np.abs(rzs) - np.min(np.abs(rzs))) / (np.ptp(np.abs(rzs)) + 1e-9)
    df["score"] = (0.7 * z_norm) + (0.3 * iso_flag.astype(float))
    return df

def plot_anoms(an_df: pd.DataFrame, title: str):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=an_df["ds"], y=an_df["y"], mode="lines+markers", name="Billing"))
    flag = an_df[an_df["anomaly"]]
    if not flag.empty:
        fig.add_trace(go.Scatter(x=flag["ds"], y=flag["y"], mode="markers", name="Anomaly",
                                 marker=dict(size=10, symbol="x", color="#FF7A70")))
    fig.update_layout(title=title, height=420, margin=dict(l=10,r=10,b=10,t=50))
    st.plotly_chart(fig, use_container_width=True)

# ---------------- LOS HELPERS ----------------
def los_bucket(days: float) -> str:
    if pd.isna(days): return np.nan
    d = float(days)
    if d <= 5: return "Short"
    if d <= 15: return "Medium"
    if d <= 45: return "Long"
    return "Very Long"

def los_prep(data: pd.DataFrame):
    if "length_of_stay" not in data: return None
    d = data.dropna(subset=["length_of_stay"]).copy()
    d["los_bucket"] = d["length_of_stay"].apply(los_bucket)
    num_cols = [c for c in ["age","billing_amount","length_of_stay","dow","month","is_emergency"] if c in d.columns]
    cat_cols = [c for c in ["admission_type","insurer","hospital","condition","doctor","age_group","icd_code"] if c in d.columns]
    X = d[num_cols + cat_cols].copy(); y = d["los_bucket"].copy()
    for c in num_cols: X[c] = pd.to_numeric(X[c], errors="coerce").fillna(X[c].median())
    for c in cat_cols: X[c] = X[c].fillna("Unknown")
    return X, y, num_cols, cat_cols, d

# ---------------- OpenAI helpers (safe, no hardcoding) ----------------
def _get_openai_client():
    """
    Initialize OpenAI client from (priority):
      1) st.secrets["OPENAI_API_KEY"]
      2) os.environ["OPENAI_API_KEY"]
      3) st.session_state["OPENAI_API_KEY"] (manual entry)
    Validates key prefix and falls back gracefully.
    """
    key = (
        st.secrets.get("OPENAI_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
        or st.session_state.get("OPENAI_API_KEY")
    )
    if not key:
        st.info("OpenAI disabled: no OPENAI_API_KEY in Secrets/env/session.")
        return None
    key = key.strip()
    if not key.startswith("sk-"):
        st.error("OPENAI_API_KEY looks invalid (must start with 'sk-').")
        return None
    try:
        from openai import OpenAI  # lazy import for robustness
        return OpenAI(api_key=key)
    except Exception as e:
        st.error(f"Could not initialize OpenAI client: {e}")
        return None

def _model_fallback():
    preferred = (st.secrets.get("PREFERRED_OPENAI_MODEL","") or os.environ.get("PREFERRED_OPENAI_MODEL","")).strip()
    cascade = [m for m in [
        preferred or None,
        "gpt-4o",
        "gpt-4o-mini",
        "o3-mini",
        "gpt-4-turbo",
    ] if m]
    return cascade

def _try_chat(client, model, messages, temperature=0.2):
    try:
        rsp = client.chat.completions.create(model=model, messages=messages, temperature=temperature)
        return rsp.choices[0].message.content, None
    except Exception as e:
        return None, str(e)

# ---------------- AI WRITER (per section, Exec/Analyst) ----------------
def ai_write(section_title: str, payload: dict):
    client = _get_openai_client()

    col1, col2 = st.columns([1, 2])
    use_ai = col1.checkbox(
        f"Use AI for {section_title}",
        value=client is not None,
        key=f"ai_{section_title}"
    )
    analyst = col2.toggle("Analyst mode (more detail)", value=False, key=f"ai_mode_{section_title}")

    if use_ai and client:
        prompt = textwrap.dedent(f"""
        You are a product-analytics writer for a hospital ops platform.
        Audience: hospital leaders. Tone: crisp, actionable, non-technical.
        Write a concise {'executive' if not analyst else 'analyst-grade'} narrative for "{section_title}" (140–220 words).
        Use ONLY this JSON: {json.dumps(payload, default=str)[:6000]}
        Include:
          1) What the results imply for operations ("what / so-what / now-what"),
          2) A short "Performance & Use Case" table,
          3) 3–5 concrete recommendations (owners and suggested SLAs).
        """).strip()

        messages = [
            {"role": "system", "content": "Be precise, concise, and actionable. Avoid marketing fluff."},
            {"role": "user", "content": prompt},
        ]

        chosen = None
        last_err = None
        for model in _model_fallback():
            content, err = _try_chat(client, model, messages, temperature=0.2)
            if err is None and content:
                chosen = model
                st.markdown(content)
                with st.expander("AI diagnostics (model used)", expanded=False):
                    st.caption(f"Model: `{model}`")
                break
            last_err = err

        if chosen is None:
            st.error("OpenAI call failed on all fallback models.")
            if last_err:
                st.caption(f"Last error: {last_err}")
            use_ai = False

    if not use_ai:
        st.markdown(f"**Deterministic summary ({section_title})**")
        st.json(payload)
        st.caption(
            "ℹ️ Add OPENAI_API_KEY in Secrets and (optionally) set "
            "`PREFERRED_OPENAI_MODEL` to a model your project can use. "
            "The app will auto-fallback if a model is blocked."
        )

# ---------------- ACTION FOOTER + DECISION LOG ----------------
if "decision_log" not in st.session_state:
    st.session_state["decision_log"] = []

def action_footer(section: str):
    st.markdown("#### Action footer")
    c1, c2, c3, c4 = st.columns([1.2,1,1,1])
    owner = c1.selectbox("Owner", ["House Supervisor","Revenue Integrity","Case Mgmt","Unit Manager","Finance Lead"])
    decision = c2.selectbox("Decision", ["Promote","Hold","Tune","Investigate"])
    sla_date = c3.date_input("SLA Date", value=date.today())
    sla_time = c4.time_input("SLA Time", value=datetime.now().time())
    note = st.text_input("Notes (optional)")
    colA, colB = st.columns([1,1])
    if colA.button(f"Save to Decision Log ({section})"):
        st.session_state["decision_log"].append({
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "section": section, "owner": owner, "decision": decision,
            "sla": f"{sla_date} {sla_time}", "note": note
        })
        st.success("Saved to Decision Log.")
    # Always show a download button so execs can export any time
    df_log = pd.DataFrame(st.session_state["decision_log"])
    st.download_button(
        label="Download Decision Log (CSV)",
        data=df_log.to_csv(index=False).encode("utf-8"),
        file_name="decision_log.csv", mime="text/csv",
        key=f"dl_{section}"
    )

# ---------------- NAV ----------------
tabs = st.tabs(["📈 Admissions Control", "🧾 Revenue Watch", "🛏️ LOS Planner"])

# ===== 1) Admissions Control =====
with tabs[0]:
    st.subheader("📈 Admissions Control — Forecast → Staffing Targets")
    # Cohort Explorer
    c1, c2, c3, c4 = st.columns(4)
    cohort_dim = c1.selectbox("Cohort dimension", ["All","hospital","insurer","condition"])
    cohort_val = c2.selectbox(
        "Cohort value",
        ["(all)"] + (sorted(fdf[cohort_dim].dropna().unique().tolist()) if cohort_dim!="All" and cohort_dim in fdf else []),
    )
    agg = c3.selectbox("Aggregation", ["Daily","Weekly"], index=0)
    horizon = c4.slider("Forecast horizon (days)", 7, 90, 30)
    freq = "D" if agg=="Daily" else "W"

    # Filter to chosen cohort
    fdx = fdf.copy()
    if cohort_dim!="All" and cohort_dim in fdx and cohort_val and cohort_val!="(all)":
        fdx = fdx[fdx[cohort_dim]==cohort_val]

    ts = build_timeseries(fdx, metric="intake", freq=freq)
    if ts.empty:
        st.info("No admissions series for the current cohort/filters.")
    else:
        # Seasonality heatmap (weekly view)
        with st.expander("Seasonality calendar (avg admissions by week-of-year × weekday)"):
            s = fdx.set_index("admit_date").assign(_one=1)["_one"].resample("D").sum().fillna(0.0)
            cal = pd.DataFrame({"dow": s.index.weekday, "woy": s.index.isocalendar().week.values, "val": s.values})
            heat = cal.groupby(["woy","dow"])["val"].mean().reset_index()
            heat_pivot = heat.pivot(index="woy", columns="dow", values="val").fillna(0)
            fig = px.imshow(heat_pivot, aspect="auto", labels=dict(x="Weekday (0=Mon)", y="Week of Year", color="Avg admits"))
            fig.update_layout(height=360, margin=dict(l=10,r=10,b=10,t=30))
            st.plotly_chart(fig, use_container_width=True)

        # Scenario sliders
        sc1, sc2 = st.columns(2)
        flu_pct = sc1.slider("Flu surge scenario (±%)", -30, 50, 0, 5)
        weather_pct = sc2.slider("Weather impact (±%)", -20, 20, 0, 5)

        # Model selection
        candidates = ["Holt-Winters","ARIMA (SARIMAX)"] + (["Prophet"] if _HAS_PROPHET else [])
        chosen = st.multiselect("Models to compare", candidates, default=candidates)

        # Forecast + scenario adjustment
        fc = run_forecasts(ts, horizon=horizon, models=chosen)
        adj_factor = (100 + flu_pct + weather_pct) / 100.0
        fc_adj = {k: v.assign(yhat=v["yhat"] * adj_factor) for k,v in fc.items()}

        plot_ts(
            ts,
            fc_adj,
            f"Admissions Forecast — Cohort: {cohort_dim} = {cohort_val if cohort_dim!='All' else 'All'} (with scenario)"
        )

        # Backtest comparison
        metrics = compare_backtests(ts, horizon=horizon, models=chosen)
        if metrics:
            tbl = pd.DataFrame(metrics).T[["MAPE%","MAE","RMSE"]].sort_values("MAPE%")
            st.markdown("#### 📊 Model Performance (backtests)")
            st.markdown(style_lower_better(tbl), unsafe_allow_html=True)
            st.caption("Lower is better. Green → Red indicates rank per metric.")
        else:
            tbl = None
            st.info("Backtests unavailable (insufficient history).")

        # Staffing target heuristic (illustrative)
        st.markdown("#### Staffing targets (illustrative heuristic)")
        if fc_adj:
            first_model = list(fc_adj.keys())[0]
            daily_fc = fc_adj[first_model]["yhat"].to_numpy()
            rn_per_shift = np.ceil(daily_fc / 5.0).astype(int)  # illustrative heuristic
            targets = pd.DataFrame({
                "Date": pd.to_datetime(fc_adj[first_model]["ds"]).dt.date,
                "Expected Admissions": np.round(daily_fc, 1),
                "RN/Day": rn_per_shift, "RN/Evening": rn_per_shift, "RN/Night": rn_per_shift
            })
            st.dataframe(targets, use_container_width=True, hide_index=True)
        else:
            st.info("No forecast available to compute staffing targets.")

        # AI Explainer (per-section)
        ai_payload = {
            "cohort": {"dimension": cohort_dim, "value": cohort_val},
            "aggregation": agg,
            "horizon_days": horizon,
            "scenario": {"flu_pct": flu_pct, "weather_pct": weather_pct},
            "models_compared": chosen,
            "metrics_table": (tbl.to_dict() if tbl is not None else None)
        }
        st.markdown("---")
        ai_write("Admissions Control", ai_payload)

        # Action footer
        action_footer("Admissions Control")

# ===== 2) Revenue Watch =====
with tabs[1]:
    st.subheader("🧾 Revenue Watch — Anomalies → Cash Protection")
    c1, c2, c3 = st.columns(3)
    agg = c1.selectbox("Aggregation", ["Daily","Weekly"], index=0, key="bill_agg")
    sensitivity = c2.slider("Sensitivity (higher = fewer alerts)", 1.5, 5.0, 3.0, 0.1, key="bill_sens")
    baseline_weeks = c3.slider("Baseline window (weeks)", 2, 12, 4, 1, key="bill_base")
    freq = "D" if agg=="Daily" else "W"

    ts_bill = build_timeseries(fdf, metric="billing_amount", freq=freq)
    if ts_bill.empty:
        st.info("No billing series for the current filters.")
        an = None
    else:
        an = detect_anomalies(ts_bill, sensitivity)
        plot_anoms(an, "Billing Amount with Anomalies")

        # Root-cause drilldowns
        st.markdown("#### Root-cause explorer")
        dims = [d for d in ["insurer","hospital","condition","doctor"] if d in fdf.columns]
        if not dims:
            st.info("No categorical dimensions available for drilldown.")
        else:
            drill = st.selectbox("Group anomalies by", dims, index=0)

            if isinstance(an, pd.DataFrame) and not an.empty:
                # Build anomaly-day flags using .dt.date (fix for AttributeError)
                fdf_agg = fdf.set_index("admit_date")
                an_series = pd.to_datetime(an.loc[an["anomaly"], "ds"])
                an_days = set(an_series.dt.date.tolist())
                fdf_agg["is_anom_day"] = pd.Series(fdf_agg.index.date, index=fdf_agg.index).isin(an_days).astype(int)

                grp = fdf_agg.groupby(drill).agg(
                    billing_total=("billing_amount","sum"),
                    encounters=("billing_amount","count"),
                    anomaly_days=("is_anom_day","sum")
                ).reset_index().sort_values("anomaly_days", ascending=False)

                st.dataframe(grp, use_container_width=True)
            else:
                st.info("No anomalies available for drilldown in the current window.")

        # Recent anomaly facts (guarded)
        if isinstance(an, pd.DataFrame) and not an.empty:
            recent = an.tail(30)
            flagged = recent[recent["anomaly"]]
            with st.expander("Recent anomaly details (last 30 periods)"):
                if flagged.empty:
                    st.success("No recent anomalies.")
                else:
                    st.dataframe(
                        flagged[["ds","y","rzs","score"]].rename(columns={"ds":"When","y":"Value"}),
                        use_container_width=True
                    )
        else:
            st.info("No recent anomalies to display.")

    ai_payload = {
        "aggregation": agg,
        "sensitivity": sensitivity,
        "baseline_weeks": baseline_weeks,
        "recent_window": 30,
        "recent_points": int(len(an.tail(30))) if isinstance(an, pd.DataFrame) and not an.empty else 0,
        "recent_anomalies": int(an.tail(30)["anomaly"].sum()) if isinstance(an, pd.DataFrame) and not an.empty else 0,
        "max_severity_score": (
            float(an.tail(30).loc[an.tail(30)["anomaly"], "score"].max())
            if isinstance(an, pd.DataFrame) and not an.empty and an.tail(30)["anomaly"].any()
            else 0.0
        )
    }
    st.markdown("---")
    ai_write("Revenue Watch", ai_payload)
    action_footer("Revenue Watch")

# ===== 3) LOS Planner =====
with tabs[2]:
    st.subheader("🛏️ LOS Planner — Risk Buckets → Discharge Orchestration")
    prep = los_prep(fdf)
    if prep is None:
        st.info("`length_of_stay` not found.")
        perf = None
    else:
        X, y, num_cols, cat_cols, d_full = prep
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.25, random_state=42, stratify=y
            )
        except ValueError:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.25, random_state=42
            )

        # Proper preprocessing: scale nums, ONE-HOT encode cats (fix for LOS tab showing nothing)
        pre = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), [c for c in num_cols if c in X.columns]),
                ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), [c for c in cat_cols if c in X.columns]),
            ],
            remainder="drop"
        )

        models = {
            "Logistic Regression": Pipeline([("pre", pre), ("clf", LogisticRegression(max_iter=600, multi_class="multinomial"))]),
            "Random Forest": Pipeline([("pre", pre), ("clf", RandomForestClassifier(n_estimators=400, random_state=42))]),
        }
        if _HAS_XGB:
            models["XGBoost (optional)"] = Pipeline([("pre", pre), ("clf", XGBClassifier(
                n_estimators=500, learning_rate=0.05, max_depth=6, subsample=0.9, colsample_bytree=0.9,
                objective="multi:softprob", eval_metric="mlogloss", random_state=42
            ))])

        classes = sorted(y.dropna().unique().tolist())
        try:
            y_test_bin = label_binarize(y[y.index.isin(X_test.index)], classes=classes)
        except Exception:
            y_test_bin = None

        results = {}; roc_curves = {}
        for name, pipe in models.items():
            pipe.fit(X_train, y_train)
            y_pred = pipe.predict(X_test)
            try:
                y_proba = pipe.predict_proba(X_test)
            except Exception:
                y_proba = None

            acc = accuracy_score(y_test, y_pred)
            pr, rc, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="weighted", zero_division=0)
            auc = None; fprs={}; tprs={}
            if (y_test_bin is not None) and (y_proba is not None) and (y_proba.shape[1]==len(classes)):
                try:
                    auc = roc_auc_score(label_binarize(y_test, classes=classes), y_proba, average="weighted", multi_class="ovr")
                    for i, cls in enumerate(classes):
                        fpr, tpr, _ = roc_curve(label_binarize(y_test, classes=classes)[:,i], y_proba[:,i])
                        fprs[cls] = fpr; tprs[cls] = tpr
                except Exception:
                    pass
            results[name] = {"Accuracy":acc, "Precision":pr, "Recall":rc, "F1":f1, "ROC-AUC":auc}
            roc_curves[name] = (fprs, tprs)

        perf = pd.DataFrame(results).T
        for c in ["Accuracy","Precision","Recall","F1","ROC-AUC"]:
            if c not in perf: perf[c] = np.nan
        perf = perf[["Accuracy","Precision","Recall","F1","ROC-AUC"]]

        st.markdown("#### 📊 Model Performance (Classification)")
        st.markdown(style_higher_better(perf), unsafe_allow_html=True)
        st.caption("Higher is better. Green → Red indicates rank per metric.")

        top_model = perf["F1"].astype(float).idxmax()
        st.markdown(f"#### ROC Curves (one-vs-rest) — **{top_model}**")
        fprs, tprs = roc_curves[top_model]
        fig = go.Figure()
        for cls in classes:
            if cls in fprs and cls in tprs:
                fig.add_trace(go.Scatter(x=fprs[cls], y=tprs[cls], mode="lines", name=f"{top_model} — {cls}"))
        fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode="lines", name="Chance", line=dict(dash="dash")))
        fig.update_layout(height=420, xaxis_title="False Positive Rate", yaxis_title="True Positive Rate")
        st.plotly_chart(fig, use_container_width=True)

        # Equity slices (group performance by insurer or age_group)
        st.markdown("#### Equity slices")
        slice_dim_options = [d for d in ["insurer","age_group"] if d in d_full.columns]
        slice_dim = st.selectbox("Slice by", slice_dim_options, index=0 if "insurer" in slice_dim_options else 0) if slice_dim_options else None
        if slice_dim:
            best_pipe = models[top_model]
            X_test_idx = X_test.index
            slice_vals = d_full.loc[X_test_idx, slice_dim].fillna("Unknown")
            y_pred_best = best_pipe.predict(X_test)
            grp_rows = []
            for sv in sorted(slice_vals.unique().tolist()):
                mask = (slice_vals==sv)
                acc_g = accuracy_score(y_test[mask], y_pred_best[mask]) if mask.any() else np.nan
                grp_rows.append({"Group": str(sv), "Accuracy": acc_g, "N": int(mask.sum())})
            st.dataframe(pd.DataFrame(grp_rows).sort_values("Accuracy", ascending=False), use_container_width=True)

        # Per-section AI
        ai_payload = {
            "buckets": {"Short":"<=5","Medium":"6-15","Long":"16-45","Very Long":">45"},
            "metrics_table": perf.to_dict(),
            "top_model": top_model,
            "equity_slice": slice_dim
        }
        st.markdown("---")
        ai_write("LOS Planner", ai_payload)
        action_footer("LOS Planner")

# --------------- FOOTER: Decision Log quick peek ---------------
with st.expander("Decision Log (peek)"):
    df_log = pd.DataFrame(st.session_state["decision_log"])
    if df_log.empty:
        st.info("No decisions logged yet.")
    else:
        st.dataframe(df_log, use_container_width=True)
        st.download_button(
            label="Download Decision Log (CSV)",
            data=df_log.to_csv(index=False).encode("utf-8"),
            file_name="decision_log.csv", mime="text/csv",
            key="dl_footer"
        )
