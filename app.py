# app.py — Hospital Ops Studio (Enterprise Single Page)
# Product design: PM-first clarity + DS rigor
# Sections: 📈 Forecast Studio | 🛏️ LOS Risk & Planning | 🧠 AI Summary
# Data: Loads default CSV from GitHub raw URL; no sidebar; full-width layouts
# Notes:
# - Forecast Studio: ARIMA, Prophet, Holt-Winters with backtesting + comparison
# - Billing Anomalies: IsolationForest + Robust-MAD with confidence-ish scoring
# - LOS Risk & Planning: bucketed LOS + LogReg/RandomForest/XGBoost; full metrics + ROC curves
# - AI Summary: OpenAI (if key provided) to generate data-driven narrative & recommendations
# - Graceful degradation: missing packages or insufficient data won’t crash UX

import os
import json
import textwrap
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

# Optional deps (graceful degradation)
try:
    from prophet import Prophet
    _HAS_PROPHET = True
except Exception:
    _HAS_PROPHET = False

try:
    import pmdarima as pm  # ARIMA
    _HAS_PM = True
except Exception:
    _HAS_PM = False

from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error,
    accuracy_score, precision_recall_fscore_support,
    roc_auc_score, roc_curve, classification_report
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

try:
    from xgboost import XGBClassifier
    _HAS_XGB = True
except Exception:
    _HAS_XGB = False

# -------------------- PAGE SETUP --------------------
st.set_page_config(page_title="Hospital Ops Studio", layout="wide", page_icon="🏥")

RAW_URL = "https://raw.githubusercontent.com/baheldeepti/hospital-streamlit-app/main/modified_healthcare_dataset.csv"

# Minimal CSS polish (no sidebar, clean paddings, enterprise feel)
st.markdown("""
<style>
.block-container {padding-top: 1.2rem; padding-bottom: 1rem; max-width: 1500px;}
h1,h2,h3 { font-weight: 700; }
.kpi { padding:10px 12px; border-radius:12px; border:1px solid #eaeaea; }
.tbl-note { color:#6b7280; font-size:0.9rem; }
.metric-better { background: rgba(46, 204, 113, 0.12); }
.metric-worse { background: rgba(231, 76, 60, 0.12); }
hr { border-top: 1px solid #eee; }
</style>
""", unsafe_allow_html=True)

st.title("Hospital Ops Studio")
st.caption("Forecasts • Billing Anomalies • LOS Risk — designed like a product, not a demo")

# -------------------- DATA LOADING --------------------
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
    for k, v in rename_map.items():
        if k in df.columns:
            df.rename(columns={k: v}, inplace=True)

    # Types
    if "admit_date" in df:
        df["admit_date"] = pd.to_datetime(df["admit_date"], errors="coerce")
    if "discharge_date" in df:
        df["discharge_date"] = pd.to_datetime(df["discharge_date"], errors="coerce")
    for c in ["length_of_stay", "billing_amount", "age"]:
        if c in df:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=["admit_date"]).copy()

    # Feature Engineering (seasonality, emergency flags, age groups, ICD)
    if "admit_date" in df:
        df["dow"] = df["admit_date"].dt.weekday    # 0=Mon
        df["week"] = df["admit_date"].dt.isocalendar().week.astype(int)
        df["month"] = df["admit_date"].dt.month
        df["admit_day"] = df["admit_date"].dt.date

    # Emergency admissions flag (simple heuristic on Admission Type if present)
    if "admission_type" in df:
        df["is_emergency"] = df["admission_type"].str.contains("Emergency|ER|Urgent", case=False, na=False).astype(int)
    else:
        df["is_emergency"] = 0

    # Age groups
    if "age" in df:
        bins = [-np.inf, 17, 40, 65, np.inf]
        labels = ["Child", "Adult", "Senior", "Elder"]
        df["age_group"] = pd.cut(df["age"], bins=bins, labels=labels)

    # ICD mapping from condition
    if "condition" in df:
        df["icd_code"] = df["condition"].map(ICD_MAPPING).fillna("R69")  # R69 Unknown/unspecified

    return df

df = load_data()

# Optional override (small expander, not a sidebar)
with st.expander("Data source (optional override)"):
    up = st.file_uploader("Upload CSV to override default", type=["csv"])
    if up is not None:
        try:
            df = pd.read_csv(up)
            st.success(f"Loaded {len(df):,} rows from upload.")
        except Exception as e:
            st.error(f"Failed to read uploaded CSV: {e}")

# -------------------- GLOBAL FILTERS --------------------
def render_filters(data: pd.DataFrame) -> pd.DataFrame:
    st.subheader("Filters")
    c1, c2, c3, c4 = st.columns(4)
    hospital = c1.multiselect("Hospitals", sorted(data["hospital"].dropna().unique()) if "hospital" in data else [])
    insurer = c2.multiselect("Insurers", sorted(data["insurer"].dropna().unique()) if "insurer" in data else [])
    adm     = c3.multiselect("Admission Types", sorted(data["admission_type"].dropna().unique()) if "admission_type" in data else [])
    cond    = c4.multiselect("Conditions", sorted(data["condition"].dropna().unique()) if "condition" in data else [])
    q = pd.Series(True, index=data.index)
    if hospital and "hospital" in data: q &= data["hospital"].isin(hospital)
    if insurer and "insurer" in data:   q &= data["insurer"].isin(insurer)
    if adm and "admission_type" in data: q &= data["admission_type"].isin(adm)
    if cond and "condition" in data:    q &= data["condition"].isin(cond)
    return data[q].copy()

fdf = render_filters(df)

# -------------------- HELPERS: TIME SERIES --------------------
def build_timeseries(data: pd.DataFrame, metric: str, freq: str = "D") -> pd.DataFrame:
    if "admit_date" not in data:
        return pd.DataFrame(columns=["ds", "y"])
    idx = data.set_index("admit_date")
    if metric == "billing_amount" and "billing_amount" in idx:
        s = idx["billing_amount"].resample(freq).sum().fillna(0.0)
    elif metric == "intake":
        s = idx.assign(_one=1)["_one"].resample(freq).sum().fillna(0.0)
    elif metric == "length_of_stay" and "length_of_stay" in idx:
        s = idx["length_of_stay"].resample(freq).mean().fillna(method="ffill").fillna(0.0)
    else:
        return pd.DataFrame(columns=["ds", "y"])
    return pd.DataFrame({"ds": s.index, "y": s.values})

def ts_metrics(y_true, y_pred) -> dict:
    """Return MAE, RMSE, MAPE%. Compatible with older sklearn versions."""
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()

    # Align lengths just in case
    n = min(len(y_true), len(y_pred))
    y_true = y_true[:n]
    y_pred = y_pred[:n]

    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    denom = np.clip(np.abs(y_true), 1e-9, None)
    mape = float(np.mean(np.abs((y_true - y_pred) / denom)) * 100.0)

    return {"MAE": mae, "RMSE": rmse, "MAPE%": mape}


def backtest_prophet(ts: pd.DataFrame, horizon: int = 14, folds: int = 3):
    if not _HAS_PROPHET: return None
    dfp = ts.dropna().rename(columns={"ds":"ds","y":"y"})
    if len(dfp) < (folds+1)*horizon + 10: return None
    preds, trues = [], []
    split_points = [len(dfp) - (i+1)*horizon for i in range(folds)][::-1]
    for sp in split_points:
        train = dfp.iloc[:sp]
        test  = dfp.iloc[sp:sp+horizon]
        m = Prophet(weekly_seasonality=True, daily_seasonality=True, yearly_seasonality=True)
        m.fit(train)
        future = m.make_future_dataframe(periods=horizon, freq="D")
        fc = m.predict(future).tail(horizon)[["ds","yhat"]]
        preds.append(fc["yhat"].values); trues.append(test["y"].values)
    return np.concatenate(preds), np.concatenate(trues)

def backtest_holt(ts: pd.DataFrame, horizon: int = 14, folds: int = 3):
    dfh = ts.copy().set_index("ds")["y"].asfreq("D").fillna(method="ffill")
    if len(dfh) < (folds+1)*horizon + 10: return None
    preds, trues = [], []
    for i in range(folds, 0, -1):
        split = len(dfh) - i*horizon
        train = dfh.iloc[:split]
        test  = dfh.iloc[split: split+horizon]
        try:
            model = ExponentialSmoothing(train, trend="add", seasonal="add", seasonal_periods=7).fit()
        except Exception:
            model = ExponentialSmoothing(train, trend="add").fit()
        fc = model.forecast(horizon)
        preds.append(fc.values); trues.append(test.values)
    return np.concatenate(preds), np.concatenate(trues)

def backtest_arima(ts: pd.DataFrame, horizon: int = 14, folds: int = 3):
    if not _HAS_PM: return None
    s = ts.set_index("ds")["y"].asfreq("D").fillna(method="ffill")
    if len(s) < (folds+1)*horizon + 10: return None
    preds, trues = [], []
    for i in range(folds, 0, -1):
        split = len(s) - i*horizon
        train = s.iloc[:split]
        test  = s.iloc[split: split+horizon]
        try:
            model = pm.auto_arima(train, seasonal=True, m=7, error_action="ignore", suppress_warnings=True)
            fc = model.predict(n_periods=horizon)
            preds.append(fc); trues.append(test.values)
        except Exception:
            return None
    return np.concatenate(preds), np.concatenate(trues)

def forecast_full(ts: pd.DataFrame, horizon: int = 30) -> Dict[str, pd.DataFrame]:
    """Forecast from full history for visualization (Prophet/Holt/ARIMA if available). Returns dict of model -> df(ds,yhat)."""
    out = {}

    # Prophet
    if _HAS_PROPHET:
        try:
            m = Prophet(weekly_seasonality=True, daily_seasonality=True, yearly_seasonality=True)
            m.fit(ts.rename(columns={"ds":"ds","y":"y"}))
            future = m.make_future_dataframe(periods=horizon, freq="D")
            fc = m.predict(future)[["ds","yhat"]].tail(horizon)
            out["Prophet"] = fc
        except Exception:
            pass

    # Holt-Winters
    try:
        s = ts.set_index("ds")["y"].asfreq("D").fillna(method="ffill")
        model = ExponentialSmoothing(s, trend="add", seasonal="add", seasonal_periods=7).fit()
        fc = model.forecast(horizon)
        out["Holt-Winters"] = pd.DataFrame({"ds": pd.date_range(s.index.max()+pd.Timedelta(days=1), periods=horizon, freq="D"),
                                            "yhat": fc.values})
    except Exception:
        pass

    # ARIMA
    if _HAS_PM:
        try:
            s = ts.set_index("ds")["y"].asfreq("D").fillna(method="ffill")
            ar = pm.auto_arima(s, seasonal=True, m=7, error_action="ignore", suppress_warnings=True)
            fc = ar.predict(n_periods=horizon)
            out["ARIMA"] = pd.DataFrame({"ds": pd.date_range(s.index.max()+pd.Timedelta(days=1), periods=horizon, freq="D"),
                                         "yhat": fc})
        except Exception:
            pass

    return out

def plot_ts(history: pd.DataFrame, models: Dict[str, pd.DataFrame], title: str):
    fig = go.Figure()
    if not history.empty:
        fig.add_trace(go.Scatter(x=history["ds"], y=history["y"], name="History", mode="lines"))
    for name, dfp in models.items():
        if dfp is not None and not dfp.empty:
            fig.add_trace(go.Scatter(x=dfp["ds"], y=dfp["yhat"], name=f"{name}", mode="lines"))
    fig.update_layout(title=title, height=420, margin=dict(l=10,r=10,b=10,t=50))
    st.plotly_chart(fig, use_container_width=True)

# -------------------- BILLING ANOMALIES --------------------
def detect_anomalies(ts: pd.DataFrame, sensitivity: float = 3.0) -> pd.DataFrame:
    """Robust MAD z-scores + IsolationForest vote, returns DataFrame with anomaly, score."""
    if ts.empty: return ts.assign(anomaly=False, score=0.0)

    df = ts.copy().sort_values("ds")
    y = df["y"].values

    med = np.median(y)
    mad = np.median(np.abs(y - med)) + 1e-9
    rzs = 0.6745 * (y - med) / mad
    z_flags = np.abs(rzs) > sensitivity

    # IsolationForest
    try:
        iso = IsolationForest(n_estimators=300, contamination="auto", random_state=42)
        iso_scores = -iso.fit_predict(y.reshape(-1,1))  # 2 = anomaly, 0 = normal (after negation)
        iso_flag = iso_scores > 1  # crude mapping
    except Exception:
        iso_scores = np.zeros_like(y)
        iso_flag = np.zeros_like(z_flags, dtype=bool)

    df["rzs"] = rzs
    df["iso_score"] = iso_scores
    df["anomaly"] = z_flags | iso_flag
    # Confidence-ish score (0-1): normalized |rzs|
    z_norm = (np.abs(rzs) - np.min(np.abs(rzs))) / (np.ptp(np.abs(rzs)) + 1e-9)
    df["score"] = (0.6 * z_norm) + (0.4 * (iso_scores / (iso_scores.max() + 1e-9)))
    return df

def plot_anomalies(an_df: pd.DataFrame, title: str):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=an_df["ds"], y=an_df["y"], mode="lines+markers", name="Billing"))
    an = an_df[an_df["anomaly"]]
    if not an.empty:
        fig.add_trace(go.Scatter(x=an["ds"], y=an["y"], mode="markers", name="Anomaly",
                                 marker=dict(size=10, symbol="x")))
    fig.update_layout(title=title, height=420, margin=dict(l=10,r=10,b=10,t=50))
    st.plotly_chart(fig, use_container_width=True)

# -------------------- LOS BUCKETING --------------------
def los_bucket(days: float) -> str:
    if pd.isna(days): return np.nan
    d = float(days)
    if d <= 5: return "Short"
    if d <= 15: return "Medium"
    if d <= 45: return "Long"
    return "Very Long"

def los_data_prep(data: pd.DataFrame):
    if "length_of_stay" not in data:
        return None
    d = data.dropna(subset=["length_of_stay"]).copy()
    d["los_bucket"] = d["length_of_stay"].apply(los_bucket)
    num_cols = [c for c in ["age","billing_amount","length_of_stay","dow","month"] if c in d.columns]
    cat_cols = [c for c in ["admission_type","insurer","hospital","condition","doctor","age_group","icd_code"] if c in d.columns]
    X = d[num_cols + cat_cols].copy()
    y = d["los_bucket"].copy()

    for c in num_cols:
        if c in X:
            X[c] = pd.to_numeric(X[c], errors="coerce").fillna(X[c].median())
    for c in cat_cols:
        if c in X:
            X[c] = X[c].fillna("Unknown")

    return X, y, num_cols, cat_cols

# -------------------- NAVIGATION --------------------
tabs = st.tabs(["📈 Forecast Studio", "🧾 Billing Anomalies", "🛏️ LOS Risk & Planning", "🧠 AI Summary"])

# ========== TAB 1: FORECAST STUDIO ==========
with tabs[0]:
    st.subheader("📈 Forecast Studio")
    cc1, cc2, cc3 = st.columns(3)
    metric_label = cc1.selectbox("Metric", ["Patient Intake", "Billing", "Length of Stay (avg)"], index=0)
    agg = cc2.selectbox("Aggregation", ["Daily","Weekly"], index=0)
    horizon = cc3.slider("Forecast horizon (days)", 7, 90, 30)
    freq = "D" if agg == "Daily" else "W"
    metric_map = {"Patient Intake": "intake", "Billing": "billing_amount", "Length of Stay (avg)": "length_of_stay"}
    metric = metric_map[metric_label]

    ts = build_timeseries(fdf, metric=metric, freq=freq)
    if ts.empty:
        st.info("No data for this metric and filter selection.")
    else:
        # Forecast (full history → future horizon)
        forecasts = forecast_full(ts, horizon=horizon)
        plot_ts(ts, forecasts, title=f"{metric_label}: History + Competing Forecasts")

        # Backtests (3 folds @ min(14,horizon))
        st.markdown("#### 📊 Model Performance Comparison (Backtest)")
        metrics_table: Dict[str, Dict[str, float]] = {}
        H = min(horizon, 14)

        if _HAS_PROPHET:
            p = backtest_prophet(ts, horizon=H, folds=3)
            if p is not None:
                yhat, ytrue = p
                metrics_table["Prophet"] = ts_metrics(ytrue, yhat)
        h = backtest_holt(ts, horizon=H, folds=3)
        if h is not None:
            yhat, ytrue = h
            metrics_table["Holt-Winters"] = ts_metrics(ytrue, yhat)
        if _HAS_PM:
            a = backtest_arima(ts, horizon=H, folds=3)
            if a is not None:
                yhat, ytrue = a
                metrics_table["ARIMA"] = ts_metrics(ytrue, yhat)

        if metrics_table:
            tbl = pd.DataFrame(metrics_table).T[["MAPE%","MAE","RMSE"]]
            tbl = tbl.sort_values("MAPE%")
            # Conditional formatting (lower better): render HTML
            def style_table(df_):
                df = df_.copy()
                # ranks
                ranks = df.rank(ascending=True, method="min")
                # color gradient green(best)->red(worst)
                def cell(col, val):
                    r = ranks.loc[df.index, col]
                    rmin, rmax = r.min(), r.max()
                    if rmax == rmin:
                        color = "rgba(255,255,255,0.6)"
                    else:
                        pct = (r.loc[val.name] - rmin) / (rmax - rmin + 1e-9)
                        red = int(255*pct); green = int(255*(1-pct)); blue = 200
                        color = f"rgba({red},{green},{blue},0.25)"
                    return f"background-color:{color}"
                styled = df.style.format({"MAPE%":"{:.2f}","MAE":"{:.2f}","RMSE":"{:.2f}"})
                for col in df.columns:
                    styled = styled.apply(lambda s: [cell(col, s) for _ in s], axis=1)
                return styled
            html = style_table(tbl).to_html()
            st.markdown(html, unsafe_allow_html=True)
            st.caption("Lower is better. Color runs green (best) → red (worst).")

            # Keep in session for AI Summary
            st.session_state["forecast_metrics"] = tbl.to_dict()
        else:
            st.info("Backtests unavailable (insufficient history or optional packages missing).")
            st.session_state["forecast_metrics"] = None

# ========== TAB 2: BILLING ANOMALIES ==========
with tabs[1]:
    st.subheader("🧾 Billing Anomalies")
    cc1, cc2 = st.columns(2)
    sensitivity = cc1.slider("Sensitivity (higher = fewer alerts)", 1.5, 5.0, 3.0, 0.1)
    agg = cc2.selectbox("Aggregation", ["Daily","Weekly"], index=0)
    freq = "D" if agg == "Daily" else "W"

    ts_bill = build_timeseries(fdf, metric="billing_amount", freq=freq)
    if ts_bill.empty:
        st.info("No billing series available for the current filters.")
        st.session_state["anomaly_summary"] = None
    else:
        an = detect_anomalies(ts_bill, sensitivity=sensitivity)
        plot_anomalies(an, title="Billing Amount with Anomalies")

        recent = an.tail(30)
        flagged = recent[recent["anomaly"]]
        with st.expander("Recent anomaly details (last 30 periods)"):
            if flagged.empty:
                st.success("No anomalies in the recent window.")
            else:
                st.dataframe(flagged[["ds","y","rzs","iso_score","score"]].rename(
                    columns={"ds":"When","y":"Value"}), use_container_width=True)
        # Keep for AI Summary
        st.session_state["anomaly_summary"] = {
            "recent_points": int(len(recent)),
            "recent_anomalies": int(flagged.shape[0]),
            "max_score": float(flagged["score"].max()) if not flagged.empty else 0.0
        }

# ========== TAB 3: LOS RISK & PLANNING ==========
with tabs[2]:
    st.subheader("🛏️ LOS Risk & Planning")
    prep = los_data_prep(fdf)
    if prep is None:
        st.info("Length of stay column not found.")
        st.session_state["los_metrics"] = None
    else:
        X, y, num_cols, cat_cols = prep
        # Split
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.25, random_state=42, stratify=y
            )
        except ValueError:
            st.warning("Not enough class balance to perform a stratified split. Using random split.")
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.25, random_state=42
            )

        pre = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), [c for c in num_cols if c in X.columns]),
                ("cat", "passthrough", [c for c in cat_cols if c in X.columns])
            ],
            remainder="drop"
        )

        models = {
            "LogisticRegression": Pipeline([("pre", pre),
                                            ("clf", LogisticRegression(max_iter=400, multi_class="multinomial"))]),
            "RandomForest": Pipeline([("pre", pre),
                                      ("clf", RandomForestClassifier(n_estimators=400, random_state=42))]),
        }
        if _HAS_XGB:
            models["XGBoost"] = Pipeline([("pre", pre),
                                          ("clf", XGBClassifier(
                                              n_estimators=500, learning_rate=0.05, max_depth=6,
                                              subsample=0.9, colsample_bytree=0.9, objective="multi:softprob",
                                              eval_metric="mlogloss", random_state=42))])

        results = {}
        roc_curves = {}
        classes = sorted(y.unique())
        try:
            y_test_bin = label_binarize(y_test, classes=classes)
        except Exception:
            y_test_bin = None

        for name, pipe in models.items():
            pipe.fit(X_train, y_train)
            y_pred = pipe.predict(X_test)
            try:
                y_proba = pipe.predict_proba(X_test)
            except Exception:
                y_proba = None

            acc = accuracy_score(y_test, y_pred)
            pr, rc, f1, _ = precision_recall_fscore_support(
                y_test, y_pred, average="weighted", zero_division=0
            )
            auc = None
            fprs, tprs = {}, {}
            if (y_test_bin is not None) and (y_proba is not None) and (y_proba.shape[1] == len(classes)):
                try:
                    auc = roc_auc_score(y_test_bin, y_proba, average="weighted", multi_class="ovr")
                    for i, cls in enumerate(classes):
                        fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_proba[:, i])
                        fprs[cls] = fpr; tprs[cls] = tpr
                except Exception:
                    pass

            results[name] = {"Accuracy": acc, "Precision": pr, "Recall": rc, "F1": f1, "ROC-AUC": auc}
            roc_curves[name] = (fprs, tprs)

        perf = pd.DataFrame(results).T
        order_cols = ["Accuracy","Precision","Recall","F1","ROC-AUC"]
        for c in order_cols:
            if c not in perf: perf[c] = np.nan
        perf = perf[order_cols]

        # Conditional formatting (higher better): HTML table
        def style_higher_better(df_):
            df = df_.copy()
            ranks = df.rank(ascending=False, method="min")
            def style_cell(col, row_name):
                r = ranks.loc[row_name, col]
                rmin, rmax = ranks[col].min(), ranks[col].max()
                if rmax == rmin:
                    return "background-color: rgba(255,255,255,0.6)"
                pct = (r - rmin) / (rmax - rmin + 1e-9)
                red = int(255*pct); green = int(255*(1-pct)); blue = 200
                return f"background-color: rgba({red},{green},{blue},0.25)"
            styled = df.style.format("{:.3f}")
            for col in df.columns:
                styled = styled.apply(lambda s: [style_cell(col, s.name) for _ in s], axis=1)
            return styled

        st.markdown("#### 📊 Model Performance Comparison (Classification)")
        html = style_higher_better(perf).to_html()
        st.markdown(html, unsafe_allow_html=True)
        st.caption("Higher is better. Color runs green (best) → red (worst).")

        # Per-class breakdown (best F1)
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

        # Per-class metrics table
        st.markdown("#### Per-class metrics")
        # Recompute per-class metrics for the winning model
        best_pipe = models[top_model]
        y_pred_best = best_pipe.predict(X_test)
        rep = classification_report(y_test, y_pred_best, output_dict=True, zero_division=0)
        cls_rows = {k:v for k,v in rep.items() if k in classes}
        per_class_df = pd.DataFrame(cls_rows).T[["precision","recall","f1-score","support"]]
        st.dataframe(per_class_df, use_container_width=True)

        # Save for AI
        st.session_state["los_metrics"] = perf.to_dict()
        st.session_state["los_best_model"] = top_model

# ========== TAB 4: AI SUMMARY ==========
with tabs[3]:
    st.subheader("🧠 AI Summary")
    st.caption("Narrative, Performance & Use Case Table, and Recommendations from real metrics (not hardcoded).")

    forecast_metrics = st.session_state.get("forecast_metrics", None)
    anomaly_summary  = st.session_state.get("anomaly_summary", None)
    los_metrics      = st.session_state.get("los_metrics", None)
    los_best_model   = st.session_state.get("los_best_model", None)

    openai_key = st.secrets.get("OPENAI_API_KEY", os.environ.get("OPENAI_API_KEY", ""))
    use_ai = st.checkbox("Use OpenAI (gpt-4o-mini) for narrative & recommendations", value=bool(openai_key))

    payload = {
        "forecast_metrics": forecast_metrics,
        "anomaly_summary": anomaly_summary,
        "los_metrics": los_metrics,
        "los_best_model": los_best_model,
        "filters_hint": {
            "n_rows": int(fdf.shape[0]),
            "hospitals": sorted(fdf["hospital"].dropna().unique().tolist()) if "hospital" in fdf else [],
        }
    }

    if use_ai:
        try:
            from openai import OpenAI
            client = OpenAI(api_key="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")
            prompt = textwrap.dedent(f"""
            You are a product-analytics writer for a hospital operations platform.
            Using ONLY the JSON below (no inventions), write:
            1) A crisp executive narrative (140–220 words) summarizing forecasting performance, billing anomalies risk,
               and LOS classification performance. Use non-technical language where possible.
            2) A "Performance & Use Case Table" that maps each model to when it is preferable (based on the metrics).
            3) A "Recommendations" list with 3–6 concrete actions (e.g., "prefer Prophet for weekly seasonal intake",
               "pilot RandomForest for LOS buckets", "tighten anomaly sensitivity if false positives > X/week").

            JSON:
            {json.dumps(payload, default=str)[:6000]}
            """).strip()
            rsp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role":"system","content":"Be precise, concise, and actionable. Avoid marketing fluff."},
                          {"role":"user","content": prompt}],
                temperature=0
            )
            st.markdown(rsp.choices[0].message.content)
        except Exception as e:
            st.error(f"OpenAI call failed: {e}")
            use_ai = False

    if not use_ai:
        # Deterministic fallback
        st.markdown("**Executive Narrative (deterministic)**")
        fct = forecast_metrics
        los = los_metrics
        an = anomaly_summary or {"recent_points": 0, "recent_anomalies": 0, "max_score": 0}
        fct_hint = f"- Forecasts: {', '.join(fct.keys())}" if fct else "- Forecasts: Not enough data for backtests."
        los_hint = f"- LOS models: {', '.join(los.keys())}" if los else "- LOS models: Not enough signals."
        st.write(
            f"In the current slice of data, the platform ran a fair bake-off across time-series models and LOS classifiers.\n"
            f"{fct_hint}\n"
            f"- Billing anomalies flagged in last {an.get('recent_points',0)} periods: {an.get('recent_anomalies',0)} "
            f"(peak severity score ≈ {an.get('max_score',0):.2f}).\n"
            f"{los_hint}\n\n"
            "Rules of thumb:\n"
            "• Prophet tends to win when weekly/seasonal patterns are strong; Holt–Winters is stable with short or noisy series; ARIMA helps when the series is near-stationary.\n"
            "• For LOS buckets, tree ensembles (Random Forest/XGBoost) handle mixed tabular signals well; Logistic Regression is best for transparency.\n\n"
            "Recommendations:\n"
            "• Promote the best-validated forecasting model to production for this cohort; require backtests before changes.\n"
            "• Tune anomaly sensitivity to keep noise low; investigate any sustained billing spikes.\n"
            "• Pilot the LOS top model and monitor F1/ROC-AUC; cross-check per-class errors to reduce bias on Very Long stays."
        )
