# app.py — Hospital Ops Studio (conference build)
# Streamlit 1.49+, Python 3.11, sklearn 1.5+
# Features: Prophet+ARIMA forecasting, model comparison with color coding,
# polished AI narratives with safe fallbacks, robust tabs, data-quality, LOS classifier.

import os, json, textwrap, asyncio, gc, logging
from datetime import datetime, date, timedelta
from typing import Dict, Optional, List
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import streamlit as st

# Viz / ML
import plotly.graph_objects as go
import plotly.express as px
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, label_binarize
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, roc_curve
import gc
# Add periodic garbage collection
if st.button("Clear Cache & GC"):
    st.cache_data.clear()
    gc.collect()

# Optional extras — degrade gracefully if missing
try:
    from prophet import Prophet  # fbprophet renamed to prophet
    _HAS_PROPHET = True
except Exception:
    _HAS_PROPHET = False

try:
    from xgboost import XGBClassifier  # type: ignore
    _HAS_XGB = True
except Exception:
    _HAS_XGB = False

# System & monitoring
try:
    import psutil
    _HAS_PSUTIL = True
except Exception:
    _HAS_PSUTIL = False

import structlog
from contextlib import contextmanager

# ---------------- CONFIG / THEME ----------------
st.set_page_config(page_title="Hospital Ops Studio — Control Tower", layout="wide", page_icon="🏥")
RAW_URL = "https://raw.githubusercontent.com/baheldeepti/hospital-streamlit-app/main/modified_healthcare_dataset.csv"

# ------------- Logging (structured) -------------
structlog.configure(
    processors=[structlog.processors.TimeStamper(fmt="iso"), structlog.dev.ConsoleRenderer()],
    wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
)
logger = structlog.get_logger()

# ----------------- Styles -----------------------
st.markdown("""
<style>
:root{
  --pri:#0F4C81; --teal:#159E99; --ink:#1F2937; --sub:#6B7280;
  --ok:#14A38B; --warn:#F59E0B; --alert:#EF4444;
}
.block-container{max-width:1500px;padding-top:12px}
h1,h2,h3{font-weight:700;color:var(--ink)}
a {color:var(--teal)}
.stButton>button{background:var(--pri);color:#fff;border-radius:10px}
.small{color:#6B7280;font-size:0.92rem}
hr{border-top:1px solid #eee}
.stTabs [data-baseweb="tab-list"], .stTabs div[role="tablist"]{
  position: sticky; top: 0; z-index: 5; background: white;
  border-bottom: 1px solid #eee; padding-top:.25rem; margin-top:.25rem;
}
.stTabs [data-baseweb="tab"], .stTabs div[role="tab"]{font-weight:600}
.exec-box{
  border-left:6px solid var(--pri); background:#f9fafb; padding:12px 14px; border-radius:8px; margin:.25rem 0 .75rem;
}
.kpi{
  display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:12px; margin:6px 0 8px;
}
.kpi .card{
  border:1px solid #e5e7eb; border-radius:14px; padding:12px 14px; background:#fff;
  box-shadow:0 1px 2px rgba(0,0,0,.04);
}
.kpi .h{color:#6b7280;font-size:.85rem}
.kpi .v{font-size:1.3rem; font-weight:700; color:#111827}
</style>
""", unsafe_allow_html=True)

st.title("Hospital Ops Studio — Control Tower")
st.caption("Admissions Control • Revenue Watch • LOS Planner — decisions first, dashboards second")

# ---------------- SIDEBAR SETTINGS ----------------
with st.sidebar:
    st.header("Settings")
    AI_TOGGLE = st.toggle("Enable AI narratives", value=True, key="ai_toggle")
    st.caption("Tip: set OPENAI_API_KEY and OPENAI_MODEL in Secrets or environment.")
    if _HAS_PSUTIL:
        process = psutil.Process()
        mem_mb = process.memory_info().rss / 1024 / 1024
        st.metric("Memory (MB)", f"{mem_mb:.1f}")
        if mem_mb > 1000:
            st.warning("High memory usage. Running GC.")
            gc.collect()
    else:
        st.caption("psutil not installed — memory metrics unavailable.")

# ---------------- DATA LAYER ----------------
ICD_MAPPING = {
    'Infections': 'A49.9', 'Flu': 'J10.1', 'Cancer': 'C80.1', 'Asthma': 'J45.909',
    'Heart Disease': 'I51.9', "Alzheimer's": 'G30.9', 'Diabetes': 'E11.9', 'Obesity': 'E66.9'
}

class HospitalDataManager:
    BASE_SCHEMA = {
        'admit_date': 'datetime64[ns]',
        'discharge_date': 'datetime64[ns]',
        'billing_amount': 'float64',
        'length_of_stay': 'float64',
        'condition': 'string',
        'admission_type': 'string',
        'insurer': 'string',
        'hospital': 'string',
        'doctor': 'string',
        'test_results': 'string',
        'age': 'float64',
    }
    RENAME_MAP = {
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

    def _normalize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        for k, v in self.RENAME_MAP.items():
            if k in df.columns:
                df.rename(columns={k: v}, inplace=True)
        return df

    def _coerce_types(self, df: pd.DataFrame) -> pd.DataFrame:
        if "admit_date" in df:
            df["admit_date"] = pd.to_datetime(df["admit_date"], errors="coerce")
        if "discharge_date" in df:
            df["discharge_date"] = pd.to_datetime(df["discharge_date"], errors="coerce")
        for c in ["length_of_stay", "billing_amount", "age"]:
            if c in df:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        for c in ["condition","admission_type","insurer","hospital","doctor","test_results"]:
            if c in df:
                df[c] = df[c].astype("string")
        return df

    def _feature_engineer(self, df: pd.DataFrame) -> pd.DataFrame:
        if "admit_date" in df:
            df["dow"] = df["admit_date"].dt.weekday
            df["weekofyear"] = df["admit_date"].dt.isocalendar().week.astype(int)
            df["month"] = df["admit_date"].dt.month
            df["admit_day"] = df["admit_date"].dt.date
        if "admission_type" in df:
            df["is_emergency"] = df["admission_type"].str.contains(
                "Emergency|ER|Urgent", case=False, na=False
            ).astype(int)
        else:
            df["is_emergency"] = 0
        if "age" in df:
            bins = [-np.inf, 17, 40, 65, np.inf]
            labels = ["Child", "Adult", "Senior", "Elder"]
            df["age_group"] = pd.cut(df["age"], bins=bins, labels=labels)
        if "condition" in df:
            df["icd_code"] = df["condition"].map(ICD_MAPPING).fillna("R69")
        return df

    def _validate_chunk(self, chunk: pd.DataFrame) -> pd.DataFrame:
        if "admit_date" in chunk:
            chunk = chunk.dropna(subset=["admit_date"])
        if "length_of_stay" in chunk:
            chunk["length_of_stay"] = chunk["length_of_stay"].clip(lower=0, upper=365)
        if "billing_amount" in chunk:
            chunk["billing_amount"] = chunk["billing_amount"].clip(lower=0)
        return chunk

    def _get_fallback_data(self) -> pd.DataFrame:
        # Deterministic-ish fallback for demo
        rng = np.random.default_rng(42)
        dates = pd.date_range(end=pd.Timestamp.today(), periods=120, freq="D")
        df = pd.DataFrame({
            "admit_date": dates,
            "billing_amount": rng.gamma(5, 220, len(dates)),
            "length_of_stay": rng.integers(1, 12, len(dates)).astype(float),
            "condition": rng.choice(list(ICD_MAPPING.keys()), len(dates)),
            "admission_type": rng.choice(["Emergency","Elective","Urgent"], len(dates)),
            "insurer": rng.choice(["Aetna","BlueCross","Kaiser","United"], len(dates)),
            "hospital": rng.choice(["North Campus","East Wing","South Pavilion"], len(dates)),
            "doctor": rng.choice(["Dr. Smith","Dr. Lee","Dr. Patel","Dr. Moore"], len(dates)),
            "age": np.clip(rng.normal(55, 15, len(dates)), 0, 95),
        })
        df = self._coerce_types(self._feature_engineer(df))
        return df

    @staticmethod
    @st.cache_data(ttl=3600, max_entries=3, show_spinner=False)
    def load_data_chunked(url: str, chunk_size: int = 10000) -> pd.DataFrame:
        temp_mgr = HospitalDataManager()
        try:
            chunks = []
            for chunk in pd.read_csv(url, chunksize=chunk_size):
                chunk = temp_mgr._normalize_columns(chunk)
                chunk = temp_mgr._coerce_types(chunk)
                chunk = temp_mgr._validate_chunk(chunk)
                chunks.append(chunk)
            if not chunks:
                return temp_mgr._get_fallback_data()
            df = pd.concat(chunks, ignore_index=True)
            df = temp_mgr._feature_engineer(df)
            logger.info("data_loaded", rows=len(df))
            return df
        except Exception as e:
            logger.error("data_load_failed", error=str(e))
            st.error(f"Data load failed. Using fallback dataset. Details: {e}")
            return temp_mgr._get_fallback_data()

data_mgr = HospitalDataManager()
df = data_mgr.load_data_chunked(RAW_URL)

# Optional upload override
with st.expander("Data source (optional override)"):
    up = st.file_uploader("Upload CSV to override default", type=["csv"], key="upload_csv")
    if up is not None:
        try:
            df = pd.read_csv(up)
            df = data_mgr._normalize_columns(df)
            df = data_mgr._coerce_types(df)
            df = df.dropna(subset=["admit_date"])
            df = data_mgr._feature_engineer(df)
            st.success(f"Loaded {len(df):,} rows from upload.")
        except Exception as e:
            st.error(f"Failed to read uploaded CSV: {e}")

# ---------------- Data Quality Monitor ----------------
class DataQualityMonitor:
    def _detect_outliers(self, df: pd.DataFrame) -> Dict[str, int]:
        out = {}
        for col in ["billing_amount","length_of_stay","age"]:
            if col in df:
                s = pd.to_numeric(df[col], errors="coerce")
                q1, q3 = np.nanpercentile(s, [25, 75])
                iqr = q3 - q1
                lo, hi = q1 - 1.5*iqr, q3 + 1.5*iqr
                out[col] = int(((s < lo) | (s > hi)).sum())
        return out

    def _check_schema(self, df: pd.DataFrame) -> Dict[str, bool]:
        must = ["admit_date","billing_amount","length_of_stay"]
        return {c: (c in df.columns) for c in must}

    def generate_quality_report(self, df: pd.DataFrame) -> dict:
        completeness = (1 - df.isnull().sum() / max(len(df), 1))
        completeness = completeness[[c for c in completeness.index if completeness[c] == completeness[c]]]
        return {
            "rows": int(len(df)),
            "duplicates": int(df.duplicated().sum()),
            "completeness": {k: float(v) for k, v in completeness.to_dict().items()},
            "outliers": self._detect_outliers(df),
            "schema_compliance": self._check_schema(df),
        }

dq = DataQualityMonitor()
with st.expander("Data Quality Report"):
    st.json(dq.generate_quality_report(df))

# ---------------- SHARED FILTERS ----------------
def render_filters(data: pd.DataFrame) -> pd.DataFrame:
    st.subheader("Filters")
    c1, c2, c3, c4 = st.columns(4)
    hosp = c1.multiselect("Hospitals", sorted(data["hospital"].dropna().unique()) if "hospital" in data else [], key="f_hosp")
    insurer = c2.multiselect("Insurers", sorted(data["insurer"].dropna().unique()) if "insurer" in data else [], key="f_ins")
    adm = c3.multiselect("Admission Types", sorted(data["admission_type"].dropna().unique()) if "admission_type" in data else [], key="f_adm")
    cond = c4.multiselect("Conditions", sorted(data["condition"].dropna().unique()) if "condition" in data else [], key="f_cond")
    q = pd.Series(True, index=data.index)
    if hosp and "hospital" in data: q &= data["hospital"].isin(hosp)
    if insurer and "insurer" in data: q &= data["insurer"].isin(insurer)
    if adm and "admission_type" in data: q &= data["admission_type"].isin(adm)
    if cond and "condition" in data: q &= data["condition"].isin(cond)
    return data[q].copy()

fdf = render_filters(df)

# ---------------- UTIL: SERIES + METRICS ----------------
def build_timeseries(data: pd.DataFrame, metric: str, freq: str = "D") -> pd.DataFrame:
    if "admit_date" not in data or data.empty:
        return pd.DataFrame(columns=["ds","y"])
    idx = data.set_index("admit_date").sort_index()
    if metric == "intake":
        s = idx.assign(_one=1)["_one"].resample(freq).sum().fillna(0.0)
    elif metric == "billing_amount" and "billing_amount" in idx:
        s = idx["billing_amount"].resample(freq).sum().fillna(0.0)
    elif metric == "length_of_stay" and "length_of_stay" in idx:
        s = idx["length_of_stay"].resample(freq).mean().ffill().fillna(0.0)
    else:
        return pd.DataFrame(columns=["ds","y"])
    return pd.DataFrame({"ds": s.index, "y": s.values})

def ts_metrics(y_true, y_pred) -> dict:
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    n = min(len(y_true), len(y_pred))
    y_true = y_true[:n]; y_pred = y_pred[:n]
    mae = float(np.mean(np.abs(y_true - y_pred))) if n else np.nan
    rmse = float(np.sqrt(np.mean((y_true - y_pred)**2))) if n else np.nan
    denom = np.clip(np.abs(y_true), 1e-9, None)
    mape = float(np.mean(np.abs((y_true - y_pred)/denom))*100.0) if n else np.nan
    return {"MAPE%": mape, "MAE": mae, "RMSE": rmse}

def style_lower_better(df: pd.DataFrame) -> str:
    if df.empty: return ""
    ranks = df.rank(ascending=True, method="min")
    def bg(cell_val, rank, col):
        if pd.isna(rank): return "background-color: rgba(255,255,255,0.6)"
        rmin, rmax = ranks[col].min(), ranks[col].max()
        if rmax == rmin: return "background-color: rgba(255,255,255,0.6)"
        pct = (rank - rmin)/(rmax - rmin + 1e-9)
        red = int(255*pct); green = int(255*(1-pct)); blue = 210
        return f"background-color: rgba({red},{green},{blue},0.25)"
    styled = df.style.format({"MAPE%":"{:.2f}","MAE":"{:.2f}","RMSE":"{:.2f}"})
    for col in df.columns:
        styled = styled.apply(lambda s: [bg(val, ranks.loc[s.name, col], col) for val in s], axis=1)
    return styled.to_html()

def style_higher_better(df: pd.DataFrame) -> str:
    if df.empty: return ""
    ranks = df.rank(ascending=False, method="min")
    def bg(cell_val, rank, col):
        if pd.isna(rank): return "background-color: rgba(255,255,255,0.6)"
        rmin, rmax = ranks[col].min(), ranks[col].max()
        if rmax == rmin: return "background-color: rgba(255,255,255,0.6)"
        pct = (rank - rmin)/(rmax - rmin + 1e-9)
        red = int(255*pct); green = int(255*(1-pct)); blue = 210
        return f"background-color: rgba({red},{green},{blue},0.25)"
    styled = df.style.format("{:.3f}")
    for col in df.columns:
        styled = styled.apply(lambda s: [bg(val, ranks.loc[s.name, col], col) for val in s], axis=1)
    return styled.to_html()

# ---------------- SAFE ZONE ----------------
@contextmanager
def safe_zone(label: str):
    try:
        yield
    except Exception as e:
        logger.error("section_error", section=label, error=str(e))
        st.error(f"{label}: something went wrong.")
        st.exception(e)

# ---------------- Model Manager (async-ready) ----------------
class ModelManager:
    def __init__(self):
        self.executor = ThreadPoolExecutor(max_workers=2)

    def _holt_forecast(self, s: pd.Series, horizon: int) -> np.ndarray:
        try:
            m = ExponentialSmoothing(s, trend="add", seasonal="add", seasonal_periods=7).fit()
        except Exception:
            m = ExponentialSmoothing(s, trend="add").fit()
        return np.asarray(m.forecast(horizon))

    def _arima_forecast(self, s: pd.Series, horizon: int) -> np.ndarray:
        try:
            model = SARIMAX(s, order=(1,1,1), seasonal_order=(1,0,1,7),
                            enforce_stationarity=False, enforce_invertibility=False)
            res = model.fit(disp=False)
            fc = res.forecast(steps=horizon)
            return np.asarray(fc)
        except Exception:
            diff = s.diff().dropna()
            drift = diff.mean() if len(diff) else 0.0
            last = s.iloc[-1]
            return np.asarray([last + drift*(i+1) for i in range(horizon)])

    def _prophet_forecast(self, ts: pd.DataFrame, horizon: int) -> np.ndarray:
        if not _HAS_PROPHET:
            raise RuntimeError("Prophet not available")
        dfp = ts.rename(columns={"ds":"ds","y":"y"}).copy()
        m = Prophet(weekly_seasonality=True, daily_seasonality=False, yearly_seasonality=False)
        m.fit(dfp)
        future = m.make_future_dataframe(periods=horizon, freq="D", include_history=False)
        fc = m.predict(future)["yhat"].to_numpy()
        return fc

    def forecast_all(self, ts: pd.DataFrame, horizon: int) -> Dict[str, pd.DataFrame]:
        s = ts.set_index("ds")["y"].asfreq("D").ffill()
        idx = pd.date_range(s.index.max() + pd.Timedelta(days=1), periods=horizon, freq="D")
        out: Dict[str, pd.DataFrame] = {}
        out["Holt-Winters"] = pd.DataFrame({"ds": idx, "yhat": self._holt_forecast(s, horizon)})
        out["ARIMA"] = pd.DataFrame({"ds": idx, "yhat": self._arima_forecast(s, horizon)})
        if _HAS_PROPHET:
            try:
                out["Prophet"] = pd.DataFrame({"ds": idx, "yhat": self._prophet_forecast(ts, horizon)})
            except Exception as e:
                logger.warning("prophet_failed", error=str(e))
        return out

    def backtest(self, ts: pd.Series, horizon: int, windows: int = 3) -> Dict[str, Dict[str,float]]:
        metrics: Dict[str, List[float]] = {"Holt-Winters":[],"ARIMA":[]}
        if _HAS_PROPHET: metrics["Prophet"] = []
        n = len(ts); step = horizon
        needed = (windows+1)*horizon + 7
        if n < needed:
            windows = max(1, (n - 14) // max(1, horizon))

        def eval_pair(y_true, y_pred):
            return ts_metrics(y_true, y_pred)["MAPE%"]

        for w in range(windows, 0, -1):
            split = n - w*step
            train = ts.iloc[:split].asfreq("D").ffill()
            test = ts.iloc[split:split+step]
            for name in ["Holt-Winters","ARIMA"] + (["Prophet"] if _HAS_PROPHET else []):
                try:
                    if name == "Holt-Winters":
                        yhat = self._holt_forecast(train, step)
                    elif name == "ARIMA":
                        yhat = self._arima_forecast(train, step)
                    else:
                        dfp = pd.DataFrame({"ds": train.index, "y": train.values})
                        yhat = self._prophet_forecast(dfp, step)
                    metrics[name].append(eval_pair(test.to_numpy(), np.asarray(yhat)))
                except Exception:
                    metrics[name].append(np.nan)

        def concat_eval(model_name):
            preds, trues = [], []
            for w in range(windows, 0, -1):
                split = n - w*step
                train = ts.iloc[:split].asfreq("D").ffill()
                test = ts.iloc[split:split+step]
                if model_name == "Holt-Winters":
                    yhat = self._holt_forecast(train, step)
                elif model_name == "ARIMA":
                    yhat = self._arima_forecast(train, step)
                else:
                    dfp = pd.DataFrame({"ds": train.index, "y": train.values})
                    yhat = self._prophet_forecast(dfp, step)
                preds.append(np.asarray(yhat)); trues.append(test.to_numpy())
            yhat_all, ytrue_all = np.concatenate(preds), np.concatenate(trues)
            return ts_metrics(ytrue_all, yhat_all)

        table = {}
        for m in ["Holt-Winters","ARIMA"] + (["Prophet"] if _HAS_PROPHET else []):
            try:
                table[m] = concat_eval(m)
            except Exception:
                table[m] = {"MAPE%": np.nan, "MAE": np.nan, "RMSE": np.nan}
        return table

    async def run_async(self, fn, *args, **kwargs):
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.executor, lambda: fn(*args, **kwargs))

model_mgr = ModelManager()

# ---------------- ANOMALY DETECTION ----------------
def detect_anomalies(ts: pd.DataFrame, sensitivity: float = 3.0) -> pd.DataFrame:
    if ts.empty: return ts.assign(anomaly=False, score=0.0)
    df2 = ts.copy().sort_values("ds"); y = df2["y"].values
    med = np.median(y); mad = np.median(np.abs(y - med)) + 1e-9
    rzs = 0.6745 * (y - med) / mad
    z_flag = np.abs(rzs) > sensitivity
    try:
        iso = IsolationForest(n_estimators=300, contamination="auto", random_state=42)
        iso_flag = (iso.fit_predict(y.reshape(-1,1)) == -1)
    except Exception:
        iso_flag = np.zeros_like(z_flag, dtype=bool)
    df2["rzs"] = rzs
    df2["anomaly"] = z_flag | iso_flag
    z_norm = (np.abs(rzs) - np.min(np.abs(rzs))) / (np.ptp(np.abs(rzs)) + 1e-9)
    df2["score"] = (0.7 * z_norm) + (0.3 * iso_flag.astype(float))
    return df2

def plot_anoms(an_df: pd.DataFrame, title: str):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=an_df["ds"], y=an_df["y"], mode="lines+markers", name="Billing"))
    flag = an_df[an_df["anomaly"]]
    if not flag.empty:
        fig.add_trace(go.Scatter(x=flag["ds"], y=flag["y"], mode="markers", name="Anomaly",
                                 marker=dict(size=10, symbol="x")))
    fig.update_layout(title=title, height=420, margin=dict(l=10,r=10,b=10,t=50))
    st.plotly_chart(fig, width="stretch")

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
    for c in num_cols:
        d[c] = pd.to_numeric(d[c], errors="coerce").fillna(d[c].median())

    try:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)

    cat_cols = [c for c in ["admission_type","insurer","hospital","condition","doctor","age_group","icd_code"] if c in d.columns]
    for c in cat_cols:
        d[c] = d[c].astype("string").fillna("Unknown")

    X = d[num_cols + cat_cols].copy()
    y = d["los_bucket"].copy()
    return X, y, num_cols, cat_cols, d, ohe

# ---------------- Circuit Breaker for External APIs ----------------
class CircuitBreaker:
    def __init__(self, failure_threshold=3, timeout=90):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.failure_count = 0
        self.last_failure_time: Optional[datetime] = None

    def _is_open(self) -> bool:
        if self.failure_count < self.failure_threshold:
            return False
        if self.last_failure_time is None:
            return False
        return (datetime.utcnow() - self.last_failure_time) < timedelta(seconds=self.timeout)

    def call(self, func, *args, **kwargs):
        if self._is_open():
            raise RuntimeError("Circuit breaker is open")
        try:
            res = func(*args, **kwargs)
            self.failure_count = 0
            self.last_failure_time = None
            return res
        except Exception as e:
            self.failure_count += 1
            self.last_failure_time = datetime.utcnow()
            raise e

if "cb_ai" not in st.session_state:
    st.session_state["cb_ai"] = CircuitBreaker(failure_threshold=3, timeout=90)

# ---------------- AI (OpenAI) ----------------
_ALLOWED_MODELS = [
    (st.secrets.get("OPENAI_MODEL") or os.environ.get("OPENAI_MODEL") or "").strip(),
    "gpt-4", "gpt-3.5-turbo"
]
_seen=set(); _MODEL_PREFS=[m for m in _ALLOWED_MODELS if m and (m not in _seen and not _seen.add(m))]
AI_MODEL = _MODEL_PREFS[0] if _MODEL_PREFS else "gpt-3.5-turbo"

def _get_openai_client():
    key = (
        st.secrets.get("OPENAI_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
        or st.session_state.get("OPENAI_API_KEY")
    )
    if not key:
        return None
    try:
        from openai import OpenAI
        return OpenAI(api_key=key)
    except Exception as e:
        logger.error("openai_init_failed", error=str(e))
        return None

def _try_completion(client, model, messages, temperature=0.2, max_tokens=450):
    return st.session_state["cb_ai"].call(
        client.chat.completions.create,
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens
    )

def _truncate_json(d: dict, limit: int = 6000) -> str:
    s = json.dumps(d, default=str)
    return (s[:limit] + "…") if len(s) > limit else s

def ai_write(section_title: str, payload: dict):
    """Polished executive/analyst narratives with safe fallbacks, never crash."""
    try:
        client = _get_openai_client()
        use_ai_default = bool(AI_TOGGLE and (client is not None))

        col1, col2 = st.columns([1, 2])
        use_ai = col1.checkbox(f"Use AI for {section_title}", value=use_ai_default, key=f"ai_use_{section_title}")
        analyst = col2.toggle("Analyst mode (more detail)", value=False, key=f"ai_mode_{section_title}")

        voice = "Executive Summary" if not analyst else "Analyst Report"
        sys_msg = (
            "You are a healthcare analytics writer. Be clear, factual, and business-focused. "
            "Avoid buzzwords. Produce tidy markdown."
        )

        # Keep tokens modest; long prompts + big max_tokens are a common failure mode
        max_out_tokens = 420 if not analyst else 520

        # Trim the JSON more aggressively in analyst mode to avoid over-long prompts
        json_trim_limit = 4500 if analyst else 5500

        content = textwrap.dedent(f"""
        Create a {voice} for **{section_title}** from the JSON below. Two audiences:

        1) **Executives** — What, So-What, Now-What. Crisp, 5–7 bullet points max.
        2) **Operations** — next steps with owners and target due dates.
        3) **Performance & Use Cases** — a compact markdown table (2–5 rows).
        4) **Risk/Watchlist** — 2–4 bullets, if relevant.

        Style:
        - Start with a one-paragraph headline takeaway in bold.
        - Use short sentences and numbers (e.g., “↑12% WoW”).
        - Keep under 220 words for Executive mode; up to 350 for Analyst mode.

        JSON (trimmed): { _truncate_json(payload, json_trim_limit) }
        """).strip()

        if use_ai and client:
            messages = [{"role": "system", "content": sys_msg}, {"role": "user", "content": content}]
            tried = []
            # Only try models we actually listed
            for mdl in _MODEL_PREFS:
                try:
                    rsp = _try_completion(client, mdl, messages, temperature=0.25, max_tokens=max_out_tokens)
                    st.markdown(rsp.choices[0].message.content)
                    with st.expander("AI diagnostics", expanded=False):
                        st.caption(f"Model: `{mdl}` | tokens: ≤{max_out_tokens}")
                    return
                except Exception as e:
                    tried.append(f"{mdl}: {type(e).__name__} — {str(e)[:120]}")

            with st.expander("AI unavailable (showing deterministic summary)"):
                st.caption("Tried: " + " | ".join(tried))

        # Fallback deterministic summary — never throw
        st.markdown(f"**Deterministic summary — {section_title}**")
        st.markdown("""
<div class="exec-box">
<b>Headline:</b> Platform generated summary is unavailable. Use the operational details below to brief the team.
</div>
""", unsafe_allow_html=True)
        st.json(payload)
        st.caption("ℹ️ Set OPENAI_API_KEY and optionally OPENAI_MODEL to enable narratives.")
    except Exception as e:
        # Last-resort safety net: show the error instead of the white 'Oh no' page
        st.error("Narrative generation failed.")
        st.exception(e)
        logger.exception("ai_write_error", section=section_title)


# ---------------- ACTION FOOTER + DECISION LOG ----------------
if "decision_log" not in st.session_state:
    st.session_state["decision_log"] = []

def action_footer(section: str):
    st.markdown("#### Action footer")
    c1, c2, c3, c4 = st.columns([1.2, 1, 1, 1])

    owner = c1.selectbox(
        "Owner",
        ["House Supervisor", "Revenue Integrity", "Case Mgmt", "Unit Manager", "Finance Lead"],
        key=f"owner_{section}",
    )
    decision = c2.selectbox("Decision", ["Promote", "Hold", "Tune", "Investigate"], key=f"decision_{section}")
    sla_date = c3.date_input("SLA Date", value=date.today(), key=f"sla_date_{section}")
    # ✅ fixed: closed string + closed call
    sla_time = c4.time_input("SLA Time", value=datetime.now().time(), key=f"sla_time_{section}")
    note = st.text_input("Notes (optional)", key=f"note_{section}")

    colA, colB = st.columns([1, 1])
    if colA.button(f"Save to Decision Log ({section})", key=f"save_{section}"):
        st.session_state["decision_log"].append({
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "section": section,
            "owner": owner,
            "decision": decision,
            "sla": f"{sla_date} {sla_time}",
            "note": note,
        })
        st.success("Saved to Decision Log.")

    df_log = pd.DataFrame(st.session_state["decision_log"])
    st.download_button(
        label="Download Decision Log (CSV)",
        data=df_log.to_csv(index=False).encode("utf-8"),
        file_name="decision_log.csv",
        mime="text/csv",
        key=f"dl_{section}",
    )

# ---------------- TABS (top-level) ----------------
def run_app():
    tabs = st.tabs(["📈 Admissions Control","🧾 Revenue Watch","🛏️ LOS Planner"])
    
    # ===== 1) Admissions Control =====
    with tabs[0], safe_zone("Admissions Control"):
        st.subheader("📈 Admissions Control — Forecast → Staffing Targets")
        c1, c2, c3, c4 = st.columns(4)
        cohort_dim = c1.selectbox("Cohort dimension", ["All","hospital","insurer","condition"], key="adm_dim")
        cohort_val = c2.selectbox(
            "Cohort value",
            ["(all)"] + (sorted(fdf[cohort_dim].dropna().unique().tolist()) if cohort_dim!="All" and cohort_dim in fdf else []),
            key="adm_val",
        )
        agg = c3.selectbox("Aggregation", ["Daily","Weekly"], index=0, key="adm_agg")
        horizon = c4.slider("Forecast horizon (days)", 7, 90, 30, key="adm_hz")
        freq = "D" if agg=="Daily" else "W"
    
        fdx = fdf.copy()
        if cohort_dim!="All" and cohort_dim in fdx and cohort_val and cohort_val!="(all)":
            fdx = fdx[fdx[cohort_dim]==cohort_val]
    
        ts = build_timeseries(fdx, metric="intake", freq=freq)
        if ts.empty:
            st.info("No admissions series for the current cohort/filters.")
        else:
            with st.expander("Seasonality calendar (avg admissions by week-of-year × weekday)"):
                s = fdx.set_index("admit_date").assign(_one=1)["_one"].resample("D").sum().fillna(0.0)
                cal = pd.DataFrame({"dow": s.index.weekday, "woy": s.index.isocalendar().week.values, "val": s.values})
                heat = cal.groupby(["woy","dow"])["val"].mean().reset_index()
                heat_pivot = heat.pivot(index="woy", columns="dow", values="val").fillna(0)
                fig = px.imshow(heat_pivot, aspect="auto",
                                labels=dict(x="Weekday (0=Mon)", y="Week of Year", color="Avg admits"))
                fig.update_layout(height=360, margin=dict(l=10,r=10,b=10,t=30))
                st.plotly_chart(fig, width="stretch")
    
            sc1, sc2 = st.columns(2)
            flu_pct = sc1.slider("Flu surge scenario (±%)", -30, 50, 0, 5, key="adm_flu")
            weather_pct = sc2.slider("Weather impact (±%)", -20, 20, 0, 5, key="adm_wthr")
    
            fc_dict = model_mgr.forecast_all(ts, horizon=horizon)
            adj_factor = (100 + flu_pct + weather_pct) / 100.0
    
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=ts["ds"], y=ts["y"], name="History", mode="lines"))
            for name, fc in fc_dict.items():
                fig.add_trace(go.Scatter(x=fc["ds"], y=fc["yhat"]*adj_factor, name=name, mode="lines"))
            fig.update_layout(title="Admissions Forecast", height=420, margin=dict(l=10,r=10,b=10,t=50))
            st.plotly_chart(fig, width="stretch")
    
            # Backtests & selection
            try:
                s = ts.set_index("ds")["y"].asfreq("D").ffill()
                H = min(14, horizon)
                table = model_mgr.backtest(s, H, windows=3)
                comp = pd.DataFrame(table).T[["MAPE%","MAE","RMSE"]]
                st.markdown("#### 🧪 Model Comparison (rolling backtests)")
                st.markdown(style_lower_better(comp), unsafe_allow_html=True)
                comp_rank = comp.rank(ascending=True, method="min")
                best = comp_rank.sum(axis=1).sort_values().index[0]
            except Exception:
                best = "Holt-Winters"
    
            st.markdown(f"**Selected model:** `{best}`")
            best_fc = fc_dict.get(best, next(iter(fc_dict.values())))
            daily_fc = (best_fc["yhat"] * adj_factor).to_numpy()
    
            st.markdown("#### Staffing targets (illustrative)")
            rn_per_shift = np.ceil(daily_fc / 5.0).astype(int)
            targets = pd.DataFrame({
                "Date": pd.to_datetime(best_fc["ds"]).dt.date,
                "Expected Admissions": np.round(daily_fc, 1),
                "RN/Day": rn_per_shift, "RN/Evening": rn_per_shift, "RN/Night": rn_per_shift
            })
            st.dataframe(targets, width="stretch", hide_index=True)
    
            ai_payload = {"cohort": {"dimension": cohort_dim, "value": cohort_val},
                          "aggregation": agg, "horizon_days": horizon,
                          "scenario": {"flu_pct": flu_pct, "weather_pct": weather_pct},
                          "models": list(fc_dict.keys()), "selected_model": best}
            st.markdown("---")
            ai_write("Admissions Control", ai_payload)
            action_footer("Admissions Control")
    
    # ===== 2) Revenue Watch =====
    with tabs[1], safe_zone("Revenue Watch"):
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
    
            st.markdown("#### Root-cause explorer")
            dims = [d for d in ["insurer","hospital","condition","doctor"] if d in fdf.columns]
            if dims:
                drill = st.selectbox("Group anomalies by", dims, index=0, key="bill_drill")
                fdf_agg = fdf.set_index("admit_date")
                if isinstance(an, pd.DataFrame) and not an.empty:
                    an_days = set(pd.to_datetime(an.loc[an["anomaly"], "ds"]).dt.date.tolist())
                    fdf_agg["is_anom_day"] = pd.Series(fdf_agg.index.date, index=fdf_agg.index).isin(an_days).astype(int)
                    grp = fdf_agg.groupby(drill).agg(
                        billing_total=("billing_amount","sum"),
                        encounters=("billing_amount","count"),
                        anomaly_days=("is_anom_day","sum")
                    ).reset_index().sort_values("anomaly_days", ascending=False)
                    st.dataframe(grp, width="stretch")
    
        ai_payload = {"aggregation": agg, "sensitivity": sensitivity, "baseline_weeks": baseline_weeks}
        st.markdown("---")
        ai_write("Revenue Watch", ai_payload)
        action_footer("Revenue Watch")
    
    # ===== 3) LOS Planner =====
    # ===== 3) LOS Planner =====
    with tabs[2], safe_zone("LOS Planner"):
        st.subheader("🛏️ LOS Planner — Risk Buckets → Discharge Orchestration")
        prep = los_prep(fdf)
        if prep is None:
            st.info("`length_of_stay` not found.")
        else:
            X, y, num_cols, cat_cols, d_full, ohe = prep
    
            # ---- Guard: need at least 2 classes to train a classifier
            classes_all = pd.Series(y).dropna().unique().tolist()
            if len(classes_all) < 2:
                st.warning("Not enough class variety in the filtered data to train (need ≥2 buckets). Try loosening filters.")
                st.stop()
    
            # Train/test split — fall back if stratify can’t be satisfied
            try:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.25, random_state=42, stratify=y
                )
            except Exception:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.25, random_state=42
                )
    
            # ---- Build ColumnTransformer only with non-empty feature groups
            num_feats = [c for c in num_cols if c in X.columns]
            cat_feats = [c for c in cat_cols if c in X.columns]
    
            transformers = []
            if num_feats:
                transformers.append(("num", StandardScaler(), num_feats))
            if cat_feats:
                transformers.append(("cat", ohe, cat_feats))
    
            if not transformers:
                st.warning("No usable features found after filtering. Add features (numeric/categorical) or relax filters.")
                st.stop()
    
            pre = ColumnTransformer(transformers=transformers, remainder="drop")
    
            # ---- Models
            models = {
                "Logistic Regression": Pipeline([
                    ("pre", pre),
                    ("clf", LogisticRegression(max_iter=600, multi_class="multinomial"))
                ]),
                "Random Forest": Pipeline([
                    ("pre", pre),
                    ("clf", RandomForestClassifier(n_estimators=400, random_state=42))
                ]),
            }
            if _HAS_XGB:
                models["XGBoost (optional)"] = Pipeline([
                    ("pre", pre),
                    ("clf", XGBClassifier(
                        n_estimators=500, learning_rate=0.05, max_depth=6,
                        subsample=0.9, colsample_bytree=0.9,
                        objective="multi:softprob", eval_metric="mlogloss",
                        random_state=42
                    ))
                ])
    
            # ---- Fit/evaluate robustly
            classes = sorted(pd.Series(y).dropna().unique().tolist())
            results, roc_curves = {}, {}
    
            for name, pipe in models.items():
                try:
                    pipe.fit(X_train, y_train)
                    y_pred = pipe.predict(X_test)
                except Exception as e:
                    st.warning(f"{name} failed to train: {e}")
                    continue
    
                # Metrics
                try:
                    y_proba = pipe.predict_proba(X_test)
                except Exception:
                    y_proba = None
    
                acc = accuracy_score(y_test, y_pred)
                pr, rc, f1, _ = precision_recall_fscore_support(
                    y_test, y_pred, average="weighted", zero_division=0
                )
    
                auc = None; fprs = {}; tprs = {}
                # Need at least 2 classes present in y_test to compute ROC
                if (y_proba is not None) and (len(np.unique(pd.Series(y_test).dropna())) >= 2):
                    try:
                        y_test_b = label_binarize(y_test, classes=classes)
                        # Only compute AUC if classifier produced probs for all classes we expect
                        if y_proba.shape[1] == len(classes):
                            auc = roc_auc_score(y_test_b, y_proba, average="weighted", multi_class="ovr")
                            for i, cls in enumerate(classes):
                                fpr, tpr, _ = roc_curve(y_test_b[:, i], y_proba[:, i])
                                fprs[cls] = fpr; tprs[cls] = tpr
                    except Exception:
                        pass
    
                results[name] = {"Accuracy": acc, "Precision": pr, "Recall": rc, "F1": f1, "ROC-AUC": auc}
                roc_curves[name] = (fprs, tprs)
    
            if not results:
                st.error("All models failed to train on the current slice. Try relaxing filters or checking data quality.")
                st.stop()
    
            perf = pd.DataFrame(results).T
            for c in ["Accuracy","Precision","Recall","F1","ROC-AUC"]:
                if c not in perf: perf[c] = np.nan
            perf = perf[["Accuracy","Precision","Recall","F1","ROC-AUC"]]
    
            st.markdown("#### 📊 Model Performance")
            st.markdown(style_higher_better(perf), unsafe_allow_html=True)
    
            # ---- ROC plot (best by F1), only if curves exist
            def _render_roc():
                try:
                    top_model = perf["F1"].astype(float).idxmax()
                except Exception:
                    top_model = next(iter(results.keys()))
                fprs, tprs = roc_curves.get(top_model, ({}, {}))
                if not fprs:
                    st.info("ROC not available for the current split.")
                    return
                fig = go.Figure()
                for cls in classes:
                    if cls in fprs and cls in tprs and len(fprs[cls]) and len(tprs[cls]):
                        fig.add_trace(go.Scatter(x=fprs[cls], y=tprs[cls], mode="lines", name=f"{top_model} — {cls}"))
                fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode="lines", name="Chance", line=dict(dash="dash")))
                fig.update_layout(height=420, xaxis_title="False Positive Rate", yaxis_title="True Positive Rate")
                st.plotly_chart(fig, width="stretch")
    
            with st.expander("ROC Curves (one-vs-rest)"):
                _render_roc()

        # ---- Equity slices (optional)
        st.markdown("#### Equity slices")
        slice_dim_options = [d for d in ["insurer","age_group"] if d in d_full.columns]
        slice_dim = st.selectbox("Slice by", slice_dim_options, index=0 if slice_dim_options else 0, key="los_slice") if slice_dim_options else None
        if slice_dim:
            try:
                top_model = perf["F1"].astype(float).idxmax()
            except Exception:
                top_model = next(iter(results.keys()))
            best_pipe = models[top_model]
            X_test_idx = X_test.index
            slice_vals = d_full.loc[X_test_idx, slice_dim].fillna("Unknown")
            y_pred_best = best_pipe.predict(X_test)

            grp_rows = []
            for sv in sorted(pd.Series(slice_vals).unique().tolist()):
                mask = (slice_vals == sv)
                try:
                    acc_g = accuracy_score(y_test[mask], y_pred_best[mask]) if mask.any() else np.nan
                except Exception:
                    # index misalignment guard
                    acc_g = np.nan
                grp_rows.append({"Group": str(sv), "Accuracy": acc_g, "N": int(mask.sum())})

            st.dataframe(pd.DataFrame(grp_rows).sort_values("Accuracy", ascending=False), width="stretch")

        ai_payload = {
            "buckets": {"Short":"<=5","Medium":"6-15","Long":"16-45","Very Long":">45"},
            "metrics_table": perf.to_dict(),
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
            st.dataframe(df_log, width="stretch")
            st.download_button(
                label="Download Decision Log (CSV)",
                data=df_log.to_csv(index=False).encode("utf-8"),
                file_name="decision_log.csv",
                mime="text/csv",
                key="dl_footer"
        )
# Call it with a guard + global safety net
if __name__ == "__main__":
    try:
        run_app()
    except Exception as e:
        st.error(f"Application error: {e}")
        logger.exception("app_error")
