import os, json, textwrap, asyncio, gc, logging, hashlib
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pandas as pd
import streamlit as st
----Viz / ML--------
import plotly.graph_objects as go
import plotly.express as px
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, label_binarize
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, roc_curve
----Optional extras — degrade gracefully if missing----
try:
from xgboost import XGBClassifier  # type: ignore
_HAS_XGB = True
except Exception:
_HAS_XGB = False
System & monitoring
import psutil
import structlog
---------------- CONFIG / THEME ----------------
st.set_page_config(page_title="Hospital Ops Studio — Control Tower", layout="wide", page_icon="🏥")
RAW_URL = "https://raw.githubusercontent.com/baheldeepti/hospital-streamlit-app/main/modified_healthcare_dataset.csv"
------------- Logging (structured) -------------
structlog.configure(
processors=[structlog.processors.TimeStamper(fmt="iso"), structlog.dev.ConsoleRenderer()],
wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
)
logger = structlog.get_logger()
----------------- Styles -----------------------
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
---------------- SIDEBAR SETTINGS ----------------
with st.sidebar:
st.header("Settings")
AI_TOGGLE = st.toggle("Enable AI narratives", value=True, key="ai_toggle")
st.caption("Tip: configure OPENAI_API_KEY and OPENAI_MODEL in Secrets.")
# Resource monitor
process = psutil.Process()
mem_mb = process.memory_info().rss / 1024 / 1024
st.metric("Memory (MB)", f"{mem_mb:.1f}")
if mem_mb > 1000:
st.warning("High memory usage. Running GC.")
gc.collect()
---------------- DATA LAYER ----------------
ICD_MAPPING = {
'Infections': 'A49.9', 'Flu': 'J10.1', 'Cancer': 'C80.1', 'Asthma': 'J45.909',
'Heart Disease': 'I51.9', "Alzheimer's": 'G30.9', 'Diabetes': 'E11.9', 'Obesity': 'E66.9'
}
class HospitalDataManager:
"""Chunked loader with schema normalization + simple validation."""
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
Copydef __init__(self): ...

@staticmethod
def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
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
    return df

@staticmethod
def _coerce_types(df: pd.DataFrame) -> pd.DataFrame:
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

@staticmethod
def _feature_engineer(df: pd.DataFrame) -> pd.DataFrame:
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

@staticmethod
def _validate_chunk(chunk: pd.DataFrame) -> pd.DataFrame:
    # Basic validation — drop rows without admit_date and clip impossible values
    chunk = chunk.dropna(subset=["admit_date"])
    if "length_of_stay" in chunk:
        chunk["length_of_stay"] = chunk["length_of_stay"].clip(lower=0, upper=365)
    if "billing_amount" in chunk:
        chunk["billing_amount"] = chunk["billing_amount"].clip(lower=0)
    return chunk

@staticmethod
def _get_fallback_data() -> pd.DataFrame:
    # Minimal synthetic fallback to keep the app operational
    dates = pd.date_range(end=pd.Timestamp.today(), periods=60, freq="D")
    df = pd.DataFrame({
        "admit_date": dates,
        "billing_amount": np.random.gamma(5, 200, len(dates)),
        "length_of_stay": np.random.randint(1, 10, len(dates)).astype(float),
        "condition": np.random.choice(list(ICD_MAPPING.keys()), len(dates)),
        "admission_type": np.random.choice(["Emergency","Elective","Urgent"], len(dates)),
        "insurer": np.random.choice(["Aetna","BlueCross","Kaiser"], len(dates)),
        "hospital": np.random.choice(["North Campus","East Wing"], len(dates)),
        "doctor": np.random.choice(["Dr. Smith","Dr. Lee","Dr. Patel"], len(dates)),
        "age": np.random.normal(55, 15, len(dates)).clip(0, 95),
    })
    df = HospitalDataManager._coerce_types(HospitalDataManager._feature_engineer(df))
    return df

def load_data_chunked(self, url: str, chunk_size: int = 10000) -> pd.DataFrame:
    return load_data_chunked_cached(url, chunk_size)
Create a standalone cached function
@st.cache_data(ttl=3600, max_entries=3, show_spinner=False)
def load_data_chunked_cached(url: str, chunk_size: int = 10000) -> pd.DataFrame:
try:
chunks = []
for chunk in pd.read_csv(url, chunksize=chunk_size):
chunk = HospitalDataManager._normalize_columns(chunk)
chunk = HospitalDataManager._coerce_types(chunk)
chunk = HospitalDataManager._validate_chunk(chunk)
chunks.append(chunk)
if not chunks:
return HospitalDataManager._get_fallback_data()
df = pd.concat(chunks, ignore_index=True)
df = HospitalDataManager._feature_engineer(df)
logger.info("data_loaded", rows=len(df))
return df
except Exception as e:
logger.error("data_load_failed", error=str(e))
st.error(f"Data load failed. Using fallback dataset. Details: {e}")
return HospitalDataManager._get_fallback_data()
data_mgr = HospitalDataManager()
df = data_mgr.load_data_chunked(RAW_URL)
Optional upload override
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
---------------- Data Quality Monitor ----------------
class DataQualityMonitor:
def _detect_outliers(self, df: pd.DataFrame) -> Dict[str, int]:
out = {}
for col in ["billing_amount","length_of_stay","age"]:
if col in df:
s = pd.to_numeric(df[col], errors="coerce")
q1, q3 = np.nanpercentile(s, [25, 75])
iqr = q3 - q1
lo, hi = q1 - 1.5iqr, q3 + 1.5iqr
out[col] = int(((s < lo) | (s > hi)).sum())
return out
Copydef _check_schema(self, df: pd.DataFrame) -> Dict[str, bool]:
    must = ["admit_date","billing_amount","length_of_stay"]
    return {c: (c in df.columns) for c in must}

def generate_quality_report(self, df: pd.DataFrame) -> dict:
    completeness = (1 - df.isnull().sum() / max(len(df), 1)).to_dict()
    return {
        "rows": int(len(df)),
        "duplicates": int(df.duplicated().sum()),
        "completeness": {k: float(v) for k,v in completeness.items() if isinstance(v,(int,float,np.floating))},
        "outliers": self._detect_outliers(df),
        "schema_compliance": self._check_schema(df),
    }
dq = DataQualityMonitor()
with st.expander("Data Quality Report"):
st.json(dq.generate_quality_report(df))
---------------- SHARED FILTERS ----------------
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
---------------- UTIL: SERIES + METRICS ----------------
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
y_true = y_true[]; y_pred = y_pred[]
mae = float(np.mean(np.abs(y_true - y_pred))) if n else np.nan
rmse = float(np.sqrt(np.mean((y_true - y_pred)**2))) if n else np.nan
denom = np.clip(np.abs(y_true), 1e-9, None)
mape = float(np.mean(np.abs((y_true - y_pred)/denom))*100.0) if n else np.nan
return {"MAPE%": mape, "MAE": mae, "RMSE": rmse}
def style_lower_better(df: pd.DataFrame) -> str:
if df.empty: return ""
ranks = df.rank(ascending=True, method="min")
def bg(col, row):
r = ranks.loc[row, col]; rmin, rmax = ranks[col].min(), ranks[col].max()
if rmax == rmin: return "background-color: rgba(255,255,255,0.6)"
pct = (r - rmin)/(rmax - rmin + 1e-9)
red = int(255pct); green = int(255(1-pct)); blue = 200
return f"background-color: rgba({red},{green},{blue},0.25)"
styled = df.style.format({"MAPE%":"{:.2f}","MAE":"{:.2f}","RMSE":"{:.2f}"})
for col in df.columns:
styled = styled.apply(lambda s: [bg(col, s.name) for _ in s], axis=1)
return styled.to_html()
def style_higher_better(df: pd.DataFrame) -> str:
if df.empty: return ""
ranks = df.rank(ascending=False, method="min")
def bg(col, row):
r = ranks.loc[row, col]; rmin, rmax = ranks[col].min(), ranks[col].max()
if rmax == rmin: return "background-color: rgba(255,255,255,0.6)"
pct = (r - rmin)/(rmax - rmin + 1e-9)
red = int(255pct); green = int(255(1-pct)); blue = 200
return f"background-color: rgba({red},{green},{blue},0.25)"
styled = df.style.format("{:.3f}")
for col in df.columns:
styled = styled.apply(lambda s: [bg(col, s.name) for _ in s], axis=1)
return styled.to_html()
---------------- Model Manager (async-ready + cache) ----------------
class ModelManager:
def init(self):
self.executor = ThreadPoolExecutor(max_workers=2)
Copydef _hash_series(self, ts_df: pd.DataFrame, horizon: int) -> str:
    h = hashlib.md5()
    h.update(np.asarray(ts_df["y"]).tobytes())
    h.update(str(ts_df["ds"].min()).encode())
    h.update(str(ts_df["ds"].max()).encode())
    h.update(str(horizon).encode())
    return h.hexdigest()

@st.cache_resource(show_spinner=False)
def cached_forecast(self, y_bytes: bytes, start: str, end: str, horizon: int) -> pd.DataFrame:
    """Cache key is derived by Streamlit from args; returns future dataframe."""
    # Reconstruct a simple sequence just to keep Streamlit caching happy
    # (we only need horizon for deterministic forecast here)
    fc_idx = pd.date_range(pd.to_datetime(end) + pd.Timedelta(days=1), periods=horizon, freq="D")
    # Placeholder array (actual values are computed in forecast_one below)
    return pd.DataFrame({"ds": fc_idx, "yhat": np.zeros(horizon)})

def _holt_forecast(self, s: pd.Series, horizon: int) -> np.ndarray:
    try:
        m = ExponentialSmoothing(s, trend="add", seasonal="add", seasonal_periods=7).fit()
    except Exception:
        m = ExponentialSmoothing(s, trend="add").fit()
    return m.forecast(horizon)

def forecast_one(self, ts: pd.DataFrame, horizon: int) -> pd.DataFrame:
    # Single robust model for scalability (Holt-Winters)
    s = ts.set_index("ds")["y"].asfreq("D").ffill()
    yhat = self._holt_forecast(s, horizon)
    idx = pd.date_range(s.index.max() + pd.Timedelta(days=1), periods=horizon, freq="D")
    return pd.DataFrame({"ds": idx, "yhat": np.asarray(yhat)})

async def run_async(self, fn, *args, **kwargs):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(self.executor, lambda: fn(*args, **kwargs))
model_mgr = ModelManager()
---------------- ANOMALY DETECTION ----------------
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
st.plotly_chart(fig, use_container_width=True)
---------------- LOS HELPERS ----------------
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
Copy# Numeric columns
num_cols = [c for c in ["age","billing_amount","length_of_stay","dow","month","is_emergency"] if c in d.columns]
for c in num_cols:
    d[c] = pd.to_numeric(d[c], errors="coerce").fillna(d[c].median())

# Categorical columns — cast to string before fillna to avoid Categorical TypeError
cat_cols = [c for c in ["admission_type","insurer","hospital","condition","doctor","age_group","icd_code"] if c in d.columns]
for c in cat_cols:
    d[c] = d[c].astype("string").fillna("Unknown")

X = d[num_cols + cat_cols].copy()
y = d["los_bucket"].copy()
return X, y, num_cols, cat_cols, d
---------------- Circuit Breaker for External APIs ----------------
class CircuitBreaker:
def init(self, failure_threshold=3, timeout=60):
self.failure_threshold = failure_threshold
self.timeout = timeout
self.failure_count = 0
self.last_failure_time: Optional[datetime] = None
Copydef _is_open(self) -> bool:
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
---------------- AI (single model, robust) ----------------
AI_MODEL = (st.secrets.get("OPENAI_MODEL")
or os.environ.get("OPENAI_MODEL")
or "gpt-4o-mini").strip()
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
return OpenAI(api_key=key)  # no proxies/extra kwargs
except Exception as e:
logger.error("openai_init_failed", error=str(e))
return None
def ai_write(section_title: str, payload: dict):
client = get_openai_client()
use_ai = AI_TOGGLE and (client is not None)
col1, col2 = st.columns([1, 2])
use_ai = col1.checkbox(f"Use AI for {section_title}", value=use_ai, key=f"ai_use{section_title}")
analyst = col2.toggle("Analyst mode (more detail)", value=False, key=f"ai_mode_{section_title}")
Copyif use_ai and client:
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
    try:
        rsp = st.session_state["cb_ai"].call(
            client.chat.completions.create,
            model=AI_MODEL,
            messages=[
                {"role": "system", "content": "Be precise, concise, and actionable. Avoid marketing fluff."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
        )
        st.markdown(rsp.choices[0].message.content)
        with st.expander("AI diagnostics", expanded=False):
            st.caption(f"Model: `{AI_MODEL}`")
        return
    except Exception as e:
        logger.warning("ai_call_failed", error=str(e))
        with st.expander("AI unavailable (showing deterministic summary)", expanded=False):
            st.caption(f"{type(e).__name__}: {e}")

# Deterministic fallback
st.markdown(f"**Deterministic summary ({section_title})**")
st.json(payload)
st.caption("ℹ️ To enable AI narratives, set OPENAI_API_KEY and OPENAI_MODEL in Secrets.")
---------------- SAFETY WRAPPER ----------------
from contextlib import contextmanager
@contextmanager
def safe_zone(label: str):
try:
yield
except Exception as e:
logger.error("section_error", section=label, error=str(e))
st.error(f"{label}: something went wrong.")
st.exception(e)
---------------- ACTION FOOTER + DECISION LOG ----------------
if "decision_log" not in st.session_state:
st.session_state["decision_log"] = []
def action_footer(section: str):
st.markdown("#### Action footer")
c1, c2, c3, c4 = st.columns([1.2,1,1,1])
owner = c1.selectbox("Owner",
["House Supervisor","Revenue Integrity","Case Mgmt","Unit Manager","Finance Lead"],
key=f"owner_{section}"
)
decision = c2.selectbox("Decision", ["Promote","Hold","Tune","Investigate"], key=f"decision_{section}")
sla_date = c3.date_input("SLA Date", value=date.today(), key=f"sla_date_{section}")
sla_time = c4.time_input("SLA Time", value=datetime.now().time(), key=f"sla_time_{section}")
note = st.text_input("Notes (optional)", key=f"note_{section}")
CopycolA, colB = st.columns([1,1])
if colA.button(f"Save to Decision Log ({section})", key=f"save_{section}"):
    st.session_state["decision_log"].append({
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "section": section, "owner": owner, "decision": decision,
        "sla": f"{sla_date} {sla_time}", "note": note
    })
    st.success("Saved to Decision Log.")
df_log = pd.DataFrame(st.session_state["decision_log"])
st.download_button(
    label="Download Decision Log (CSV)",
    data=df_log.to_csv(index=False).encode("utf-8"),
    file_name="decision_log.csv", mime="text/csv",
    key=f"dl_{section}"
)
---------------- Progressive UI ----------------
class ProgressiveUI:
@staticmethod
def lazy_component(component_name: str, load_func):
placeholder = st.empty()
if placeholder.button(f"Load {component_name}", key=f"btn_{component_name}"):
with st.spinner(f"Loading {component_name}..."):
placeholder.empty()
return load_func()
else:
st.info(f"Click to load {component_name}")
return None
---------------- NAV ----------------
tabs = st.tabs(["📈 Admissions Control", "🧾 Revenue Watch", "🛏️ LOS Planner"])
===== 1) Admissions Control =====
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
Copyfdx = fdf.copy()
if cohort_dim!="All" and cohort_dim in fdx and cohort_val and cohort_val!="(all)":
    fdx = fdx[fdx[cohort_dim]==cohort_val]

ts = build_timeseries(fdx, metric="intake", freq=freq)
if ts.empty:
    st.info("No admissions series for the current cohort/filters.")
else:
    # Progressive loading for heavy calendar
    def _render_calendar():
        s = fdx.set_index("admit_date").assign(_one=1)["_one"].resample("D").sum().fillna(0.0)
        cal = pd.DataFrame({"dow": s.index.weekday, "woy": s.index.isocalendar().week.values, "val": s.values})
        heat = cal.groupby(["woy","dow"])["val"].mean().reset_index()
        heat_pivot = heat.pivot(index="woy", columns="dow", values="val").fillna(0)
        fig = px.imshow(heat_pivot, aspect="auto", labels=dict(x="Weekday (0=Mon)", y="Week of Year", color="Avg admits"))
        fig.update_layout(height=360, margin=dict(l=10,r=10,b=10,t=30))
        st.plotly_chart(fig, use_container_width=True)
    with st.expander("Seasonality calendar (avg admissions by week-of-year × weekday)"):
        ProgressiveUI.lazy_component("Seasonality Calendar", _render_calendar)

    sc1, sc2 = st.columns(2)
    flu_pct = sc1.slider("Flu surge scenario (±%)", -30, 50, 0, 5, key="adm_flu")
    weather_pct = sc2.slider("Weather impact (±%)", -20, 20, 0, 5, key="adm_wthr")

    # Single scalable forecast (Holt-Winters) — run via cached/async manager
    fc = model_mgr.forecast_one(ts, horizon=horizon)
    adj_factor = (100 + flu_pct + weather_pct) / 100.0
    fc_adj = fc.assign(yhat=fc["yhat"] * adj_factor)

    # Plot
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ts["ds"], y=ts["y"], name="History", mode="lines"))
    fig.add_trace(go.Scatter(x=fc_adj["ds"], y=fc_adj["yhat"], name="Holt-Winters (adj.)", mode="lines"))
    fig.update_layout(title=f"Admissions Forecast — Cohort: {cohort_dim} = {cohort_val if cohort_dim!='All' else 'All'}",
                      height=420, margin=dict(l=10,r=10,b=10,t=50))
    st.plotly_chart(fig, use_container_width=True)

    # Backtest (simple rolling) for transparency
    try:
        s = ts.set_index("ds")["y"].asfreq("D").ffill()
        H = min(14, horizon)
        preds, trues = [], []
        if len(s) >= 4*H + 10:
            for i in range(3,0,-1):
                split = len(s) - i*H
                train = s.iloc[:split]; test = s.iloc[split:split+H]
                yhat = model_mgr._holt_forecast(train, H)
                preds.append(np.asarray(yhat)); trues.append(test.to_numpy())
            yhat_bt, ytrue_bt = np.concatenate(preds), np.concatenate(trues)
            bt_tbl = pd.DataFrame([ts_metrics(ytrue_bt, yhat_bt)], index=["Holt-Winters"])
            st.markdown("#### 📊 Model Performance (backtests)")
            st.markdown(style_lower_better(bt_tbl), unsafe_allow_html=True)
            st.caption("Lower is better. Green → Red indicates rank per metric.")
    except Exception as e:
        logger.warning("backtest_failed", error=str(e))

    # Staffing targets (illustrative)
    st.markdown("#### Staffing targets (illustrative heuristic)")
    daily_fc = fc_adj["yhat"].to_numpy()
    rn_per_shift = np.ceil(daily_fc / 5.0).astype(int)
    targets = pd.DataFrame({
        "Date": pd.to_datetime(fc_adj["ds"]).dt.date,
        "Expected Admissions": np.round(daily_fc, 1),
        "RN/Day": rn_per_shift, "RN/Evening": rn_per_shift, "RN/Night": rn_per_shift
    })
    st.dataframe(targets, use_container_width=True, hide_index=True)

    # AI narrative
    ai_payload = {
        "cohort": {"dimension": cohort_dim, "value": cohort_val},
        "aggregation": agg,
        "horizon_days": horizon,
        "scenario": {"flu_pct": flu_pct, "weather_pct": weather_pct},
        "model": "Holt-Winters",
    }
    st.markdown("---")
    ai_write("Admissions Control", ai_payload)
    action_footer("Admissions Control")
===== 2) Revenue Watch =====
with tabs[1], safe_zone("Revenue Watch"):
st.subheader("🧾 Revenue Watch — Anomalies → Cash Protection")
c1, c2, c3 = st.columns(3)
agg = c1.selectbox("Aggregation", ["Daily","Weekly"], index=0, key="bill_agg")
sensitivity = c2.slider("Sensitivity (higher = fewer alerts)", 1.5, 5.0, 3.0, 0.1, key="bill_sens")
baseline_weeks = c3.slider("Baseline window (weeks)", 2, 12, 4, 1, key="bill_base")
freq = "D" if agg=="Daily" else "W"
Copyts_bill = build_timeseries(fdf, metric="billing_amount", freq=freq)
if ts_bill.empty:
    st.info("No billing series for the current filters.")
    an = None
else:
    an = detect_anomalies(ts_bill, sensitivity)
    plot_anoms(an, "Billing Amount with Anomalies")

    st.markdown("#### Root-cause explorer")
    dims = [d for d in ["insurer","hospital","condition","doctor"] if d in fdf.columns]
    if not dims:
        st.info("No categorical dimensions available for drilldown.")
    else:
        drill = st.selectbox("Group anomalies by", dims, index=0, key="bill_drill")

        if isinstance(an, pd.DataFrame) and not an.empty:
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
===== 3) LOS Planner =====
with tabs[2], safe_zone("LOS Planner"):
st.subheader("🛏️ LOS Planner — Risk Buckets → Discharge Orchestration")
prep = los_prep(fdf)
if prep is None:
st.info("length_of_stay not found.")
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
Copy    pre = ColumnTransformer(
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
            except Exception as e:
                logger.warning("roc_calc_failed", error=str(e))
        results[name] = {"Accuracy":acc, "Precision":pr, "Recall":rc, "F1":f1, "ROC-AUC":auc}
        roc_curves[name] = (fprs, tprs)

    perf = pd.DataFrame(results).T
    for c in ["Accuracy","Precision","Recall","F1","ROC-AUC"]:
        if c not in perf: perf[c] = np.nan
    perf = perf[["Accuracy","Precision","Recall","F1","ROC-AUC"]]

    st.markdown("#### 📊 Model Performance (Classification)")
    st.markdown(style_higher_better(perf), unsafe_allow_html=True)
    st.caption("Higher is better. Green → Red indicates rank per metric.")

    # Progressive loading for ROC (can be heavy)
    def _render_roc():
        top_model = perf["F1"].astype(float).idxmax()
        fprs, tprs = roc_curves[top_model]
        fig = go.Figure()
        for cls in classes:
            if cls in fprs and cls in tprs and len(fprs[cls]) and len(tprs[cls]):
                fig.add_trace(go.Scatter(x=fprs[cls], y=tprs[cls], mode="lines", name=f"{top_model} — {cls}"))
        fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode="lines", name="Chance", line=dict(dash="dash")))
        fig.update_layout(height=420, xaxis_title="False Positive Rate", yaxis_title="True Positive Rate")
        st.plotly_chart(fig, use_container_width=True)
    with st.expander("ROC Curves (one-vs-rest)"):
        ProgressiveUI.lazy_component("ROC Curves", _render_roc)

    # Equity slices
    st.markdown("#### Equity slices")
    slice_dim_options = [d for d in ["insurer","age_group"] if d in d_full.columns]
    slice_dim = st.selectbox("Slice by", slice_dim_options, index=0 if "insurer" in slice_dim_options else 0, key="los_slice") if slice_dim_options else None
    if slice_dim:
        # Choose best model by F1
        top_model = perf["F1"].astype(float).idxmax()
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

    ai_payload = {
        "buckets": {"Short":"<=5","Medium":"6-15","Long":"16-45","Very Long":">45"},
        "metrics_table": perf.to_dict(),
        "equity_slice": slice_dim
    }
    st.markdown("---")
    ai_write("LOS Planner", ai_payload)
    action_footer("LOS Planner")
--------------- FOOTER: Decision Log quick peek ---------------
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

      
