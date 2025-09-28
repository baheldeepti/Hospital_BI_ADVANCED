# app.py — Hospital Ops Studio: World-Class Predictive Analytics Platform
# Enhanced version with comprehensive model selection, AI narratives, and professional UX

import os
import json
import warnings
import traceback
from datetime import datetime, date, timedelta
from typing import Dict, Optional, List, Tuple, Any
import asyncio
import logging

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Core libraries
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# ML and forecasting libraries
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
import seaborn as sns

# Optional libraries with graceful fallback
try:
    from prophet import Prophet
    HAS_PROPHET = True
except ImportError:
    HAS_PROPHET = False

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

try:
    import openai
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

# Configuration
st.set_page_config(
    page_title="Hospital Ops Studio - Predictive Analytics",
    layout="wide",
    page_icon="🏥",
    initial_sidebar_state="expanded"
)

# Enhanced CSS with professional styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        margin-bottom: 2rem;
        text-align: center;
    }
    
    .business-context {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 8px;
        border-left: 4px solid #007bff;
        margin: 1rem 0;
    }
    
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border: 1px solid #e9ecef;
        margin: 0.5rem 0;
    }
    
    .model-comparison {
        background: #fff;
        padding: 1rem;
        border-radius: 8px;
        border: 2px solid #28a745;
        margin: 1rem 0;
    }
    
    .ai-narrative {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    
    .code-section {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #dee2e6;
        font-family: 'Courier New', monospace;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding-left: 20px;
        padding-right: 20px;
        background-color: #f1f3f4;
        border-radius: 8px 8px 0px 0px;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #007bff;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'model_results' not in st.session_state:
    st.session_state.model_results = {}
if 'selected_features' not in st.session_state:
    st.session_state.selected_features = {}
if 'decision_log' not in st.session_state:
    st.session_state.decision_log = []

# Header
st.markdown("""
<div class="main-header">
    <h1>🏥 Hospital Operations Studio</h1>
    <h3>World-Class Predictive Analytics Platform</h3>
    <p>Transforming Healthcare Operations Through Data-Driven Decision Making</p>
</div>
""", unsafe_allow_html=True)

# Data Loading and Management
@st.cache_data(ttl=3600, show_spinner="Loading hospital data...")
def load_hospital_data():
    """Load and prepare hospital operations data"""
    try:
        # Try to load from the provided URL
        url = "https://raw.githubusercontent.com/baheldeepti/hospital-streamlit-app/main/modified_healthcare_dataset.csv"
        df = pd.read_csv(url)
    except Exception as e:
        st.warning(f"Could not load data from URL: {e}. Using synthetic data.")
        # Generate synthetic data
        np.random.seed(42)
        n_records = 10000
        
        dates = pd.date_range(start='2023-01-01', end='2024-01-01', periods=n_records)
        
        df = pd.DataFrame({
            'Date of Admission': dates,
            'Discharge Date': dates + pd.to_timedelta(np.random.exponential(5, n_records), unit='D'),
            'Age': np.random.normal(55, 20, n_records).clip(0, 100),
            'Medical Condition': np.random.choice(['Diabetes', 'Hypertension', 'Heart Disease', 'Cancer', 'Asthma'], n_records),
            'Admission Type': np.random.choice(['Emergency', 'Elective', 'Urgent'], n_records, p=[0.4, 0.4, 0.2]),
            'Hospital': np.random.choice(['General Hospital', 'Children Hospital', 'University Hospital'], n_records),
            'Insurance Provider': np.random.choice(['Medicare', 'Medicaid', 'Blue Cross', 'Aetna', 'UnitedHealth'], n_records),
            'Billing Amount': np.random.lognormal(8, 1, n_records),
            'Doctor': [f'Dr. {name}' for name in np.random.choice(['Smith', 'Johnson', 'Williams', 'Brown', 'Jones'], n_records)],
            'Test Results': np.random.choice(['Normal', 'Abnormal', 'Inconclusive'], n_records, p=[0.6, 0.3, 0.1])
        })
    
    # Data preprocessing
    df.columns = df.columns.str.replace(' ', '_').str.lower()
    
    # Handle date columns
    date_cols = ['date_of_admission', 'discharge_date']
    for col in date_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')
    
    # Calculate length of stay
    if 'date_of_admission' in df.columns and 'discharge_date' in df.columns:
        df['length_of_stay'] = (df['discharge_date'] - df['date_of_admission']).dt.days.clip(1, 365)
    
    # Feature engineering
    if 'date_of_admission' in df.columns:
        df['admission_month'] = df['date_of_admission'].dt.month
        df['admission_day_of_week'] = df['date_of_admission'].dt.dayofweek
        df['admission_quarter'] = df['date_of_admission'].dt.quarter
    
    # Handle billing amount
    if 'billing_amount' in df.columns:
        df['billing_amount'] = pd.to_numeric(df['billing_amount'], errors='coerce').fillna(0).clip(0, None)
    
    # Age groups
    if 'age' in df.columns:
        df['age'] = pd.to_numeric(df['age'], errors='coerce').fillna(df['age'].median())
        df['age_group'] = pd.cut(df['age'], bins=[0, 18, 35, 50, 65, 100], 
                                labels=['Child', 'Young Adult', 'Adult', 'Senior', 'Elderly'])
    
    # LOS categories
    if 'length_of_stay' in df.columns:
        df['los_category'] = pd.cut(df['length_of_stay'], 
                                   bins=[0, 3, 7, 14, float('inf')], 
                                   labels=['Short', 'Medium', 'Long', 'Extended'])
    
    return df.dropna(subset=['date_of_admission'])

# Load data
df = load_hospital_data()

# Sidebar
st.sidebar.header("📊 Analytics Configuration")

# Data overview
st.sidebar.subheader("Data Overview")
st.sidebar.metric("Total Records", f"{len(df):,}")
st.sidebar.metric("Date Range", f"{df['date_of_admission'].min().strftime('%Y-%m-%d')} to {df['date_of_admission'].max().strftime('%Y-%m-%d')}")

# Filters
st.sidebar.subheader("🔍 Data Filters")
hospitals = st.sidebar.multiselect("Select Hospitals", df['hospital'].unique() if 'hospital' in df.columns else [])
conditions = st.sidebar.multiselect("Medical Conditions", df['medical_condition'].unique() if 'medical_condition' in df.columns else [])
admission_types = st.sidebar.multiselect("Admission Types", df['admission_type'].unique() if 'admission_type' in df.columns else [])

# Apply filters
filtered_df = df.copy()
if hospitals:
    filtered_df = filtered_df[filtered_df['hospital'].isin(hospitals)]
if conditions:
    filtered_df = filtered_df[filtered_df['medical_condition'].isin(conditions)]
if admission_types:
    filtered_df = filtered_df[filtered_df['admission_type'].isin(admission_types)]

st.sidebar.metric("Filtered Records", f"{len(filtered_df):,}")

# AI Configuration
st.sidebar.subheader("🤖 AI Configuration")
use_ai = st.sidebar.toggle("Enable AI Narratives", value=True)
ai_mode = st.sidebar.selectbox("AI Report Type", ["Executive Summary", "Technical Analysis", "Comprehensive Report"])

# OpenAI API configuration
openai_api_key = st.sidebar.text_input("OpenAI API Key", type="password", 
                                       value=os.getenv("OPENAI_API_KEY", ""))

# Utility Functions
def generate_ai_narrative(section_title: str, data_summary: dict, model_results: dict = None):
    """Generate AI-powered business narratives"""
    if not use_ai or not openai_api_key:
        return f"""
        **{section_title} - Automated Summary**
        
        📊 **Key Findings:**
        • Data analysis completed successfully
        • {len(filtered_df):,} records analyzed
        • Multiple models trained and evaluated
        
        💡 **Business Impact:**
        • Predictive models ready for deployment
        • Insights available for operational decision-making
        • Continuous monitoring recommended
        
        🎯 **Next Steps:**
        • Review model performance metrics
        • Implement top-performing model
        • Schedule regular model retraining
        
        *Enable AI narratives with OpenAI API key for detailed insights*
        """
    
    try:
        openai.api_key = openai_api_key
        
        prompt = f"""
        You are a senior healthcare analytics consultant. Generate a comprehensive business report for {section_title}.
        
        Data Summary: {json.dumps(data_summary, default=str)}
        Model Results: {json.dumps(model_results, default=str) if model_results else "Not available"}
        
        Create a structured report with:
        1. Executive Summary (business impact, key findings, ROI implications)
        2. Technical Analysis (model performance, statistical insights)
        3. Operational Recommendations (specific actions, timelines, owners)
        4. Risk Assessment (potential issues, mitigation strategies)
        5. Success Metrics (KPIs to track, expected outcomes)
        
        Make it {ai_mode.lower()} focused. Use specific numbers and actionable insights.
        """
        
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1000,
            temperature=0.3
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
        return f"""
        **{section_title} - Analysis Summary**
        
        📊 **Data Overview:** {len(filtered_df):,} records processed
        
        🔍 **Key Insights:**
        • Analysis completed successfully
        • Multiple predictive models evaluated
        • Performance metrics calculated
        
        ⚠️ **Note:** AI narrative generation unavailable ({str(e)[:100]}...)
        
        💡 **Recommendation:** Review detailed metrics and model comparisons below
        """

def create_model_comparison_table(results: dict) -> pd.DataFrame:
    """Create a formatted model comparison table with conditional formatting"""
    if not results:
        return pd.DataFrame()
    
    df_results = pd.DataFrame(results).T
    return df_results

def plot_model_performance(results: dict, metric: str = 'accuracy'):
    """Create model performance visualization"""
    if not results:
        return None
    
    models = list(results.keys())
    scores = [results[model].get(metric, 0) for model in models]
    
    fig = go.Figure(data=[
        go.Bar(x=models, y=scores, 
               marker_color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'][:len(models)],
               text=[f'{score:.3f}' for score in scores],
               textposition='auto')
    ])
    
    fig.update_layout(
        title=f'Model Performance Comparison - {metric.title()}',
        xaxis_title='Models',
        yaxis_title=metric.title(),
        height=400,
        showlegend=False
    )
    
    return fig

def generate_python_code(model_type: str, features: list, target: str) -> str:
    """Generate Python code for model training"""
    code = f'''
# Hospital Operations Predictive Model - {model_type}
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Load and prepare data
df = pd.read_csv('hospital_data.csv')

# Feature selection
features = {features}
target = '{target}'

X = df[features]
y = df[target]

# Preprocessing
numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_features = X.select_dtypes(include=['object']).columns.tolist()

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_features)
    ]
)

# Model setup
'''
    
    if model_type == "Random Forest":
        code += '''
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
'''
    elif model_type == "Logistic Regression":
        code += '''
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(random_state=42)
'''
    elif model_type == "XGBoost":
        code += '''
from xgboost import XGBClassifier
model = XGBClassifier(random_state=42)
'''
    elif model_type == "SVM":
        code += '''
from sklearn.svm import SVC
model = SVC(probability=True, random_state=42)
'''
    
    code += '''
# Create pipeline
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', model)
])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
pipeline.fit(X_train, y_train)

# Make predictions
y_pred = pipeline.predict(X_test)
y_pred_proba = pipeline.predict_proba(X_test)

# Evaluate model
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
print(f"Accuracy: {accuracy_score(y_test, y_pred):.3f}")
print("\\nClassification Report:")
print(classification_report(y_test, y_pred))
'''
    
    return code

# Main Application Tabs
tabs = st.tabs(["📈 Admissions Forecasting", "💰 Revenue Anomaly Detection", "🛏️ Length of Stay Prediction"])

# Tab 1: Admissions Forecasting
with tabs[0]:
    st.markdown("### 📈 Admissions Control - Predictive Staffing & Resource Planning")
    
    # Business context
    st.markdown("""
    <div class="business-context">
        <h4>🎯 Business Problem</h4>
        <p><strong>Challenge:</strong> Hospitals struggle with unpredictable patient admission patterns, leading to staffing shortages, overcrowding, and increased costs.</p>
        <p><strong>Solution:</strong> Advanced time series forecasting predicts daily admissions 7-30 days ahead, enabling proactive staffing decisions and resource allocation.</p>
        <p><strong>Impact:</strong> Reduce overtime costs by 15-25%, improve patient satisfaction scores, and optimize bed utilization rates.</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📊 Current Admission Trends")
        
        # Create time series
        daily_admissions = filtered_df.groupby(filtered_df['date_of_admission'].dt.date).size().reset_index()
        daily_admissions.columns = ['date', 'admissions']
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=daily_admissions['date'], y=daily_admissions['admissions'],
                                mode='lines+markers', name='Daily Admissions',
                                line=dict(color='#007bff', width=2)))
        
        # Add trend line
        x_numeric = np.arange(len(daily_admissions))
        z = np.polyfit(x_numeric, daily_admissions['admissions'], 1)
        p = np.poly1d(z)
        fig.add_trace(go.Scatter(x=daily_admissions['date'], y=p(x_numeric),
                                mode='lines', name='Trend', line=dict(color='red', dash='dash')))
        
        fig.update_layout(title='Daily Hospital Admissions Over Time', 
                         xaxis_title='Date', yaxis_title='Number of Admissions',
                         height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("📈 Key Metrics")
        
        avg_daily = daily_admissions['admissions'].mean()
        max_daily = daily_admissions['admissions'].max()
        std_daily = daily_admissions['admissions'].std()
        
        st.markdown(f"""
        <div class="metric-card">
            <h4>Average Daily Admissions</h4>
            <h2 style="color: #007bff;">{avg_daily:.1f}</h2>
        </div>
        <div class="metric-card">
            <h4>Peak Daily Admissions</h4>
            <h2 style="color: #28a745;">{max_daily}</h2>
        </div>
        <div class="metric-card">
            <h4>Daily Variability (σ)</h4>
            <h2 style="color: #ffc107;">{std_daily:.1f}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    st.subheader("🔧 Model Configuration")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        forecast_days = st.slider("Forecast Horizon (days)", 7, 30, 14)
        
    with col2:
        seasonality = st.selectbox("Seasonality Pattern", 
                                  ["Auto-detect", "Weekly", "Monthly", "None"])
    
    with col3:
        confidence_level = st.slider("Confidence Level", 0.8, 0.99, 0.95)
    
    # Model Selection and Training
    st.subheader("🤖 Forecasting Models")
    
    selected_models = st.multiselect(
        "Select Forecasting Models",
        ["Linear Trend", "ARIMA", "Exponential Smoothing", "Prophet"],
        default=["Linear Trend", "ARIMA", "Exponential Smoothing"]
    )
    
    if st.button("🚀 Train Forecasting Models", key="train_forecast"):
        with st.spinner("Training forecasting models..."):
            results = {}
            forecasts = {}
            
            # Prepare time series data
            ts_data = daily_admissions.set_index('date')['admissions']
            train_size = len(ts_data) - forecast_days
            train_data = ts_data[:train_size]
            test_data = ts_data[train_size:]
            
            # Linear Trend
            if "Linear Trend" in selected_models:
                try:
                    x = np.arange(len(train_data))
                    coeffs = np.polyfit(x, train_data.values, 1)
                    
                    # Forecast
                    future_x = np.arange(len(train_data), len(train_data) + forecast_days)
                    forecast = np.polyval(coeffs, future_x)
                    forecasts["Linear Trend"] = forecast
                    
                    # Calculate metrics
                    if len(test_data) > 0:
                        test_forecast = np.polyval(coeffs, np.arange(len(train_data), len(ts_data)))
                        mae = mean_absolute_error(test_data.values, test_forecast)
                        rmse = np.sqrt(mean_squared_error(test_data.values, test_forecast))
                        mape = np.mean(np.abs((test_data.values - test_forecast) / test_data.values)) * 100
                        results["Linear Trend"] = {"MAE": mae, "RMSE": rmse, "MAPE": mape}
                except Exception as e:
                    st.warning(f"Linear Trend failed: {str(e)}")
            
            # ARIMA
            if "ARIMA" in selected_models:
                try:
                    model = ARIMA(train_data, order=(1,1,1))
                    fitted_model = model.fit()
                    forecast = fitted_model.forecast(steps=forecast_days)
                    forecasts["ARIMA"] = forecast
                    
                    if len(test_data) > 0:
                        test_forecast = fitted_model.forecast(steps=len(test_data))
                        mae = mean_absolute_error(test_data.values, test_forecast)
                        rmse = np.sqrt(mean_squared_error(test_data.values, test_forecast))
                        mape = np.mean(np.abs((test_data.values - test_forecast) / test_data.values)) * 100
                        results["ARIMA"] = {"MAE": mae, "RMSE": rmse, "MAPE": mape}
                except Exception as e:
                    st.warning(f"ARIMA failed: {str(e)}")
            
            # Exponential Smoothing
            if "Exponential Smoothing" in selected_models:
                try:
                    model = ExponentialSmoothing(train_data, trend='add', seasonal='add', seasonal_periods=7)
                    fitted_model = model.fit()
                    forecast = fitted_model.forecast(steps=forecast_days)
                    forecasts["Exponential Smoothing"] = forecast
                    
                    if len(test_data) > 0:
                        test_forecast = fitted_model.forecast(steps=len(test_data))
                        mae = mean_absolute_error(test_data.values, test_forecast)
                        rmse = np.sqrt(mean_squared_error(test_data.values, test_forecast))
                        mape = np.mean(np.abs((test_data.values - test_forecast) / test_data.values)) * 100
                        results["Exponential Smoothing"] = {"MAE": mae, "RMSE": rmse, "MAPE": mape}
                except Exception as e:
                    st.warning(f"Exponential Smoothing failed: {str(e)}")
            
            # Prophet
            if "Prophet" in selected_models and HAS_PROPHET:
                try:
                    prophet_data = train_data.reset_index()
                    prophet_data.columns = ['ds', 'y']
                    
                    model = Prophet(daily_seasonality=False, weekly_seasonality=True, yearly_seasonality=False)
                    model.fit(prophet_data)
                    
                    future = model.make_future_dataframe(periods=forecast_days)
                    forecast = model.predict(future)
                    forecasts["Prophet"] = forecast['yhat'].tail(forecast_days).values
                    
                    if len(test_data) > 0:
                        test_future = model.make_future_dataframe(periods=len(test_data))
                        test_forecast_full = model.predict(test_future)
                        test_forecast = test_forecast_full['yhat'].tail(len(test_data)).values
                        mae = mean_absolute_error(test_data.values, test_forecast)
                        rmse = np.sqrt(mean_squared_error(test_data.values, test_forecast))
                        mape = np.mean(np.abs((test_data.values - test_forecast) / test_data.values)) * 100
                        results["Prophet"] = {"MAE": mae, "RMSE": rmse, "MAPE": mape}
                except Exception as e:
                    st.warning(f"Prophet failed: {str(e)}")
            
            # Store results
            st.session_state.model_results['forecasting'] = results
            
            # Display results
            if results:
                st.subheader("📊 Model Performance Comparison")
                
                results_df = pd.DataFrame(results).T
                
                # Color coding for better models (lower is better for these metrics)
                def highlight_best(s):
                    is_min = s == s.min()
                    return ['background-color: #d4edda; color: #155724' if v else '' for v in is_min]
                
                styled_df = results_df.style.apply(highlight_best, axis=0).format("{:.2f}")
                st.dataframe(styled_df, use_container_width=True)
                
                # Forecast visualization
                st.subheader("🔮 Admission Forecasts")
                
                fig = go.Figure()
                
                # Historical data
                fig.add_trace(go.Scatter(x=ts_data.index, y=ts_data.values,
                                        mode='lines', name='Historical',
                                        line=dict(color='#007bff')))
                
                # Forecasts
                future_dates = pd.date_range(start=ts_data.index[-1] + pd.Timedelta(days=1), 
                                           periods=forecast_days, freq='D')
                
                colors = ['#28a745', '#dc3545', '#ffc107', '#17a2b8']
                for i, (model_name, forecast) in enumerate(forecasts.items()):
                    fig.add_trace(go.Scatter(x=future_dates, y=forecast,
                                            mode='lines+markers', name=f'{model_name} Forecast',
                                            line=dict(color=colors[i % len(colors)], dash='dash')))
                
                fig.update_layout(title='Admission Forecasts by Model',
                                 xaxis_title='Date', yaxis_title='Predicted Admissions',
                                 height=500)
                st.plotly_chart(fig, use_container_width=True)
                
                # Business Impact
                st.subheader("💼 Business Impact Analysis")
                
                best_model = min(results.keys(), key=lambda x: results[x]['MAPE'])
                best_forecast = forecasts[best_model]
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    avg_forecast = np.mean(best_forecast)
                    st.metric("Avg Daily Admissions (Forecast)", f"{avg_forecast:.1f}", 
                             f"{((avg_forecast - avg_daily) / avg_daily * 100):+.1f}%")
                
                with col2:
                    peak_forecast = np.max(best_forecast)
                    nurses_needed = int(np.ceil(peak_forecast / 8))  # Assume 8 patients per nurse
                    st.metric("Peak Day Nursing Staff", f"{nurses_needed} nurses", 
                             f"{peak_forecast:.0f} admissions")
                
                with col3:
                    capacity_utilization = (avg_forecast / max_daily) * 100
                    st.metric("Capacity Utilization", f"{capacity_utilization:.1f}%")
                
                # Generate staffing recommendations
                st.subheader("👥 Staffing Recommendations")
                
                staffing_df = pd.DataFrame({
                    'Date': future_dates,
                    'Predicted Admissions': best_forecast,
                    'Nurses Needed (Day)': np.ceil(best_forecast / 8).astype(int),
                    'Nurses Needed (Night)': np.ceil(best_forecast / 12).astype(int),
                    'Total Staff Budget ($)': (np.ceil(best_forecast / 8) * 350 + np.ceil(best_forecast / 12) * 400).astype(int)
                })
                
                st.dataframe(staffing_df, use_container_width=True)
                
                # Python code generation
                st.subheader("💻 Python Code for Implementation")
                
                code = f'''
# Hospital Admissions Forecasting Model
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error

# Load data
df = pd.read_csv('hospital_admissions.csv')
df['date'] = pd.to_datetime(df['date'])
daily_admissions = df.groupby('date').size()

# Train {best_model} model (best performing)
forecast_days = {forecast_days}
train_size = len(daily_admissions) - forecast_days
train_data = daily_admissions[:train_size]

# Model training and forecasting
model = ARIMA(train_data, order=(1,1,1))
fitted_model = model.fit()
forecast = fitted_model.forecast(steps=forecast_days)

print(f"Model Performance - MAPE: {results[best_model]['MAPE']:.2f}%")
print(f"Average daily admissions forecast: {np.mean(forecast):.1f}")

# Staffing recommendations
nurses_per_day = np.ceil(forecast / 8).astype(int)
print("Recommended nursing staff by day:")
for i, nurses in enumerate(nurses_per_day):
    print(f"Day {i+1}: {nurses} nurses")
'''
                
                with st.expander("View Complete Implementation Code"):
                    st.code(code, language='python')
    
    # AI Narrative
    if results:
        st.markdown("---")
        st.markdown("## 🤖 AI-Generated Business Report")
        
        data_summary = {
            "total_records": len(filtered_df),
            "forecast_horizon": forecast_days,
            "average_daily_admissions": float(avg_daily),
            "peak_admissions": int(max_daily),
            "best_model": best_model,
            "best_model_mape": float(results[best_model]['MAPE']),
            "predicted_avg_admissions": float(np.mean(best_forecast)),
            "capacity_utilization": float(capacity_utilization)
        }
        
        narrative = generate_ai_narrative("Admissions Forecasting", data_summary, results)
        
        st.markdown(f"""
        <div class="ai-narrative">
            {narrative}
        </div>
        """, unsafe_allow_html=True)

# Tab 2: Revenue Anomaly Detection
with tabs[1]:
    st.markdown("### 💰 Revenue Watch - Intelligent Anomaly Detection & Loss Prevention")
    
    # Business context
    st.markdown("""
    <div class="business-context">
        <h4>🎯 Business Problem</h4>
        <p><strong>Challenge:</strong> Revenue leakage through billing errors, fraud, and system anomalies costs hospitals millions annually.</p>
        <p><strong>Solution:</strong> Machine learning algorithms detect unusual billing patterns, flagging potential revenue risks in real-time.</p>
        <p><strong>Impact:</strong> Recover 2-5% of revenue through early anomaly detection, prevent fraud losses, and improve billing accuracy.</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("💳 Revenue Trends Analysis")
        
        # Daily revenue
        daily_revenue = filtered_df.groupby(filtered_df['date_of_admission'].dt.date)['billing_amount'].sum().reset_index()
        daily_revenue.columns = ['date', 'revenue']
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=daily_revenue['date'], y=daily_revenue['revenue'],
                                mode='lines+markers', name='Daily Revenue',
                                line=dict(color='#28a745', width=2)))
        
        fig.update_layout(title='Daily Hospital Revenue',
                         xaxis_title='Date', yaxis_title='Revenue ($)',
                         height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("💰 Revenue Metrics")
        
        total_revenue = filtered_df['billing_amount'].sum()
        avg_daily_revenue = daily_revenue['revenue'].mean()
        avg_bill = filtered_df['billing_amount'].mean()
        
        st.markdown(f"""
        <div class="metric-card">
            <h4>Total Revenue</h4>
            <h2 style="color: #28a745;">${total_revenue:,.0f}</h2>
        </div>
        <div class="metric-card">
            <h4>Avg Daily Revenue</h4>
            <h2 style="color: #007bff;">${avg_daily_revenue:,.0f}</h2>
        </div>
        <div class="metric-card">
            <h4>Avg Bill Amount</h4>
            <h2 style="color: #ffc107;">${avg_bill:,.0f}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    st.subheader("🔧 Anomaly Detection Configuration")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        detection_method = st.selectbox("Detection Method", 
                                       ["Isolation Forest", "Statistical Outliers", "Ensemble"])
    
    with col2:
        sensitivity = st.slider("Sensitivity", 0.01, 0.1, 0.05, 0.01)
    
    with col3:
        features_for_anomaly = st.multiselect(
            "Features for Detection",
            ['billing_amount', 'age', 'length_of_stay', 'admission_month', 'admission_day_of_week'],
            default=['billing_amount', 'length_of_stay']
        )
    
    if st.button("🚀 Run Anomaly Detection", key="detect_anomalies"):
        if features_for_anomaly:
            with st.spinner("Detecting anomalies..."):
                # Prepare data
                X = filtered_df[features_for_anomaly].dropna()
                
                results = {}
                anomaly_predictions = {}
                
                # Isolation Forest
                if detection_method in ["Isolation Forest", "Ensemble"]:
                    iso_forest = IsolationForest(contamination=sensitivity, random_state=42)
                    anomaly_pred = iso_forest.fit_predict(X)
                    anomaly_predictions["Isolation Forest"] = anomaly_pred
                    
                    n_anomalies = sum(anomaly_pred == -1)
                    anomaly_rate = n_anomalies / len(X)
                    results["Isolation Forest"] = {
                        "anomalies_detected": n_anomalies,
                        "anomaly_rate": anomaly_rate,
                        "avg_anomaly_amount": filtered_df.loc[X.index[anomaly_pred == -1], 'billing_amount'].mean() if n_anomalies > 0 else 0
                    }
                
                # Statistical Outliers
                if detection_method in ["Statistical Outliers", "Ensemble"]:
                    # Z-score based detection
                    z_scores = np.abs((X - X.mean()) / X.std())
                    threshold = 3  # 3 standard deviations
                    stat_anomalies = (z_scores > threshold).any(axis=1)
                    
                    n_anomalies = sum(stat_anomalies)
                    anomaly_rate = n_anomalies / len(X)
                    results["Statistical Outliers"] = {
                        "anomalies_detected": n_anomalies,
                        "anomaly_rate": anomaly_rate,
                        "avg_anomaly_amount": filtered_df.loc[X.index[stat_anomalies], 'billing_amount'].mean() if n_anomalies > 0 else 0
                    }
                    
                    anomaly_predictions["Statistical Outliers"] = stat_anomalies.astype(int) * 2 - 1  # Convert to -1/1
                
                st.session_state.model_results['anomaly'] = results
                
                # Display results
                st.subheader("🎯 Anomaly Detection Results")
                
                results_df = pd.DataFrame(results).T
                st.dataframe(results_df.style.format({
                    'anomalies_detected': '{:.0f}',
                    'anomaly_rate': '{:.2%}',
                    'avg_anomaly_amount': '${:,.0f}'
                }), use_container_width=True)
                
                # Visualization
                st.subheader("📊 Anomaly Visualization")
                
                # Use the best method (lowest anomaly rate that's still meaningful)
                best_method = min(results.keys(), key=lambda x: abs(results[x]['anomaly_rate'] - 0.05))
                best_predictions = anomaly_predictions[best_method]
                
                # Create anomaly visualization
                if len(features_for_anomaly) >= 2:
                    fig = px.scatter(x=X[features_for_anomaly[0]], 
                                   y=X[features_for_anomaly[1]],
                                   color=best_predictions,
                                   title=f'Anomaly Detection Results - {best_method}',
                                   color_discrete_map={1: 'blue', -1: 'red'},
                                   labels={'color': 'Anomaly'})
                    fig.update_layout(height=500)
                    st.plotly_chart(fig, use_container_width=True)
                
                # Anomaly details
                anomaly_indices = X.index[best_predictions == -1]
                if len(anomaly_indices) > 0:
                    st.subheader("🚨 Flagged Cases")
                    
                    anomaly_details = filtered_df.loc[anomaly_indices, 
                                                    ['date_of_admission', 'billing_amount', 'medical_condition', 
                                                     'hospital', 'insurance_provider', 'doctor']].head(20)
                    st.dataframe(anomaly_details, use_container_width=True)
                    
                    # Financial impact
                    total_anomaly_amount = filtered_df.loc[anomaly_indices, 'billing_amount'].sum()
                    potential_recovery = total_anomaly_amount * 0.15  # Assume 15% recovery rate
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Flagged Amount", f"${total_anomaly_amount:,.0f}")
                    with col2:
                        st.metric("Potential Recovery (15%)", f"${potential_recovery:,.0f}")
                    with col3:
                        st.metric("Cases to Review", f"{len(anomaly_indices)}")
                
                # Python code
                st.subheader("💻 Implementation Code")
                
                code = f'''
# Revenue Anomaly Detection System
from sklearn.ensemble import IsolationForest
import pandas as pd
import numpy as np

# Load and prepare data
df = pd.read_csv('hospital_revenue.csv')
features = {features_for_anomaly}
X = df[features].dropna()

# Train {best_method} model
model = IsolationForest(contamination={sensitivity}, random_state=42)
anomaly_predictions = model.fit_predict(X)

# Identify anomalies
anomalies = X[anomaly_predictions == -1]
print(f"Detected {{len(anomalies)}} anomalies out of {{len(X)}} records")

# Calculate potential impact
anomaly_amounts = df.loc[anomalies.index, 'billing_amount']
total_flagged = anomaly_amounts.sum()
potential_recovery = total_flagged * 0.15  # 15% recovery rate

print(f"Total flagged amount: ${{total_flagged:,.0f}}")
print(f"Potential recovery: ${{potential_recovery:,.0f}}")

# Real-time scoring function
def score_new_case(case_data):
    score = model.decision_function([case_data])[0]
    is_anomaly = model.predict([case_data])[0] == -1
    return score, is_anomaly
'''
                
                with st.expander("View Complete Implementation Code"):
                    st.code(code, language='python')
        else:
            st.warning("Please select at least one feature for anomaly detection.")
    
    # AI Narrative for Revenue Watch
    if 'anomaly' in st.session_state.model_results:
        st.markdown("---")
        st.markdown("## 🤖 AI-Generated Business Report")
        
        results = st.session_state.model_results['anomaly']
        best_method = min(results.keys(), key=lambda x: abs(results[x]['anomaly_rate'] - 0.05))
        
        data_summary = {
            "total_revenue": float(total_revenue),
            "avg_daily_revenue": float(avg_daily_revenue),
            "detection_method": detection_method,
            "best_method": best_method,
            "anomalies_detected": int(results[best_method]['anomalies_detected']),
            "anomaly_rate": float(results[best_method]['anomaly_rate']),
            "potential_recovery": float(results[best_method]['avg_anomaly_amount'] * results[best_method]['anomalies_detected'] * 0.15)
        }
        
        narrative = generate_ai_narrative("Revenue Anomaly Detection", data_summary, results)
        
        st.markdown(f"""
        <div class="ai-narrative">
            {narrative}
        </div>
        """, unsafe_allow_html=True)

# Tab 3: Length of Stay Prediction
with tabs[2]:
    st.markdown("### 🛏️ Length of Stay Prediction - Smart Discharge Planning")
    
    # Business context
    st.markdown("""
    <div class="business-context">
        <h4>🎯 Business Problem</h4>
        <p><strong>Challenge:</strong> Unpredictable discharge timing leads to bed management issues, delayed admissions, and increased costs.</p>
        <p><strong>Solution:</strong> ML models predict patient LOS at admission, enabling proactive discharge planning and resource optimization.</p>
        <p><strong>Impact:</strong> Improve bed turnover by 10-20%, reduce average LOS by 0.5-1 days, and enhance patient flow efficiency.</p>
    </div>
    """, unsafe_allow_html=True)
    
    if 'length_of_stay' not in filtered_df.columns:
        st.error("Length of stay data not available. Please check your dataset.")
    else:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("📊 Length of Stay Distribution")
            
            fig = px.histogram(filtered_df, x='length_of_stay', nbins=30,
                             title='Distribution of Length of Stay',
                             color_discrete_sequence=['#007bff'])
            fig.update_layout(height=400, xaxis_title='Length of Stay (days)', yaxis_title='Number of Patients')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("🏥 LOS Metrics")
            
            avg_los = filtered_df['length_of_stay'].mean()
            median_los = filtered_df['length_of_stay'].median()
            max_los = filtered_df['length_of_stay'].max()
            
            st.markdown(f"""
            <div class="metric-card">
                <h4>Average LOS</h4>
                <h2 style="color: #007bff;">{avg_los:.1f} days</h2>
            </div>
            <div class="metric-card">
                <h4>Median LOS</h4>
                <h2 style="color: #28a745;">{median_los:.1f} days</h2>
            </div>
            <div class="metric-card">
                <h4>Maximum LOS</h4>
                <h2 style="color: #dc3545;">{max_los:.0f} days</h2>
            </div>
            """, unsafe_allow_html=True)
        
        # LOS by category analysis
        st.subheader("📈 LOS by Category")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if 'medical_condition' in filtered_df.columns:
                fig = px.box(filtered_df, x='medical_condition', y='length_of_stay',
                           title='LOS by Medical Condition')
                fig.update_xaxes(tickangle=45)
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            if 'admission_type' in filtered_df.columns:
                fig = px.box(filtered_df, x='admission_type', y='length_of_stay',
                           title='LOS by Admission Type')
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("🔧 Model Configuration")
        
        # Feature selection
        available_features = [col for col in filtered_df.columns if col not in 
                            ['length_of_stay', 'los_category', 'date_of_admission', 'discharge_date']]
        
        col1, col2 = st.columns(2)
        
        with col1:
            selected_features = st.multiselect(
                "Select Features for Prediction",
                available_features,
                default=[f for f in ['age', 'medical_condition', 'admission_type', 'hospital'] if f in available_features]
            )
        
        with col2:
            target_type = st.selectbox("Prediction Target", 
                                      ["Length of Stay (Regression)", "LOS Category (Classification)"])
            
            selected_models = st.multiselect(
                "Select Models",
                ["Random Forest", "Logistic Regression", "XGBoost", "SVM"],
                default=["Random Forest", "Logistic Regression"]
            )
        
        if st.button("🚀 Train LOS Prediction Models", key="train_los"):
            if selected_features:
                with st.spinner("Training LOS prediction models..."):
                    # Prepare data
                    feature_data = filtered_df[selected_features + ['length_of_stay', 'los_category']].dropna()
                    
                    if target_type == "Length of Stay (Regression)":
                        target = 'length_of_stay'
                        is_classification = False
                    else:
                        target = 'los_category'
                        is_classification = True
                    
                    X = feature_data[selected_features]
                    y = feature_data[target]
                    
                    # Preprocessing
                    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
                    categorical_features = X.select_dtypes(include=['object']).columns.tolist()
                    
                    preprocessor = ColumnTransformer(
                        transformers=[
                            ('num', StandardScaler(), numeric_features),
                            ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_features)
                        ]
                    )
                    
                    # Train-test split
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                    
                    results = {}
                    models_trained = {}
                    
                    for model_name in selected_models:
                        try:
                            # Select model
                            if is_classification:
                                if model_name == "Random Forest":
                                    model = RandomForestClassifier(n_estimators=100, random_state=42)
                                elif model_name == "Logistic Regression":
                                    model = LogisticRegression(random_state=42, max_iter=1000)
                                elif model_name == "XGBoost" and HAS_XGBOOST:
                                    model = xgb.XGBClassifier(random_state=42)
                                elif model_name == "SVM":
                                    model = SVC(probability=True, random_state=42)
                                else:
                                    continue
                            else:
                                if model_name == "Random Forest":
                                    model = RandomForestRegressor(n_estimators=100, random_state=42)
                                elif model_name == "Linear Regression":
                                    model = LinearRegression()
                                elif model_name == "XGBoost" and HAS_XGBOOST:
                                    model = xgb.XGBRegressor(random_state=42)
                                elif model_name == "SVM":
                                    model = SVR()
                                else:
                                    continue
                            
                            # Create pipeline
                            pipeline = Pipeline([
                                ('preprocessor', preprocessor),
                                ('classifier' if is_classification else 'regressor', model)
                            ])
                            
                            # Train model
                            pipeline.fit(X_train, y_train)
                            models_trained[model_name] = pipeline
                            
                            # Make predictions
                            y_pred = pipeline.predict(X_test)
                            
                            if is_classification:
                                # Classification metrics
                                accuracy = accuracy_score(y_test, y_pred)
                                precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')
                                
                                try:
                                    y_pred_proba = pipeline.predict_proba(X_test)
                                    roc_auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr')
                                except:
                                    roc_auc = None
                                
                                results[model_name] = {
                                    'accuracy': accuracy,
                                    'precision': precision,
                                    'recall': recall,
                                    'f1_score': f1,
                                    'roc_auc': roc_auc
                                }
                            else:
                                # Regression metrics
                                mae = mean_absolute_error(y_test, y_pred)
                                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                                r2 = r2_score(y_test, y_pred)
                                
                                results[model_name] = {
                                    'mae': mae,
                                    'rmse': rmse,
                                    'r2_score': r2
                                }
                        
                        except Exception as e:
                            st.warning(f"Failed to train {model_name}: {str(e)}")
                            continue
                    
                    if results:
                        st.session_state.model_results['los'] = results
                        
                        # Display results
                        st.subheader("📊 Model Performance Comparison")
                        
                        results_df = pd.DataFrame(results).T
                        
                        # Apply conditional formatting
                        if is_classification:
                            def highlight_max(s):
                                is_max = s == s.max()
                                return ['background-color: #d4edda; color: #155724' if v else '' for v in is_max]
                            styled_df = results_df.style.apply(highlight_max, axis=0)
                        else:
                            def highlight_best(s):
                                if s.name == 'r2_score':
                                    is_best = s == s.max()
                                else:
                                    is_best = s == s.min()
                                return ['background-color: #d4edda; color: #155724' if v else '' for v in is_best]
                            styled_df = results_df.style.apply(highlight_best, axis=0)
                        
                        st.dataframe(styled_df.format("{:.3f}"), use_container_width=True)
                        
                        # Performance visualization
                        if is_classification:
                            metric_to_plot = 'accuracy'
                        else:
                            metric_to_plot = 'r2_score'
                        
                        fig = plot_model_performance(results, metric_to_plot)
                        if fig:
                            st.plotly_chart(fig, use_container_width=True)
                        
                        # Feature importance (for tree-based models)
                        st.subheader("🎯 Feature Importance Analysis")
                        
                        best_model_name = max(results.keys(), key=lambda x: results[x].get(metric_to_plot, 0))
                        best_model = models_trained[best_model_name]
                        
                        try:
                            if hasattr(best_model.named_steps['classifier' if is_classification else 'regressor'], 'feature_importances_'):
                                # Get feature names after preprocessing
                                preprocessor_fitted = best_model.named_steps['preprocessor']
                                
                                feature_names = []
                                if numeric_features:
                                    feature_names.extend(numeric_features)
                                if categorical_features:
                                    cat_feature_names = preprocessor_fitted.named_transformers_['cat'].get_feature_names_out(categorical_features)
                                    feature_names.extend(cat_feature_names)
                                
                                importances = best_model.named_steps['classifier' if is_classification else 'regressor'].feature_importances_
                                
                                importance_df = pd.DataFrame({
                                    'feature': feature_names[:len(importances)],
                                    'importance': importances
                                }).sort_values('importance', ascending=False).head(10)
                                
                                fig = px.bar(importance_df, x='importance', y='feature', orientation='h',
                                           title=f'Top 10 Feature Importances - {best_model_name}')
                                fig.update_layout(height=400)
                                st.plotly_chart(fig, use_container_width=True)
                        
                        except Exception as e:
                            st.info(f"Feature importance not available for {best_model_name}")
                        
                        # Confusion Matrix for Classification
                        if is_classification:
                            st.subheader("🔍 Confusion Matrix")
                            
                            y_pred_best = models_trained[best_model_name].predict(X_test)
                            cm = confusion_matrix(y_test, y_pred_best)
                            
                            fig = px.imshow(cm, text_auto=True, aspect="auto",
                                          title=f'Confusion Matrix - {best_model_name}')
                            fig.update_layout(height=400)
                            st.plotly_chart(fig, use_container_width=True)
                        
                        # ROC Curves for Classification
                        if is_classification and len(np.unique(y)) > 2:
                            st.subheader("📈 ROC Curves")
                            
                            fig = go.Figure()
                            
                            for model_name, pipeline in models_trained.items():
                                try:
                                    y_pred_proba = pipeline.predict_proba(X_test)
                                    
                                    # For multiclass, plot ROC for each class
                                    classes = pipeline.classes_
                                    for i, class_name in enumerate(classes):
                                        if len(classes) > 2:
                                            # One-vs-rest ROC
                                            y_binary = (y_test == class_name).astype(int)
                                            if len(np.unique(y_binary)) > 1:
                                                fpr, tpr, _ = roc_curve(y_binary, y_pred_proba[:, i])
                                                auc_score = roc_auc_score(y_binary, y_pred_proba[:, i])
                                                fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines',
                                                                       name=f'{model_name} - {class_name} (AUC: {auc_score:.3f})'))
                                
                                except Exception:
                                    continue
                            
                            # Add diagonal line
                            fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines',
                                                   line=dict(dash='dash', color='gray'),
                                                   name='Random Classifier'))
                            
                            fig.update_layout(title='ROC Curves by Class',
                                             xaxis_title='False Positive Rate',
                                             yaxis_title='True Positive Rate',
                                             height=500)
                            st.plotly_chart(fig, use_container_width=True)
                        
                        # Business Impact Analysis
                        st.subheader("💼 Business Impact Analysis")
                        
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            if is_classification:
                                accuracy = results[best_model_name]['accuracy']
                                st.metric("Best Model Accuracy", f"{accuracy:.1%}")
                            else:
                                r2 = results[best_model_name]['r2_score']
                                st.metric("Best Model R²", f"{r2:.3f}")
                        
                        with col2:
                            if is_classification:
                                predicted_categories = models_trained[best_model_name].predict(X_test)
                                short_stay_rate = sum(predicted_categories == 'Short') / len(predicted_categories)
                                st.metric("Predicted Short Stays", f"{short_stay_rate:.1%}")
                            else:
                                y_pred_best = models_trained[best_model_name].predict(X_test)
                                avg_predicted_los = np.mean(y_pred_best)
                                st.metric("Avg Predicted LOS", f"{avg_predicted_los:.1f} days")
                        
                        with col3:
                            # Calculate potential bed savings
                            if not is_classification:
                                current_avg_los = filtered_df['length_of_stay'].mean()
                                potential_reduction = max(0, current_avg_los - avg_predicted_los)
                                daily_admissions = len(filtered_df) / (filtered_df['date_of_admission'].max() - filtered_df['date_of_admission'].min()).days
                                beds_saved = potential_reduction * daily_admissions
                                st.metric("Potential Bed Days Saved", f"{beds_saved:.1f}/day")
                        
                        # Prediction examples
                        st.subheader("🔮 Sample Predictions")
                        
                        sample_predictions = []
                        for i in range(min(10, len(X_test))):
                            sample_input = X_test.iloc[[i]]
                            prediction = models_trained[best_model_name].predict(sample_input)[0]
                            
                            if is_classification:
                                prob = models_trained[best_model_name].predict_proba(sample_input)[0].max()
                                sample_predictions.append({
                                    'Patient_ID': f'Patient_{i+1}',
                                    'Predicted_Category': prediction,
                                    'Confidence': f"{prob:.2%}",
                                    'Actual': y_test.iloc[i]
                                })
                            else:
                                sample_predictions.append({
                                    'Patient_ID': f'Patient_{i+1}',
                                    'Predicted_LOS': f"{prediction:.1f} days",
                                    'Actual_LOS': f"{y_test.iloc[i]:.1f} days",
                                    'Difference': f"{abs(prediction - y_test.iloc[i]):.1f} days"
                                })
                        
                        pred_df = pd.DataFrame(sample_predictions)
                        st.dataframe(pred_df, use_container_width=True)
                        
                        # Python code generation
                        st.subheader("💻 Implementation Code")
                        
                        code = generate_python_code(best_model_name, selected_features, target)
                        
                        with st.expander("View Complete Implementation Code"):
                            st.code(code, language='python')
                    
                    else:
                        st.error("All models failed to train. Please check your feature selection and data quality.")
            
            else:
                st.warning("Please select at least one feature for model training.")
        
        # AI Narrative for LOS Prediction
        if 'los' in st.session_state.model_results:
            st.markdown("---")
            st.markdown("## 🤖 AI-Generated Business Report")
            
            results = st.session_state.model_results['los']
            
            if is_classification:
                best_model = max(results.keys(), key=lambda x: results[x].get('accuracy', 0))
                performance_metric = results[best_model]['accuracy']
            else:
                best_model = max(results.keys(), key=lambda x: results[x].get('r2_score', 0))
                performance_metric = results[best_model]['r2_score']
            
            data_summary = {
                "prediction_type": target_type,
                "best_model": best_model,
                "performance_metric": float(performance_metric),
                "features_used": selected_features,
                "avg_los": float(avg_los),
                "median_los": float(median_los),
                "total_patients": len(filtered_df)
            }
            
            narrative = generate_ai_narrative("Length of Stay Prediction", data_summary, results)
            
            st.markdown(f"""
            <div class="ai-narrative">
                {narrative}
            </div>
            """, unsafe_allow_html=True)

# Footer with Decision Log
st.markdown("---")
st.markdown("## 📋 Decision Log & Action Items")

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Add Decision")
    
    decision_col1, decision_col2 = st.columns(2)
    
    with decision_col1:
        decision_section = st.selectbox("Section", ["Admissions Forecasting", "Revenue Anomaly Detection", "Length of Stay Prediction"])
        decision_action = st.selectbox("Decision", ["Approve for Production", "Needs Review", "Requires Additional Data", "Reject"])
    
    with decision_col2:
        decision_owner = st.text_input("Owner/Responsible Party")
        decision_date = st.date_input("Target Date")
    
    decision_notes = st.text_area("Notes/Comments")
    
    if st.button("Add to Decision Log"):
        new_decision = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "section": decision_section,
            "action": decision_action,
            "owner": decision_owner,
            "target_date": decision_date.strftime("%Y-%m-%d"),
            "notes": decision_notes
        }
        
        st.session_state.decision_log.append(new_decision)
        st.success("Decision added to log!")

with col2:
    st.subheader("Quick Stats")
    
    if st.session_state.decision_log:
        decisions_df = pd.DataFrame(st.session_state.decision_log)
        
        st.metric("Total Decisions", len(decisions_df))
        
        if 'action' in decisions_df.columns:
            approved = sum(decisions_df['action'] == 'Approve for Production')
            st.metric("Approved for Production", approved)
            
            pending = sum(decisions_df['action'] == 'Needs Review')
            st.metric("Pending Review", pending)
    else:
        st.info("No decisions logged yet")

# Display Decision Log
if st.session_state.decision_log:
    st.subheader("Decision History")
    decisions_df = pd.DataFrame(st.session_state.decision_log)
    st.dataframe(decisions_df, use_container_width=True)
    
    # Download decision log
    csv = decisions_df.to_csv(index=False)
    st.download_button(
        label="Download Decision Log as CSV",
        data=csv,
        file_name=f"hospital_decisions_{datetime.now().strftime('%Y%m%d')}.csv",
        mime="text/csv"
    )

# Sidebar summary
with st.sidebar:
    st.markdown("---")
    st.subheader("📊 Session Summary")
    
    if st.session_state.model_results:
        for section, results in st.session_state.model_results.items():
            st.write(f"✅ {section.title()}: {len(results)} models trained")
    else:
        st.write("No models trained yet")
    
    if st.session_state.decision_log:
        st.write(f"📋 {len(st.session_state.decision_log)} decisions logged")
    
    st.markdown("---")
    st.caption("Hospital Ops Studio v2.0")
    st.caption("Built with Streamlit & scikit-learn")
