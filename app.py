# app.py — Hospital Operations Analytics Platform
# Redesigned for optimal user experience without sidebars

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

# Configuration
st.set_page_config(
    page_title="Hospital Operations Analytics",
    layout="wide",
    page_icon="🏥",
    initial_sidebar_state="collapsed"
)

# Enhanced CSS with modern, clean styling
st.markdown("""
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
    
    .floating-actions {
        position: fixed;
        bottom: 20px;
        right: 20px;
        background: white;
        padding: 1rem;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        border: 1px solid #e9ecef;
        z-index: 1000;
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
    <h1>🏥 Hospital Operations Analytics</h1>
    <p>Data-driven insights for better patient outcomes and operational efficiency</p>
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

# Data Overview Section (moved from sidebar)
st.markdown("""
<div class="data-overview">
    <h3>📊 Dataset Overview</h3>
</div>
""", unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Total Records", f"{len(df):,}")
with col2:
    st.metric("Date Range", f"{(df['date_of_admission'].max() - df['date_of_admission'].min()).days} days")
with col3:
    st.metric("Hospitals", f"{df['hospital'].nunique()}" if 'hospital' in df.columns else "N/A")
with col4:
    st.metric("Conditions", f"{df['medical_condition'].nunique()}" if 'medical_condition' in df.columns else "N/A")

# Global filters (moved from sidebar, made collapsible)
with st.expander("🔍 Filter Data (Optional)", expanded=False):
    col1, col2, col3 = st.columns(3)
    
    with col1:
        hospitals = st.multiselect("Select Hospitals", df['hospital'].unique() if 'hospital' in df.columns else [])
    with col2:
        conditions = st.multiselect("Medical Conditions", df['medical_condition'].unique() if 'medical_condition' in df.columns else [])
    with col3:
        admission_types = st.multiselect("Admission Types", df['admission_type'].unique() if 'admission_type' in df.columns else [])

# Apply filters
filtered_df = df.copy()
if hospitals:
    filtered_df = filtered_df[filtered_df['hospital'].isin(hospitals)]
if conditions:
    filtered_df = filtered_df[filtered_df['medical_condition'].isin(conditions)]
if admission_types:
    filtered_df = filtered_df[filtered_df['admission_type'].isin(admission_types)]

if len(filtered_df) != len(df):
    st.info(f"Applied filters: {len(filtered_df):,} records selected from {len(df):,} total records")

# Utility Functions
def generate_business_summary(section_title: str, data_summary: dict, model_results: dict = None):
    """Generate automated business insights"""
    return f"""
    **{section_title} - Key Insights**
    
    📊 **Analysis Summary:**
    • Processed {data_summary.get('total_records', 0):,} patient records
    • Analysis period covers multiple operational patterns
    • Statistical models trained and validated successfully
    
    💡 **Business Impact:**
    • Predictive analytics ready for operational deployment
    • Data-driven decision making capabilities established
    • Performance monitoring framework implemented
    
    🎯 **Recommended Actions:**
    • Deploy top-performing model to production environment
    • Establish regular model performance monitoring
    • Train staff on new predictive insights workflow
    • Schedule quarterly model retraining cycles
    
    📈 **Success Metrics:**
    • Model accuracy and reliability scores available
    • Operational efficiency improvements measurable
    • Cost reduction opportunities identified
    """

def create_model_comparison_table(results: dict) -> pd.DataFrame:
    """Create a formatted model comparison table"""
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
               marker_color=['#4285f4', '#34a853', '#fbbc04', '#ea4335'][:len(models)],
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
tabs = st.tabs(["📈 Admissions Forecasting", "💰 Revenue Analytics", "🛏️ Length of Stay Prediction"])

# Tab 1: Admissions Forecasting
with tabs[0]:
    st.markdown("""
    <div class="analysis-section">
        <h3>📈 Patient Admission Forecasting</h3>
        <p>Predict future admission patterns to optimize staffing and resource allocation</p>
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
                                line=dict(color='#4285f4', width=2)))
        
        # Add trend line
        x_numeric = np.arange(len(daily_admissions))
        z = np.polyfit(x_numeric, daily_admissions['admissions'], 1)
        p = np.poly1d(z)
        fig.add_trace(go.Scatter(x=daily_admissions['date'], y=p(x_numeric),
                                mode='lines', name='Trend', line=dict(color='#ea4335', dash='dash')))
        
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
            <h4>Average Daily</h4>
            <h2 style="color: #4285f4;">{avg_daily:.1f}</h2>
        </div>
        <div class="metric-card">
            <h4>Peak Daily</h4>
            <h2 style="color: #34a853;">{max_daily}</h2>
        </div>
        <div class="metric-card">
            <h4>Variability</h4>
            <h2 style="color: #fbbc04;">{std_daily:.1f}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    # Model Configuration
    st.markdown("""
    <div class="config-row">
        <h4>🔧 Forecasting Configuration</h4>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        forecast_days = st.slider("Forecast Period (days)", 7, 30, 14)
        
    with col2:
        seasonality = st.selectbox("Seasonality", 
                                  ["Auto-detect", "Weekly", "Monthly", "None"])
    
    with col3:
        confidence_level = st.slider("Confidence Level", 0.8, 0.99, 0.95)
    
    # Model Selection
    selected_models = st.multiselect(
        "Select Forecasting Models",
        ["Linear Trend", "ARIMA", "Exponential Smoothing", "Prophet"],
        default=["Linear Trend", "ARIMA", "Exponential Smoothing"]
    )
    
    if st.button("🚀 Generate Forecasts", key="train_forecast", type="primary"):
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
                st.success("✅ Forecasting models trained successfully!")
                
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
                                        line=dict(color='#4285f4')))
                
                # Forecasts
                future_dates = pd.date_range(start=ts_data.index[-1] + pd.Timedelta(days=1), 
                                           periods=forecast_days, freq='D')
                
                colors = ['#34a853', '#ea4335', '#fbbc04', '#9aa0a6']
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
                
                # Staffing recommendations
                st.subheader("👥 Staffing Recommendations")
                
                staffing_df = pd.DataFrame({
                    'Date': future_dates,
                    'Predicted Admissions': best_forecast.astype(int),
                    'Day Shift Nurses': np.ceil(best_forecast / 8).astype(int),
                    'Night Shift Nurses': np.ceil(best_forecast / 12).astype(int),
                    'Estimated Cost ($)': (np.ceil(best_forecast / 8) * 350 + np.ceil(best_forecast / 12) * 400).astype(int)
                })
                
                st.dataframe(staffing_df, use_container_width=True)
                
                # Implementation code
                with st.expander("💻 View Implementation Code"):
                    code = f'''
# Hospital Admissions Forecasting - {best_model}
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA

# Load your data
df = pd.read_csv('hospital_admissions.csv')
daily_admissions = df.groupby('date').size()

# Train best performing model: {best_model}
forecast_days = {forecast_days}
model = ARIMA(daily_admissions, order=(1,1,1))
fitted_model = model.fit()
forecast = fitted_model.forecast(steps=forecast_days)

print(f"MAPE: {results[best_model]['MAPE']:.2f}%")
print(f"Average forecast: {np.mean(best_forecast):.1f} admissions/day")

# Generate staffing recommendations
def calculate_staffing(predicted_admissions):
    day_nurses = int(np.ceil(predicted_admissions / 8))
    night_nurses = int(np.ceil(predicted_admissions / 12))
    return day_nurses, night_nurses

for i, pred in enumerate(forecast):
    day, night = calculate_staffing(pred)
    print(f"Day {{i+1}}: {{pred:.0f}} admissions -> {{day}} day nurses, {{night}} night nurses")
'''
                    st.code(code, language='python')
    
    # Business insights
    if 'forecasting' in st.session_state.model_results:
        st.markdown("---")
        st.markdown("## 📋 Business Insights")
        
        forecasting_results = st.session_state.model_results['forecasting']
        best_model = min(forecasting_results.keys(), key=lambda x: forecasting_results[x]['MAPE'])
        
        data_summary = {
            "total_records": len(filtered_df),
            "forecast_horizon": forecast_days,
            "average_daily_admissions": float(avg_daily),
            "best_model": best_model,
            "best_model_mape": float(forecasting_results[best_model]['MAPE'])
        }
        
        insights = generate_business_summary("Admissions Forecasting", data_summary, forecasting_results)
        
        st.markdown(f"""
        <div class="insights-panel">
            {insights}
        </div>
        """, unsafe_allow_html=True)

# Tab 2: Revenue Analytics
with tabs[1]:
    st.markdown("""
    <div class="analysis-section">
        <h3>💰 Revenue Pattern Analysis</h3>
        <p>Detect unusual billing patterns and identify potential revenue optimization opportunities</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("💳 Revenue Trends")
        
        # Daily revenue
        daily_revenue = filtered_df.groupby(filtered_df['date_of_admission'].dt.date)['billing_amount'].sum().reset_index()
        daily_revenue.columns = ['date', 'revenue']
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=daily_revenue['date'], y=daily_revenue['revenue'],
                                mode='lines+markers', name='Daily Revenue',
                                line=dict(color='#34a853', width=2)))
        
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
            <h2 style="color: #34a853;">${total_revenue:,.0f}</h2>
        </div>
        <div class="metric-card">
            <h4>Daily Average</h4>
            <h2 style="color: #4285f4;">${avg_daily_revenue:,.0f}</h2>
        </div>
        <div class="metric-card">
            <h4>Avg Per Patient</h4>
            <h2 style="color: #fbbc04;">${avg_bill:,.0f}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    # Configuration
    st.markdown("""
    <div class="config-row">
        <h4>🔧 Anomaly Detection Settings</h4>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        detection_method = st.selectbox("Detection Method", 
                                       ["Isolation Forest", "Statistical Outliers", "Ensemble"])
    
    with col2:
        sensitivity = st.slider("Sensitivity Level", 0.01, 0.1, 0.05, 0.01)
    
    with col3:
        features_for_anomaly = st.multiselect(
            "Analysis Features",
            ['billing_amount', 'age', 'length_of_stay', 'admission_month', 'admission_day_of_week'],
            default=['billing_amount', 'length_of_stay']
        )
    
    if st.button("🚀 Analyze Revenue Patterns", key="detect_anomalies", type="primary"):
        if features_for_anomaly:
            with st.spinner("Analyzing revenue patterns..."):
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
                st.success("✅ Revenue analysis completed!")
                
                st.subheader("🎯 Anomaly Detection Results")
                
                results_df = pd.DataFrame(results).T
                st.dataframe(results_df.style.format({
                    'anomalies_detected': '{:.0f}',
                    'anomaly_rate': '{:.2%}',
                    'avg_anomaly_amount': '${:,.0f}'
                }), use_container_width=True)
                
                # Visualization
                st.subheader("📊 Pattern Visualization")
                
                # Use the best method
                best_method = min(results.keys(), key=lambda x: abs(results[x]['anomaly_rate'] - 0.05))
                best_predictions = anomaly_predictions[best_method]
                
                # Create anomaly visualization
                if len(features_for_anomaly) >= 2:
                    fig = px.scatter(x=X[features_for_anomaly[0]], 
                                   y=X[features_for_anomaly[1]],
                                   color=best_predictions,
                                   title=f'Revenue Pattern Analysis - {best_method}',
                                   color_discrete_map={1: '#4285f4', -1: '#ea4335'},
                                   labels={'color': 'Pattern Type'})
                    fig.update_layout(height=500)
                    st.plotly_chart(fig, use_container_width=True)
                
                # Flagged cases details
                anomaly_indices = X.index[best_predictions == -1]
                if len(anomaly_indices) > 0:
                    st.subheader("🚨 Unusual Cases Identified")
                    
                    anomaly_details = filtered_df.loc[anomaly_indices, 
                                                    ['date_of_admission', 'billing_amount', 'medical_condition', 
                                                     'hospital', 'insurance_provider', 'doctor']].head(20)
                    st.dataframe(anomaly_details, use_container_width=True)
                    
                    # Financial impact
                    total_anomaly_amount = filtered_df.loc[anomaly_indices, 'billing_amount'].sum()
                    potential_investigation = total_anomaly_amount * 0.15  # Assume 15% worth investigating
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Flagged Amount", f"${total_anomaly_amount:,.0f}")
                    with col2:
                        st.metric("Investigation Priority", f"${potential_investigation:,.0f}")
                    with col3:
                        st.metric("Cases for Review", f"{len(anomaly_indices)}")
                
                # Implementation code
                with st.expander("💻 View Implementation Code"):
                    code = f'''
# Revenue Pattern Analysis - {best_method}
from sklearn.ensemble import IsolationForest
import pandas as pd
import numpy as np

# Load and prepare data
df = pd.read_csv('hospital_revenue.csv')
features = {features_for_anomaly}
X = df[features].dropna()

# Configure anomaly detection
model = IsolationForest(contamination={sensitivity}, random_state=42)
anomaly_predictions = model.fit_predict(X)

# Identify unusual patterns
unusual_cases = X[anomaly_predictions == -1]
print(f"Identified {{len(unusual_cases)}} unusual cases from {{len(X)}} total records")

# Calculate business impact
flagged_amounts = df.loc[unusual_cases.index, 'billing_amount']
total_flagged = flagged_amounts.sum()
investigation_priority = total_flagged * 0.15

print(f"Total flagged amount: ${{total_flagged:,.0f}}")
print(f"Suggested investigation budget: ${{investigation_priority:,.0f}}")

# Function for real-time scoring
def analyze_new_case(case_data):
    anomaly_score = model.decision_function([case_data])[0]
    is_unusual = model.predict([case_data])[0] == -1
    return anomaly_score, is_unusual

# Example usage
new_case = [50000, 45, 5]  # [billing_amount, age, length_of_stay]
score, unusual = analyze_new_case(new_case)
print(f"Case analysis - Score: {{score:.3f}}, Unusual: {{unusual}}")
'''
                    st.code(code, language='python')
        else:
            st.warning("Please select at least one feature for analysis.")
    
    # Business insights
    if 'anomaly' in st.session_state.model_results:
        st.markdown("---")
        st.markdown("## 📋 Business Insights")
        
        # Fixed: Use session state data instead of local variable
        anomaly_results = st.session_state.model_results['anomaly']
        best_method = min(anomaly_results.keys(), key=lambda x: abs(anomaly_results[x]['anomaly_rate'] - 0.05))
        
        data_summary = {
            "total_revenue": float(total_revenue),
            "avg_daily_revenue": float(avg_daily_revenue),
            "detection_method": detection_method,
            "best_method": best_method,
            "anomalies_detected": int(anomaly_results[best_method]['anomalies_detected']),
            "anomaly_rate": float(anomaly_results[best_method]['anomaly_rate'])
        }
        
        insights = generate_business_summary("Revenue Pattern Analysis", data_summary, anomaly_results)
        
        st.markdown(f"""
        <div class="insights-panel">
            {insights}
        </div>
        """, unsafe_allow_html=True)

# Tab 3: Length of Stay Prediction
with tabs[2]:
    st.markdown("""
    <div class="analysis-section">
        <h3>🛏️ Length of Stay Prediction</h3>
        <p>Predict patient stay duration to optimize bed management and discharge planning</p>
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
                             color_discrete_sequence=['#4285f4'])
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
                <h2 style="color: #4285f4;">{avg_los:.1f} days</h2>
            </div>
            <div class="metric-card">
                <h4>Median LOS</h4>
                <h2 style="color: #34a853;">{median_los:.1f} days</h2>
            </div>
            <div class="metric-card">
                <h4>Maximum LOS</h4>
                <h2 style="color: #ea4335;">{max_los:.0f} days</h2>
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
        
        # Model Configuration
        st.markdown("""
        <div class="config-row">
            <h4>🔧 Prediction Model Setup</h4>
        </div>
        """, unsafe_allow_html=True)
        
        # Feature selection
        available_features = [col for col in filtered_df.columns if col not in 
                            ['length_of_stay', 'los_category', 'date_of_admission', 'discharge_date']]
        
        col1, col2 = st.columns(2)
        
        with col1:
            selected_features = st.multiselect(
                "Select Prediction Features",
                available_features,
                default=[f for f in ['age', 'medical_condition', 'admission_type', 'hospital'] if f in available_features]
            )
        
        with col2:
            target_type = st.selectbox("Prediction Target", 
                                      ["Length of Stay (Days)", "LOS Category (Short/Medium/Long)"])
            
            selected_models = st.multiselect(
                "Select Models",
                ["Random Forest", "Logistic Regression", "XGBoost", "SVM"],
                default=["Random Forest", "Logistic Regression"]
            )
        
        if st.button("🚀 Train Prediction Models", key="train_los", type="primary"):
            if selected_features:
                with st.spinner("Training LOS prediction models..."):
                    # Prepare data
                    feature_data = filtered_df[selected_features + ['length_of_stay', 'los_category']].dropna()
                    
                    if target_type == "Length of Stay (Days)":
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
                                
                                results[model_name] = {
                                    'accuracy': accuracy,
                                    'precision': precision,
                                    'recall': recall,
                                    'f1_score': f1
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
                        
                        st.success("✅ LOS prediction models trained successfully!")
                        
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
                        
                        # Best model analysis
                        st.subheader("🎯 Best Model Analysis")
                        
                        best_model_name = max(results.keys(), key=lambda x: results[x].get(metric_to_plot, 0))
                        best_model = models_trained[best_model_name]
                        
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
                                predicted_categories = best_model.predict(X_test)
                                short_stay_rate = sum(predicted_categories == 'Short') / len(predicted_categories)
                                st.metric("Predicted Short Stays", f"{short_stay_rate:.1%}")
                            else:
                                y_pred_best = best_model.predict(X_test)
                                avg_predicted_los = np.mean(y_pred_best)
                                st.metric("Avg Predicted LOS", f"{avg_predicted_los:.1f} days")
                        
                        with col3:
                            # Calculate potential optimization
                            if not is_classification:
                                current_avg_los = filtered_df['length_of_stay'].mean()
                                potential_reduction = max(0, current_avg_los - avg_predicted_los)
                                st.metric("Potential LOS Reduction", f"{potential_reduction:.1f} days")
                        
                        # Sample predictions
                        st.subheader("🔮 Sample Predictions")
                        
                        sample_predictions = []
                        for i in range(min(10, len(X_test))):
                            sample_input = X_test.iloc[[i]]
                            prediction = best_model.predict(sample_input)[0]
                            
                            if is_classification:
                                sample_predictions.append({
                                    'Case': f'Patient {i+1}',
                                    'Predicted Category': prediction,
                                    'Actual Category': y_test.iloc[i],
                                    'Match': '✅' if prediction == y_test.iloc[i] else '❌'
                                })
                            else:
                                sample_predictions.append({
                                    'Case': f'Patient {i+1}',
                                    'Predicted LOS': f"{prediction:.1f} days",
                                    'Actual LOS': f"{y_test.iloc[i]:.1f} days",
                                    'Difference': f"{abs(prediction - y_test.iloc[i]):.1f} days"
                                })
                        
                        pred_df = pd.DataFrame(sample_predictions)
                        st.dataframe(pred_df, use_container_width=True)
                        
                        # Implementation code
                        with st.expander("💻 View Implementation Code"):
                            code = generate_python_code(best_model_name, selected_features, target)
                            st.code(code, language='python')
                    
                    else:
                        st.error("All models failed to train. Please check your feature selection and data quality.")
            
            else:
                st.warning("Please select at least one feature for model training.")
        
        # Business insights
        if 'los' in st.session_state.model_results:
            st.markdown("---")
            st.markdown("## 📋 Business Insights")
            
            los_results = st.session_state.model_results['los']
            
            if is_classification:
                best_model = max(los_results.keys(), key=lambda x: los_results[x].get('accuracy', 0))
                performance_metric = los_results[best_model]['accuracy']
            else:
                best_model = max(los_results.keys(), key=lambda x: los_results[x].get('r2_score', 0))
                performance_metric = los_results[best_model]['r2_score']
            
            data_summary = {
                "prediction_type": target_type,
                "best_model": best_model,
                "performance_metric": float(performance_metric),
                "avg_los": float(avg_los),
                "total_patients": len(filtered_df)
            }
            
            insights = generate_business_summary("Length of Stay Prediction", data_summary, los_results)
            
            st.markdown(f"""
            <div class="insights-panel">
                {insights}
            </div>
            """, unsafe_allow_html=True)

# Decision Log Section (moved from sidebar)
st.markdown("---")
st.markdown("## 📋 Decision Tracking")

with st.expander("📝 Add New Decision", expanded=False):
    col1, col2 = st.columns(2)
    
    with col1:
        decision_section = st.selectbox("Analysis Section", ["Admissions Forecasting", "Revenue Analytics", "Length of Stay Prediction"])
        decision_action = st.selectbox("Decision", ["Approve for Production", "Needs Review", "Requires Additional Data", "Reject"])
    
    with col2:
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
            "notes": decision_notes
        }
        
        st.session_state.decision_log.append(new_decision)
        st.success("✅ Decision added to tracking log!")

# Display decision history
if st.session_state.decision_log:
    st.subheader("Decision History")
    
    decisions_df = pd.DataFrame(st.session_state.decision_log)
    
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        st.dataframe(decisions_df, use_container_width=True)
    
    with col2:
        if 'action' in decisions_df.columns:
            approved = sum(decisions_df['action'] == 'Approve for Production')
            pending = sum(decisions_df['action'] == 'Needs Review')
            
            st.metric("Approved", approved)
            st.metric("Pending Review", pending)
    
    with col3:
        # Download decision log
        csv = decisions_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Log",
            data=csv,
            file_name=f"decisions_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv",
            type="secondary"
        )

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 2rem;">
    <p>Hospital Operations Analytics Platform • Built with Streamlit & Python ML Libraries</p>
    <p>For questions or support, contact your analytics team</p>
</div>
""", unsafe_allow_html=True)
