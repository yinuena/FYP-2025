# utils.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder
import os
import plotly.express as px

# --- FILE PATH CONSTANTS ---
MODEL_PATH = "rf_model4.pkl"
DATASET_PATH = "pipeline_dataset4.csv"
PRODUCT_ENCODER_PATH = "le_product.pkl"
SERVICE_ENCODER_PATH = "le_service.pkl"

# --- Material Colors for consistent charting ---
MATERIAL_COLORS = {
    'Carbon Steel': '#3b82f6', # Blue
    'Flexible': '#a855f7',     # Purple
    'IFL': '#10b981',          # Green
    'RTP': '#f59e0b'           # Amber
}

# --- Shared UI Components ---
def set_page_config_and_css():
    st.set_page_config(
        page_title="PMAT - Pipeline Material Assessment Tool",
        page_icon="🔧",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    st.markdown("""
    <style>
    .main-header { background: linear-gradient(90deg, #1e3a8a 0%, #3b82f6 100%); padding: 2rem; border-radius: 10px; color: white; margin-bottom: 2rem; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);}
    .main-header h1 { color: white; margin: 0; font-size: 2.5rem;}
    .main-header p { color: #e0e7ff; margin: 0.3rem 0 0 0; font-size: 0.95rem;}
    .stButton>button { width: 100%; background: linear-gradient(90deg,#3b82f6 0%,#2563eb 100%); color: white; font-weight:bold; padding:0.75rem; border-radius:8px; border:none; font-size:1.1rem; transition: all 0.3s ease;}
    .stButton>button:hover { background: linear-gradient(90deg,#2563eb 0%,#1d4ed8 100%); box-shadow:0 4px 8px rgba(37,99,235,0.3); transform: translateY(-2px);}
    .info-box {background-color: #eff6ff; border-left: 4px solid #3b82f6; padding:1rem; border-radius:4px; margin:1rem 0;}
    .warning-box {background-color:#fef3c7; border-left:4px solid #f59e0b; padding:1rem; border-radius:4px; margin:1rem 0;}
    .success-box {background-color:#d1fae5; border-left:4px solid #10b981; padding:1rem; border-radius:4px; margin:1rem 0;}
    h1,h2,h3 {color:#1e3a8a;}
    @keyframes slideIn { from {opacity:0; transform:translateY(20px);} to {opacity:1; transform:translateY(0);} }
    .result-section { animation: slideIn 0.5s ease-out;}
    .stDataFrame { margin-top: 15px; }
    </style>
    """, unsafe_allow_html=True)

# --- LOAD MODEL & DATA ---
@st.cache_resource
def load_model_and_encoders():
    """Load the trained model and label encoders"""
    model = None
    le_product = None
    le_service = None
    
    if os.path.exists(MODEL_PATH):
        try:
            model = joblib.load(MODEL_PATH)
        except Exception as e:
            st.error(f"Error loading model: {e}")
            return None, None, None
    
    if os.path.exists(PRODUCT_ENCODER_PATH):
        try:
            le_product = joblib.load(PRODUCT_ENCODER_PATH)
        except Exception as e:
            st.error(f"Error loading product encoder: {e}")
    
    if os.path.exists(SERVICE_ENCODER_PATH):
        try:
            le_service = joblib.load(SERVICE_ENCODER_PATH)
        except Exception as e:
            st.error(f"Error loading service encoder: {e}")
    
    return model, le_product, le_service

@st.cache_data
def load_dataset(path=DATASET_PATH):
    """Load dataset and create encoders if not available"""
    if not os.path.exists(path):
        st.error(f"Dataset file '{path}' not found.")
        return None
    try:
        df = pd.read_csv(path)
        return df
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        return None

# --- HELPER FUNCTIONS ---
def get_dataset_stats(df):
    """Get basic statistics from the dataset"""
    if df is None or df.empty:
        return None
    return {
        'total_pipelines': len(df),
        'material_distribution': df['Type'].value_counts().to_dict() if 'Type' in df.columns else {}
    }

def get_rejection_reasoning(material_type, inputs):
    """
    Generates a human-readable reason for why a material might not be the primary choice. (Local XAI)
    """
    reasons = []
    
    # Define simple, illustrative constraint thresholds 
    P_HIGH = 110 # barg
    T_HIGH = 85  # degC
    SIZE_LARGE = 24 # inch
    SIZE_SMALL = 8 # inch
    P_LOW = 40 # barg

    pressure = inputs['pressure']
    temperature = inputs['temperature']
    size = inputs['size']
    service = inputs['service']

    # --- Material-Specific Constraints ---

    if material_type == 'RTP':
        if pressure > P_HIGH:
            reasons.append(f"Design Pressure ({pressure:.1f} barg) exceeds the typical limit for **RTP** (generally below {P_HIGH} barg).")
        if temperature > T_HIGH:
            reasons.append(f"Design Temperature ({temperature:.1f}°C) is high, approaching the operational limit for thermoplastic materials.")
        if size > SIZE_SMALL:
            reasons.append(f"Pipeline Size ({size} inch) is large; **RTP** is typically favored for smaller diameter pipes (<{SIZE_SMALL} inch).")
            
    elif material_type == 'Carbon Steel':
        if service == 'Sour':
            reasons.append(f"Service is **'Sour'** (corrosive), which often requires specialized corrosion resistant materials or internal lining (like **IFL**).")
        if pressure < P_LOW and size < SIZE_SMALL:
            reasons.append(f"The combined low pressure ({pressure:.1f} barg) and small size ({size} inch) mean a more cost-effective material (**RTP** or **Flexible**) might be preferable.")
            
    elif material_type == 'IFL':
        if service == 'Sweet':
            reasons.append("Service is **'Sweet'** (non-corrosive), eliminating the primary justification for the high cost of the internal lining (IFL).")
        if pressure < P_LOW:
             reasons.append(f"Design Pressure ({pressure:.1f} barg) is low; **IFL** is generally reserved for corrosive, high-pressure applications.")

    elif material_type == 'Flexible':
        if pressure > P_HIGH:
            reasons.append(f"Design Pressure ({pressure:.1f} barg) exceeds the operational limit for many common **Flexible** pipe designs.")
        if size > SIZE_LARGE:
             reasons.append(f"Pipeline Size ({size} inch) is large, making installation and procurement of high-diameter **Flexible** pipe complex.")
    
    # Fallback reason
    if not reasons and st.session_state.prediction_made:
        confidence = st.session_state.prediction_result['confidence']
        reasons.append(f"The AI's confidence ({confidence*100:.1f}%) in the primary recommendation was significantly higher, suggesting a more optimal fit based on the historical feature patterns.")

    # Return as a bulleted list string
    return "<ul>" + "".join([f"<li>{r}</li>" for r in reasons]) + "</ul>"
