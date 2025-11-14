import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import time
import joblib
from sklearn.preprocessing import LabelEncoder
import os

# --- FILE PATH CONSTANTS ---
MODEL_PATH = "rf_model4.pkl"
DATASET_PATH = "pipeline_dataset4.csv"
PRODUCT_ENCODER_PATH = "le_product.pkl"
SERVICE_ENCODER_PATH = "le_service.pkl"

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="PMAT - Pipeline Material Assessment Tool",
    page_icon="🔧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CUSTOM CSS ---
st.markdown("""
<style>
.main-header { background: linear-gradient(90deg, #1e3a8a 0%, #3b82f6 100%); padding: 2rem; border-radius: 10px; color: white; margin-bottom: 2rem; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);}
.main-header h1 { color: white; margin: 0; font-size: 2.5rem;}
.main-header p { color: #e0e7ff; margin: 0.3rem 0 0 0; font-size: 0.95rem;}
.metric-card { background-color: #f8fafc; padding: 1.5rem; border-radius: 8px; border-left: 4px solid #3b82f6; box-shadow: 0 2px 4px rgba(0,0,0,0.05);}
.stButton>button { width: 100%; background: linear-gradient(90deg,#3b82f6 0%,#2563eb 100%); color: white; font-weight:bold; padding:0.75rem; border-radius:8px; border:none; font-size:1.1rem; transition: all 0.3s ease;}
.stButton>button:hover { background: linear-gradient(90deg,#2563eb 0%,#1d4ed8 100%); box-shadow:0 4px 8px rgba(37,99,235,0.3); transform: translateY(-2px);}
.info-box {background-color: #eff6ff; border-left: 4px solid #3b82f6; padding:1rem; border-radius:4px; margin:1rem 0;}
.warning-box {background-color:#fef3c7; border-left:4px solid #f59e0b; padding:1rem; border-radius:4px; margin:1rem 0;}
.success-box {background-color:#d1fae5; border-left:4px solid #10b981; padding:1rem; border-radius:4px; margin:1rem 0;}
h1,h2,h3 {color:#1e3a8a;}
@keyframes slideIn { from {opacity:0; transform:translateY(20px);} to {opacity:1; transform:translateY(0);} }
.result-section { animation: slideIn 0.5s ease-out;}
</style>
""", unsafe_allow_html=True)

# --- LOAD MODEL & DATA ---
@st.cache_resource
def load_model_and_encoders():
    """Load the trained model and label encoders"""
    model = None
    le_product = None
    le_service = None
    
    # Load model
    if os.path.exists(MODEL_PATH):
        try:
            model = joblib.load(MODEL_PATH)
        except Exception as e:
            st.error(f"Error loading model: {e}")
            return None, None, None
    else:
        st.error(f"Model file '{MODEL_PATH}' not found.")
        return None, None, None
    
    # Load encoders
    if os.path.exists(PRODUCT_ENCODER_PATH):
        try:
            le_product = joblib.load(PRODUCT_ENCODER_PATH)
        except Exception as e:
            st.error(f"Error loading product encoder: {e}")
    else:
        st.warning(f"Product encoder '{PRODUCT_ENCODER_PATH}' not found. Will create from dataset.")
    
    if os.path.exists(SERVICE_ENCODER_PATH):
        try:
            le_service = joblib.load(SERVICE_ENCODER_PATH)
        except Exception as e:
            st.error(f"Error loading service encoder: {e}")
    else:
        st.warning(f"Service encoder '{SERVICE_ENCODER_PATH}' not found. Will create from dataset.")
    
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

# --- DATA & MODEL ---
df = load_dataset()
rf_model, le_product, le_service = load_model_and_encoders()

# Create encoders from dataset if not loaded
if df is not None and (le_product is None or le_service is None):
    if le_product is None and 'Product' in df.columns:
        le_product = LabelEncoder()
        le_product.fit(df['Product'])
    if le_service is None and 'Service' in df.columns:
        le_service = LabelEncoder()
        le_service.fit(df['Service'])

# --- SESSION STATE ---
if 'prediction_made' not in st.session_state:
    st.session_state.prediction_made = False
if 'prediction_result' not in st.session_state:
    st.session_state.prediction_result = None
if 'prediction_history' not in st.session_state:
    st.session_state.prediction_history = []

# --- HELPER FUNCTIONS ---
def get_dataset_stats(df):
    """Get basic statistics from the dataset"""
    if df is None or df.empty:
        return None
    return {
        'total_pipelines': len(df),
        'material_distribution': df['Type'].value_counts().to_dict() if 'Type' in df.columns else {}
    }

def make_prediction(model, le_product, le_service, inputs):
    """Make prediction using the trained model"""
    try:
        # Encode categorical features
        product_encoded = le_product.transform([inputs['product']])[0]
        service_encoded = le_service.transform([inputs['service']])[0]
    except ValueError as e:
        st.error(f"Categorical encoding error: {e}")
        return None

    # Prepare features in correct order
    feature_order = ['Length', 'Pressure', 'Temperature', 'Pipeline Size', 'Product_Encoded', 'Service_Encoded']
    features = np.array([
        inputs['length'], 
        inputs['pressure'], 
        inputs['temperature'],
        inputs['size'], 
        product_encoded, 
        service_encoded
    ]).reshape(1, -1)

    # Make prediction
    pred_class = model.predict(features)[0]
    pred_probs = model.predict_proba(features)[0]
    class_names = model.classes_

    # Get class name if prediction is numeric
    if isinstance(pred_class, (int, np.integer)):
        pred_class = class_names[pred_class]

    # Create probabilities dictionary
    probabilities_dict = dict(zip(class_names, pred_probs))
    confidence = probabilities_dict.get(pred_class, 0.0)
    
    # Get alternatives (top 3 excluding the predicted class)
    alternatives = sorted(
        [(c, p) for c, p in probabilities_dict.items() if c != pred_class], 
        key=lambda x: x[1], 
        reverse=True
    )[:3]

    # Get feature importance
    feature_importance = []
    if hasattr(model, 'feature_importances_'):
        feature_importance = sorted(
            zip(feature_order, model.feature_importances_), 
            key=lambda x: x[1], 
            reverse=True
        )

    return {
        'material': pred_class,
        'confidence': confidence,
        'alternatives': [{'material': c, 'score': p} for c, p in alternatives],
        'feature_importance': feature_importance,
        'probabilities': probabilities_dict
    }

# --- HEADER ---
st.markdown("""
<div class="main-header">
<h1>🔧 PMAT | Pipeline Material Assessment Tool</h1>
<p>AI-Powered Material Selection System</p>
<p>Universiti Teknologi PETRONAS | PETRONAS Carigali</p>
</div>
""", unsafe_allow_html=True)

# Info box
st.markdown("""
<div class="info-box">
<strong>ℹ️ About This Tool:</strong><br>
Uses ML (Random Forest) to recommend pipeline materials. Trained on 400 real configurations with ~98% accuracy.
</div>
""", unsafe_allow_html=True)

# --- SIDEBAR ---
with st.sidebar:
    st.header("📊 Model Info")
    if rf_model:
        st.metric("Algorithm", "Random Forest")
        st.metric("Accuracy", "98%")
        st.metric("Materials", str(len(rf_model.classes_)))
        st.success("🟢 Model: Active")
    else:
        st.error("❌ Model: Not Loaded")
    
    st.markdown("---")
    st.header("🎯 Material Types")
    materials_info = {
        "Carbon Steel": "High pressure, general purpose",
        "Flexible": "Medium sizes, moderate conditions",
        "IFL": "Corrosive environments",
        "RTP": "Small sizes, low pressure"
    }
    for m, desc in materials_info.items():
        st.markdown(f"**{m}:** {desc}")
    
    # Dataset stats
    stats = get_dataset_stats(df)
    if stats:
        st.markdown("---")
        st.header("📈 Dataset Stats")
        st.metric("Total Pipelines", stats['total_pipelines'])
        st.subheader("Material Distribution")
        for material, count in stats['material_distribution'].items():
            st.write(f"{material}: {count} ({count/stats['total_pipelines']*100:.1f}%)")

# --- CHECK IF MODEL IS LOADED ---
if rf_model is None or le_product is None or le_service is None:
    st.error("⚠️ Model or encoders not loaded properly. Please check your files.")
    st.stop()

# --- USER INPUT FORM ---
st.header("Pipeline Parameters Input")

with st.form("pipeline_inputs"):
    col1, col2 = st.columns(2)

    # Get options from dataset or use defaults
    product_options = df['Product'].unique().tolist() if df is not None and 'Product' in df.columns else ['Gas', 'Oil', 'Condensate', 'Water']
    service_options = df['Service'].unique().tolist() if df is not None and 'Service' in df.columns else ['Sweet', 'Sour']

    with col1:
        st.subheader("Physical Parameters")
        pipeline_size = st.number_input("Pipeline Size (inch)", 2, 36, 20, 2)
        length = st.number_input("Length (km)", 1.0, 75.0, 48.4, 0.1)
        product = st.selectbox("Product Type", options=product_options)

    with col2:
        st.subheader("Operating Conditions")
        service = st.selectbox("Service Type", options=service_options)
        pressure = st.number_input("Design Pressure (barg)", 20.0, 180.0, 168.4, 0.1)
        temperature = st.number_input("Design Temperature (°C)", 25.0, 130.0, 28.2, 0.1)

    submitted = st.form_submit_button("Generate AI Recommendation", use_container_width=True)

# --- HANDLE SUBMISSION ---
if submitted:
    inputs = {
        'size': pipeline_size, 
        'length': length, 
        'product': product, 
        'service': service,
        'pressure': pressure, 
        'temperature': temperature
    }
    
    with st.spinner(" Generating recommendation..."):
        time.sleep(0.5)
        result = make_prediction(rf_model, le_product, le_service, inputs)
        
        if result:
            st.session_state.prediction_result = result
            st.session_state.prediction_made = True
            st.session_state.prediction_history.append({
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'inputs': inputs,
                'result': result['material'],
                'confidence': result['confidence']
            })
            st.rerun()

# --- DISPLAY RESULTS ---
if st.session_state.prediction_made and st.session_state.prediction_result:
    result = st.session_state.prediction_result
    st.markdown("---")
    st.header("AI Recommendation Results")
    st.success("✅ Analysis Complete!")

    # Recommendation card
    material_colors = {
        'Carbon Steel': '#3b82f6',
        'Flexible': '#a855f7',
        'IFL': '#10b981',
        'RTP': '#f59e0b'
    }
    color = material_colors.get(result['material'], '#94a3b8')
    
    st.markdown(f"""
    <div style="background-color:{color}33; padding:2rem; border-radius:12px; border-left: 6px solid {color};">
    <h2 style="margin:0; color:{color};">Recommended Material: {result['material']}</h2>
    <p style="font-size:1.2rem; margin:0.5rem 0 0 0;">Confidence: <strong>{result['confidence']*100:.1f}%</strong></p>
    </div>
    """, unsafe_allow_html=True)

    # Alternatives
    if result['alternatives']:
        st.subheader("Alternative Materials")
        alt_cols = st.columns(len(result['alternatives']))
        for idx, alt in enumerate(result['alternatives']):
            with alt_cols[idx]:
                st.metric(
                    label=alt['material'],
                    value=f"{alt['score']*100:.1f}%"
                )

    # Probability distribution chart
    st.subheader("Material Probability Distribution")
    prob_df = pd.DataFrame([
        {'Material': k, 'Probability': v*100} 
        for k, v in result['probabilities'].items()
    ])
    prob_df = prob_df.sort_values('Probability', ascending=False)
    
    fig_prob = px.bar(
        prob_df, 
        x='Material', 
        y='Probability',
        color='Material',
        color_discrete_map=material_colors,
        text='Probability'
    )
    fig_prob.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
    fig_prob.update_layout(
        showlegend=False,
        yaxis_title="Probability (%)",
        xaxis_title="Material Type",
        height=400
    )
    st.plotly_chart(fig_prob, use_container_width=True)

    # Feature importance
    if result['feature_importance']:
        st.subheader("Feature Importance Analysis")
        fi_df = pd.DataFrame(result['feature_importance'], columns=['Feature', 'Importance'])
        
        fig_fi = px.bar(
            fi_df, 
            x='Importance', 
            y='Feature', 
            orientation='h',
            color='Importance',
            color_continuous_scale='Blues'
        )
        fig_fi.update_layout(
            yaxis=dict(autorange="reversed"),
            showlegend=False,
            height=400
        )
        st.plotly_chart(fig_fi, use_container_width=True)

# --- PREDICTION HISTORY ---
if st.session_state.prediction_history:
    st.markdown("---")
    st.header("Prediction History")
    
    history_display = []
    for h in st.session_state.prediction_history[-10:]:
        inputs_str = f"Size: {h['inputs']['size']}\", Length: {h['inputs']['length']} km, {h['inputs']['product']}, {h['inputs']['service']}"
        history_display.append({
            'Timestamp': h['timestamp'],
            'Parameters': inputs_str,
            'Recommended Material': h['result'],
            'Confidence': f"{h['confidence']*100:.1f}%"
        })
    
    history_df = pd.DataFrame(history_display)
    st.dataframe(history_df, use_container_width=True, hide_index=True)
    
    if st.button("Clear History"):
        st.session_state.prediction_history = []
        st.rerun()

# --- FOOTER ---
st.markdown("---")
st.markdown("""
<div style="text-align:center;color:#64748b;">
<p>PMAT v1.0 | Random Forest Classifier | UTP & PETRONAS Carigali | © 2025 Nureen Binti Mohd Erzan</p>
</div>
""", unsafe_allow_html=True)