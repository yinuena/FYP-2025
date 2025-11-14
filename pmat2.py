# PMAT_Home.py (The Main Page)
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import time
# Import all necessary components from utils
from utils import (
    set_page_config_and_css, 
    load_dataset, 
    load_model_and_encoders, 
    get_dataset_stats, 
    get_rejection_reasoning,
    MATERIAL_COLORS
)

# --- INITIAL SETUP ---
set_page_config_and_css()

# --- DATA & MODEL ---
df = load_dataset()
rf_model, le_product, le_service = load_model_and_encoders()

# Create encoders from dataset if not loaded (Safety check)
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
if 'current_inputs' not in st.session_state:
    st.session_state.current_inputs = {}


# --- HELPER FUNCTIONS ---
def make_prediction(model, le_product, le_service, inputs):
    """Make prediction using the trained model (moved from original script)"""
    try:
        product_encoded = le_product.transform([inputs['product']])[0]
        service_encoded = le_service.transform([inputs['service']])[0]
    except ValueError as e:
        st.error(f"Categorical encoding error: {e}")
        return None

    feature_order = ['Length', 'Pressure', 'Temperature', 'Pipeline Size', 'Product_Encoded', 'Service_Encoded']
    features = np.array([
        inputs['length'], 
        inputs['pressure'], 
        inputs['temperature'],
        inputs['size'], 
        product_encoded, 
        service_encoded
    ]).reshape(1, -1)

    pred_class = model.predict(features)[0]
    pred_probs = model.predict_proba(features)[0]
    class_names = model.classes_

    if isinstance(pred_class, (int, np.integer)):
        pred_class = class_names[pred_class]

    probabilities_dict = dict(zip(class_names, pred_probs))
    confidence = probabilities_dict.get(pred_class, 0.0)
    
    alternatives = sorted(
        [(c, p) for c, p in probabilities_dict.items() if c != pred_class], 
        key=lambda x: x[1], 
        reverse=True
    )[:3]

    feature_importance = []
    if hasattr(model, 'feature_importances_'):
        ordered_feature_names = ['Length (km)', 'Design Pressure (barg)', 'Design Temperature (°C)', 'Pipeline Size (inch)', 'Product_Encoded', 'Service_Encoded']
        feature_importance = sorted(
            zip(ordered_feature_names, model.feature_importances_), 
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
<h1>🔧 PMAT | AI Material Recommendation</h1>
<p>Input Parameters and Get Instant, Explainable Material Selection</p>
</div>
""", unsafe_allow_html=True)

# Info box
st.markdown("""
<div class="info-box">
<strong>ℹ️ About This Tool:</strong> Uses ML (Random Forest) to recommend pipeline materials. Trained on 400 real configurations with ~98% accuracy.
</div>
""", unsafe_allow_html=True)

# --- SIDEBAR (Now simpler, focused on context) ---
with st.sidebar:
    st.header("📊 Model & Material Info")
    if rf_model:
        st.metric("Algorithm", "Random Forest Classifier")
        st.metric("Model Accuracy", "98%")
        st.success("🟢 Model: Active")
    else:
        st.error("❌ Model: Not Loaded")
    
    st.markdown("---")
    st.header("🎯 Material Types")
    materials_info = {
        "Carbon Steel": "High pressure, general purpose",
        "Flexible": "Medium sizes, moderate conditions",
        "IFL": "Corrosive environments (Sour service)",
        "RTP": "Small sizes, low pressure"
    }
    for m, desc in materials_info.items():
        st.markdown(f"**{m}:** {desc}")
    
# --- CHECK IF MODEL IS LOADED ---
if rf_model is None or le_product is None or le_service is None:
    st.error("⚠️ Model or encoders not loaded properly. Please check your files.")
    st.stop()


# --- USER INPUT FORM ---
st.header("Pipeline Parameters Input")

with st.form("pipeline_inputs"):
    col1, col2 = st.columns(2)

    # Get options from dataset
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

    # Capture inputs immediately after form submission logic
    st.session_state.current_inputs = {
        'size': pipeline_size, 
        'length': length, 
        'product': product, 
        'service': service,
        'pressure': pressure, 
        'temperature': temperature
    }

# --- HANDLE SUBMISSION ---
if submitted:
    inputs = st.session_state.current_inputs
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

# --- DISPLAY RESULTS (Enhanced with Local XAI & Pinpoint Plot) ---
if st.session_state.prediction_made and st.session_state.prediction_result:
    result = st.session_state.prediction_result
    inputs = st.session_state.current_inputs
    st.markdown("---")
    st.header("AI Recommendation Results")
    st.success("✅ Analysis Complete!")

    # Recommendation card
    color = MATERIAL_COLORS.get(result['material'], '#94a3b8')
    
    st.markdown(f"""
    <div style="background-color:{color}33; padding:2rem; border-radius:12px; border-left: 6px solid {color};">
    <h2 style="margin:0; color:{color};">Recommended Material: {result['material']}</h2>
    <p style="font-size:1.2rem; margin:0.5rem 0 0 0;">Confidence: <strong>{result['confidence']*100:.1f}%</strong></p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")

    # --- Interactive P-T Plot Pinpoint (WOW FACTOR A) ---
    st.subheader("Visual Validation: Your Input on the Design Envelope")
    
    if df is not None and 'Type' in df.columns:
        # 1. Base Scatter Plot (Historical Data)
        fig_pinpoint = px.scatter(
            df,
            x='Design Temperature degC',
            y='Design Pressure (barg)',
            color='Type',
            opacity=0.5,
            color_discrete_map=MATERIAL_COLORS,
            title='Your Design Point vs. Historical Material Envelope',
            template='plotly_white'
        )
        
        # 2. Add the User's Prediction as a distinct marker
        fig_pinpoint.add_trace(
            go.Scatter(
                x=[inputs['temperature']],
                y=[inputs['pressure']],
                mode='markers',
                marker=dict(
                    size=18,
                    color=color, 
                    symbol='star', 
                    line=dict(width=2, color='white')
                ),
                name=f"Your Input: {result['material']}",
                hovertemplate=f"Predicted: {result['material']}<br>P: {inputs['pressure']} barg<br>T: {inputs['temperature']} °C<extra></extra>"
            )
        )

        fig_pinpoint.update_layout(
            height=550,
            legend_title='Material Type',
            xaxis_title='Design Temperature ($^{\circ}$C)',
            yaxis_title='Design Pressure (barg)',
        )
        st.plotly_chart(fig_pinpoint, use_container_width=True)
        st.markdown("""
        <div class="success-box">
        <strong>Insight:</strong> The **star** shows your design point. If it lands deep within a historical cluster, your confidence is high. If it's near a boundary, or outside all clusters, consider the alternatives seriously!
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")

    col_xai_1, col_xai_2 = st.columns(2)

    with col_xai_1:
        # Alternatives and Reasoning (Enhanced Section - Local XAI)
        if result['alternatives']:
            st.subheader("Alternatives & Rejection Analysis (Local XAI)")
            st.markdown("""
            <div class="warning-box">
            <strong>🔍 Rejection Analysis:</strong><br>
            Materials below are less likely because their operational or cost profile does not align optimally with your input parameters.
            </div>
            """, unsafe_allow_html=True)
            
            alt_data = []
            for alt in result['alternatives']:
                reasoning = get_rejection_reasoning(alt['material'], inputs) 
                alt_data.append({
                    'Material': alt['material'],
                    'Score': f"{alt['score']*100:.1f}%",
                    'Reason for Lower Score': reasoning
                })
                
            alt_df = pd.DataFrame(alt_data)
            
            # Display the alternative materials in a simple HTML table
            st.markdown("### Detailed Alternative Breakdown")
            st.markdown(alt_df.to_html(escape=False, index=False), unsafe_allow_html=True)

    with col_xai_2:
        # Feature importance
        if result['feature_importance']:
            st.subheader("Feature Importance Analysis")
            st.markdown("Shows which input factors drove the model's decision for this specific case.")
            fi_df = pd.DataFrame(result['feature_importance'], columns=['Feature', 'Importance'])
            
            fig_fi = px.bar(
                fi_df, 
                x='Importance', 
                y='Feature', 
                orientation='h',
                color='Importance',
                color_continuous_scale='Blues',
                text='Importance'
            )
            fig_fi.update_traces(texttemplate='%{text:.2f}', textposition='outside')
            fig_fi.update_layout(
                yaxis=dict(autorange="reversed"),
                showlegend=False,
                height=450,
                xaxis_title="Relative Importance Score (0 to 1)"
            )
            st.plotly_chart(fig_fi, use_container_width=True)


# --- FOOTER ---
st.markdown("---")
st.markdown("""
<div style="text-align:center;color:#64748b;">
<p>PMAT v1.0 | UTP & PETRONAS Carigali | © 2025</p>
</div>
""", unsafe_allow_html=True)
