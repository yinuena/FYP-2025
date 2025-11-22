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
import base64
from fpdf import FPDF # Requires pip install fpdf2

# --- FILE PATH CONSTANTS ---
MODEL_PATH = "rf_model4.pkl"
DATASET_PATH = "pipeline_dataset4.csv"
PRODUCT_ENCODER_PATH = "le_product.pkl"
SERVICE_ENCODER_PATH = "le_service.pkl"
LOGO_PATH = "pmat_logo.png"

# --- Material Colors for consistent charting ---
MATERIAL_COLORS = {
    'Carbon Steel': '#3b82f6', # Blue
    'Flexible': '#a855f7',     # Purple
    'IFL': '#10b981',          # Green
    'RTP': '#f59e0b'           # Amber
}

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="PMAT - Pipeline Material Assessment Tool",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CUSTOM CSS ---
st.markdown("""
<style>
/* Header Styling */
.main-header { 
    background: linear-gradient(90deg, #1e3a8a 0%, #3b82f6 100%); 
    padding: 2rem; 
    border-radius: 10px; 
    color: white; 
    margin-bottom: 2rem; 
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    /* For logo and text alignment */
    display: flex;
    align-items: center;
    justify-content: center;
    text-align: center;
}
.main-header h1 { color: white; margin: 0; font-size: 2.5rem;}
.main-header p { color: #e0e7ff; margin: 0.3rem 0 0 0; font-size: 0.95rem;}
/* Button Styling */
.stButton>button { width: 100%; background: linear-gradient(90deg,#3b82f6 0%,#2563eb 100%); color: white; font-weight:bold; padding:0.75rem; border-radius:8px; border:none; font-size:1.1rem; transition: all 0.3s ease;}
.stButton>button:hover { background: linear-gradient(90deg,#2563eb 0%,#1d4ed8 100%); box-shadow:0 4px 8px rgba(37,99,235,0.3); transform: translateY(-2px);}
/* Headings and Spacing */
h1,h2,h3 {color:#1e3a8a;}
.stMarkdown h3 {margin-top: 2rem;}

/* Style the radio buttons for a horizontal 'top menu' feel, especially on mobile */
div.stRadio > label {
    padding: 0 0.5rem; /* Reduced padding */
    margin-right: 0.5rem; /* Space between buttons */
    border-radius: 5px;
    white-space: normal; /* Allow text wrapping */
}
div.stRadio {
    width: 100%;
}
/* Reduce vertical spacing around radio buttons */
.stRadio > div {
    padding-top: 0;
    padding-bottom: 0;
}
</style>
""", unsafe_allow_html=True)

# --- LOAD MODEL & DATA ---
@st.cache_resource
def load_model_and_encoders():
    model = None
    le_product = None
    le_service = None
    if os.path.exists(MODEL_PATH):
        try:
            model = joblib.load(MODEL_PATH)
        except:
            pass
    if os.path.exists(PRODUCT_ENCODER_PATH):
        try:
            le_product = joblib.load(PRODUCT_ENCODER_PATH)
        except:
            pass
    if os.path.exists(SERVICE_ENCODER_PATH):
        try:
            le_service = joblib.load(SERVICE_ENCODER_PATH)
        except:
            pass
    return model, le_product, le_service

@st.cache_data
def load_dataset(path=DATASET_PATH):
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_csv(path)
        return df
    except:
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
if 'current_inputs' not in st.session_state:
    st.session_state.current_inputs = {'size': 20, 'length': 48.4, 'product': 'Gas', 'service': 'Sweet', 'pressure': 168.4, 'temperature': 28.2}
if 'selected_page' not in st.session_state: 
    st.session_state.selected_page = "1. AI Analysis: Input & Results"

# --- HELPER FUNCTIONS ---
def make_prediction(model, le_product, le_service, inputs):
    try:
        product_encoded = le_product.transform([inputs['product']])[0]
        service_encoded = le_service.transform([inputs['service']])[0]
    except ValueError as e:
        st.error(f"Categorical encoding error: {e}")
        return None

    features = np.array([
        inputs['length'], inputs['pressure'], inputs['temperature'],
        inputs['size'], product_encoded, service_encoded
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
        key=lambda x: x[1], reverse=True
    )[:3]
    
    feature_importance = []
    if hasattr(model, 'feature_importances_'):
        ordered_feature_names = ['Length (km)', 'Design Pressure (barg)', 'Design Temperature (°C)', 'Pipeline Size (inch)', 'Product_Encoded', 'Service_Encoded']
        feature_importance = sorted(
            zip(ordered_feature_names, model.feature_importances_), 
            key=lambda x: x[1], reverse=True
        )

    return {
        'material': pred_class,
        'confidence': confidence,
        'alternatives': [{'material': c, 'score': p} for c, p in alternatives],
        'feature_importance': feature_importance,
        'probabilities': probabilities_dict
    }

def get_rejection_reasoning(material_type, inputs):
    # This function is used both for the Streamlit UI (HTML) and the PDF report (plain text/list)
    reasons = []
    P_HIGH = 110 # barg
    T_HIGH = 85  # degC
    SIZE_LARGE = 24 # inch
    SIZE_SMALL = 8 # inch
    P_LOW = 40 # barg

    pressure = inputs['pressure']
    temperature = inputs['temperature']
    size = inputs['size']
    service = inputs['service']

    # Text formatting for HTML (Streamlit UI)
    def format_reason(reason):
        return f"<li>{reason}</li>"
    
    # Text formatting for PDF (Plain Text)
    def format_reason_pdf(reason):
        return f"- {reason}"
    
    # Check if we are formatting for PDF (a simple way to differentiate based on environment variable)
    is_pdf = 'PDF_CONTEXT' in os.environ and os.environ['PDF_CONTEXT'] == 'TRUE'
    
    reason_formatter = format_reason_pdf if is_pdf else format_reason
    
    # NOTE: HTML bolding is kept in the string for the UI, but FPDF will render it as plain text.
    if material_type == 'RTP':
        if pressure > P_HIGH: reasons.append(reason_formatter(f"Design Pressure ({pressure:.1f} barg) exceeds the typical limit for **RTP** (generally below {P_HIGH} barg)."))
        if temperature > T_HIGH: reasons.append(reason_formatter(f"Design Temperature ({temperature:.1f}°C) is high, approaching the operational limit for thermoplastic materials."))
        if size > SIZE_SMALL: reasons.append(reason_formatter(f"Pipeline Size ({size} inch) is large; **RTP** is typically favored for smaller diameter pipes (<{SIZE_SMALL} inch)."))
    elif material_type == 'Carbon Steel':
        if service == 'Sour': reasons.append(reason_formatter(f"Service is **'Sour'** (corrosive), which often requires specialized corrosion resistant materials or internal lining (like **IFL**)."))
        if pressure < P_LOW and size < SIZE_SMALL: reasons.append(reason_formatter(f"The combined low pressure ({pressure:.1f} barg) and small size ({size} inch) mean a more cost-effective material (**RTP** or **Flexible**) might be preferable."))
    elif material_type == 'IFL':
        if service == 'Sweet': reasons.append(reason_formatter("Service is **'Sweet'** (non-corrosive), eliminating the primary justification for the high cost of the internal lining (IFL)."))
        if pressure < P_LOW: reasons.append(reason_formatter(f"Design Pressure ({pressure:.1f} barg) is low; **IFL** is generally reserved for corrosive, high-pressure applications."))
    elif material_type == 'Flexible':
        if pressure > P_HIGH: reasons.append(reason_formatter(f"Design Pressure ({pressure:.1f} barg) exceeds the operational limit for many common **Flexible** pipe designs."))
        if size > SIZE_LARGE: reasons.append(reason_formatter(f"Pipeline Size ({size} inch) is large, making installation and procurement of high-diameter **Flexible** pipe complex."))
    
    if not reasons and st.session_state.prediction_made and not is_pdf: # Only for UI, not for PDF
        confidence = st.session_state.prediction_result['confidence']
        reasons.append(reason_formatter(f"The AI's confidence ({confidence*100:.1f}%) in the primary recommendation was significantly higher, suggesting a more optimal fit based on the historical feature patterns."))
        
    if is_pdf:
        # Return plain text list for PDF
        return "\n".join(reasons)
    else:
        # Return HTML list for Streamlit UI
        return "<ul>" + "".join(reasons) + "</ul>"

def get_img_as_base64(file):
    try:
        with open(file, "rb") as f:
            return base64.b64encode(f.read()).decode()
    except FileNotFoundError:
        return None

# --- NEW: PDF GENERATION FUNCTION ---
def create_prediction_report_pdf(inputs, result):
    # Temporarily set environment variable for PDF formatting in get_rejection_reasoning
    os.environ['PDF_CONTEXT'] = 'TRUE' 
    
    pdf = FPDF(orientation='P', unit='mm', format='A4')
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    
    # Title
    pdf.set_font("Arial", "B", 18)
    pdf.set_text_color(30, 58, 138) # Dark Blue
    pdf.cell(0, 10, "PMAT AI Material Assessment Report", 0, 1, 'C')
    pdf.set_line_width(0.5)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(5)

    # Metadata
    pdf.set_font("Arial", "", 10)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(0, 5, f"Date Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", 0, 1, 'R')
    pdf.ln(5)
    
    # 1. Inputs
    pdf.set_font("Arial", "B", 12)
    pdf.set_text_color(30, 58, 138)
    pdf.cell(0, 7, "1. Input Parameters", 0, 1, 'L')
    pdf.set_font("Arial", "", 10)
    pdf.set_text_color(0, 0, 0)
    
    input_data = [
        ("Pipeline Size (inch):", inputs['size']),
        ("Length (km):", inputs['length']),
        ("Design Pressure (barg):", inputs['pressure']),
        ("Design Temperature (°C):", inputs['temperature']),
        ("Product Type:", inputs['product']),
        ("Service Type:", inputs['service']),
    ]
    
    for label, value in input_data:
        pdf.cell(60, 5, label, 0, 0, 'L')
        pdf.cell(0, 5, str(value), 0, 1, 'L')
    pdf.ln(8)

    # 2. Recommendation
    pdf.set_font("Arial", "B", 14)
    
    # Get the recommended material color (R, G, B integers)
    rec_color = MATERIAL_COLORS.get(result['material'], '#94a3b8').lstrip('#')
    r, g, b = tuple(int(rec_color[i:i+2], 16) for i in (0, 2, 4))
    
    pdf.set_fill_color(r, g, b)
    pdf.set_text_color(255, 255, 255) # White text for dark background
    
    pdf.cell(0, 10, f"2. AI Recommended Material: {result['material']}", 0, 1, 'C', 1)
    
    pdf.set_font("Arial", "B", 12)
    pdf.set_text_color(0, 0, 0) # Back to black for next line
    pdf.cell(0, 7, f"AI Confidence Score: {result['confidence']*100:.1f}%", 0, 1, 'C')
    pdf.ln(5)

    # 3. Probability Distribution
    pdf.set_font("Arial", "B", 12)
    pdf.set_text_color(30, 58, 138)
    pdf.cell(0, 7, "3. Material Probability Distribution", 0, 1, 'L')
    pdf.set_font("Arial", "", 10)
    pdf.set_text_color(0, 0, 0)
    
    # Prepare data for a simple table
    pdf.set_fill_color(240, 240, 240)
    pdf.cell(60, 7, "Material", 1, 0, 'C', 1)
    pdf.cell(30, 7, "Probability (%)", 1, 1, 'C', 1)

    sorted_probs = sorted(result['probabilities'].items(), key=lambda x: x[1], reverse=True)
    for material, prob in sorted_probs:
        pdf.cell(60, 7, material, 1, 0, 'L')
        pdf.cell(30, 7, f"{prob*100:.1f}", 1, 1, 'R')
    pdf.ln(5)
    
    # 4. Rejection Analysis
    pdf.set_font("Arial", "B", 12)
    pdf.set_text_color(30, 58, 138)
    pdf.cell(0, 7, "4. Rejection Analysis (Why Alternatives Scored Lower)", 0, 1, 'L')
    pdf.set_font("Arial", "", 10)
    pdf.set_text_color(0, 0, 0)
    
    for alt in result['alternatives']:
        material = alt['material']
        score = alt['score']
        # This will call get_rejection_reasoning in PDF context mode
        reasoning_text = get_rejection_reasoning(material, inputs)
        
        pdf.set_font("Arial", "B", 11)
        pdf.cell(0, 6, f"Material: {material} (AI Score: {score*100:.1f}%)", 0, 1, 'L')
        pdf.set_font("Arial", "", 10)
        
        # Output the reasoning text (multicell handles line breaks)
        pdf.multi_cell(0, 5, reasoning_text)
        pdf.ln(3)

    # Cleanup environment variable
    del os.environ['PDF_CONTEXT']

    # --- FIX APPLIED HERE: pdf.output(dest='S') already returns bytes, remove .encode() ---
    return pdf.output(dest='S')
# --- END PDF GENERATION FUNCTION ---


# --- HEADER (Updated with Logo) ---
img_b64 = get_img_as_base64(LOGO_PATH)

if img_b64:
    header_content = f"""
    <div style="display: flex; align-items: center; justify-content: center; text-align: center;">
        <img src="data:image/png;base66,{img_b64}" style="height: 120px; margin-right: 20px;">
        <div>
            <h1 style="margin: 0; color: white;">PMAT | Pipeline Material Assessment Tool</h1>
            <p style="margin: 0.3rem 0 0 0; color: #e0e7ff;">AI-Powered Material Selection and Decision Support Dashboard</p>
        </div>
    </div>
    """
    st.markdown(f"""
    <div class="main-header" style="padding: 1rem 2rem;">
        {header_content}
    </div>
    """, unsafe_allow_html=True)
else:
    # Fallback to text-only header if the logo file is missing
    st.markdown("""
    <div class="main-header">
        <h1>PMAT | Pipeline Material Assessment Tool</h1>
        <p>AI-Powered Material Selection and Decision Support Dashboard</p>
    </div>
    """, unsafe_allow_html=True)


# --- NAVIGATION DEFINITION ---
PAGES = {
    "1. AI Analysis: Input & Results": "ai_analysis",
    "2. Engineering Decision Matrix": "decision_matrix",
    "3. Global XAI & Historical Data": "global_xai",
    "4. Prediction History": "history", 
}
PAGE_KEYS = list(PAGES.keys())

# --- SIDEBAR (ONLY Model Context) ---
with st.sidebar:
    st.header("Model Context")
    
    if rf_model:
        st.metric("Algorithm", "Random Forest Classifier")
        st.metric("Model Accuracy", "88%")
        st.success("Model: Active")
    else:
        st.error("Model: Not Loaded")
    
    st.markdown("---")

    # Pipe Types / Materials Information
    st.subheader("Pipeline Materials / Pipe Types")

    pipe_info = {
        "Carbon Steel (CS)": "Most widely used. Strong, durable, cost-effective; suitable for high-pressure service.",
        "Infield Line (IFL)": "Short-distance pipelines within a field (well → manifold). Usually carbon steel.",
        "Reinforced Thermoplastic Pipe (RTP)": "Lightweight, corrosion-proof composite pipe. Fast installation; good for sour service.",
        "Flexible Pipe": "Multi-layer dynamic pipe for offshore environments; excellent fatigue resistance."
    }

    for pipe, desc in pipe_info.items():
        st.markdown(
            f"**{pipe}**<br>"
            f"<span style='font-size:13px;color:#888;'>{desc}</span>",
            unsafe_allow_html=True
        )



# --- MAIN BODY NAVIGATION (Top Menu/Radio Button) ---
# This is the primary navigation widget.
st.radio(
    "Select Step:", 
    options=PAGE_KEYS, 
    index=PAGE_KEYS.index(st.session_state.selected_page),
    key='selected_page', # Uses the main session state key
    horizontal=True
)

# The logic below uses the synchronized state variable
page_selection = st.session_state.selected_page


# --- CHECK IF MODEL IS LOADED ---
if rf_model is None or le_product is None or le_service is None:
    st.error("Model or encoders not loaded properly. Please check your files.")
    st.stop()


# --- SECTION 1: AI ANALYSIS (INPUT & RESULTS) ---
if page_selection == "1. AI Analysis: Input & Results":
    st.header("1. AI Analysis: Pipeline Parameters Input & Results")

    # --- INPUT FORM ---
    st.subheader("Pipeline Parameters Input")
    st.markdown("Define the technical specifications for the AI analysis.")

    with st.form("pipeline_inputs"):
        col1, col2 = st.columns(2)

        product_options = df['Product'].unique().tolist() if df is not None and 'Product' in df.columns else ['Gas', 'Oil', 'Condensate', 'Water']
        service_options = df['Service'].unique().tolist() if df is not None and 'Service' in df.columns else ['Sweet', 'Sour']

        with col1:
            st.subheader("Physical Parameters")
            pipeline_size = st.number_input("Pipeline Size (inch)", 2, 36, st.session_state.current_inputs['size'], 2)
            length = st.number_input("Length (km)", 1.0, 75.0, st.session_state.current_inputs['length'], 0.1)
            product_index = product_options.index(st.session_state.current_inputs['product']) if st.session_state.current_inputs['product'] in product_options else 0
            product = st.selectbox("Product Type", options=product_options, index=product_index)

        with col2:
            st.subheader("Operating Conditions")
            service_index = service_options.index(st.session_state.current_inputs['service']) if st.session_state.current_inputs['service'] in service_options else 0
            service = st.selectbox("Service Type", options=service_options, index=service_index)
            pressure = st.number_input("Design Pressure (barg)", 20.0, 180.0, st.session_state.current_inputs['pressure'], 0.1)
            temperature = st.number_input("Design Temperature (°C)", 25.0, 130.0, st.session_state.current_inputs['temperature'], 0.1)

        submitted = st.form_submit_button("Generate Prediction & View Results", use_container_width=True)

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
        with st.spinner("Generating recommendation..."):
            time.sleep(0.5)
            result = make_prediction(rf_model, le_product, le_service, st.session_state.current_inputs)
            
            if result:
                st.session_state.prediction_result = result
                st.session_state.prediction_made = True
                st.session_state.prediction_history.append({
                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'inputs': st.session_state.current_inputs,
                    'result': result['material'],
                    'confidence': result['confidence']
                })
                st.rerun() 
    
    # --- DISPLAY RESULTS (Continuation of Section 1) ---
    if st.session_state.prediction_made and st.session_state.prediction_result:
        result = st.session_state.prediction_result
        inputs = st.session_state.current_inputs
        st.markdown("---")
        st.subheader("AI Recommendation & Explainable AI (XAI) Results")
        st.success("Prediction Analysis Complete.")

        # Recommendation card
        color = MATERIAL_COLORS.get(result['material'], '#94a3b8')
        
        col_rec, col_download = st.columns([3, 1])

        with col_rec:
            st.markdown(f"""
            <div style="background-color:{color}33; padding:2rem; border-radius:12px; border-left: 6px solid {color};">
            <h2 style="margin:0; color:{color};">Recommended Material: {result['material']}</h2>
            <p style="font-size:1.2rem; margin:0.5rem 0 0 0;">Confidence (AI Score): <strong>{result['confidence']*100:.1f}%</strong></p>
            </div>
            """, unsafe_allow_html=True)
        
        # --- PDF DOWNLOAD BUTTON (NEW) ---
        with col_download:
            pdf_bytes = create_prediction_report_pdf(inputs, result)
            b64_pdf = base64.b64encode(pdf_bytes).decode('latin-1')
            
            # The download link HTML
            pdf_download_link = f"""
            <a href="data:application/pdf;base64,{b64_pdf}" download="PMAT_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf">
                <button style="
                    width: 100%; 
                    background: linear-gradient(90deg, #10b981 0%, #059669 100%); 
                    color: white; 
                    font-weight: bold; 
                    padding: 0.75rem; 
                    border-radius: 8px; 
                    border: none; 
                    font-size: 1.1rem; 
                    transition: all 0.3s ease; 
                    margin-top: 2rem;">
                    ⬇️ Download PDF Report
                </button>
            </a>
            """
            st.markdown(pdf_download_link, unsafe_allow_html=True)
        # --- END PDF DOWNLOAD BUTTON ---
            
        st.markdown("---")

        # --- Probability Distribution (Sequential Visual) ---
        st.subheader("Material Probability Distribution")
        prob_df = pd.DataFrame([
            {'Material': k, 'Probability': v*100} 
            for k, v in result['probabilities'].items()
        ])
        prob_df = prob_df.sort_values('Probability', ascending=False)
        
        fig_prob = px.bar(
            prob_df, x='Material', y='Probability', color='Material', color_discrete_map=MATERIAL_COLORS, text='Probability'
        )
        fig_prob.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
        fig_prob.update_layout(showlegend=False, yaxis_title="Probability (%)", xaxis_title="Material Type", height=400)
        st.plotly_chart(fig_prob, use_container_width=True)

        st.markdown("---")

        # --- Feature Importance (Sequential Visual) ---
        if result['feature_importance']:
            st.subheader("Feature Importance Analysis (Local XAI)")
            st.markdown("Shows which input factors drove the model's decision for this specific case.")
            fi_df = pd.DataFrame(result['feature_importance'], columns=['Feature', 'Importance'])
            
            fig_fi = px.bar(
                fi_df, x='Importance', y='Feature', orientation='h', color='Importance', color_continuous_scale='Blues'
            )
            fig_fi.update_layout(
                yaxis=dict(autorange="reversed"), showlegend=False, height=400, xaxis_title="Relative Importance Score"
            )
            st.plotly_chart(fig_fi, use_container_width=True)
        
        st.markdown("---")

        # --- Visual Validation (Sequential Visual) ---
        st.subheader("Visual Validation: Your Design Point on the Historical Envelope")
        
        fig_pinpoint = px.scatter(
            df, x='Design Temperature degC', y='Design Pressure (barg)', color='Type', opacity=0.5,
            color_discrete_map=MATERIAL_COLORS, title='Historical P-T Data by Material Type', template='plotly_white'
        )
        
        fig_pinpoint.add_trace(
            go.Scatter(
                x=[inputs['temperature']], y=[inputs['pressure']], mode='markers',
                marker=dict(size=18, color=color, symbol='star', line=dict(width=2, color='white')),
                name=f"Your Input: {result['material']}",
                hovertemplate=f"Predicted: {result['material']}<br>P: {inputs['pressure']} barg<br>T: {inputs['temperature']} °C<extra></extra>"
            )
        )

        fig_pinpoint.update_layout(height=500, legend_title='Material Type', xaxis_title='Design Temperature ($^{\circ}$C)', yaxis_title='Design Pressure (barg)')
        st.plotly_chart(fig_pinpoint, use_container_width=True)

        st.markdown("---")

        # Local XAI: Rejection Analysis (kept as a table/list)
        st.subheader("Rejection Analysis (Local XAI)")
        st.info("Why the alternatives scored lower, based on engineering constraints and input parameters.")
        
        alt_data = []
        for alt in result['alternatives']:
            alt_data.append({
                'Material': alt['material'],
                'AI Score': f"{alt['score']*100:.1f}%",
                'Reasoning': get_rejection_reasoning(alt['material'], inputs) 
            })
            
        alt_df = pd.DataFrame(alt_data)
        st.markdown(alt_df.to_html(escape=False, index=False), unsafe_allow_html=True)
        
        st.markdown("---")


# --- SECTION 2: DECISION MATRIX ---
if page_selection == "2. Engineering Decision Matrix":
    if not st.session_state.prediction_made:
        st.warning("Please input parameters and generate a prediction in the '1. AI Analysis: Input & Results' section first.")
    else:
        result = st.session_state.prediction_result
        st.header("2. Engineering Decision Support Matrix")
        st.markdown("Use your expert judgment to weigh the AI's prediction against non-technical factors to determine the best material.")
        
        # --- Compile material list for matrix ---
        matrix_materials = [{'Material': result['material'], 'AI Probability': result['confidence'], 'Type': 'Primary AI Pick'}]
        for alt in result['alternatives']:
            matrix_materials.append({'Material': alt['material'], 'AI Probability': alt['score'], 'Type': 'Alternative'})
        df_matrix = pd.DataFrame(matrix_materials)
        
        with st.container(border=True):
            
            # --- User Weights ---
            st.subheader("User Weights: Importance of Non-AI Factors")
            st.markdown("**Slider Meaning:** 0 = Factor is irrelevant or not considered. 5 = Factor is critically important in the final decision.")

            col_w1, col_w2, col_w3 = st.columns(3)

            weight_cost = col_w1.slider("Relative Cost/CAPEX Importance", 0, 5, 4, key='cost_weight', help="How much cost (CAPEX/OPEX) should influence the final decision (5=High).")
            weight_risk = col_w2.slider("Failure/Safety Consequence Importance", 0, 5, 3, key='risk_weight', help="How much risk/safety consequence should influence the final decision (5=High).")
            weight_availability = col_w3.slider("Lead Time/Availability Importance", 0, 5, 2, key='avail_weight', help="How much material availability and procurement time should influence the final decision (5=High).")

            weights = {
                'Cost Score': weight_cost,
                'Risk Score': weight_risk,
                'Availability Score': weight_availability
            }
        
        # --- Scoring Table (Updated Guide) ---
        st.subheader("Scoring Table: Manual Engineering Score (1-5)")
        st.markdown("**Scoring Guide (Your Expert Judgment):**")
        st.markdown("""
        <ul style="padding-left: 20px;">
            <li><strong style="color: #0d9488;">Cost Score:</strong> 5 = Lowest estimated CAPEX/OPEX. 1 = Highest estimated CAPEX/OPEX.</li>
            <li><strong style="color: #7e22ce;">Risk Score:</strong> 5 = Lowest risk/consequence of failure (Highest Safety). 1 = Highest risk/consequence of failure.</li>
            <li><strong style="color: #b45309;">Availability Score:</strong> 5 = Fastest lead time and best availability. 1 = Longest lead time and low availability.</li>
        </ul>
        """, unsafe_allow_html=True)

        score_cols = st.columns(len(df_matrix))
        manual_scores = {}

        for i, row in df_matrix.iterrows():
            mat = row['Material']
            with score_cols[i]:
                st.markdown(f"**{mat}** ({row['AI Probability']*100:.1f}%)")
                
                # Sliders with unique keys generated dynamically
                cost_s = st.slider(f"Cost Score (1-5)", 1, 5, key=f"cost_score_{mat}", value=3, help="5=Low Cost, 1=High Cost")
                risk_s = st.slider(f"Risk Score (1-5)", 1, 5, key=f"risk_score_{mat}", value=4, help="5=Low Risk, 1=High Risk")
                avail_s = st.slider(f"Availability Score (1-5)", 1, 5, key=f"avail_score_{mat}", value=5, help="5=Fast Lead Time, 1=Slow Lead Time")
                
                manual_scores[mat] = {
                    'Cost Score': cost_s,
                    'Risk Score': risk_s,
                    'Availability Score': avail_s
                }

        # --- Final Score Calculation ---
        st.subheader("Final Score: Combined AI and Engineering Judgment")
        
        st.markdown(r"""
        The **Final Score** is calculated as the sum of AI probability and the weighted user scores:
        $$\text{Final Score} = (\text{AI Probability}) + \sum (\text{User Score} \times \text{User Weight})$$
        The highest Final Score indicates the recommended material based on **both** data and expert input.
        """)
        
        final_scores_list = []
        
        for i, row in df_matrix.iterrows():
            mat = row['Material']
            scores = manual_scores[mat]
            
            # Calculate weighted manual score: Sum (User Score * User Weight)
            weighted_manual_sum = (
                scores['Cost Score'] * weights['Cost Score'] +
                scores['Risk Score'] * weights['Risk Score'] +
                scores['Availability Score'] * weights['Availability Score']
            )
            
            # Calculate Final Score: AI Probability + Weighted Manual Sum
            final_score = row['AI Probability'] + weighted_manual_sum
            
            final_scores_list.append({
                'Material': mat,
                'AI Probability': f"{row['AI Probability']*100:.1f}%",
                'Weighted Manual Score': f"{weighted_manual_sum:.1f}",
                'Final Weighted Score': final_score,
            })

        df_final = pd.DataFrame(final_scores_list)
        df_final = df_final.sort_values(by='Final Weighted Score', ascending=False).reset_index(drop=True)
        
        top_material = df_final.iloc[0]['Material']
        
        st.markdown("### Decision Table")
        st.dataframe(
            df_final.style.bar(
                subset=['Final Weighted Score'], 
                color=MATERIAL_COLORS['Flexible']
            ).format({'Final Weighted Score': "{:.2f}"}),
            use_container_width=True
        )

        st.success(f"### Final Recommended Material (Based on Weighted Score): {top_material}")
        
        st.markdown("---")


# --- SECTION 3: GLOBAL XAI & HISTORICAL DATA ---
if page_selection == "3. Global XAI & Historical Data":
    if df is not None:
        st.header("3. Global XAI & Historical Data")
        
        # Raw Historical Data
        st.subheader("Raw Historical Dataset")
        st.info("The complete 400-point dataset used to train the AI model.")
        with st.expander("Expand to view and filter the full dataset"):
            st.dataframe(df, use_container_width=True)
        
        st.markdown("---")

        st.subheader("Global Data Analysis (Global XAI Context)")
        st.markdown("Review the overall patterns the AI model was trained on.")
        
        # 1. Corrosivity Profile
        st.markdown("#### Corrosivity Profile: Product and Service Breakdown")
        fig_breakdown = px.histogram(
            df, x='Product', color='Service', barmode='group', text_auto=True,
            color_discrete_map={'Sweet': MATERIAL_COLORS['Carbon Steel'], 'Sour': MATERIAL_COLORS['IFL']},
            title='Pipeline Count by Product and Corrosivity Service', template='plotly_white'
        )
        st.plotly_chart(fig_breakdown, use_container_width=True)
        
        # 2. Design Pressure Distribution
        st.markdown("#### Design Pressure Distribution by Material Type")
        fig_box = px.box(
            df, x='Type', y='Design Pressure (barg)', color='Type', color_discrete_map=MATERIAL_COLORS,
            points="all", title='Distribution of Design Pressure for Each Material', template='plotly_white'
        )
        st.plotly_chart(fig_box, use_container_width=True)
        
    else:
        st.error("Historical dataset could not be loaded. Please check the file path.")

# --- SECTION 4: PREDICTION HISTORY ---
if page_selection == "4. Prediction History":
    st.header("4. Prediction History")
    
    if not st.session_state.prediction_history:
        st.info("No prediction history available yet. Run a prediction in the first section to start tracking.")
    else:
        st.subheader(f"Recent Prediction History ({len(st.session_state.prediction_history)} total runs)")
        st.info("Showing the last 10 entries for brevity.")
        
        history_display = []
        # Reverse and take the last 10
        for h in st.session_state.prediction_history[::-1][:10]:
            inputs_str = f"Size: {h['inputs']['size']} inch, Length: {h['inputs']['length']} km, {h['inputs']['product']}, {h['inputs']['service']}"
            history_display.append({
                'Timestamp': h['timestamp'],
                'Parameters': inputs_str,
                'Recommended Material': h['result'],
                'Confidence': f"{h['confidence']*100:.1f}%"
            })
        
        history_df = pd.DataFrame(history_display)
        st.dataframe(history_df, use_container_width=True, hide_index=True)
        
        st.markdown("---")
        if st.button("Clear All History", key='clear_history', use_container_width=True):
            st.session_state.prediction_history = []
            st.session_state.prediction_made = False
            st.rerun()

# --- FOOTER ---
st.markdown("---")
st.markdown("""
<div style="text-align:center;color:#64748b;">
<p>PMAT v1.0 | UTP & PETRONAS Carigali | &copy; 2025</p>
</div>
""", unsafe_allow_html=True)

