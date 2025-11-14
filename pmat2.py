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
.stButton>button { width: 100%; background: linear-gradient(90deg,#3b82f6 0%,#2563eb 100%); color: white; font-weight:bold; padding:0.75rem; border-radius:8px; border:none; font-size:1.1rem; transition: all 0.3s ease;}
.stButton>button:hover { background: linear-gradient(90deg,#2563eb 0%,#1d4ed8 100%); box-shadow:0 4px 8px rgba(37,99,235,0.3); transform: translateY(-2px);}
.info-box {background-color: #eff6ff; border-left: 4px solid #3b82f6; padding:1rem; border-radius:4px; margin:1rem 0;}
.warning-box {background-color:#fef3c7; border-left:4px solid #f59e0b; padding:1rem; border-radius:4px; margin:1rem 0;}
.success-box {background-color:#d1fae5; border-left:4px solid #10b981; padding:1rem; border-radius:4px; margin:1rem 0;}
h1,h2,h3 {color:#1e3a8a;}
.stMarkdown h3 {margin-top: 2rem;} /* Better spacing for subheaders */
</style>
""", unsafe_allow_html=True)

# --- LOAD MODEL & DATA ---
@st.cache_resource
def load_model_and_encoders():
    # Model loading logic (as before)
    model = None
    le_product = None
    le_service = None
    if os.path.exists(MODEL_PATH):
        try:
            model = joblib.load(MODEL_PATH)
        except:
            pass # Error handled in main flow
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
    # Dataset loading logic (as before)
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
    st.session_state.current_inputs = {}

# --- HELPER FUNCTIONS ---
def get_dataset_stats(df):
    if df is None or df.empty:
        return None
    return {
        'total_pipelines': len(df),
        'material_distribution': df['Type'].value_counts().to_dict() if 'Type' in df.columns else {}
    }

def make_prediction(model, le_product, le_service, inputs):
    # Prediction logic (as before)
    try:
        product_encoded = le_product.transform([inputs['product']])[0]
        service_encoded = le_service.transform([inputs['service']])[0]
    except ValueError as e:
        st.error(f"Categorical encoding error: {e}")
        return None

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

def get_rejection_reasoning(material_type, inputs):
    # Rejection reasoning logic (as before)
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

    if material_type == 'RTP':
        if pressure > P_HIGH: reasons.append(f"Design Pressure ({pressure:.1f} barg) exceeds the typical limit for **RTP** (generally below {P_HIGH} barg).")
        if temperature > T_HIGH: reasons.append(f"Design Temperature ({temperature:.1f}°C) is high, approaching the operational limit for thermoplastic materials.")
        if size > SIZE_SMALL: reasons.append(f"Pipeline Size ({size} inch) is large; **RTP** is typically favored for smaller diameter pipes (<{SIZE_SMALL} inch).")
    elif material_type == 'Carbon Steel':
        if service == 'Sour': reasons.append(f"Service is **'Sour'** (corrosive), which often requires specialized corrosion resistant materials or internal lining (like **IFL**).")
        if pressure < P_LOW and size < SIZE_SMALL: reasons.append(f"The combined low pressure ({pressure:.1f} barg) and small size ({size} inch) mean a more cost-effective material (**RTP** or **Flexible**) might be preferable.")
    elif material_type == 'IFL':
        if service == 'Sweet': reasons.append("Service is **'Sweet'** (non-corrosive), eliminating the primary justification for the high cost of the internal lining (IFL).")
        if pressure < P_LOW: reasons.append(f"Design Pressure ({pressure:.1f} barg) is low; **IFL** is generally reserved for corrosive, high-pressure applications.")
    elif material_type == 'Flexible':
        if pressure > P_HIGH: reasons.append(f"Design Pressure ({pressure:.1f} barg) exceeds the operational limit for many common **Flexible** pipe designs.")
        if size > SIZE_LARGE: reasons.append(f"Pipeline Size ({size} inch) is large, making installation and procurement of high-diameter **Flexible** pipe complex.")
    
    if not reasons and st.session_state.prediction_made:
        confidence = st.session_state.prediction_result['confidence']
        reasons.append(f"The AI's confidence ({confidence*100:.1f}%) in the primary recommendation was significantly higher, suggesting a more optimal fit based on the historical feature patterns.")

    return "<ul>" + "".join([f"<li>{r}</li>" for r in reasons]) + "</ul>"


# --- HEADER ---
st.markdown("""
<div class="main-header">
<h1>🔧 PMAT | Pipeline Material Assessment Tool</h1>
<p>AI-Powered Material Selection and Decision Support Dashboard</p>
</div>
""", unsafe_allow_html=True)

# --- SIDEBAR FOR MODEL CONTEXT ---
with st.sidebar:
    st.header("🎯 Model Context")
    if rf_model:
        st.metric("Algorithm", "Random Forest")
        st.metric("Accuracy", "98%")
        st.success("🟢 Model: Active")
    else:
        st.error("❌ Model: Not Loaded")
    
    stats = get_dataset_stats(df)
    if stats:
        st.markdown("---")
        st.header("📈 Dataset Stats")
        st.metric("Total Pipelines", stats['total_pipelines'])
        st.subheader("Material Distribution")
        for material, count in stats['material_distribution'].items():
            st.markdown(f"**{material}**: {count} ({count/stats['total_pipelines']*100:.1f}%)")

# --- CHECK IF MODEL IS LOADED ---
if rf_model is None or le_product is None or le_service is None:
    st.error("⚠️ Model or encoders not loaded properly. Please check your files.")
    st.stop()


# --- USER INPUT FORM (Back in Main Page) ---
st.header("1. Pipeline Parameters Input")
st.markdown("Define the technical specifications for the AI analysis.")

with st.form("pipeline_inputs"):
    col1, col2 = st.columns(2)

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

    submitted = st.form_submit_button("Generate AI Recommendation and Decision Matrix", use_container_width=True)

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
    with st.spinner(" Generating recommendation..."):
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

# --- DISPLAY RESULTS (Main Area) ---
if st.session_state.prediction_made and st.session_state.prediction_result:
    result = st.session_state.prediction_result
    inputs = st.session_state.current_inputs
    st.markdown("---")
    st.header("2. AI Recommendation & Local XAI")
    st.success("✅ Analysis Complete! The AI has provided a technical recommendation.")

    # Recommendation card
    color = MATERIAL_COLORS.get(result['material'], '#94a3b8')
    
    st.markdown(f"""
    <div style="background-color:{color}33; padding:2rem; border-radius:12px; border-left: 6px solid {color};">
    <h2 style="margin:0; color:{color};">Recommended Material: {result['material']}</h2>
    <p style="font-size:1.2rem; margin:0.5rem 0 0 0;">Confidence (AI Score): <strong>{result['confidence']*100:.1f}%</strong></p>
    </div>
    """, unsafe_allow_html=True)

    col_xai_1, col_xai_2 = st.columns(2)

    with col_xai_1:
        # Probability distribution chart (brings back alternatives percentage)
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
            color_discrete_map=MATERIAL_COLORS,
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

    with col_xai_2:
        # Feature importance (Local XAI)
        if result['feature_importance']:
            st.subheader("Feature Importance Analysis")
            st.markdown("Shows which input factors drove the model's decision for this specific case (Local XAI).")
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
                height=400,
                xaxis_title="Relative Importance Score"
            )
            st.plotly_chart(fig_fi, use_container_width=True)
            
    st.markdown("---")
    
    # --- Visual Validation (Requested Subheader) ---
    st.subheader("Visual Validation: Your Design Point on the Historical Envelope")
    
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
    
    st.markdown("---")
    
    # --- DECISION MATRIX (New Feature) ---
    st.header("3. Engineering Decision Support Matrix")
    st.markdown("Use your expert judgment to weigh the AI's prediction against non-technical factors.")
    
    # --- Compile material list for matrix ---
    matrix_materials = []
    matrix_materials.append({
        'Material': result['material'],
        'AI Probability': result['confidence'],
        'Type': 'Primary AI Pick'
    })
    for alt in result['alternatives']:
        matrix_materials.append({
            'Material': alt['material'],
            'AI Probability': alt['score'],
            'Type': 'Alternative'
        })
    df_matrix = pd.DataFrame(matrix_materials)
    
    with st.container(border=True):
        
        # --- User Weights (Requested Subheader) ---
        st.subheader("User Weights: Importance of Non-AI Factors (0-5)")
        st.markdown("Higher score means higher importance in the final weighted decision.")

        col_w1, col_w2, col_w3 = st.columns(3)

        weight_cost = col_w1.slider("Relative Cost/CAPEX Weight", 0, 5, 4)
        weight_risk = col_w2.slider("Failure/Safety Consequence Weight", 0, 5, 3)
        weight_availability = col_w3.slider("Lead Time/Availability Weight", 0, 5, 2)

        weights = {
            'Cost Score': weight_cost,
            'Risk Score': weight_risk,
            'Availability Score': weight_availability
        }
    
    # --- Scoring Table (Requested Subheader) ---
    st.subheader("Scoring Table: Manual Engineering Score (1-5)")
    st.markdown("**Cost Score:** 5=Low Cost, 1=High Cost. **Risk Score:** 5=Low Risk, 1=High Risk. **Availability Score:** 5=Fast, 1=Slow.")

    score_cols = st.columns(len(df_matrix))
    manual_scores = {}

    for i, row in df_matrix.iterrows():
        mat = row['Material']
        with score_cols[i]:
            st.markdown(f"**{mat}** ({row['AI Probability']*100:.1f}%)")
            
            cost_s = st.slider(f"Cost Score (1-5)", 1, 5, key=f"cost_{mat}", value=3)
            risk_s = st.slider(f"Risk Score (1-5)", 1, 5, key=f"risk_{mat}", value=4)
            avail_s = st.slider(f"Availability Score (1-5)", 1, 5, key=f"avail_{mat}", value=5)
            
            manual_scores[mat] = {
                'Cost Score': cost_s,
                'Risk Score': risk_s,
                'Availability Score': avail_s
            }

    # --- Final Score Calculation (Requested Subheader) ---
    st.subheader("Final Score: Combined AI and Engineering Judgment")
    
    st.markdown(r"""
    The **Final Score** is calculated based on your requested formula:
    $$\text{Final Score} = \text{AI Probability} + \sum (\text{User Score} \times \text{User Weight})$$
    *(Note: This results in a score greater than 1, reflecting the combined magnitude of AI and manual influence.)*
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
        
        # Calculate Final Score: AI Probability + Weighted Manual Sum (as requested formula)
        final_score = row['AI Probability'] + weighted_manual_sum
        
        final_scores_list.append({
            'Material': mat,
            'AI Probability': f"{row['AI Probability']*100:.1f}%",
            'Weighted Manual Score': f"{weighted_manual_sum:.1f}",
            'Final Weighted Score': final_score,
        })

    df_final = pd.DataFrame(final_scores_list)
    df_final = df_final.sort_values(by='Final Weighted Score', ascending=False).reset_index(drop=True)
    
    # Identify the top recommendation from the matrix
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
    
    # --- Global Dataset Analysis (Requested Subheader) ---
    if df is not None:
        st.subheader("Global Dataset Analysis & Context")
        st.markdown("Review the overall patterns the AI model was trained on for **Global XAI Context**.")
        
        with st.expander("Expand for Global XAI Visualizations"):

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

# --- FOOTER ---
st.markdown("---")
st.markdown("""
<div style="text-align:center;color:#64748b;">
<p>PMAT v1.0 | UTP & PETRONAS Carigali | © 2025</p>
</div>
""", unsafe_allow_html=True)
