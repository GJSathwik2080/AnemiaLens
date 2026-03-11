"""
AnemiaLens Web App — Beautiful, Clean UI for Non-Invasive Anemia Detection
Built with Streamlit with Enhanced Error Handling & Instructions
"""

import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import os
import cv2
from segmentation import ROISegmenter
from config import Config
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input


# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="AnemiaLens",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="auto",
)

# Beautiful custom CSS with improved styling
st.markdown("""
<style>
    /* Main background gradient */
    [data-testid="stAppViewContainer"] {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    
    /* Header styling */
    .header-title {
        text-align: center;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 40px 20px;
        border-radius: 15px;
        margin-bottom: 30px;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }
    
    .header-title h1 {
        margin: 0;
        font-size: 2.8em;
    }
    
    .header-title p {
        margin: 10px 0 0 0;
        font-size: 1.2em;
        opacity: 0.95;
    }
    
    /* Card styling */
    .upload-card {
        background: white;
        border-radius: 15px;
        padding: 25px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        margin: 15px 0;
        border-top: 5px solid;
        transition: all 0.3s ease;
    }
    
    .upload-card:hover {
        box-shadow: 0 6px 20px rgba(0,0,0,0.15);
        transform: translateY(-2px);
    }
    
    .upload-card-eye { border-top-color: #3498db; }
    .upload-card-nail { border-top-color: #e74c3c; }
    .upload-card-palm { border-top-color: #f39c12; }
    
    /* Result boxes */
    .result-box {
        padding: 30px;
        border-radius: 15px;
        margin: 20px 0;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    
    .result-healthy {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        color: white;
    }
    
    .result-anemic {
        background: linear-gradient(135deg, #ee0979 0%, #ff6a00 100%);
        color: white;
    }
    
    .result-box h2 {
        margin-top: 0;
        font-size: 2.2em;
    }
    
    .result-box p {
        font-size: 1.1em;
        margin: 15px 0;
    }
    
    /* Score display */
    .score-display {
        background: white;
        padding: 30px;
        border-radius: 15px;
        margin: 20px 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    
    .score-display-large {
        font-size: 4em;
        font-weight: bold;
        color: #667eea;
        margin: 20px 0;
    }
    
    /* Metric card */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 25px;
        border-radius: 15px;
        text-align: center;
        margin: 15px 0;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }
    
    .metric-card-value {
        font-size: 2.5em;
        font-weight: bold;
        margin: 15px 0;
    }
    
    .metric-card-label {
        font-size: 0.95em;
        opacity: 0.9;
    }
    
    .metric-card-status {
        font-size: 0.9em;
        margin-top: 10px;
        padding: 8px 15px;
        background: rgba(255,255,255,0.2);
        border-radius: 20px;
        display: inline-block;
    }
    
    /* Severity badge */
    .severity-badge {
        display: inline-block;
        padding: 10px 20px;
        border-radius: 25px;
        font-weight: bold;
        font-size: 0.95em;
        margin: 10px 5px;
    }
    
    .severity-mild { background: #fff3cd; color: #856404; }
    .severity-moderate { background: #ffe0b2; color: #e65100; }
    .severity-severe { background: #ffcdd2; color: #b71c1c; }
    
    /* Instructions box */
    .instructions-box {
        background: #ecf0f1;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #667eea;
        margin: 20px 0;
    }
    
    /* Color boxes */
    .upload-info { background: #e3f2fd; color: #1565c0; padding: 12px 15px; border-radius: 8px; margin: 10px 0; border-left: 4px solid #667eea; }
    .warning-box { background: #fff3cd; border-left: 5px solid #ffc107; padding: 15px; border-radius: 8px; color: #856404; margin: 15px 0; }
    .error-box { background: #f8d7da; border-left: 5px solid #dc3545; padding: 15px; border-radius: 8px; color: #721c24; margin: 15px 0; }
    .success-box { background: #d4edda; border-left: 5px solid #28a745; padding: 15px; border-radius: 8px; color: #155724; margin: 15px 0; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# LOAD MODELS (Cached for performance)
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource
def load_models():
    """Load all trained models once at startup"""
    models = {}
    try:
        for modality in ["conjunctiva", "nail", "palm"]:
            path = f"saved_models/{modality}_final.keras"
            if not os.path.exists(path):
                st.error(f"❌ Model not found: {path}")
                return None
            models[modality] = tf.keras.models.load_model(path)
        return models
    except Exception as e:
        st.error(f"❌ Error loading models: {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# PREDICTION FUNCTION
# ─────────────────────────────────────────────────────────────────────────────

def predict_from_image(model, pil_image, modality):
    """
    Run inference on an uploaded PIL image.
    Handles proper image conversion and preprocessing.
    """
    try:
        # Convert PIL Image to numpy array
        img_array = np.array(pil_image)
        
        # Handle different image formats
        if len(img_array.shape) == 2:  # Grayscale
            img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2BGR)
        elif img_array.shape[2] == 4:  # RGBA
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2BGR)
        elif img_array.shape[2] == 3:  # RGB
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        else:
            raise ValueError(f"Unexpected image shape: {img_array.shape}")
        
        # Segment ROI using the dedicated segmentation routine
        roi = ROISegmenter.segment(img_array, modality)
        
        # Resize to model input size
        roi_resized = cv2.resize(roi, Config.IMG_SIZE, 
                                 interpolation=cv2.INTER_AREA)
        
        # Convert BGR to RGB for preprocessing
        rgb = cv2.cvtColor(roi_resized, cv2.COLOR_BGR2RGB)
        
        # Prepare for model (float32, expand dims, preprocess)
        arr = np.expand_dims(rgb.astype(np.float32), axis=0)
        arr = preprocess_input(arr.copy())
        
        # Get prediction
        pred = model.predict(arr, verbose=0)[0][0]
        return float(pred)
        
    except Exception as e:
        st.error(f"❌ Error processing {modality} image: {str(e)}")
        raise


def get_severity(score):
    """Determine severity level from combined score"""
    if score > 0.8:
        return "SEVERE", "severity-severe", "🔴"
    elif score > 0.65:
        return "MODERATE", "severity-moderate", "🟠"
    elif score > 0.5:
        return "MILD", "severity-mild", "🟡"
    return None, None, None


# ─────────────────────────────────────────────────────────────────────────────
# MAIN APP
# ─────────────────────────────────────────────────────────────────────────────

def main():
    # Header
    st.markdown("""
    <div class="header-title">
        <h1>🩺 AnemiaLens</h1>
        <p>Non-Invasive Anemia Screening Using AI</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Instructions - Compact version
    with st.expander("ℹ️ Instructions", expanded=False):
        st.markdown("""
        **Upload 3 images:**
        - 👁️ Eye (inner eyelid)
        - 💅 Nail (nail bed)  
        - 🤚 Palm area
        
        **Tips:** Clear photos with good lighting give best results.
        
        **⚠️ Important:** This is a screening tool only. Always confirm results with a blood test.
        """)
    
    st.markdown("---")
    
    # Create 3 columns for image uploads
    col1, col2, col3 = st.columns(3)
    
    uploaded_files = {}
    images_dict = {}
    
    with col1:
        st.markdown("""
        <div class="upload-card upload-card-eye">
            <h3 style="color: #3498db; margin-top: 0;">👁️ Eye (Conjunctiva)</h3>
            <p style="color: #7f8c8d; font-size: 0.95em;">Inner eyelid area</p>
        </div>
        """, unsafe_allow_html=True)
        
        uploaded_files['conjunctiva'] = st.file_uploader(
            "Upload eye image",
            type=['png', 'jpg', 'jpeg'],
            key='conjunctiva',
            label_visibility="collapsed"
        )
        
        if uploaded_files['conjunctiva']:
            img = Image.open(uploaded_files['conjunctiva'])
            images_dict['conjunctiva'] = img
            st.image(img, use_column_width=True, caption="Conjunctiva Preview")
            st.markdown(
                '<div class="success-box">✓ Image uploaded successfully</div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                '<div class="upload-info">📤 Click to upload eye image</div>',
                unsafe_allow_html=True
            )
    
    with col2:
        st.markdown("""
        <div class="upload-card upload-card-nail">
            <h3 style="color: #e74c3c; margin-top: 0;">💅 Nail</h3>
            <p style="color: #7f8c8d; font-size: 0.95em;">Nail bed photograph</p>
        </div>
        """, unsafe_allow_html=True)
        
        uploaded_files['nail'] = st.file_uploader(
            "Upload nail image",
            type=['png', 'jpg', 'jpeg'],
            key='nail',
            label_visibility="collapsed"
        )
        
        if uploaded_files['nail']:
            img = Image.open(uploaded_files['nail'])
            images_dict['nail'] = img
            st.image(img, use_column_width=True, caption="Nail Preview")
            st.markdown(
                '<div class="success-box">✓ Image uploaded successfully</div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                '<div class="upload-info">📤 Click to upload nail image</div>',
                unsafe_allow_html=True
            )
    
    with col3:
        st.markdown("""
        <div class="upload-card upload-card-palm">
            <h3 style="color: #f39c12; margin-top: 0;">🤚 Palm</h3>
            <p style="color: #7f8c8d; font-size: 0.95em;">Palm area image</p>
        </div>
        """, unsafe_allow_html=True)
        
        uploaded_files['palm'] = st.file_uploader(
            "Upload palm image",
            type=['png', 'jpg', 'jpeg'],
            key='palm',
            label_visibility="collapsed"
        )
        
        if uploaded_files['palm']:
            img = Image.open(uploaded_files['palm'])
            images_dict['palm'] = img
            st.image(img, use_column_width=True, caption="Palm Preview")
            st.markdown(
                '<div class="success-box">✓ Image uploaded successfully</div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                '<div class="upload-info">📤 Click to upload palm image</div>',
                unsafe_allow_html=True
            )
    
    st.markdown("---")
    
    # Upload status indicator
    uploaded_count = sum(1 for f in uploaded_files.values() if f is not None)
    status_color = "#28a745" if uploaded_count == 3 else "#ffc107"
    status_text = f"✓ Ready to analyze" if uploaded_count == 3 else f"⏳ {uploaded_count}/3 images uploaded"
    
    st.markdown(f"""
    <div style="text-align: center; padding: 10px; background: {status_color}20; 
                border-left: 4px solid {status_color}; border-radius: 8px; margin: 15px 0;">
        <span style="color: {status_color}; font-weight: bold;">{status_text}</span>
    </div>
    """, unsafe_allow_html=True)
    
    # Analyze Button
    col_button = st.columns([1, 2, 1])
    with col_button[1]:
        analyze_button = st.button(
            "🔍 Analyze All Images",
            use_container_width=True,
            key="analyze",
            disabled=(uploaded_count < 3)
        )
    
    # Run Analysis
    if analyze_button:
        # Load models
        with st.spinner("🔄 Loading AI models..."):
            models = load_models()
            if models is None:
                st.error("❌ Could not load models. Please check that saved_models/ directory exists.")
                return
        
        st.markdown("---")
        st.markdown("## 📊 Analyzing Your Images...")
        
        # Run predictions
        try:
            progress_placeholder = st.empty()
            
            # Conjunctiva prediction
            with st.spinner("👁️ Analyzing eye..."):
                progress_placeholder.info("Processing conjunctiva image... (1/3)")
                eye_pred = predict_from_image(models['conjunctiva'], images_dict['conjunctiva'], 'conjunctiva')
            
            # Nail prediction
            with st.spinner("💅 Analyzing nail..."):
                progress_placeholder.info("Processing nail image... (2/3)")
                nail_pred = predict_from_image(models['nail'], images_dict['nail'], 'nail')
            
            # Palm prediction
            with st.spinner("🤚 Analyzing palm..."):
                progress_placeholder.info("Processing palm image... (3/3)")
                palm_pred = predict_from_image(models['palm'], images_dict['palm'], 'palm')
            
            # Clear progress message
            progress_placeholder.empty()
            
            # Calculate combined score (weighted)
            combined_score = eye_pred * 0.5 + nail_pred * 0.3 + palm_pred * 0.2
            
            st.markdown("---")
            st.markdown("# ✅ Analysis Complete!")
            st.markdown("---")
            
            # INDIVIDUAL RESULTS
            st.markdown("## 📋 Individual Predictions")
            
            result_col1, result_col2, result_col3 = st.columns(3)
            
            with result_col1:
                st.markdown(f"""
                <div class="metric-card">
                    <div style="font-size: 2.5em; margin: 10px 0;">👁️</div>
                    <div class="metric-card-label">Conjunctiva (Eye)</div>
                    <div class="metric-card-value">{eye_pred:.1%}</div>
                    <div class="metric-card-status">
                        {'⚠️ Anemic' if eye_pred > 0.5 else '✓ Healthy'}
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            with result_col2:
                st.markdown(f"""
                <div class="metric-card">
                    <div style="font-size: 2.5em; margin: 10px 0;">💅</div>
                    <div class="metric-card-label">Nail Bed</div>
                    <div class="metric-card-value">{nail_pred:.1%}</div>
                    <div class="metric-card-status">
                        {'⚠️ Anemic' if nail_pred > 0.5 else '✓ Healthy'}
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            with result_col3:
                st.markdown(f"""
                <div class="metric-card">
                    <div style="font-size: 2.5em; margin: 10px 0;">🤚</div>
                    <div class="metric-card-label">Palm Area</div>
                    <div class="metric-card-value">{palm_pred:.1%}</div>
                    <div class="metric-card-status">
                        {'⚠️ Anemic' if palm_pred > 0.5 else '✓ Healthy'}
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("---")
            
            # COMBINED SCORE
            st.markdown("## 🎯 Combined Screening Score")
            
            st.markdown(f"""
            <div class="score-display">
                <div style="font-size: 1.1em; color: #7f8c8d; margin-bottom: 10px;">
                    Based on weighted analysis: Conjunctiva 50% + Nail 30% + Palm 20%
                </div>
                <div class="score-display-large">{combined_score:.1%}</div>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("---")
            
            # FINAL DIAGNOSIS
            st.markdown("## 🏥 Screening Result")
            
            if combined_score > 0.5:
                severity, severity_class, severity_emoji = get_severity(combined_score)
                st.markdown(f"""
                <div class="result-box result-anemic">
                    <h2 style="margin-top: 0;">{severity_emoji} ANEMIC SIGNS DETECTED</h2>
                    <p style="font-size: 1.2em; margin: 20px 0;"><strong>Severity Level:</strong></p>
                    <span class="severity-badge {severity_class}">{severity}</span>
                    <p>The AI screening detected visual indicators consistent with anemia based on analysis of eye, nail, and palm images.</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="result-box result-healthy">
                    <h2 style="margin-top: 0;">✅ HEALTHY SIGNS DETECTED</h2>
                    <p>The AI screening suggests healthy hemoglobin levels based on visual indicators from the analyzed images.</p>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("---")
            
            # IMPORTANT DISCLAIMER
            st.markdown("## ⚕️ Medical Disclaimer")
            st.markdown("""
            <div class="warning-box">
                <strong>⚠️ IMPORTANT:</strong>
                <ul>
                    <li>This is a <strong>screening tool only</strong> and NOT a medical diagnosis</li>
                    <li>Results are <strong>NOT a substitute</strong> for professional blood testing (CBC)</li>
                    <li>Always <strong>consult a healthcare professional</strong> for confirmation</li>
                    <li>In case of severe symptoms, <strong>seek immediate medical attention</strong></li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("---")
            
            # Next steps (condensed)
            st.markdown("### 📋 Recommended Actions")
            
            if combined_score > 0.5:
                st.info("🔴 **Schedule a blood test (CBC)** with your doctor for professional confirmation")
            else:
                st.info("🟢 **Continue regular check-ups** and maintain a balanced diet")
        
        except Exception as e:
            st.error(f"❌ Error during analysis: {str(e)}")
            st.info("💡 Try uploading different images or checking image quality")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #7f8c8d; font-size: 0.9em; padding: 20px 0;">
        <p><strong>AnemiaLens</strong> © 2026 | Non-Invasive Anemia Screening System</p>
        <p>Built with ❤️ for global healthcare accessibility</p>
        <p style="font-size: 0.85em;">Made with Streamlit & TensorFlow</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
