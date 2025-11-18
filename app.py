import streamlit as st
import cv2
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

# Page configuration
st.set_page_config(
    page_title="COVID-19 X-Ray Diagnosis",
    page_icon="🏥",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        padding: 2rem;
        border-radius: 10px;
        text-align: center;
        margin-top: 2rem;
        font-size: 1.5rem;
        font-weight: bold;
    }
    .covid-result {
        background-color: #ffebee;
        color: #c62828;
        border: 2px solid #c62828;
    }
    .normal-result {
        background-color: #e8f5e9;
        color: #2e7d32;
        border: 2px solid #2e7d32;
    }
    .pneumonia-result {
        background-color: #fff3e0;
        color: #ef6c00;
        border: 2px solid #ef6c00;
    }
    .info-box {
        background-color: #e3f2fd;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Load model
@st.cache_resource
def load_covid_model():
    return load_model(r"D:\Work\TECH\Route\C9\Projects\Covid_APP\covid_19_model.h5")

model = load_covid_model()

class_map = {
    0: 'COVID-19 Positive', 
    1: 'Normal', 
    2: 'Viral Pneumonia'
}

class_info = {
    'COVID-19 Positive': {
        'emoji': '🦠',
        'color': 'covid-result',
        'description': 'The X-ray shows patterns consistent with COVID-19 infection.'
    },
    'Normal': {
        'emoji': '✅',
        'color': 'normal-result',
        'description': 'The X-ray appears normal with no signs of infection.'
    },
    'Viral Pneumonia': {
        'emoji': '⚠️',
        'color': 'pneumonia-result',
        'description': 'The X-ray shows patterns consistent with viral pneumonia.'
    }
}

x_resize = 224
y_resize = 224
dims = 3

# Main Functions
def preprocess_image(image, x_resize, y_resize, dims):
    """Preprocess the uploaded image for model prediction"""
    img_array = np.array(image)
    img_array = cv2.resize(img_array, (x_resize, y_resize))
    
    if img_array.ndim == 2:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
        
    img_array = img_array.astype('float32') / 255.0
    new_img = img_array.reshape(1, x_resize, y_resize, dims)
    
    return new_img

def predict_image(image):
    """Predict the class of the X-ray image"""
    preds = model.predict(image, verbose=0)
    pred_label = np.argmax(preds, axis=1)
    pred_class = class_map[pred_label[0]]
    confidence = float(preds[0][pred_label[0]] * 100)
    
    return pred_class, confidence, preds[0]

# GUI
st.markdown('<p class="main-header">🏥 COVID-19 X-Ray Diagnosis System</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">AI-Powered Chest X-Ray Analysis for COVID-19 Detection</p>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("ℹ️ About")
    st.info(
        "This application uses deep learning to analyze chest X-ray images "
        "and classify them into three categories:\n"
        "- COVID-19 Positive\n"
        "- Normal\n"
        "- Viral Pneumonia"
    )
    
    st.header("⚕️ Disclaimer")
    st.warning(
        "This tool is for educational and research purposes only. "
        "It should NOT be used as a substitute for professional medical diagnosis. "
        "Always consult with qualified healthcare professionals."
    )
    
    st.header("📋 Instructions")
    st.markdown(
        "1. Upload a chest X-ray image\n"
        "2. Click the 'Analyze X-Ray' button\n"
        "3. Review the AI prediction results"
    )

# Main content
col1, col2, col3 = st.columns([1, 3, 1])

with col2:
    uploaded_file = st.file_uploader(
        "Upload Chest X-Ray Image",
        type=["jpg", "png", "jpeg"],
        help="Supported formats: JPG, PNG, JPEG"
    )

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    
    # Display uploaded image
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image(img, caption='Uploaded X-Ray Image', use_container_width=True)
    
    # Predict button
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        predict_button = st.button('🔬 Analyze X-Ray', use_container_width=True, type="primary")
    
    if predict_button:
        with st.spinner('Analyzing X-ray image...'):
            processed_image = preprocess_image(img, x_resize, y_resize, dims)
            prediction, confidence, all_preds = predict_image(processed_image)
            
            # Display prediction
            pred_info = class_info[prediction]
            st.markdown(
                f'<div class="prediction-box {pred_info["color"]}">'
                f'{pred_info["emoji"]} Prediction: {prediction}<br>'
                f'Confidence: {confidence:.2f}%'
                f'</div>',
                unsafe_allow_html=True
            )
            
            st.markdown(f"**Analysis:** {pred_info['description']}")
            
            # Show detailed probabilities
            st.subheader("📊 Detailed Analysis")
            
            for idx, (key, value) in enumerate(class_map.items()):
                prob = float(all_preds[idx] * 100)
                st.progress(prob / 100, text=f"{value}: {prob:.2f}%")
            
            st.markdown('<div class="info-box">⚠️ <strong>Important:</strong> This analysis is generated by an AI model and should be reviewed by a qualified medical professional.</div>', unsafe_allow_html=True)
else:
    # Welcome message when no image is uploaded
    st.markdown('<div class="info-box">👆 Please upload a chest X-ray image to begin the analysis.</div>', unsafe_allow_html=True)