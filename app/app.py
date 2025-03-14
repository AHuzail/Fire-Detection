import streamlit as st
import cv2
import numpy as np
import io
import time
import os
import sys
from PIL import Image
from ultralytics import YOLO
import matplotlib.pyplot as plt
import tempfile

# Set up paths
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(ROOT_DIR, "models", "fire_best.pt")

# Add root directory to path
sys.path.insert(0, ROOT_DIR)

# Set page configuration
st.set_page_config(
    page_title="Fire Detection System",
    page_icon="🔥",
    layout="wide",
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #FF5733;
        text-align: center;
        margin-bottom: 10px;
        padding-top: 20px;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #4B4B4B;
        margin-bottom: 30px;
        text-align: center;
    }
    .stApp {
        background-color: #F8F9FA;
    }
    .info-box {
        background-color: #F0F8FF;
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .result-header {
        font-size: 1.2rem;
        font-weight: bold;
        margin-bottom: 15px;
        color: #346751;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        width: 100%;
    }
    .upload-section {
        padding: 20px;
        border: 2px dashed #ddd;
        border-radius: 10px;
        text-align: center;
    }
    .results-container {
        margin-top: 20px;
    }
    .metric-container {
        background-color: #f9f9f9;
        padding: 10px;
        border-radius: 5px;
        margin-bottom: 10px;
    }
    div[data-testid="stMetricValue"] {
        font-size: 1.2rem;
    }
    div[data-testid="stHorizontalBlock"] {
        gap: 20px;
    }
</style>
""", unsafe_allow_html=True)

# App Header
st.markdown("<h1 class='main-header'>🔥 Fire Detection System</h1>", unsafe_allow_html=True)
st.markdown("<p class='sub-header'>Upload an image to detect fire instances</p>", unsafe_allow_html=True)

# Load the YOLO model
@st.cache_resource
def load_model():
    return YOLO(MODEL_PATH)

model = load_model()

# Sidebar
with st.sidebar:
    st.header("About")
    st.markdown("""
    This application uses a YOLO model trained specifically for fire detection. 
    Upload an image to detect any fire instances.
    
    **Features:**
    - Fast and accurate fire detection
    - Visual representation of detection results
    - Confidence scores for each detection
    """)
    
    st.header("Instructions")
    st.markdown("""
    1. Upload an image using the file uploader
    2. Wait for the model to process the image
    3. View results with bounding boxes around detected fires
    """)
    
    st.divider()
    st.markdown("### Model Information")
    st.info("YOLO model for fire detection (fire_best.pt)")

# Create container for main content
main_container = st.container()

with main_container:
    # Main content with better organized columns
    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        upload_container = st.container()
        with upload_container:
            st.markdown("<div class='info-box'>", unsafe_allow_html=True)
            st.markdown("### Upload Image")
            
            # Create a centered upload section
            st.markdown("<div class='upload-section'>", unsafe_allow_html=True)
            uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])
            st.markdown("</div>", unsafe_allow_html=True)
            
            # Display the uploaded image if available
            if uploaded_file is not None:
                image = Image.open(uploaded_file)
                st.image(image, caption="Uploaded Image", use_column_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        results_container = st.container()
        with results_container:
            st.markdown("<div class='info-box'>", unsafe_allow_html=True)
            st.markdown("### Detection Results")
            
            if uploaded_file is not None:
                with st.spinner("Processing..."):
                    # Measure time
                    start_time = time.time()
                    
                    # Convert PIL Image to numpy array for processing
                    image_array = np.array(image)
                    
                    # Convert RGB to BGR (OpenCV format)
                    image_bgr = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
                    
                    # Make prediction
                    results = model.predict(image_bgr)
                    
                    # Calculate time taken
                    elapsed_time = time.time() - start_time
                    
                    # Get results
                    boxes = results[0].boxes
                    
                    # Convert BGR back to RGB for displaying
                    annotated_img = cv2.cvtColor(results[0].plot(), cv2.COLOR_BGR2RGB)
                    st.markdown("<div class='results-container'>", unsafe_allow_html=True)
                    st.image(annotated_img, caption="Detection Result", use_column_width=True)
                    
                    # Display metrics in a nicer layout
                    st.markdown("<p class='result-header'>Results:</p>", unsafe_allow_html=True)
                    
                    metrics_cols = st.columns(2)
                    with metrics_cols[0]:
                        st.markdown("<div class='metric-container'>", unsafe_allow_html=True)
                        st.metric("Detections", len(boxes))
                        st.markdown("</div>", unsafe_allow_html=True)
                    with metrics_cols[1]:
                        st.markdown("<div class='metric-container'>", unsafe_allow_html=True)
                        st.metric("Time Taken", f"{elapsed_time:.3f} seconds")
                        st.markdown("</div>", unsafe_allow_html=True)
                    
                    # Show detailed information
                    if len(boxes) > 0:
                        with st.expander("Detection Details", expanded=True):
                            for i, box in enumerate(boxes):
                                x, y, w, h = box.xywh[0].tolist()
                                conf = box.conf[0].item()
                                
                                st.markdown(f"#### Detection #{i+1}")
                                detail_cols = st.columns(4)
                                detail_cols[0].metric("X", f"{x:.1f}")
                                detail_cols[1].metric("Y", f"{y:.1f}")
                                detail_cols[2].metric("Width", f"{w:.1f}")
                                detail_cols[3].metric("Height", f"{h:.1f}")
                                
                                # Confidence bar
                                st.write(f"Confidence: {conf:.2%}")
                                st.progress(float(conf))
                                if i < len(boxes) - 1:
                                    st.divider()
                    else:
                        st.warning("No fire detected in the image!")
                    st.markdown("</div>", unsafe_allow_html=True)
            else:
                # Display placeholder when no image is uploaded
                st.markdown("<div style='text-align:center; padding:50px;'>", unsafe_allow_html=True)
                st.info("Please upload an image to get detection results.")
                st.markdown("</div>", unsafe_allow_html=True)
            
            st.markdown("</div>", unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center">
    <p>Developed with ❤️ using Streamlit and YOLO | 2023</p>
</div>
""", unsafe_allow_html=True)
