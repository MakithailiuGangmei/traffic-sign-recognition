import streamlit as st
from ultralytics import YOLO
from PIL import Image
import tempfile
import os
import glob
import subprocess

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Traffic Sign Recognition System",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# =========================
# CUSTOM CSS - CLEAN ACADEMIC STYLE
# =========================
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global Styles */
    * {
        font-family: 'Inter', sans-serif;
    }
    
    /* Main Background */
    .stApp {
        background: #f5f6f8;
        color: #212529;
    }
    
    /* Hide Streamlit Default Elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Hero Section - Simple and clean */
    .hero-section {
        text-align: center;
        padding: 2rem 0;
        margin-bottom: 2rem;
    }
    
    .hero-logo {
        max-width: 350px;
        width: 100%;
        margin: 0 auto 2rem;
        display: block;
    }
    
    .hero-title {
        font-size: 1.75rem;
        font-weight: 700;
        color: #1a202c;
        margin-bottom: 0.75rem;
        letter-spacing: -0.3px;
    }
    
    .hero-subtitle {
        font-size: 1rem;
        color: #4a5568;
        margin-bottom: 0.5rem;
        font-weight: 400;
    }
    
    .hero-group {
        font-size: 0.95rem;
        color: #2c5282;
        font-weight: 600;
        margin-top: 0.5rem;
    }
    
    /* Navigation */
    .nav-container {
        display: flex;
        gap: 1rem;
        justify-content: center;
        flex-wrap: wrap;
        margin: 2rem 0;
    }
    
    .nav-card {
        background: white;
        color: #2d3748;
        padding: 1.25rem 1.5rem;
        border-radius: 10px;
        text-align: center;
        cursor: pointer;
        transition: all 0.2s ease;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.08);
        border: 1px solid #e2e8f0;
        min-width: 160px;
        flex: 1;
        max-width: 220px;
    }
    
    .nav-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.12);
        border-color: #3182ce;
        background: #f7fafc;
    }
    
    .nav-card-icon {
        font-size: 1.75rem;
        margin-bottom: 0.5rem;
    }
    
    .nav-card-title {
        font-size: 0.95rem;
        font-weight: 600;
        color: #2d3748;
    }
    
    /* Content Cards - Small and focused */
    .content-card {
        background: white;
        color: #1a202c;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.08);
        margin-bottom: 1.25rem;
        border-left: 4px solid #3182ce;
    }
    
    .content-card-dark {
        background: #1e3a5f;
        color: #ffffff;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
        margin-bottom: 1.25rem;
        border: 1px solid #2c5282;
    }
    
    /* Section Headers */
    .section-header {
        font-size: 1.5rem;
        font-weight: 700;
        color: #1a202c;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #3182ce;
        display: inline-block;
    }
    
    /* Info Grid */
    .info-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 1rem;
        margin: 1rem 0;
    }
    
    .info-item {
        background: #f7fafc;
        padding: 1.25rem;
        border-radius: 8px;
        border-left: 3px solid #3182ce;
    }
    
    .info-label {
        font-size: 0.8rem;
        color: #4a5568;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 0.25rem;
        font-weight: 600;
    }
    
    .info-value {
        font-size: 1.1rem;
        font-weight: 600;
        color: #1a202c;
    }
    
    /* Detectable Signs */
    .signs-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
        gap: 0.75rem;
        margin: 1rem 0;
    }
    
    .sign-item {
        background: #f7fafc;
        padding: 0.875rem 1.25rem;
        border-radius: 8px;
        border-left: 3px solid #48bb78;
        font-weight: 500;
        font-size: 0.95rem;
    }
    
    .sign-item.red {
        border-left-color: #fc8181;
    }
    
    .sign-item.yellow {
        border-left-color: #f6e05e;
    }
    
    /* Detection Results Panel */
    .detection-panel {
        background: white;
        padding: 1.25rem;
        border-radius: 10px;
        margin-top: 1rem;
        border: 1px solid #e2e8f0;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.08);
    }
    
    .detection-item {
        background: #f7fafc;
        padding: 0.875rem;
        border-radius: 6px;
        margin-bottom: 0.5rem;
        border-left: 3px solid #48bb78;
    }
    
    .confidence-bar {
        background: #e2e8f0;
        height: 6px;
        border-radius: 3px;
        margin-top: 0.5rem;
        overflow: hidden;
    }
    
    .confidence-fill {
        background: linear-gradient(90deg, #48bb78 0%, #38a169 100%);
        height: 100%;
        border-radius: 3px;
        transition: width 0.3s ease;
    }
    
    /* Team Member Cards */
    .team-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
        gap: 1.5rem;
        margin: 1.5rem 0;
    }
    
    .team-card {
        background: white;
        color: #1a202c;
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.08);
        transition: all 0.2s ease;
        border-top: 4px solid #3182ce;
    }
    
    .team-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.12);
    }
    
    .team-avatar {
        width: 100px;
        height: 100px;
        border-radius: 50%;
        background: linear-gradient(135deg, #3182ce 0%, #2c5282 100%);
        margin: 0 auto 1rem;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 2.5rem;
        color: white;
        font-weight: 700;
    }
    
    .team-name {
        font-size: 1.1rem;
        font-weight: 700;
        color: #2d3748;
        margin-bottom: 0.25rem;
    }
    
    .team-role {
        font-size: 0.875rem;
        color: #718096;
        font-weight: 500;
    }
    
    /* Buttons */
    .stButton>button {
        background: #3182ce;
        color: white;
        border: none;
        padding: 0.625rem 1.5rem;
        border-radius: 8px;
        font-weight: 600;
        font-size: 0.95rem;
        transition: all 0.2s ease;
        box-shadow: 0 1px 3px rgba(49, 130, 206, 0.2);
    }
    
    .stButton>button:hover {
        background: #2c5282;
        box-shadow: 0 2px 6px rgba(49, 130, 206, 0.3);
        transform: translateY(-1px);
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.5rem;
        background: white;
        padding: 0.375rem;
        border-radius: 8px;
        border: 1px solid #e2e8f0;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        color: #4a5568;
        border-radius: 6px;
        padding: 0.625rem 1.25rem;
        font-weight: 600;
        font-size: 0.9rem;
    }
    
    .stTabs [aria-selected="true"] {
        background: #3182ce;
        color: white;
    }
    
    /* File Uploader */
    .stFileUploader {
        background: white;
        padding: 1.25rem;
        border-radius: 8px;
        border: 2px dashed #cbd5e0;
    }
    
    /* Image Container */
    .image-container {
        background: white;
        padding: 0.75rem;
        border-radius: 8px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.08);
        border: 1px solid #e2e8f0;
    }
    
    /* Spinner */
    .stSpinner {
        text-align: center;
    }
    
    /* Success/Warning/Info Messages */
    .stSuccess, .stWarning, .stInfo {
        background: #f7fafc;
        color: #1a202c;
        border-radius: 8px;
        padding: 0.875rem;
        border-left: 3px solid #48bb78;
    }
    
    .stWarning {
        border-left-color: #f6ad55;
    }
    
    .stInfo {
        border-left-color: #63b3ed;
    }
    
    /* Responsive Design */
    @media (max-width: 768px) {
        .hero-title {
            font-size: 1.5rem;
        }
        
        .hero-subtitle {
            font-size: 0.9rem;
        }
        
        .nav-container {
            flex-direction: column;
            align-items: center;
        }
        
        .nav-card {
            max-width: 100%;
            width: 100%;
        }
        
        .team-grid {
            grid-template-columns: 1fr;
        }
    }
</style>
""", unsafe_allow_html=True)

# =========================
# LOAD MODEL
# =========================
@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()

# =========================
# PAGE STATE
# =========================
if "page" not in st.session_state:
    st.session_state.page = "home"

# =========================
# VIDEO CONVERTER
# =========================
def convert_to_mp4(input_path):
    output_path = input_path.replace(".avi", ".mp4")
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            input_path,
            output_path
        ],
        check=True
    )
    return output_path

# =========================
# HOME PAGE
# =========================
if st.session_state.page == "home":
    
    # Hero Section - Simple and clean
    st.markdown('<div class="hero-section">', unsafe_allow_html=True)
    
    # Try to load logo with error handling
    try:
        st.image("logo.png", use_container_width=False, width=350)
    except Exception as e:
        st.warning("⚠️ Logo not found. Please ensure logo.png is in the project folder.")
    
    st.markdown('<h1 class="hero-title">🚦 Traffic Sign Recognition System Using YOLOv8l</h1>', unsafe_allow_html=True)
    st.markdown('<p class="hero-subtitle">An AI-powered system for traffic sign detection and recognition.</p>', unsafe_allow_html=True)
    st.markdown('<p class="hero-group">Group 2 - Butter Cups</p>', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Navigation Cards
    st.markdown('<div class="nav-container">', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="nav-card" onclick="document.querySelector('[data-testid=\'stButton\']').click()">
            <div class="nav-card-icon">📋</div>
            <div class="nav-card-title">Instructions</div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Instructions", key="nav_instructions", use_container_width=True):
            st.session_state.page = "instructions"
            st.rerun()
    
    with col2:
        st.markdown("""
        <div class="nav-card" onclick="document.querySelectorAll('[data-testid=\'stButton\']')[1].click()">
            <div class="nav-card-icon">📖</div>
            <div class="nav-card-title">About Project</div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("About Project", key="nav_about", use_container_width=True):
            st.session_state.page = "about"
            st.rerun()
    
    with col3:
        st.markdown("""
        <div class="nav-card" onclick="document.querySelectorAll('[data-testid=\'stButton\']')[2].click()">
            <div class="nav-card-icon">🎯</div>
            <div class="nav-card-title">Detection</div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Detection", key="nav_detect", use_container_width=True):
            st.session_state.page = "detect"
            st.rerun()
    
    with col4:
        st.markdown("""
        <div class="nav-card" onclick="document.querySelectorAll('[data-testid=\'stButton\']')[3].click()">
            <div class="nav-card-icon">👨‍💻</div>
            <div class="nav-card-title">Team Members</div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Team Members", key="nav_team", use_container_width=True):
            st.session_state.page = "team"
            st.rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)

# =========================
# INSTRUCTIONS PAGE
# =========================
elif st.session_state.page == "instructions":
    
    st.markdown('<h2 class="section-header">📋 How to Use</h2>', unsafe_allow_html=True)
    st.write("")
    
    # Camera Detection
    st.markdown('<h3 style="color: #1a202c; margin-bottom: 0.75rem; font-size: 1.1rem; font-weight: 600;">📷 Camera Detection</h3>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="content-card">
        <p style="color: #4a5568; line-height: 1.8; margin-bottom: 0.75rem;">
            <strong>1.</strong> Open Detection Page.<br>
            <strong>2.</strong> Select Camera Tab.<br>
            <strong>3.</strong> Capture Image.<br>
            <strong>4.</strong> Wait for Detection Result.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.write("")
    
    # Image Upload
    st.markdown('<h3 style="color: #1a202c; margin-bottom: 0.75rem; font-size: 1.1rem; font-weight: 600;">🖼 Image Upload</h3>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="content-card">
        <p style="color: #4a5568; line-height: 1.8; margin-bottom: 0.75rem;">
            <strong>1.</strong> Open Detection Page.<br>
            <strong>2.</strong> Select Image Upload Tab.<br>
            <strong>3.</strong> Upload Image.<br>
            <strong>4.</strong> View Detection Result and Confidence Scores.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.write("")
    
    # Video Upload
    st.markdown('<h3 style="color: #1a202c; margin-bottom: 0.75rem; font-size: 1.1rem; font-weight: 600;">🎥 Video Upload</h3>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="content-card">
        <p style="color: #4a5568; line-height: 1.8; margin-bottom: 0.75rem;">
            <strong>1.</strong> Open Detection Page.<br>
            <strong>2.</strong> Select Video Upload Tab.<br>
            <strong>3.</strong> Upload Video.<br>
            <strong>4.</strong> Wait for Processing.<br>
            <strong>5.</strong> View and Download Result Video.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.write("")
    st.write("")
    
    # Supported Formats
    st.markdown('<h3 style="color: #1a202c; margin-bottom: 0.75rem; font-size: 1.1rem; font-weight: 600;">📁 Supported File Formats</h3>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="content-card" style="text-align: center;">
            <h4 style="color: #2d3748; margin-bottom: 1rem; font-size: 1rem;">Image Formats</h4>
            <div style="display: flex; justify-content: center; gap: 0.75rem; flex-wrap: wrap;">
                <div style="background: #edf2f7; color: #2d3748; padding: 0.75rem 1rem; border-radius: 6px; font-weight: 600; font-size: 0.9rem;">JPG</div>
                <div style="background: #edf2f7; color: #2d3748; padding: 0.75rem 1rem; border-radius: 6px; font-weight: 600; font-size: 0.9rem;">JPEG</div>
                <div style="background: #edf2f7; color: #2d3748; padding: 0.75rem 1rem; border-radius: 6px; font-weight: 600; font-size: 0.9rem;">PNG</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="content-card" style="text-align: center;">
            <h4 style="color: #2d3748; margin-bottom: 1rem; font-size: 1rem;">Video Formats</h4>
            <div style="display: flex; justify-content: center; gap: 0.75rem; flex-wrap: wrap;">
                <div style="background: #edf2f7; color: #2d3748; padding: 0.75rem 1rem; border-radius: 6px; font-weight: 600; font-size: 0.9rem;">MP4</div>
                <div style="background: #edf2f7; color: #2d3748; padding: 0.75rem 1rem; border-radius: 6px; font-weight: 600; font-size: 0.9rem;">AVI</div>
                <div style="background: #edf2f7; color: #2d3748; padding: 0.75rem 1rem; border-radius: 6px; font-weight: 600; font-size: 0.9rem;">MOV</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    st.write("")
    
    # Return Home Button
    if st.button("🏠 Return Home", use_container_width=True):
        st.session_state.page = "home"
        st.rerun()

# =========================
# ABOUT PAGE
# =========================
elif st.session_state.page == "about":
    
    st.markdown('<h2 class="section-header">📖 About Project</h2>', unsafe_allow_html=True)
    st.write("")
    
    # Project Overview
    st.markdown("""
    <div class="content-card">
        <h3 style="color: #2d3748; margin-bottom: 0.75rem; font-size: 1.1rem;">Project Overview</h3>
        <p style="color: #4a5568; line-height: 1.7; font-size: 0.95rem;">
            Traffic Sign Recognition is an important component of intelligent transportation systems 
            and autonomous driving. This project uses the <strong>YOLOv8l</strong> object detection model 
            to identify traffic signs and traffic lights from images and videos with high accuracy 
            and real-time performance.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.write("")
    
    # Detectable Signs Section
    st.markdown('<h3 style="color: #1a202c; margin-bottom: 0.75rem; font-size: 1.1rem; font-weight: 600;">🚦 Detectable Signs</h3>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="content-card" style="text-align: center;">
            <div style="font-size: 2rem; margin-bottom: 0.75rem;">🚦</div>
            <h4 style="color: #2d3748; margin-bottom: 0.75rem; font-size: 1rem;">Traffic Lights</h4>
            <div style="text-align: left; color: #4a5568; line-height: 1.8; font-size: 0.9rem;">
                • Green Light<br>
                • Red Light
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="content-card" style="text-align: center;">
            <div style="font-size: 2rem; margin-bottom: 0.75rem;">🛑</div>
            <h4 style="color: #2d3748; margin-bottom: 0.75rem; font-size: 1rem;">Traffic Signs</h4>
            <div style="text-align: left; color: #4a5568; line-height: 1.8; font-size: 0.9rem;">
                • Stop Sign
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="content-card" style="text-align: center;">
            <div style="font-size: 2rem; margin-bottom: 0.75rem;">🚗</div>
            <h4 style="color: #2d3748; margin-bottom: 0.75rem; font-size: 1rem;">Speed Limit Signs</h4>
            <div style="text-align: left; color: #4a5568; line-height: 1.8; font-size: 0.9rem;">
                • Speed limits ranging from<br>
                &nbsp;&nbsp;20 km/h to 120 km/h
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    st.write("")
    
    # Model Information
    st.markdown('<h3 style="color: #1a202c; margin-bottom: 0.75rem; font-size: 1.1rem; font-weight: 600;">⚙️ Model Information</h3>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="info-item">
            <div class="info-label">Model</div>
            <div class="info-value">YOLOv8l</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="info-item">
            <div class="info-label">Training Epochs</div>
            <div class="info-value">60 Epochs</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="info-item">
            <div class="info-label">Framework</div>
            <div class="info-value">Ultralytics</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.write("")
    st.write("")
    
    # Before & After Image Comparison
    st.markdown('<h3 style="color: #1a202c; margin-bottom: 0.75rem; font-size: 1.1rem; font-weight: 600;">📸 Example Detection Results</h3>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="content-card" style="text-align: center;">
            <h4 style="color: #2d3748; margin-bottom: 0.75rem; font-size: 1rem;">Original Image</h4>
        """, unsafe_allow_html=True)
        st.image("sample_input.jpg", caption="Original Sample Image", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="content-card" style="text-align: center;">
            <h4 style="color: #2d3748; margin-bottom: 0.75rem; font-size: 1rem;">Detection Result</h4>
        """, unsafe_allow_html=True)
        st.image("sample_detection.jpg", caption="Sample Detection Output", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    st.write("")
    st.write("")
    
    # Sample Video Detection
    st.markdown('<h3 style="color: #1a202c; margin-bottom: 0.75rem; font-size: 1.1rem; font-weight: 600;">🎥 Sample Video Detection</h3>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="content-card" style="text-align: center;">
        <p style="color: #4a5568; margin-bottom: 0.75rem; font-size: 0.95rem;">Example Video Detection Result</p>
    """, unsafe_allow_html=True)
    
    st.video("sample.mp4")
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    st.write("")
    
    # Return Home Button
    if st.button("🏠 Return Home", use_container_width=True):
        st.session_state.page = "home"
        st.rerun()

# =========================
# DETECTION PAGE
# =========================
elif st.session_state.page == "detect":
    
    st.markdown('<h2 class="section-header">🎯 Traffic Sign Detection</h2>', unsafe_allow_html=True)
    st.write("")
    
    tab1, tab2, tab3 = st.tabs([
        "📷 Camera Detection",
        "🖼 Image Upload",
        "🎥 Video Upload"
    ])
    
    # CAMERA TAB
    with tab1:
        st.markdown("""
        <div class="content-card-dark">
            <h3 style="color: #90cdf4; margin-bottom: 0.75rem; font-size: 1.1rem;">📷 Live Camera Detection</h3>
            <p style="color: #cbd5e0; font-size: 0.95rem;">Capture an image using your camera to detect traffic signs in real-time.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.write("")
        
        camera_image = st.camera_input("Take a picture")
        
        if camera_image is not None:
            image = Image.open(camera_image)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown('<div class="image-container">', unsafe_allow_html=True)
                st.image(image, caption="Original Image", use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
            
            with st.spinner("Detecting traffic signs..."):
                results = model(image)
                result = results[0]
                detected_img = result.plot()
            
            with col2:
                st.markdown('<div class="image-container">', unsafe_allow_html=True)
                st.image(detected_img, caption="Detection Result", use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Detection Results Panel
            if len(result.boxes) > 0:
                st.markdown('<div class="detection-panel">', unsafe_allow_html=True)
                st.markdown('<h4 style="color: #2d3748; margin-bottom: 0.75rem; font-size: 1rem;">✅ Detected Objects</h4>', unsafe_allow_html=True)
                
                for box in result.boxes:
                    class_id = int(box.cls[0])
                    confidence = float(box.conf[0])
                    class_name = model.names[class_id]
                    
                    st.markdown(f"""
                    <div class="detection-item">
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <span style="font-weight: 600; color: #2d3748;">{class_name}</span>
                            <span style="color: #3182ce; font-weight: 600;">Confidence: {confidence:.1%}</span>
                        </div>
                        <div class="confidence-bar">
                            <div class="confidence-fill" style="width: {confidence * 100}%;"></div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                
                st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.warning("⚠️ No traffic signs detected in the image.")
    
    # IMAGE TAB
    with tab2:
        st.markdown("""
        <div class="content-card-dark">
            <h3 style="color: #90cdf4; margin-bottom: 0.75rem; font-size: 1.1rem;">🖼 Image Upload Detection</h3>
            <p style="color: #cbd5e0; font-size: 0.95rem;">Upload an image file to detect traffic signs. Supported formats: JPG, JPEG, PNG.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.write("")
        
        uploaded_image = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])
        
        if uploaded_image is not None:
            image = Image.open(uploaded_image)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown('<div class="image-container">', unsafe_allow_html=True)
                st.image(image, caption="Original Image", use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
            
            with st.spinner("Detecting traffic signs..."):
                results = model(image)
                result = results[0]
                detected_img = result.plot()
            
            with col2:
                st.markdown('<div class="image-container">', unsafe_allow_html=True)
                st.image(detected_img, caption="Detection Result", use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Detection Results Panel
            if len(result.boxes) > 0:
                st.markdown('<div class="detection-panel">', unsafe_allow_html=True)
                st.markdown('<h4 style="color: #2d3748; margin-bottom: 0.75rem; font-size: 1rem;">✅ Detected Objects</h4>', unsafe_allow_html=True)
                
                for box in result.boxes:
                    class_id = int(box.cls[0])
                    confidence = float(box.conf[0])
                    class_name = model.names[class_id]
                    
                    st.markdown(f"""
                    <div class="detection-item">
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <span style="font-weight: 600; color: #2d3748;">{class_name}</span>
                            <span style="color: #3182ce; font-weight: 600;">Confidence: {confidence:.1%}</span>
                        </div>
                        <div class="confidence-bar">
                            <div class="confidence-fill" style="width: {confidence * 100}%;"></div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                
                st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.warning("⚠️ No traffic signs detected in the image.")
    
    # VIDEO TAB
    with tab3:
        st.markdown("""
        <div class="content-card-dark">
            <h3 style="color: #90cdf4; margin-bottom: 0.75rem; font-size: 1.1rem;">🎥 Video Upload Detection</h3>
            <p style="color: #cbd5e0; font-size: 0.95rem;">Upload a video file to detect traffic signs throughout the video. Supported formats: MP4, AVI, MOV.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.write("")
        
        uploaded_video = st.file_uploader("Upload Video", type=["mp4", "avi", "mov"])
        
        if uploaded_video is not None:
            st.video(uploaded_video)
            
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            temp_file.write(uploaded_video.read())
            temp_file.close()
            
            with st.spinner("Processing video... This may take a moment."):
                model.predict(
                    source=temp_file.name,
                    save=True,
                    conf=0.25
                )
            
            runs = sorted(
                glob.glob("runs/detect/predict*"),
                key=os.path.getmtime
            )
            
            if runs:
                latest_run = runs[-1]
                output_files = (
                    glob.glob(os.path.join(latest_run, "*.mp4"))
                    +
                    glob.glob(os.path.join(latest_run, "*.avi"))
                )
                
                if output_files:
                    output_video = output_files[0]
                    
                    if output_video.endswith(".avi"):
                        output_video = convert_to_mp4(output_video)
                    
                    st.markdown('<div class="image-container">', unsafe_allow_html=True)
                    st.video(output_video)
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    with open(output_video, "rb") as file:
                        st.download_button(
                            "📥 Download Processed Video",
                            file,
                            file_name=os.path.basename(output_video),
                            mime="video/mp4",
                            use_container_width=True
                        )
    
    st.write("")
    st.write("")
    
    # Return Home Button
    if st.button("🏠 Return Home", use_container_width=True):
        st.session_state.page = "home"
        st.rerun()

# =========================
# TEAM PAGE
# =========================
elif st.session_state.page == "team":
    
    st.markdown('<h2 class="section-header">👨‍💻 Team Members</h2>', unsafe_allow_html=True)
    st.write("")
    
    # Group Name
    st.markdown("""
    <div class="content-card-dark" style="text-align: center;">
        <h3 style="color: #fbd38d; margin-bottom: 0.5rem; font-size: 1.1rem;">Group 2 - Butter Cups</h3>
    </div>
    """, unsafe_allow_html=True)
    
    st.write("")
    st.write("")
    
    # Team Member Cards
    st.markdown("""
    <div class="team-grid">
        <div class="team-card">
            <div class="team-avatar">L</div>
            <div class="team-name">LIMAJUNGLA I AO</div>
            <div class="team-role">Team Member</div>
        </div>
        <div class="team-card">
            <div class="team-avatar">P</div>
            <div class="team-name">P ANGSHAN NANGNYU</div>
            <div class="team-role">Team Member</div>
        </div>
        <div class="team-card">
            <div class="team-avatar">M</div>
            <div class="team-name">MAKITHAILIU GANGMEI</div>
            <div class="team-role">Team Member</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.write("")
    st.write("")
    
    # Project Info Card
    st.markdown("""
    <div class="content-card">
        <h3 style="color: #2d3748; margin-bottom: 0.75rem; font-size: 1.1rem;">🎓 Project Details</h3>
        <p style="color: #4a5568; line-height: 1.7; font-size: 0.95rem;">
            <strong>Project Title:</strong> Traffic Sign Recognition System Using YOLOv8l<br>
            <strong>Group:</strong> Group 2 - Butter Cups<br>
            <strong>Technology:</strong> YOLOv8l Deep Learning Model<br>
            <strong>Framework:</strong> Ultralytics YOLO<br>
            <strong>Application:</strong> Streamlit Web Application
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.write("")
    
    # Return Home Button
    if st.button("🏠 Return Home", use_container_width=True):
        st.session_state.page = "home"
        st.rerun()