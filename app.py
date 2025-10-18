import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
import io
import os
import time
import numpy as np

# =========================
# ⚙️ PAGE CONFIG
# =========================
st.set_page_config(
    page_title="PixelRNN Image Completion",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================
# 🎨 MODERN DARK THEME
# =========================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&display=swap');

:root {
    --primary-bg: radial-gradient(ellipse at top, #0f0f23 0%, #000000 100%);
    --secondary-bg: linear-gradient(145deg, #1a1a2e, #16213e);
    --card-bg: linear-gradient(145deg, #1e1e2e, #2a2a3e);
    --accent-color: #00f5ff;
    --accent-gradient: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
    --neon-gradient: linear-gradient(45deg, #00f5ff, #ff00f5, #f5ff00, #00f5ff);
    --text-primary: #ffffff;
    --text-secondary: #b4b4c7;
    --border-color: rgba(255, 255, 255, 0.1);
    --hover-bg: rgba(255, 255, 255, 0.05);
    --glow-color: rgba(0, 245, 255, 0.3);
}

.stApp {
    background: var(--primary-bg);
    color: var(--text-primary);
    font-family: 'Space Grotesk', sans-serif;
    position: relative;
    overflow-x: hidden;
}

.stApp::before {
    content: '';
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    background: 
        radial-gradient(circle at 20% 80%, rgba(120, 119, 198, 0.3) 0%, transparent 50%),
        radial-gradient(circle at 80% 20%, rgba(255, 119, 198, 0.3) 0%, transparent 50%),
        radial-gradient(circle at 40% 40%, rgba(120, 219, 255, 0.2) 0%, transparent 50%);
    pointer-events: none;
    z-index: -1;
    animation: backgroundShift 20s ease-in-out infinite;
}

@keyframes backgroundShift {
    0%, 100% { 
        background: 
            radial-gradient(circle at 20% 80%, rgba(120, 119, 198, 0.3) 0%, transparent 50%),
            radial-gradient(circle at 80% 20%, rgba(255, 119, 198, 0.3) 0%, transparent 50%),
            radial-gradient(circle at 40% 40%, rgba(120, 219, 255, 0.2) 0%, transparent 50%);
    }
    50% { 
        background: 
            radial-gradient(circle at 80% 20%, rgba(120, 119, 198, 0.3) 0%, transparent 50%),
            radial-gradient(circle at 20% 80%, rgba(255, 119, 198, 0.3) 0%, transparent 50%),
            radial-gradient(circle at 60% 60%, rgba(120, 219, 255, 0.2) 0%, transparent 50%);
    }
}

/* Hide Streamlit branding */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

/* Main container */
.main .block-container {
    padding-top: 2rem;
    max-width: 1200px;
}

/* Header styling */
.main-header {
    text-align: center;
    padding: 4rem 2rem;
    background: var(--card-bg);
    border-radius: 32px;
    margin-bottom: 3rem;
    border: 1px solid var(--border-color);
    position: relative;
    overflow: hidden;
    backdrop-filter: blur(20px);
    box-shadow: 
        0 8px 32px rgba(0, 0, 0, 0.3),
        inset 0 1px 0 rgba(255, 255, 255, 0.1);
}

.main-header::before {
    content: '';
    position: absolute;
    top: 0;
    left: -100%;
    width: 300%;
    height: 4px;
    background: var(--neon-gradient);
    background-size: 200% 100%;
    animation: neonFlow 3s linear infinite;
}

.main-header::after {
    content: '';
    position: absolute;
    top: 50%;
    left: 50%;
    width: 200%;
    height: 200%;
    background: radial-gradient(circle, var(--glow-color) 0%, transparent 70%);
    transform: translate(-50%, -50%);
    opacity: 0.1;
    animation: pulse 4s ease-in-out infinite;
}

.main-header h1 {
    color: var(--text-primary);
    font-size: 4rem;
    font-weight: 700;
    margin-bottom: 1rem;
    background: var(--neon-gradient);
    background-size: 200% 200%;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    animation: gradientShift 3s ease-in-out infinite;
    position: relative;
    z-index: 2;
    text-shadow: 0 0 30px var(--glow-color);
}

.main-header p {
    color: var(--text-secondary);
    font-size: 1.3rem;
    font-weight: 400;
    margin: 0 auto;
    max-width: 700px;
    position: relative;
    z-index: 2;
    line-height: 1.6;
}

@keyframes neonFlow {
    0% { transform: translateX(0); }
    100% { transform: translateX(33.333%); }
}

@keyframes gradientShift {
    0%, 100% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
}

@keyframes pulse {
    0%, 100% { opacity: 0.1; transform: translate(-50%, -50%) scale(1); }
    50% { opacity: 0.2; transform: translate(-50%, -50%) scale(1.1); }
}

/* Upload section */
.upload-section {
    background: var(--card-bg);
    border-radius: 24px;
    padding: 2.5rem;
    margin: 2rem 0;
    border: 1px solid var(--border-color);
    transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
    position: relative;
    overflow: hidden;
    backdrop-filter: blur(20px);
}

.upload-section::before {
    content: '';
    position: absolute;
    top: 0;
    left: -100%;
    width: 100%;
    height: 100%;
    background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.1), transparent);
    transition: left 0.5s;
}

.upload-section:hover {
    border-color: var(--accent-color);
    box-shadow: 
        0 20px 40px rgba(0, 245, 255, 0.15),
        0 0 0 1px rgba(0, 245, 255, 0.2);
    transform: translateY(-2px);
}

.upload-section:hover::before {
    left: 100%;
}

.upload-section h3 {
    color: var(--text-primary);
    font-weight: 600;
    margin-bottom: 0.5rem;
}

.upload-section p {
    color: var(--text-secondary);
    margin-bottom: 1.5rem;
}

/* File uploader styling */
div[data-testid="stFileUploader"] {
    background: var(--secondary-bg);
    border: 2px dashed var(--border-color);
    border-radius: 20px;
    padding: 4rem 2rem;
    text-align: center;
    transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
    position: relative;
    overflow: hidden;
}

div[data-testid="stFileUploader"]::before {
    content: '';
    position: absolute;
    top: 50%;
    left: 50%;
    width: 0;
    height: 0;
    background: radial-gradient(circle, var(--glow-color) 0%, transparent 70%);
    transform: translate(-50%, -50%);
    transition: all 0.4s ease;
    border-radius: 50%;
}

div[data-testid="stFileUploader"]:hover {
    border-color: var(--accent-color);
    background: rgba(0, 245, 255, 0.05);
    box-shadow: 
        0 0 30px rgba(0, 245, 255, 0.2),
        inset 0 0 30px rgba(0, 245, 255, 0.1);
    transform: scale(1.02);
}

div[data-testid="stFileUploader"]:hover::before {
    width: 200px;
    height: 200px;
}

div[data-testid="stFileUploader"] label {
    color: var(--text-primary) !important;
    font-weight: 500;
    font-size: 1.1rem;
}

div[data-testid="stFileUploader"] small {
    color: var(--text-secondary) !important;
}

/* Button styling */
.stButton > button {
    background: var(--neon-gradient);
    background-size: 200% 200%;
    color: white;
    border: none;
    border-radius: 16px;
    padding: 1rem 2.5rem;
    font-weight: 600;
    font-size: 1.1rem;
    transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
    box-shadow: 
        0 8px 25px rgba(0, 245, 255, 0.3),
        0 0 0 1px rgba(255, 255, 255, 0.1);
    font-family: 'Space Grotesk', sans-serif;
    position: relative;
    overflow: hidden;
    animation: gradientShift 3s ease-in-out infinite;
}

.stButton > button::before {
    content: '';
    position: absolute;
    top: 0;
    left: -100%;
    width: 100%;
    height: 100%;
    background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.2), transparent);
    transition: left 0.5s;
}

.stButton > button:hover {
    transform: translateY(-3px) scale(1.05);
    box-shadow: 
        0 15px 35px rgba(0, 245, 255, 0.4),
        0 0 0 1px rgba(255, 255, 255, 0.2),
        0 0 20px rgba(0, 245, 255, 0.6);
}

.stButton > button:hover::before {
    left: 100%;
}

.stButton > button:active {
    transform: translateY(-1px) scale(1.02);
}

/* Results section */
.result-section {
    background: var(--card-bg);
    border-radius: 20px;
    padding: 2rem;
    margin: 2rem 0;
    border: 1px solid var(--border-color);
}

.result-section h3 {
    color: var(--text-primary);
    font-weight: 600;
    margin-bottom: 1.5rem;
}

/* Image containers */
.image-container {
    background: var(--secondary-bg);
    border-radius: 20px;
    padding: 2rem;
    margin: 1rem 0;
    text-align: center;
    border: 1px solid var(--border-color);
    transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
    position: relative;
    overflow: hidden;
    backdrop-filter: blur(20px);
}

.image-container::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background: linear-gradient(45deg, transparent, rgba(255, 255, 255, 0.05), transparent);
    transform: translateX(-100%);
    transition: transform 0.6s;
}

.image-container:hover {
    border-color: var(--accent-color);
    box-shadow: 
        0 20px 40px rgba(0, 245, 255, 0.15),
        0 0 0 1px rgba(0, 245, 255, 0.3);
    transform: translateY(-5px);
}

.image-container:hover::before {
    transform: translateX(100%);
}

.image-container h3 {
    color: var(--text-primary);
    margin-bottom: 1.5rem;
    font-weight: 600;
    font-size: 1.3rem;
    position: relative;
    z-index: 2;
}

/* Stats container */
.stats-container {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
    gap: 1rem;
    margin: 2rem 0;
}

.stat-card {
    background: var(--secondary-bg);
    border-radius: 12px;
    padding: 1.5rem;
    text-align: center;
    border: 1px solid var(--border-color);
    transition: all 0.3s ease;
}

.stat-card:hover {
    border-color: var(--accent-color);
    transform: translateY(-2px);
}

.stat-number {
    font-size: 2rem;
    font-weight: 700;
    color: var(--accent-color);
    margin-bottom: 0.5rem;
}

.stat-label {
    font-size: 0.9rem;
    color: var(--text-secondary);
    font-weight: 500;
}

/* Sidebar styling */
.css-1d391kg {
    background: var(--card-bg);
}

.sidebar .stSelectbox label {
    color: var(--text-primary);
    font-weight: 500;
}

/* Info cards */
.info-card {
    background: var(--secondary-bg);
    border-radius: 12px;
    padding: 1.5rem;
    margin: 1rem 0;
    border: 1px solid var(--border-color);
    transition: all 0.3s ease;
}

.info-card:hover {
    border-color: var(--accent-color);
    box-shadow: 0 4px 20px rgba(0, 212, 255, 0.1);
}

.info-card h4 {
    color: var(--text-primary);
    margin-bottom: 0.75rem;
    font-weight: 600;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

.info-card p {
    color: var(--text-secondary);
    margin: 0;
    line-height: 1.6;
}

/* Progress bar */
.stProgress > div > div > div > div {
    background: var(--accent-gradient);
}

/* Spinner */
.stSpinner > div {
    border-top-color: var(--accent-color) !important;
}

/* Footer */
.footer {
    text-align: center;
    padding: 3rem 2rem;
    color: var(--text-secondary);
    font-size: 0.9rem;
    margin-top: 4rem;
    border-top: 1px solid var(--border-color);
}

.footer a {
    color: var(--accent-color);
    text-decoration: none;
    font-weight: 500;
    transition: all 0.3s ease;
}

.footer a:hover {
    color: var(--text-primary);
}

/* Enhanced Animations */
@keyframes slideUp {
    from { 
        opacity: 0; 
        transform: translateY(50px) scale(0.95); 
    }
    to { 
        opacity: 1; 
        transform: translateY(0) scale(1); 
    }
}

@keyframes glow {
    0%, 100% { 
        box-shadow: 
            0 0 20px rgba(0, 245, 255, 0.4),
            0 0 40px rgba(0, 245, 255, 0.2),
            0 0 60px rgba(0, 245, 255, 0.1);
    }
    50% { 
        box-shadow: 
            0 0 30px rgba(0, 245, 255, 0.6),
            0 0 60px rgba(0, 245, 255, 0.3),
            0 0 90px rgba(0, 245, 255, 0.2);
    }
}

@keyframes float {
    0%, 100% { transform: translateY(0px); }
    50% { transform: translateY(-10px); }
}

@keyframes shimmer {
    0% { background-position: -200% 0; }
    100% { background-position: 200% 0; }
}

.slide-up {
    animation: slideUp 0.8s cubic-bezier(0.4, 0, 0.2, 1);
}

.glow-effect {
    animation: glow 3s ease-in-out infinite;
    position: relative;
}

.glow-effect::after {
    content: '';
    position: absolute;
    top: -2px;
    left: -2px;
    right: -2px;
    bottom: -2px;
    background: var(--neon-gradient);
    background-size: 200% 200%;
    border-radius: 22px;
    z-index: -1;
    animation: gradientShift 3s ease-in-out infinite;
    opacity: 0.7;
}

.float-animation {
    animation: float 6s ease-in-out infinite;
}

/* Responsive design */
@media (max-width: 768px) {
    .main-header h1 {
        font-size: 2.5rem;
    }
    .main-header p {
        font-size: 1rem;
    }
    .stats-container {
        grid-template-columns: repeat(2, 1fr);
    }
}

/* Floating particles effect */
.particles {
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    pointer-events: none;
    z-index: -1;
}

.particle {
    position: absolute;
    width: 2px;
    height: 2px;
    background: var(--accent-color);
    border-radius: 50%;
    opacity: 0.6;
    animation: float-particle 20s linear infinite;
}

.particle:nth-child(1) { left: 10%; animation-delay: 0s; }
.particle:nth-child(2) { left: 20%; animation-delay: 2s; }
.particle:nth-child(3) { left: 30%; animation-delay: 4s; }
.particle:nth-child(4) { left: 40%; animation-delay: 6s; }
.particle:nth-child(5) { left: 50%; animation-delay: 8s; }
.particle:nth-child(6) { left: 60%; animation-delay: 10s; }
.particle:nth-child(7) { left: 70%; animation-delay: 12s; }
.particle:nth-child(8) { left: 80%; animation-delay: 14s; }
.particle:nth-child(9) { left: 90%; animation-delay: 16s; }

@keyframes float-particle {
    0% {
        transform: translateY(100vh) scale(0);
        opacity: 0;
    }
    10% {
        opacity: 0.6;
    }
    90% {
        opacity: 0.6;
    }
    100% {
        transform: translateY(-100px) scale(1);
        opacity: 0;
    }
}

/* Custom scrollbar */
::-webkit-scrollbar {
    width: 8px;
}

::-webkit-scrollbar-track {
    background: var(--secondary-bg);
}

::-webkit-scrollbar-thumb {
    background: var(--border-color);
    border-radius: 4px;
}

::-webkit-scrollbar-thumb:hover {
    background: var(--accent-color);
}
</style>
""", unsafe_allow_html=True)

# =========================
# 🧠 LOAD MODEL
# =========================
@st.cache_resource
def load_model():
    from pixelrnn_train import PixelRNNishUNet
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PixelRNNishUNet().to(device)

    ckpt_path = os.path.join("outputs", "pixelrnn_best_model.pth")
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        model.eval()
    
    return model, device

model, device = load_model()
checkpoint = None
if os.path.exists(os.path.join("outputs", "pixelrnn_best_model.pth")):
    checkpoint = torch.load(os.path.join("outputs", "pixelrnn_best_model.pth"), map_location=device)

# =========================
# 🔄 TRANSFORMS
# =========================
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])
to_pil = transforms.ToPILImage()

# =========================
# 🌟 MAIN HEADER
# =========================
st.markdown("""
<div class="main-header slide-up">
    <h1>� PPixelRNN Image Completion</h1>
    <p>Transform incomplete images into perfect reconstructions using cutting-edge AI technology</p>
</div>
""", unsafe_allow_html=True)

# =========================
# 📂 FILE UPLOAD SECTION
# =========================
st.markdown('<div class="upload-section slide-up">', unsafe_allow_html=True)
st.markdown("### 🎯 Upload Your Image")
st.markdown("Drop an image with missing or damaged regions and watch AI work its magic")

uploaded_file = st.file_uploader(
    "Choose an image file", 
    type=["jpg", "jpeg", "png"],
    help="Supported formats: JPG, JPEG, PNG • Best results with clear occlusions"
)
st.markdown('</div>', unsafe_allow_html=True)

# =========================
# 🎯 MAIN PROCESSING
# =========================
if uploaded_file and checkpoint:
    # Image processing
    image = Image.open(uploaded_file).convert("RGB")
    original_size = image.size
    
    # Display sizes
    display_size = (300, 300)
    display_image = image.resize(display_size, Image.Resampling.LANCZOS)
    
    # Model input
    input_tensor = transform(image).unsqueeze(0).to(device)
    
    # Processing with progress
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    start_time = time.time()
    
    with st.spinner("� AI  is reconstructing your image..."):
        progress_bar.progress(25)
        status_text.markdown("**🧠 Initializing neural networks...**")
        
        with torch.no_grad():
            progress_bar.progress(50)
            status_text.markdown("**🔍 Analyzing image patterns...**")
            
            output = model(input_tensor)
            output = torch.clamp(output, 0, 1)
            
            progress_bar.progress(75)
            status_text.markdown("**✨ Reconstructing missing regions...**")
            
    processing_time = time.time() - start_time
    progress_bar.progress(100)
    status_text.markdown("**✅ Processing Complete!**")
    
    # Convert output
    reconstructed = to_pil(output.squeeze().cpu()).resize(display_size, Image.Resampling.LANCZOS)
    
    # Results section
    st.markdown('<div class="result-section slide-up">', unsafe_allow_html=True)
    st.markdown("### ✨ AI Reconstruction Results")
    
    # Statistics
    st.markdown(f"""
    <div class="stats-container">
        <div class="stat-card">
            <div class="stat-number">{processing_time:.2f}s</div>
            <div class="stat-label">⚡ Processing Time</div>
        </div>
        <div class="stat-card">
            <div class="stat-number">{original_size[0]}×{original_size[1]}</div>
            <div class="stat-label">📐 Original Size</div>
        </div>
        <div class="stat-card">
            <div class="stat-number">128×128</div>
            <div class="stat-label">🎯 Model Input</div>
        </div>
        <div class="stat-card">
            <div class="stat-number">31M</div>
            <div class="stat-label">🧠 Parameters</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Image comparison
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="image-container">
            <h3>� Originnal Input</h3>
        </div>
        """, unsafe_allow_html=True)
        st.image(display_image, use_column_width=True)
        
    with col2:
        st.markdown("""
        <div class="image-container glow-effect">
            <h3>🚀 AI Reconstructed</h3>
        </div>
        """, unsafe_allow_html=True)
        st.image(reconstructed, use_column_width=True)
    
    # Download section
    st.markdown("### 💾 Export Your Results")
    
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        # Original size reconstruction
        full_size_output = to_pil(output.squeeze().cpu()).resize(original_size, Image.Resampling.LANCZOS)
        buf_full = io.BytesIO()
        full_size_output.save(buf_full, format="PNG")
        st.download_button(
            label="🎯 Full Resolution",
            data=buf_full.getvalue(),
            file_name=f"pixelrnn_full_{original_size[0]}x{original_size[1]}.png",
            mime="image/png",
        )
    
    with col2:
        # Display size
        buf_display = io.BytesIO()
        reconstructed.save(buf_display, format="PNG")
        st.download_button(
            label="�️ Prevoiew Size",
            data=buf_display.getvalue(),
            file_name="pixelrnn_preview.png",
            mime="image/png",
        )
    
    with col3:
        # Side by side comparison
        comparison = Image.new('RGB', (display_size[0] * 2, display_size[1]))
        comparison.paste(display_image, (0, 0))
        comparison.paste(reconstructed, (display_size[0], 0))
        buf_comp = io.BytesIO()
        comparison.save(buf_comp, format="PNG")
        st.download_button(
            label="�  Before & After",
            data=buf_comp.getvalue(),
            file_name="pixelrnn_comparison.png",
            mime="image/png",
        )
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Clear progress indicators
    time.sleep(1)
    progress_bar.empty()
    status_text.empty()

elif uploaded_file and not checkpoint:
    st.error("🚫 Model not available. Please train the model first by running `python pixelrnn_train.py`")

elif not uploaded_file:
    # Demo section
    st.markdown("""
    <div class="info-card slide-up">
        <h4>🚀 How It Works</h4>
        <p>
        • Upload an image with missing or damaged regions<br>
        • Our advanced PixelRNN-inspired U-Net analyzes the context<br>
        • AI reconstructs missing parts using learned patterns<br>
        • Download your restored image in multiple formats
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-card slide-up">
        <h4>💡 Pro Tips</h4>
        <p>
        • Clear occlusions work best (masks, scratches, holes)<br>
        • Good lighting and contrast improve results<br>
        • Natural scenes and objects are optimal<br>
        • Avoid heavily compressed or blurry images
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-card slide-up">
        <h4>⚡ Technical Specs</h4>
        <p>
        • Architecture: PixelRNN-inspired U-Net<br>
        • Model Size: 31M parameters<br>
        • Input Resolution: 128×128 pixels<br>
        • Processing: GPU-accelerated inference
        </p>
    </div>
    """, unsafe_allow_html=True)

# =========================
# 🪄 FOOTER
# =========================
st.markdown("""
<div class='footer'>
    <p>🧠 PixelRNN Image Completion • Built with PyTorch & Streamlit</p>
    <p>Crafted by <a href="https://github.com/MuhammadMaaz7">Muhammad Maaz</a> • Open Source on GitHub</p>
</div>
""", unsafe_allow_html=True)
