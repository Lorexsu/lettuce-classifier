import streamlit as st
from ultralytics import YOLO
from PIL import Image
import datetime
import pandas as pd

# Load YOLO model
model = YOLO("lettuce_project/best.pt")

# Initialize history
if "history" not in st.session_state:
    st.session_state.history = []
if "page" not in st.session_state:
    st.session_state.page = "home"

# ---------- CUSTOM CSS ----------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

* {
    font-family: 'Inter', sans-serif;
}

body {
    background-color: #f0fdf4;
    color: #064420;
}

.stApp {
    background-color: #f0fdf4;
}

header, footer {visibility: hidden;}
.main {padding: 0 !important;}

/* NAVBAR */
.nav-bar {
    background-color: white;
    padding: 1rem 3rem;
    display: flex;
    justify-content: space-between;
    align-items: center;
    box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    position: fixed;
    width: 100%;
    top: 0;
    left: 0;
    z-index: 1000;
}
.nav-logo {
    font-size: 1.5rem;
    font-weight: 700;
    color: #15803d;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}
.nav-links {
    display: flex;
    gap: 2rem;
}
.nav-links a {
    color: #475569;
    text-decoration: none;
    font-weight: 500;
    transition: color 0.3s;
}
.nav-links a:hover {
    color: #16a34a;
}

/* HERO SECTION */
.hero {
    background: linear-gradient(135deg, #15803d, #22c55e);
    color: white;
    text-align: center;
    padding: 8rem 2rem 5rem 2rem;
    margin-top: 60px;
}
.hero h1 {
    font-size: 3rem;
    font-weight: 700;
    margin-bottom: 1rem;
}
.hero p {
    font-size: 1.25rem;
    color: #dcfce7;
    margin-bottom: 2rem;
}
.hero-btn {
    background: white;
    color: #15803d;
    padding: 0.875rem 2rem;
    border-radius: 0.5rem;
    font-weight: 600;
    text-decoration: none;
    display: inline-block;
    transition: all 0.3s;
}
.hero-btn:hover {
    background: #dcfce7;
    transform: translateY(-2px);
}

/* SECTIONS */
.section {
    background: white;
    padding: 4rem 3rem;
    margin: 2rem auto;
    max-width: 1200px;
    border-radius: 1rem;
    box-shadow: 0 4px 20px rgba(0,0,0,0.08);
}

.section-title {
    text-align: center;
    font-size: 2.5rem;
    font-weight: 700;
    color: #064420;
    margin-bottom: 1rem;
}

.section-divider {
    width: 80px;
    height: 4px;
    background: #22c55e;
    margin: 0 auto 3rem auto;
    border-radius: 2px;
}

/* FEATURE CARDS */
.feature-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
    gap: 2rem;
    margin-top: 3rem;
}
.feature-card {
    background: #f0fdf4;
    padding: 2rem;
    border-radius: 1rem;
    text-align: center;
    transition: all 0.3s;
}
.feature-card:hover {
    transform: translateY(-5px);
    box-shadow: 0 8px 25px rgba(0,0,0,0.1);
}
.feature-icon {
    width: 64px;
    height: 64px;
    background: #dcfce7;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    margin: 0 auto 1.5rem;
    font-size: 2rem;
}
.feature-title {
    font-size: 1.25rem;
    font-weight: 600;
    color: #064420;
    margin-bottom: 0.75rem;
}
.feature-text {
    color: #475569;
    line-height: 1.6;
}

/* CLASSIFICATION SECTION */
.classify-container {
    max-width: 1100px;
    margin: 0 auto;
}
.result-box {
    background: #f0fdf4;
    padding: 2rem;
    border-radius: 1rem;
    border-left: 6px solid #15803d;
    margin-top: 1.5rem;
}
.result-label {
    font-size: 1.5rem;
    font-weight: 600;
    color: #064420;
    margin-bottom: 1rem;
}
.info-badge {
    background: #dcfce7;
    color: #064420;
    padding: 1rem 1.5rem;
    border-radius: 0.5rem;
    margin-top: 1rem;
    font-size: 1rem;
    border-left: 4px solid #22c55e;
}

/* FOOTER */
.footer {
    background: #0f172a;
    color: white;
    text-align: center;
    padding: 2rem;
    margin-top: 4rem;
}
.footer-text {
    color: #94a3b8;
    font-size: 0.9rem;
}

/* RESPONSIVE */
@media (max-width: 768px) {
    .hero h1 {font-size: 2rem;}
    .section-title {font-size: 2rem;}
    .nav-bar {padding: 1rem;}
    .section {padding: 2rem 1.5rem;}
}
</style>

<div class="nav-bar">
  <div class="nav-logo">🥬 Lettuce Classifier</div>
  <div class="nav-links">
    <a href="?page=home">Home</a>
    <a href="?page=classify">Classify</a>
    <a href="?page=history">History</a>
    <a href="?page=about">About</a>
  </div>
</div>
""", unsafe_allow_html=True)

# Capture navigation
if "page" in st.query_params:
    st.session_state.page = st.query_params["page"]

page = st.session_state.page

# ---------- HOME PAGE ----------
if page == "home":
    st.markdown("""
    <div class="hero">
      <h1>🌿 Lettuce Readiness Classifier</h1>
      <p>AI-powered detection system using YOLOv11 to classify lettuce readiness for harvest</p>
    </div>
    
    <div class="section">
      <h2 class="section-title">Key Features</h2>
      <div class="section-divider"></div>
      <div class="feature-grid">
        <div class="feature-card">
          <div class="feature-icon">🌱</div>
          <h3 class="feature-title">Real-time Analysis</h3>
          <p class="feature-text">Upload lettuce images and get instant readiness classification powered by YOLOv11</p>
        </div>
        <div class="feature-card">
          <div class="feature-icon">📊</div>
          <h3 class="feature-title">Track History</h3>
          <p class="feature-text">Monitor all classifications with detailed confidence scores and timestamps</p>
        </div>
        <div class="feature-card">
          <div class="feature-icon">💾</div>
          <h3 class="feature-title">Export Data</h3>
          <p class="feature-text">Download your classification history as CSV for analysis and record-keeping</p>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

#✅ Button that switches pages in the same tab
    st.markdown('<div style="text-align:center; margin-top:-2rem;">', unsafe_allow_html=True)
    if st.button("🚀 Start Classification", use_container_width=False):
        st.session_state.page = "classify"
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

# ---------- CLASSIFICATION PAGE ----------
elif page == "classify":
    st.markdown('<div class="section classify-container">', unsafe_allow_html=True)
    st.markdown('<h2 class="section-title">Upload & Classify Lettuce</h2>', unsafe_allow_html=True)
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("Choose a lettuce image", type=["jpg","jpeg","png"], label_visibility="collapsed")
    
    if uploaded_file:
        image = Image.open(uploaded_file)
        
        col1, col2 = st.columns([1.2, 1.5])
        with col1:
            st.image(image, use_column_width=True)
        
        with col2:
            results = model.predict(image, conf=0.5)
            if len(results) > 0 and len(results[0].boxes) > 0:
                box = results[0].boxes[0]
                cls_id = int(box.cls[0].item())
                conf = float(box.conf[0].item())
                label = results[0].names[cls_id]
                
                st.markdown('<div class="result-box">', unsafe_allow_html=True)
                st.markdown(f'<div class="result-label">🥬 Classification: {label}</div>', unsafe_allow_html=True)
                st.progress(conf)
                st.markdown(f'<p style="font-size:1.1rem; color:#064420; margin-top:1rem;">📊 Confidence: <strong>{conf:.2%}</strong></p>', unsafe_allow_html=True)
                st.markdown(f'<p style="font-size:1rem; color:#64748b;">📅 {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
                
                st.markdown("""
                    <div class="info-badge">
                        💡 Lettuce typically matures in 30–60 days from transplant
                    </div>
                """, unsafe_allow_html=True)
                
                st.session_state.history.append({
                    "Date/Time": datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    "Image Name": uploaded_file.name,
                    "Classification": label,
                    "Confidence": f"{conf:.2f}"
                })
            else:
                st.warning("⚠️ No lettuce detected in this image. Please upload a clearer image.")
    
    st.markdown('</div>', unsafe_allow_html=True)

# ---------- HISTORY PAGE ----------
elif page == "history":
    st.markdown('<div class="section">', unsafe_allow_html=True)
    st.markdown('<h2 class="section-title">📊 Classification History</h2>', unsafe_allow_html=True)
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    
    if len(st.session_state.history) > 0:
        df = pd.DataFrame(st.session_state.history)
        st.dataframe(df, use_container_width=True, height=400)
        
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download History as CSV",
            csv,
            "lettuce_history.csv",
            "text/csv",
            use_container_width=True
        )
    else:
        st.info("📭 No classification history yet. Start by classifying some lettuce images!")
    
    st.markdown('</div>', unsafe_allow_html=True)

# ---------- ABOUT PAGE ----------
elif page == "about":
    st.markdown('<div class="section">', unsafe_allow_html=True)
    st.markdown('<h2 class="section-title">About This System</h2>', unsafe_allow_html=True)
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div style="font-size: 1.1rem; line-height: 1.8; color: #475569;">
    
    ### 🧠 AI-Powered Lettuce Classification
    
    This application uses **YOLOv11** (You Only Look Once), a state-of-the-art deep learning model for 
    real-time object detection and classification. Our system has been specifically trained to identify 
    lettuce readiness stages with high accuracy.
    
    ### ✨ Key Capabilities
    
    - **Real-time Detection**: Upload images and receive instant classification results
    - **High Accuracy**: Trained on thousands of lettuce images for reliable predictions
    - **Confidence Scoring**: Each prediction includes a confidence score for transparency
    - **History Tracking**: Automatically logs all classifications with timestamps
    - **Data Export**: Download your classification history for further analysis
    
    ### 🛠️ Technology Stack
    
    - **Framework**: Streamlit for interactive web interface
    - **AI Model**: YOLOv11 via Ultralytics
    - **Image Processing**: PIL (Python Imaging Library)
    - **Data Management**: Pandas for history tracking
    
    ### 📈 Use Cases
    
    This tool is perfect for farmers, agricultural researchers, and commercial growers who need to:
    - Monitor crop readiness efficiently
    - Maintain consistent harvest quality
    - Track growth patterns over time
    - Make data-driven harvesting decisions
    
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

# ---------- FOOTER ----------
st.markdown("""
<div class="footer">
  <p class="footer-text">© 2025 Lettuce Classifier | Powered by YOLOv11 | Designed by Lorexsu</p>
</div>
""", unsafe_allow_html=True)



