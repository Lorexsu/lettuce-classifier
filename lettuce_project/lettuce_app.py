import streamlit as st
from ultralytics import YOLO
from PIL import Image
import datetime
import pandas as pd

# Load YOLO model
model = YOLO("lettuce_project/best.pt")

# Initialize session
if "history" not in st.session_state:
    st.session_state.history = []
if "page" not in st.session_state:
    st.session_state.page = "home"

# ---------- CUSTOM CSS ----------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
* { font-family: 'Inter', sans-serif; }
body { background-color: #f0fdf4; color: #064420; }
.stApp { background-color: #f0fdf4; }
header, footer {visibility: hidden;}
.main {padding: 0 !important;}
.nav-bar {background:white;padding:1rem 3rem;display:flex;justify-content:space-between;align-items:center;box-shadow:0 2px 8px rgba(0,0,0,0.1);position:fixed;width:100%;top:0;left:0;z-index:1000;}
.nav-logo {font-size:1.5rem;font-weight:700;color:#15803d;display:flex;align-items:center;gap:0.5rem;}
.nav-links {display:flex;gap:2rem;}
.nav-links a {color:#475569;text-decoration:none;font-weight:500;transition:color 0.3s;}
.nav-links a:hover {color:#16a34a;}
.hero {background:linear-gradient(135deg,#15803d,#22c55e);color:white;text-align:center;padding:8rem 2rem 5rem 2rem;margin-top:60px;}
.hero h1 {font-size:3rem;font-weight:700;margin-bottom:1rem;}
.hero p {font-size:1.25rem;color:#dcfce7;margin-bottom:2rem;}
.hero-btn {background:white;color:#15803d;padding:0.875rem 2rem;border-radius:0.5rem;font-weight:600;text-decoration:none;display:inline-block;transition:all 0.3s;}
.hero-btn:hover {background:#dcfce7;transform:translateY(-2px);}
.section {background:white;padding:4rem 3rem;margin:2rem auto;max-width:1200px;border-radius:1rem;box-shadow:0 4px 20px rgba(0,0,0,0.08);}
.section-title {text-align:center;font-size:2.5rem;font-weight:700;color:#064420;margin-bottom:1rem;}
.section-divider {width:80px;height:4px;background:#22c55e;margin:0 auto 3rem auto;border-radius:2px;}
.footer {background:#0f172a;color:white;text-align:center;padding:2rem;margin-top:4rem;}
.footer-text {color:#94a3b8;font-size:0.9rem;}
</style>

<div class="nav-bar">
  <div class="nav-logo">🥬 Lettuce Classifier</div>
  <div class="nav-links">
    <a href="#" onclick="window.location.reload()">Home</a>
    <a href="#" onclick="window.location.reload()">Classify</a>
    <a href="#" onclick="window.location.reload()">History</a>
    <a href="#" onclick="window.location.reload()">About</a>
  </div>
</div>
""", unsafe_allow_html=True)

# ---------- PAGE LOGIC ----------
page = st.session_state.page

# ---------- HOME ----------
if page == "home":
    st.markdown("""
    <div class="hero">
      <h1>🌿 Lettuce Readiness Classifier</h1>
      <p>AI-powered detection system using YOLOv11 to classify lettuce readiness for harvest</p>
    </div>
    """, unsafe_allow_html=True)

    # Button that stays in same tab and switches to classification
    if st.button("🚀 Start Classification", use_container_width=True):
        st.session_state.page = "classify"
        st.rerun()

# ---------- CLASSIFICATION ----------
if st.session_state.page == "classify":
    st.markdown('<div class="section">', unsafe_allow_html=True)
    st.markdown('<h2 class="section-title">Upload & Classify Lettuce</h2>', unsafe_allow_html=True)
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Upload a lettuce image", type=["jpg", "jpeg", "png"])
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

                st.markdown(f"<p style='color:#064420;font-size:20px;font-weight:600;'>🥬 Classification: {label}</p>", unsafe_allow_html=True)
                st.progress(conf)
                st.markdown(f"<p style='color:#064420;'>📊 Confidence: {conf:.2f}</p>", unsafe_allow_html=True)
                st.markdown(f"<p style='color:#64748b;'>📅 {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>", unsafe_allow_html=True)

                st.session_state.history.append({
                    "Date/Time": datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    "Image Name": uploaded_file.name,
                    "Classification": label,
                    "Confidence": f"{conf:.2f}"
                })
            else:
                st.warning("⚠️ No lettuce detected. Please upload a clearer photo.")

    st.markdown('</div>', unsafe_allow_html=True)

# ---------- FOOTER ----------
st.markdown("""
<div class="footer">
  <p class="footer-text">© 2025 Lettuce Classifier | Powered by YOLOv11 | Designed by Lorexsu</p>
</div>
""", unsafe_allow_html=True)
