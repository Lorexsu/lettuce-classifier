import streamlit as st
from ultralytics import YOLO
from PIL import Image
import datetime
import pandas as pd

# Load YOLO model
model = YOLO("lettuce_project/best.pt")

# Initialize session states
if "history" not in st.session_state:
    st.session_state.history = []
if "page" not in st.session_state:
    st.session_state.page = "home"

# ---------- CUSTOM CSS ----------
st.markdown("""
<style>
body {
    background-color: #f0fdf4;
    color: #064420;
    font-family: 'Helvetica', sans-serif;
}
header, footer {display: none;}
.main {
    background-color: #f0fdf4;
    padding: 0;
    margin: 0;
}

/* NAVBAR */
.nav-bar {
    background-color: white;
    padding: 1rem 2rem;
    display: flex;
    justify-content: space-between;
    align-items: center;
    position: fixed;
    width: 100%;
    top: 0;
    left: 0;
    box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    z-index: 100;
}
.nav-logo {
    font-size: 1.5rem;
    font-weight: bold;
    color: #15803d;
    display: flex;
    align-items: center;
}
.nav-logo span {
    margin-left: 0.5rem;
}
.nav-links a {
    color: #064420;
    text-decoration: none;
    margin-left: 1.5rem;
    font-weight: 500;
}
.nav-links a:hover {
    color: #16a34a;
}

/* HERO */
.hero {
    background: linear-gradient(90deg, #15803d, #22c55e);
    color: white;
    text-align: center;
    padding: 8rem 2rem 6rem 2rem;
    margin-top: 60px;
}
.hero h1 {
    font-size: 3rem;
    font-weight: 700;
}
.hero p {
    font-size: 1.2rem;
    margin-top: 1rem;
    color: #e0ffe9;
}
.hero a {
    background: white;
    color: #15803d;
    padding: 0.8rem 1.5rem;
    border-radius: 0.5rem;
    font-weight: 600;
    text-decoration: none;
    margin-top: 2rem;
    display: inline-block;
}
.hero a:hover {
    background: #dcfce7;
}

/* FOOTER */
.footer {
    background-color: #064420;
    color: white;
    text-align: center;
    padding: 1rem;
    font-size: 0.9rem;
    margin-top: 4rem;
}

/* CONTENT SECTIONS */
.section {
    background: white;
    padding: 4rem 2rem;
    margin: 2rem auto;
    border-radius: 12px;
    max-width: 900px;
    box-shadow: 0 0 20px rgba(0,0,0,0.05);
}

h2 {
    color: #064420;
    text-align: center;
    font-size: 2rem;
    margin-bottom: 1rem;
}
</style>

<!-- NAVBAR -->
<div class="nav-bar">
  <div class="nav-logo">🥬<span>Lettuce Classifier</span></div>
  <div class="nav-links">
    <a href="?page=home">Home</a>
    <a href="?page=classify">Classify</a>
    <a href="?page=history">History</a>
    <a href="?page=about">About</a>
  </div>
</div>
""", unsafe_allow_html=True)

# ---------- PAGE ROUTING ----------
if "page" in st.query_params:
    st.session_state.page = st.query_params["page"][0]

page = st.session_state.page

# ---------- HOME ----------
if page == "home":
    st.markdown("""
    <div class="hero">
      <h1>Lettuce Readiness Classifier</h1>
      <p>AI-powered detection system using YOLOv11 to classify lettuce readiness for harvest.</p>
      <a href="?page=classify">Try Classification</a>
    </div>
    """, unsafe_allow_html=True)

# ---------- CLASSIFICATION ----------
elif page == "classify":
    st.markdown("<div class='section'>", unsafe_allow_html=True)
    st.markdown("<h2>Upload and Classify</h2>", unsafe_allow_html=True)
    st.write("Upload a photo of lettuce to determine if it's ready for harvest.")

    uploaded_file = st.file_uploader("Upload Image", type=["jpg","jpeg","png"])
    if uploaded_file:
        image = Image.open(uploaded_file)
        col1, col2 = st.columns([1, 1])
        with col1:
            st.image(image, caption="Uploaded Image", width=320)
        with col2:
            results = model.predict(image, conf=0.5)
            if len(results) > 0 and len(results[0].boxes) > 0:
                box = results[0].boxes[0]
                cls_id = int(box.cls[0].item())
                conf = float(box.conf[0].item())
                label = results[0].names[cls_id]
                
                st.markdown(f"<p style='color:#064420; font-size:18px; font-weight:600;'>🥬 Classification: {label}</p>", unsafe_allow_html=True)
                st.progress(conf)
                st.markdown(f"<p style='color:#064420;'>📊 Confidence: {conf:.2f}</p>", unsafe_allow_html=True)
                st.markdown(f"<p style='color:#064420;'>📅 {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>", unsafe_allow_html=True)
                st.markdown("""
                    <div style='background-color:#EAF4EA; border-left: 6px solid #064420; padding:10px; border-radius:8px; color:#064420;'>
                        💡 Lettuce typically matures in 30–60 days.
                    </div>
                """, unsafe_allow_html=True)

                st.session_state.history.append({
                    "Date/Time": datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    "Image Name": uploaded_file.name,
                    "Classification": label,
                    "Confidence": f"{conf:.2f}"
                })
            else:
                st.warning("No lettuce detected in this image.")
    st.markdown("</div>", unsafe_allow_html=True)

# ---------- HISTORY ----------
elif page == "history":
    st.markdown("<div class='section'>", unsafe_allow_html=True)
    st.markdown("<h2>Classification History</h2>", unsafe_allow_html=True)
    if len(st.session_state.history) > 0:
        df = pd.DataFrame(st.session_state.history)
        st.dataframe(df, use_container_width=True)
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download History as CSV", csv, "lettuce_history.csv", "text/csv")
    else:
        st.info("No classification history yet.")
    st.markdown("</div>", unsafe_allow_html=True)

# ---------- ABOUT ----------
elif page == "about":
    st.markdown("<div class='section'>", unsafe_allow_html=True)
    st.markdown("<h2>About the System</h2>", unsafe_allow_html=True)
    st.markdown("""
    The Lettuce Readiness Classifier is an AI-based web application powered by **YOLOv11**.  
    It allows users to upload lettuce images and instantly determine whether they are ready for harvest.
    
    **Core Features:**
    - 🌿 Real-time lettuce readiness classification  
    - 📊 Result history tracking  
    - 💾 CSV export for research data  
    - 🧩 Responsive design for mobile and desktop users
    """)
    st.markdown("</div>", unsafe_allow_html=True)

# ---------- FOOTER ----------
st.markdown("""
<div class="footer">
  © 2025 Lettuce Classifier | Powered by YOLOv11 | Designed by Lorexsu
</div>
""", unsafe_allow_html=True)
