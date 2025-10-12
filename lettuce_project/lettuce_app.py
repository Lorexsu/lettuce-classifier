import streamlit as st
from ultralytics import YOLO
from PIL import Image
import datetime
import pandas as pd

# Load YOLO model
model = YOLO("lettuce_project/best.pt")

# Initialize session state
if "history" not in st.session_state:
    st.session_state.history = []
if "page" not in st.session_state:
    st.session_state.page = "classification"

# ---------- CUSTOM CSS ----------
st.markdown("""
<style>
/* ----- GENERAL PAGE SETUP ----- */
body {
    background-color: #ffffff !important;
    color: #064420 !important;
    font-family: 'Poppins', sans-serif;
}
[data-testid="stAppViewContainer"] {
    background-color: #ffffff !important;
}
[data-testid="stHeader"] {
    background: none;
}
header, footer {visibility: hidden;}

/* ----- FIXED HEADER ----- */
.nav-bar {
    background-color: #064420;
    color: white;
    padding: 1rem 2rem;
    font-size: 1.2rem;
    display: flex;
    justify-content: space-between;
    align-items: center;
    font-weight: 600;
    box-shadow: 0 2px 6px rgba(0,0,0,0.3);
    position: fixed;
    top: 0;
    left: 0;
    right: 0;
    z-index: 100;
}
.logo {
    font-size: 1.5rem;
    font-weight: 700;
    color: white;
}
.nav-links {
    display: flex;
    gap: 2rem;
}
.button-link {
    background: none;
    border: none;
    color: white;
    cursor: pointer;
    font-size: 1.1rem;
    font-weight: 400;
}
.button-link:hover {text-decoration: underline;}

/* ----- MAIN CONTENT ----- */
.main-content {
    margin-top: 110px;   /* space below header */
    margin-bottom: 100px; /* space above footer */
    padding: 2rem 3rem;
    background-color: #ffffff;
}

/* ----- FOOTER ----- */
.footer {
    background-color: #064420;
    color: white;
    text-align: center;
    padding: 1rem;
    font-size: 0.9rem;
    position: fixed;
    bottom: 0;
    left: 0;
    right: 0;
}

/* ----- TEXT COLORS ----- */
h1, h2, h3, h4, h5, h6, p, span, label, div {
    color: #064420 !important;
}
.stProgress > div > div > div {
    background-color: #2E8B57 !important;
}
</style>

<!-- ----- NAVIGATION BAR ----- -->
<div class="nav-bar">
  <div class="logo">🥬 Lettuce Classifier</div>
  <div class="nav-links">
      <form action="#" method="get">
          <button class="button-link" name="nav" value="classification">Classification</button>
          <button class="button-link" name="nav" value="history">History</button>
          <button class="button-link" name="nav" value="about">About</button>
      </form>
  </div>
</div>
""", unsafe_allow_html=True)

# ---------- PAGE LOGIC ----------
nav = st.query_params.get("nav")
if nav:
    st.session_state.page = nav
page = st.session_state.page

# ---------- MAIN CONTENT ----------
st.markdown('<div class="main-content">', unsafe_allow_html=True)

if page == "classification":
    st.title("🌿 Lettuce Readiness Classification")
    uploaded_file = st.file_uploader("Upload an image of lettuce", type=["jpg","jpeg","png"])
    if uploaded_file:
        image = Image.open(uploaded_file)
        col1, col2 = st.columns([1,1])
        with col1:
            st.image(image, caption="Uploaded Image", width=350)
        with col2:
            results = model.predict(image, conf=0.5)
            if len(results) > 0 and len(results[0].boxes) > 0:
                box = results[0].boxes[0]
                cls_id = int(box.cls[0].item())
                conf = float(box.conf[0].item())
                label = results[0].names[cls_id]

                st.subheader("Result")
                st.markdown(f"<b>🥬 Classification:</b> <span style='color:#064420;'>{label}</span>", unsafe_allow_html=True)
                st.progress(conf)
                st.markdown(f"<b>📊 Confidence:</b> {conf:.2f}")
                st.markdown(f"<b>📅 Date:</b> {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                st.info("💡 Lettuce typically matures in 30–60 days.")
                st.session_state.history.append({
                    "Date/Time": datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    "Image Name": uploaded_file.name,
                    "Classification": label,
                    "Confidence": f"{conf:.2f}"
                })
            else:
                st.warning("No lettuce detected in this image.")

elif page == "history":
    st.title("📊 Classification History")
    if len(st.session_state.history) > 0:
        df = pd.DataFrame(st.session_state.history)
        st.dataframe(df, use_container_width=True)
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download History as CSV", csv, "lettuce_history.csv", "text/csv")
    else:
        st.info("No classification history yet.")

elif page == "about":
    st.title("ℹ️ About This System")
    st.markdown("""
    ### 🧠 Lettuce Growth Classifier (YOLOv11)
    This system uses **YOLOv11** to determine whether a lettuce is ready for harvest.

    #### Features:
    - 🌿 Real-time lettuce classification  
    - 📊 Saves and exports classification history  
    - 🧩 Easy model updates  

    #### Built with:
    Python · Streamlit · Ultralytics YOLO
    """)

st.markdown('</div>', unsafe_allow_html=True)

# ---------- FOOTER ----------
st.markdown("""
<div class="footer">
  © 2025 Lettuce Classifier | Powered by YOLOv11 | Designed by Lorexsu
</div>
""", unsafe_allow_html=True)
