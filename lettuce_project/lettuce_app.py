import streamlit as st
from ultralytics import YOLO
from PIL import Image
import datetime
import pandas as pd

# Load YOLO model
model = YOLO("lettuce_project/best.pt")

# Initialize session state for history
if "history" not in st.session_state:
    st.session_state.history = []

# ---------- CUSTOM CSS ----------
st.markdown("""
    <style>
    body {
        background-color: #ffffff;
        color: #064420;
        font-family: 'Helvetica', sans-serif;
    }
    .main {
        background-color: #ffffff;
        padding: 0;
        margin: 0;
    }
    header, footer {
        display: none;
    }
    .nav-bar {
        background-color: #064420;
        color: white;
        padding: 1rem 2rem;
        font-size: 1.2rem;
        display: flex;
        justify-content: space-between;
        align-items: center;
        font-weight: 600;
        box-shadow: 0 2px 5px rgba(0,0,0,0.2);
    }
    .nav-links a {
        color: white;
        text-decoration: none;
        margin-left: 2rem;
        font-weight: 400;
    }
    .nav-links a:hover {
        text-decoration: underline;
    }
    h1, h2, h3 {
        color: #064420;
    }
    .footer {
        background-color: #064420;
        color: white;
        text-align: center;
        padding: 1rem;
        font-size: 0.9rem;
        margin-top: 3rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        justify-content: center;
    }
    </style>

    <div class="nav-bar">
        <div class="logo">🥬 Lettuce Classifier</div>
        <div class="nav-links">
            <a href="#classify">Classification</a>
            <a href="#history">History</a>
            <a href="#about">About</a>
        </div>
    </div>
""", unsafe_allow_html=True)

# ---------- Tabs ----------
tab1, tab2, tab3 = st.tabs(["🌿 Classification", "📊 History", "ℹ️ About"])

# --- Tab 1: Classification ---
with tab1:
    st.markdown("<h2 id='classify'>Lettuce Readiness Classification</h2>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload an image of lettuce", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)

        col1, col2 = st.columns([1, 1])

        with col1:
            st.image(image, caption="Uploaded Image", width=300)

        with col2:
            results = model.predict(image, conf=0.5)
            if len(results) > 0 and len(results[0].boxes) > 0:
                box = results[0].boxes[0]
                cls_id = int(box.cls[0].item())
                conf = float(box.conf[0].item())
                label = results[0].names[cls_id]

                st.subheader("Result")
                st.write(f"🥬 **Classification:** {label}")
                st.progress(conf)
                st.write(f"📊 Confidence: {conf:.2f}")
                st.write(f"📅 Date/Time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                st.info("💡 Lettuce typically matures in 30–60 days. Healthy leaves are green and crisp.")

                st.session_state.history.append({
                    "Date/Time": datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    "Image Name": uploaded_file.name,
                    "Classification": label,
                    "Confidence": f"{conf:.2f}"
                })
            else:
                st.warning("No lettuce detected in this image.")

# --- Tab 2: History ---
with tab2:
    st.markdown("<h2 id='history'>Classification History</h2>", unsafe_allow_html=True)
    if len(st.session_state.history) > 0:
        df = pd.DataFrame(st.session_state.history)
        st.dataframe(df, use_container_width=True)
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download History as CSV", csv, "lettuce_history.csv", "text/csv")
    else:
        st.info("No classification history yet.")

# --- Tab 3: About ---
with tab3:
    st.markdown("<h2 id='about'>About This System</h2>", unsafe_allow_html=True)
    st.markdown("""
    ### 🧠 Lettuce Growth Classifier (YOLOv11)
    This web-based system uses an advanced **YOLOv11 model** to automatically determine whether a lettuce crop is **Ready to Harvest** or **Not Yet Ready**.

    **Features:**
    - 🌿 Real-time image classification  
    - 📊 Classification history tracking  
    - 💾 Exportable CSV reports  
    - 🧩 Modular design for easy model updates  

    **Developed using:** Python · Streamlit · Ultralytics YOLO  
    """, unsafe_allow_html=True)

# --- Footer ---
st.markdown("""
    <div class="footer">
        © 2025 Lettuce Classifier | Powered by YOLOv11 | Designed by Lorexsu
    </div>
""", unsafe_allow_html=True)
