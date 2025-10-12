import streamlit as st
from ultralytics import YOLO
from PIL import Image
import datetime
import pandas as pd

# ---------- Custom CSS ----------
st.markdown("""
<style>
/* Global style */
body {
    background-color: #f9f9f9;
    font-family: 'Poppins', sans-serif;
    color: #1b4332;
}

/* Header */
h1, h2, h3, h4 {
    color: #1b4332;
}

/* Green highlight buttons */
.stButton>button {
    background-color: #2d6a4f;
    color: white;
    border-radius: 12px;
    border: none;
    padding: 0.5rem 1rem;
    font-weight: 600;
    transition: all 0.3s ease;
}
.stButton>button:hover {
    background-color: #1b4332;
    transform: scale(1.05);
}

/* Tabs */
[data-baseweb="tab-list"] {
    justify-content: center;
    border-bottom: 2px solid #52b788;
}
[data-baseweb="tab"] {
    color: #1b4332 !important;
    font-weight: 600;
}

/* Card style for results */
.result-card {
    background-color: #ffffff;
    border-radius: 16px;
    padding: 20px;
    box-shadow: 0px 4px 10px rgba(0,0,0,0.1);
    text-align: center;
    margin-top: 15px;
}
</style>
""", unsafe_allow_html=True)

# ---------- Model ----------
model = YOLO("lettuce_project/best.pt")

if "history" not in st.session_state:
    st.session_state.history = []

# ---------- Tabs ----------
tab1, tab2, tab3 = st.tabs(["🌿 Classification", "📊 History", "ℹ️ About"])

# ---------- Tab 1 ----------
with tab1:
    st.markdown("<h2 style='text-align:center;'>Lettuce Readiness Classification</h2>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload an image of lettuce", type=["jpg", "jpeg", "png"])

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

                # Stylish result card
                st.markdown(f"""
                <div class='result-card'>
                    <h3>🥬 Classification Result</h3>
                    <h2 style='color:#2d6a4f;'>{label}</h2>
                    <p><b>Confidence:</b> {conf:.2f}</p>
                    <div style='height:10px; background:#e9ecef; border-radius:10px;'>
                        <div style='width:{conf*100}%; height:10px; background:#2d6a4f; border-radius:10px;'></div>
                    </div>
                    <p><b>Date/Time:</b> {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
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

# ---------- Tab 2 ----------
with tab2:
    st.markdown("<h2 style='text-align:center;'>Classification History</h2>", unsafe_allow_html=True)

    if len(st.session_state.history) > 0:
        df = pd.DataFrame(st.session_state.history)
        st.dataframe(df, use_container_width=True)

        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="⬇️ Download History as CSV",
            data=csv,
            file_name="lettuce_classification_history.csv",
            mime="text/csv"
        )
    else:
        st.info("No history yet. Classify some images first.")

# ---------- Tab 3 ----------
with tab3:
    st.markdown("<h2 style='text-align:center;'>About the System</h2>", unsafe_allow_html=True)
    st.markdown("""
    <div style='text-align:justify; font-size:16px;'>
    <p>This web-based system uses a deep learning model built with <b>YOLOv11</b> to classify 
    lettuce crops as <b>Ready</b> or <b>Not Ready</b> to harvest from an uploaded image.</p>

    <p>Developed to support hydroponic farmers, the system reduces manual inspection effort 
    and helps ensure timely harvesting decisions.</p>

    <ul>
    <li><b>Framework:</b> Streamlit</li>
    <li><b>Model:</b> YOLOv11 (Ultralytics)</li>
    <li><b>Language:</b> Python</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
