import streamlit as st
from ultralytics import YOLO
from PIL import Image
import datetime
import pandas as pd

# Load YOLO model (make sure best.pt is in same folder or adjust path)
model = YOLO("lettuce_project\best.pt")

# Initialize session state for history
if "history" not in st.session_state:
    st.session_state.history = []

# --- Tab Layout ---
tab1, tab2, tab3 = st.tabs(["🌿 Classification", "📊 History", "ℹ️ About"])

# --- Tab 1: Classification ---
with tab1:
    st.header("Lettuce Readiness Classification")

    uploaded_file = st.file_uploader("Upload an image of lettuce", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)

        # Side-by-side layout
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

                # Display result
                st.subheader("Result")
                st.write(f"🥬 **Classification:** {label}")
                st.progress(conf)  # progress bar for confidence
                st.write(f"📊 Confidence: {conf:.2f}")
                st.write(f"📅 Date/Time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

                # Educational tip
                st.info("💡 Tip: Lettuce is typically ready 30–60 days after germination. Healthy leaves are green and firm.")

                # Save to history
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
    st.header("Classification History")

    if len(st.session_state.history) > 0:
        df = pd.DataFrame(st.session_state.history)
        st.dataframe(df, use_container_width=True)

        # Download as CSV
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="⬇️ Download History as CSV",
            data=csv,
            file_name="lettuce_classification_history.csv",
            mime="text/csv",
        )
    else:
        st.info("No history yet. Classify some images first.")

# --- Tab 3: About ---
with tab3:
    st.header("About the System")
    st.markdown("""
    ### Automated Lettuce Readiness Classification Using YOLOv11 in Kratky Hydroponics

    This system uses a deep learning model trained with the **YOLOv11** architecture 
    to classify whether lettuce is **Ready to Harvest** or **Not Ready** based on an uploaded image.

    **Developed by:** Your Thesis Team  
    **Technologies Used:** Python, Ultralytics YOLO, Streamlit, Google Colab, VS Code  

    The system aims to assist hydroponic farmers by providing an **automated decision support tool**, 
    reducing manual inspection effort and improving harvest accuracy.
    """)

