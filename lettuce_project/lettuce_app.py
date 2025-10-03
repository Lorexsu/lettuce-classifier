import streamlit as st
from ultralytics import YOLO
from PIL import Image
import datetime

# Load trained YOLO model
model = YOLO("lettuce_project/best.pt")

# --- Tab Layout ---
tab1, tab2 = st.tabs(["🌿 Classification", "ℹ️ About"])

# --- Tab 1: Classification ---
with tab1:
    st.header("Lettuce Readiness Classification")

    # Upload image
    uploaded_file = st.file_uploader("Upload an image of lettuce", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)

        # Create two columns: left for image, right for results
        col1, col2 = st.columns([1, 1])

        with col1:
            # Show smaller, consistent image
            st.image(image, caption="Uploaded Image", width=300)

        with col2:
            # Run YOLO model on image
            results = model.predict(image, conf=0.5)

            if len(results) > 0 and len(results[0].boxes) > 0:
                box = results[0].boxes[0]
                cls_id = int(box.cls[0].item())  
                conf = float(box.conf[0].item())
                label = results[0].names[cls_id]

                # Display classification result
                st.subheader("Result")
                st.write(f"🥬 **Classification:** {label}")
                st.write(f"📊 **Confidence:** {conf:.2f}")
                st.write(f"📅 **Date/Time:** {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            else:
                st.warning("No lettuce detected in this image. Try another one.")

# --- Tab 2: About ---
with tab2:
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

