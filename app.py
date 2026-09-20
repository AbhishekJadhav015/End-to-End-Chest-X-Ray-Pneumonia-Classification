import streamlit as st
from PIL import Image
import os
from src.pipeline.predict_pipeline import PredictPipeline

st.set_page_config(page_title="Pneumonia Detector", page_icon="🫁", layout="centered")

st.title("🫁 Chest X-Ray Pneumonia Detector")
st.markdown("""
Upload a chest X-ray image to run it through the PyTorch ResNet transfer learning model. 
The model will classify the scan as **NORMAL** or **PNEUMONIA**.
""")

@st.cache_resource
def load_model():
    model_path = "artifacts/model.pth"
    if not os.path.exists(model_path):
        return None
    return PredictPipeline(model_path=model_path, num_classes=2)

predictor = load_model()

if predictor is None:
    st.error("Model artifact not found! Please place 'model.pth' inside the 'artifacts/' directory.")
else:
    uploaded_file = st.file_uploader("Upload an X-Ray Image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.image(image, caption="Uploaded Scan", use_container_width=True)
            
        with col2:
            st.write("### Diagnostics")
            if st.button("Analyze Scan", use_container_width=True):
                with st.spinner("Analyzing..."):
                    prediction, confidence = predictor.predict(image)
                    
                    if prediction == "NORMAL":
                        st.success(f"**Diagnosis:** {prediction}")
                    else:
                        st.error(f"**Diagnosis:** {prediction}")
                        
                    st.metric(label="Model Confidence", value=f"{confidence * 100:.2f}%")
                    st.progress(float(confidence))