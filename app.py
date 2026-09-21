import streamlit as st
from PIL import Image
import os
import sys
from src.logger import logging
from src.exception import CustomException
from src.pipeline.predict_pipeline import PredictPipeline

st.set_page_config(page_title="Pneumonia Detector", page_icon="🫁", layout="centered")

st.title("🫁 Chest X-Ray Pneumonia Detector")
st.markdown("""
Upload a chest X-ray image to run it through the PyTorch ResNet transfer learning model. 
The model will classify the scan as **NORMAL** or **PNEUMONIA**.
""")

@st.cache_resource
def load_model():
    try:
        logging.info("Attempting to load the prediction model weights.")
        model_path = "artifacts/model.pth"
        
        if not os.path.exists(model_path):
            logging.warning(f"Model path {model_path} does not exist.")
            return None
            
        predictor = PredictPipeline(model_path=model_path, num_classes=2)
        logging.info("Model successfully loaded into memory.")
        return predictor
        
    except Exception as e:
        logging.error("Failed to load the model pipeline.")
        raise CustomException(e, sys)

# Load the model
predictor = load_model()

if predictor is None:
    st.error("Model artifact not found! Please place 'model.pth' inside the 'artifacts/' directory.")
else:
    uploaded_file = st.file_uploader("Upload an X-Ray Image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        try:
            logging.info(f"User uploaded a file: {uploaded_file.name}")
            image = Image.open(uploaded_file)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.image(image, caption="Uploaded Scan", use_container_width=True)
                
            with col2:
                st.write("### Diagnostics")
                if st.button("Analyze Scan", use_container_width=True):
                    with st.spinner("Analyzing..."):
                        logging.info("Initiating model prediction sequence.")
                        
                        # The prediction step
                        prediction, confidence = predictor.predict(image)
                        
                        logging.info(f"Prediction complete. Result: {prediction} | Confidence: {confidence:.4f}")
                        
                        if prediction == "NORMAL":
                            st.success(f"**Diagnosis:** {prediction}")
                        else:
                            st.error(f"**Diagnosis:** {prediction}")
                            
                        st.metric(label="Model Confidence", value=f"{confidence * 100:.2f}%")
                        st.progress(float(confidence))
                        
        except Exception as e:
            # Catch any UI or image processing errors
            custom_error = CustomException(e, sys)
            logging.error(f"Streamlit App Error: {custom_error}")
            st.error("An error occurred while processing the image. Please try uploading a different file.")