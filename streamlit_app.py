import streamlit as st
import requests
from PIL import Image
import io

st.set_page_config(page_title="Pepper Bell Leaf Disease Detection", layout="centered")

st.title("Pepper Bell Leaf Disease Detection")

# Upload an image
uploaded_file = st.file_uploader("Upload a leaf image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    st.image(uploaded_file, caption="Uploaded Image", width=400)
    
    # Convert image to bytes
    img_bytes = uploaded_file.read()
    
    # Send to FastAPI backend
    api_url = "http://127.0.0.1:8000/predict"  # Change to your API endpoint
    files = {"image": (uploaded_file.name, img_bytes, uploaded_file.type)}
    
    with st.spinner("Detecting disease..."):
        response = requests.post(api_url, files=files)
    
    if response.status_code == 200:
        result = response.json()
        pred_class = result['prediction']['predicted_class']
        confidence = result['prediction']['confidence']
        st.success(f"Prediction: {pred_class} ({confidence*100:.2f}%)")
    else:
        st.error("Error in prediction. Please check the backend.")