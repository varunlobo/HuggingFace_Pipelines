import streamlit as st
from transformers import pipeline
from PIL import Image, ImageDraw
import torch

# Load object detection pipeline
detector = pipeline("object-detection")

st.title("Object Detection App")

# Upload image
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)
    
    st.write("Detecting objects...")
    results = detector(image)
    detected_objects = []
    # Draw bounding boxes
    draw = ImageDraw.Draw(image)
    for result in results:
        box = result['box']
        label = result['label']
        score = result['score']
        
        detected_objects.append({"Object": label, "Confidence": f"{score:.2f}"})

        # Draw rectangle
        draw.rectangle([(box['xmin'], box['ymin']), (box['xmax'], box['ymax'])], outline="red", width=3)
        
        # Draw label
        draw.text((box['xmin'], box['ymin']), f"{label} ({score:.2f})", fill="red")
    
    # Display detected objects in a table
    st.write("### Detected Objects:")
    st.table(detected_objects)

    # Show image with bounding boxes
    st.image(image, caption="Detected Objects", use_container_width=True)
