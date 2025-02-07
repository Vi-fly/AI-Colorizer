import streamlit as st
from PIL import Image
import numpy as np
from io import BytesIO
import sys
from pathlib import Path
import cv2

# Add project root to Python path
sys.path.append(str(Path(__file__).parent))

# Import colorization functions
from gan.colorize import colorize_image as gan_colorize
from LAB.colorize import colorize_image as lab_colorize  # Optional

st.set_page_config(page_title="AI Colorizer", layout="centered")
st.title("🎨 AI Image Colorization")

# ===== Sidebar Controls =====
st.sidebar.header("Settings")
model_type = st.sidebar.radio(
    "Select Colorization Model",
    ["GAN (Deep Learning)", "LAB (Traditional CV)"],
    index=0
)

# GAN-specific controls
if model_type == "GAN (Deep Learning)":
    st.sidebar.markdown("**GAN Model Version:** 70")
    contrast = st.sidebar.slider(
        "Contrast Adjustment", 
        0.5, 3.0, 1.5,
        help="Adjust image contrast (1.0 = original)"
    )
    saturation = st.sidebar.slider(
        "Saturation Adjustment", 
        0.0, 2.0, 1.2,
        help="Adjust color intensity (1.0 = original)"
    )
else:  # LAB controls
    l_adjust = st.sidebar.slider(
        "Brightness Adjustment",
        -50, 50, 0,
        help="Adjust lightness/brightness level"
    )

# ===== Main Interface =====
uploaded_file = st.file_uploader(
    "📤 Upload Black & White Photo",
    type=["png", "jpg", "jpeg"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Original Image", use_container_width=True)

    if st.button(f"✨ Colorize with {model_type.split()[0]}"):
        with st.spinner("Working magic... ⏳"):
            try:
                if model_type == "GAN (Deep Learning)":
                    colorized = gan_colorize(
                        image, 
                        contrast_factor=contrast,
                        saturation_factor=saturation
                    )
                    # Convert PIL Image to bytes
                    img_buffer = BytesIO()
                    colorized.save(img_buffer, format="JPEG")
                    img_bytes = img_buffer.getvalue()
                else:
                    colorized = lab_colorize(image, l_adjust=l_adjust)
                    # Convert numpy array to bytes
                    _, img_bytes = cv2.imencode(".jpg", colorized)
                    img_bytes = img_bytes.tobytes()

                # Display results
                st.image(colorized, caption="Colorized Result", use_container_width=True)
                
                # Download button
                st.download_button(
                    "💾 Download Colorized Image",
                    data=img_bytes,
                    file_name="colorized.jpg",
                    mime="image/jpeg",
                    help="Save the colorized image to your device"
                )

            except Exception as e:
                st.error(f"Oops! Colorization failed: {str(e)}")
                st.stop()

# ===== Instructions =====
st.sidebar.markdown("---")
st.sidebar.info("""
**How to use:**
1. Upload a B&W photo
2. Select colorization model
3. Adjust parameters (if needed)
4. Click 'Colorize' button
5. Download result
""")