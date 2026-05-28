"""
Streamlit UI for surface roughness classification.
Calls the FastAPI backend at API_URL (set via environment variable).
"""
import os
import io

import requests
import streamlit as st
from PIL import Image

API_URL = os.getenv("API_URL", "http://localhost:8000")
GRIT_VALUES = [60, 80, 100, 120, 150, 180, 240, 320, 400, 600]
LABEL_COLOURS = {"Smooth": "#2ecc71", "Medium": "#f39c12", "Rough": "#e74c3c"}

st.set_page_config(page_title="Surface Roughness Classifier", layout="centered")
st.title("Surface Roughness Classifier")
st.caption("Upload a microscopy image of a machined surface to classify its roughness.")

with st.sidebar:
    st.header("Settings")
    grit = st.selectbox("Grit value used during machining", GRIT_VALUES, index=4)

uploaded = st.file_uploader("Upload image", type=["png", "jpg", "jpeg"])

if uploaded:
    image = Image.open(uploaded).convert("RGB")
    st.image(image, caption="Uploaded image", use_container_width=True)

    if st.button("Classify", type="primary"):
        with st.spinner("Running inference…"):
            buf = io.BytesIO()
            image.save(buf, format="PNG")
            buf.seek(0)

            try:
                resp = requests.post(
                    f"{API_URL}/predict",
                    files={"file": ("image.png", buf, "image/png")},
                    data={"grit": grit},
                    timeout=30,
                )
                resp.raise_for_status()
                result = resp.json()
            except requests.exceptions.ConnectionError:
                st.error("Cannot reach the API. Make sure the backend is running.")
                st.stop()
            except Exception as e:
                st.error(f"Error: {e}")
                st.stop()

        label = result["label"]
        confidence = result["confidence"]
        probs = result["probabilities"]

        colour = LABEL_COLOURS.get(label, "#999")
        st.markdown(
            f"<h2 style='color:{colour}'>Prediction: {label}</h2>",
            unsafe_allow_html=True,
        )
        st.metric("Confidence", f"{confidence:.1%}")

        st.subheader("Class probabilities")
        for cls, prob in probs.items():
            c = LABEL_COLOURS.get(cls, "#999")
            st.markdown(f"**{cls}**")
            st.progress(prob, text=f"{prob:.1%}")
