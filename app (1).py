
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import json

# Page settings
st.set_page_config(
    page_title="Skin Disease Detection",
    page_icon="🔬",
    layout="centered"
)

# Load CNN model
@st.cache_resource
def load_cnn_model():
    return tf.keras.models.load_model(
        "skin_disease_cnn.keras",
        compile=False
    )

@st.cache_data
def load_labels():
    with open("class_names.json", "r") as f:
        return json.load(f)

try:
    model = load_cnn_model()
    class_names = load_labels()
except Exception as e:
    st.error(f"Model loading error: {e}")
    st.stop()

# App heading
st.title("🔬 Skin Disease Detection")
st.write("AI-Powered Skin Image Classification")
st.divider()

st.info(
    "Educational demonstration only. "
    "This app cannot diagnose skin conditions. "
    "Consult a qualified healthcare professional "
    "for medical advice."
)

# Image upload
uploaded_file = st.file_uploader(
    "Upload a Skin Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded Skin Image",
        use_container_width=True
    )

    if st.button("🔍 Analyze Image"):

        with st.spinner("Analyzing image..."):

            # Preprocess image
            img = image.resize((128, 128))
            img_array = np.array(
                img, dtype=np.float32
            ) / 255.0

            img_array = np.expand_dims(
                img_array, axis=0
            )

            # CNN prediction
            prediction = model.predict(
                img_array, verbose=0
            )[0]

            predicted_index = int(
                np.argmax(prediction)
            )

            predicted_disease = class_names[
                predicted_index
            ]

            confidence = float(
                prediction[predicted_index] * 100
            )

        st.success("Analysis Completed!")

        st.subheader("Prediction Result")

        st.write(
            f"**Predicted Category:** "
            f"{predicted_disease}"
        )

        st.write(
            f"**Model Confidence Score:** "
            f"{confidence:.2f}%"
        )

        st.progress(
            min(max(confidence / 100, 0.0), 1.0)
        )

        st.caption(
            "Confidence is a model score, not "
            "the probability of a correct medical "
            "diagnosis."
        )

        # Show top predictions
        st.subheader("Top Predictions")

        top_indices = np.argsort(
            prediction
        )[::-1][:3]

        for idx in top_indices:
            st.write(
                f"{class_names[idx]}: "
                f"{prediction[idx] * 100:.2f}%"
            )

        st.warning(
            "The model only predicts categories "
            "it was trained on. Results may be "
            "incorrect, especially for unfamiliar "
            "images or skin tones."
        )

st.divider()
st.caption("CNN Deep Learning Project | Streamlit")
