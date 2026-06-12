import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np

# ==========================================
# PAGE CONFIG
# ==========================================

st.set_page_config(
    page_title="AI Driving Risk Prediction",
    page_icon="🚗",
    layout="centered"
)

# ==========================================
# LOAD MODEL
# ==========================================

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("models/emotion_model.keras")

model = load_model()

# ==========================================
# EMOTION LABELS
# ==========================================
# Si tu as supprimé "disgusted", garde ces 6 classes.
# L'ordre doit correspondre à l'ordre d'entraînement.
emotion_labels = {
    0: "Angry",
    1: "Fearful",
    2: "Happy",
    3: "Neutral",
    4: "Sad",
    5: "Surprised"
}

# ==========================================
# RISK ENGINE
# ==========================================

def calculate_risk(emotion, behaviour):
    if emotion in ["Angry", "Fearful"] and behaviour == "AGGRESSIVE":
        return "HIGH", 90
    elif emotion == "Sad" and behaviour == "AGGRESSIVE":
        return "HIGH", 85
    elif emotion == "Neutral" and behaviour == "NORMAL":
        return "LOW", 20
    elif emotion == "Happy" and behaviour == "SLOW":
        return "LOW", 10
    else:
        return "MEDIUM", 50

# ==========================================
# TITLE
# ==========================================

st.title("🚗 AI Driving Risk Prediction System")
st.write("Multi-Modal AI System using Emotional and Behavioural Data")

# ==========================================
# SIDE INFO
# ==========================================

with st.sidebar:
    st.header("Model Info")
    st.write("Input size: 48×48")
    st.write("Color mode: Grayscale")
    st.write("Expected classes: 6")
    st.write("Model file: `models/emotion_model.keras`")

    st.subheader("Emotion Classes")
    for k, v in emotion_labels.items():
        st.write(f"{k}: {v}")

# ==========================================
# IMAGE UPLOAD
# ==========================================

uploaded_file = st.file_uploader(
    "Upload Driver Image",
    type=["jpg", "jpeg", "png"]
)

# ==========================================
# MAIN LOGIC
# ==========================================

if uploaded_file is not None:

    # Open image
    image = Image.open(uploaded_file)

    st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True
    )

    # ==========================================
    # PREPROCESSING
    # ==========================================
    # Convert to grayscale, resize to 48x48,
    # normalize, and reshape to (1, 48, 48, 1)

    image = ImageOps.grayscale(image)
    image = image.resize((48, 48))

    img_array = np.array(image).astype(np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=-1)   # (48,48,1)
    img_array = np.expand_dims(img_array, axis=0)    # (1,48,48,1)

    st.write(f"Input shape: {img_array.shape}")

    # ==========================================
    # PREDICTION
    # ==========================================

    prediction = model.predict(img_array, verbose=0)
    predicted_class = int(np.argmax(prediction))
    confidence = float(np.max(prediction) * 100)

    emotion = emotion_labels.get(predicted_class, "Unknown")

    # ==========================================
    # DEBUG INFORMATION
    # ==========================================

    with st.expander("Debug Information", expanded=False):
        st.write("Prediction probabilities:")

        for i, prob in enumerate(prediction[0]):
            label = emotion_labels.get(i, f"Class {i}")
            st.write(f"{label}: {prob:.4f}")

        st.write(f"Predicted class index: {predicted_class}")
        st.write(f"Prediction shape: {prediction.shape}")
        st.write(f"Model output shape: {model.output_shape}")

    # ==========================================
    # EMOTION RESULT
    # ==========================================

    st.subheader("Detected Emotion")
    st.success(emotion)
    st.write(f"Confidence: {confidence:.2f}%")

    # ==========================================
    # BEHAVIOUR INPUT
    # ==========================================

    st.subheader("Driving Behaviour")

    behaviour = st.selectbox(
        "Select Driving Behaviour",
        ["AGGRESSIVE", "NORMAL", "SLOW"]
    )

    # ==========================================
    # RISK PREDICTION
    # ==========================================

    if st.button("Predict Risk"):

        risk_level, score = calculate_risk(emotion, behaviour)

        st.subheader("Risk Assessment")
        st.write(f"Risk Score: {score}/100")

        if risk_level == "HIGH":
            st.error(f"Risk Level: {risk_level}")
        elif risk_level == "MEDIUM":
            st.warning(f"Risk Level: {risk_level}")
        else:
            st.success(f"Risk Level: {risk_level}")

        st.write("---")
        st.write(f"Detected Emotion: {emotion}")
        st.write(f"Driving Behaviour: {behaviour}")

else:
    st.info("Please upload a face image to start the prediction.")