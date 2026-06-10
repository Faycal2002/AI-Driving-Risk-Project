import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# ==========================================
# PAGE CONFIG
# ==========================================

st.set_page_config(
    page_title="AI Driving Risk Prediction",
    page_icon="🚗"
)

# ==========================================
# LOAD MODEL
# ==========================================

model = tf.keras.models.load_model(
    "models/emotion_model.keras"
)

# ==========================================
# EMOTION LABELS
# ==========================================

emotion_labels = {
    0: "Angry",
    1: "Disgust",
    2: "Fear",
    3: "Happy",
    4: "Sad",
    5: "Surprise",
    6: "Neutral"
}

# ==========================================
# RISK ENGINE
# ==========================================

def calculate_risk(emotion, behaviour):

    if emotion in ["Angry", "Fear"] and behaviour == "AGGRESSIVE":
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

st.write(
    "Multi-Modal AI System using Emotional and Behavioural Data"
)

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

    image = Image.open(uploaded_file)

    st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True
    )

    # Preprocessing

    image = image.convert("RGB")
    image = image.resize((48, 48))

    img_array = np.array(image)

    img_array = img_array / 255.0

    img_array = np.expand_dims(
        img_array,
        axis=0
    )

    # Prediction

    prediction = model.predict(
        img_array,
        verbose=0
    )

    predicted_class = np.argmax(
        prediction
    )

    confidence = (
        np.max(prediction) * 100
    )

    emotion = emotion_labels[
        predicted_class
    ]

    # Emotion Result

    st.subheader("Detected Emotion")

    st.success(
        f"{emotion}"
    )

    st.write(
        f"Confidence: {confidence:.2f}%"
    )

    # Behaviour Selection

    st.subheader("Driving Behaviour")

    behaviour = st.selectbox(
        "Select Driving Behaviour",
        [
            "AGGRESSIVE",
            "NORMAL",
            "SLOW"
        ]
    )

    # Risk Prediction

    if st.button("Predict Risk"):

        risk_level, score = calculate_risk(
            emotion,
            behaviour
        )

        st.subheader(
            "Risk Assessment"
        )

        st.write(
            f"Risk Score: {score}/100"
        )

        if risk_level == "HIGH":

            st.error(
                f"Risk Level: {risk_level}"
            )

        elif risk_level == "MEDIUM":

            st.warning(
                f"Risk Level: {risk_level}"
            )

        else:

            st.success(
                f"Risk Level: {risk_level}"
            )

        st.write("---")

        st.write(
            f"Detected Emotion: {emotion}"
        )

        st.write(
            f"Driving Behaviour: {behaviour}"
        )