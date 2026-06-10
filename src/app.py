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
    1: "Disgusted",
    2: "Fearful",
    3: "Happy",
    4: "Neutral",
    5: "Sad",
    6: "Surprised"
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

    # ==========================================
    # PREPROCESSING
    # ==========================================

    image = image.convert("RGB")
    image = image.resize((48, 48))

    img_array = np.array(image)

    # SAME NORMALIZATION AS TRAINING
    img_array = img_array.astype(np.float32) / 255.0

    img_array = np.expand_dims(
        img_array,
        axis=0
    )

    # ==========================================
    # PREDICTION
    # ==========================================

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

    # ==========================================
    # DEBUG INFORMATION
    # ==========================================

    st.subheader("Debug Information")

    st.write("Prediction probabilities:")

    for i, prob in enumerate(prediction[0]):
        st.write(
            f"Class {i}: {prob:.4f}"
        )

    st.write(
        f"Predicted class index: {predicted_class}"
    )

    # ==========================================
    # EMOTION RESULT
    # ==========================================

    st.subheader(
        "Detected Emotion"
    )

    st.success(
        emotion
    )

    st.write(
        f"Confidence: {confidence:.2f}%"
    )

    # ==========================================
    # BEHAVIOUR INPUT
    # ==========================================

    st.subheader(
        "Driving Behaviour"
    )

    behaviour = st.selectbox(
        "Select Driving Behaviour",
        [
            "AGGRESSIVE",
            "NORMAL",
            "SLOW"
        ]
    )

    # ==========================================
    # RISK PREDICTION
    # ==========================================

    if st.button(
        "Predict Risk"
    ):

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