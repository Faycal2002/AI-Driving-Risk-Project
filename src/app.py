from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps

# ==================================================
# PAGE CONFIG
# ==================================================

st.set_page_config(
    page_title="AI Driving Risk Prediction",
    page_icon="🚗",
    layout="centered"
)

# ==================================================
# PATHS
# ==================================================

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"

EMOTION_MODEL_PATH = MODELS_DIR / "emotion_model.keras"
BEHAVIOUR_MODEL_PATH = MODELS_DIR / "behaviour_model.pkl"
BEHAVIOUR_ENCODER_PATH = MODELS_DIR / "behaviour_encoder.pkl"

# ==================================================
# CHECK FILES
# ==================================================

missing_files = []
for file_path in [EMOTION_MODEL_PATH, BEHAVIOUR_MODEL_PATH, BEHAVIOUR_ENCODER_PATH]:
    if not file_path.exists():
        missing_files.append(str(file_path))

if missing_files:
    st.error("Missing model files:")
    for f in missing_files:
        st.write(f"- {f}")
    st.stop()

# ==================================================
# LABELS
# ==================================================

emotion_labels = {
    0: "Angry",
    1: "Disgust",
    2: "Fear",
    3: "Happy",
    4: "Sad",
    5: "Surprise",
    6: "Neutral"
}

FEATURES = ["AccX", "AccY", "AccZ", "GyroX", "GyroY", "GyroZ"]

# ==================================================
# DRIVING PROFILES
# ==================================================

DRIVING_PROFILES = {
    "Smooth / Calm": {
        "values": {
            "AccX": 0.05,
            "AccY": 0.08,
            "AccZ": 0.02,
            "GyroX": 1.0,
            "GyroY": 1.2,
            "GyroZ": 0.8,
        },
        "description": "Gentle acceleration, light braking, stable lane position, and soft steering.",
        "cards": [
            ("🟢 Smooth acceleration", "The vehicle moves gently and steadily."),
            ("🟢 Light braking", "The driver brakes softly."),
            ("🟢 Stable lane position", "The vehicle stays well aligned."),
            ("🟢 Gentle steering", "Turning happens smoothly.")
        ]
    },
    "Normal / Balanced": {
        "values": {
            "AccX": 0.20,
            "AccY": 0.40,
            "AccZ": 0.08,
            "GyroX": 6.0,
            "GyroY": 7.0,
            "GyroZ": 6.0,
        },
        "description": "Moderate acceleration, regular braking, normal lane movement, and controlled steering.",
        "cards": [
            ("🟡 Moderate acceleration", "The vehicle speeds up in a normal way."),
            ("🟡 Regular braking", "The driver brakes normally."),
            ("🟡 Normal lane movement", "The vehicle moves in a standard pattern."),
            ("🟡 Regular steering", "Turning is controlled and average.")
        ]
    },
    "Aggressive / Fast": {
        "values": {
            "AccX": 0.95,
            "AccY": 1.55,
            "AccZ": 0.35,
            "GyroX": 20.0,
            "GyroY": 24.0,
            "GyroZ": 28.0,
        },
        "description": "Strong acceleration, hard braking, frequent lane changes, and sharp steering.",
        "cards": [
            ("🔴 Strong acceleration", "The vehicle speeds up quickly."),
            ("🔴 Hard braking", "The driver brakes suddenly."),
            ("🔴 Frequent lane changes", "The vehicle moves side to side more often."),
            ("🔴 Sharp steering / turning", "The wheel turns aggressively.")
        ]
    }
}

# ==================================================
# RISK WEIGHTS
# ==================================================

EMOTION_RISK = {
    "Happy": 5,
    "Neutral": 10,
    "Surprise": 35,
    "Sad": 60,
    "Fear": 75,
    "Disgust": 70,
    "Angry": 90
}

BEHAVIOUR_RISK = {
    "SLOW": 15,
    "NORMAL": 35,
    "AGGRESSIVE": 90
}

# ==================================================
# LOAD MODELS
# ==================================================

@st.cache_resource
def load_models():
    emotion_model = tf.keras.models.load_model(EMOTION_MODEL_PATH)
    behaviour_model = joblib.load(BEHAVIOUR_MODEL_PATH)
    behaviour_encoder = joblib.load(BEHAVIOUR_ENCODER_PATH)
    return emotion_model, behaviour_model, behaviour_encoder


emotion_model, behaviour_model, behaviour_encoder = load_models()

# ==================================================
# HELPERS
# ==================================================

def render_card(title: str, text: str, kind: str = "neutral") -> None:
    styles = {
        "green": ("#e9f7ef", "#1e8449", "#1e8449"),
        "yellow": ("#fff8e1", "#b7950b", "#7d6608"),
        "red": ("#fdecea", "#c0392b", "#922b21"),
        "blue": ("#eef5ff", "#2e86de", "#1f4e79"),
        "neutral": ("#f5f5f5", "#d0d0d0", "#2c3e50"),
    }

    bg, border, txt = styles.get(kind, styles["neutral"])
    st.markdown(
        f"""
        <div style="
            background:{bg};
            border:1px solid {border};
            padding:14px 16px;
            border-radius:16px;
            margin-bottom:10px;
            box-shadow:0 1px 3px rgba(0,0,0,0.06);
        ">
            <div style="font-weight:700; color:{txt}; font-size:1rem; margin-bottom:4px;">
                {title}
            </div>
            <div style="color:{txt}; opacity:0.95; line-height:1.4;">
                {text}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


def predict_emotion(uploaded_file):
    """
    The emotion model was trained with grayscale 48x48 images,
    so the uploaded image must be converted to grayscale first.
    """
    image = Image.open(uploaded_file)
    display_image = image.copy()

    image = ImageOps.grayscale(image)
    image = image.resize((48, 48))

    img_array = np.array(image).astype(np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=-1)  # (48, 48, 1)
    img_array = np.expand_dims(img_array, axis=0)   # (1, 48, 48, 1)

    probs = emotion_model.predict(img_array, verbose=0)[0]
    pred_idx = int(np.argmax(probs))
    confidence = float(np.max(probs) * 100)
    emotion = emotion_labels.get(pred_idx, "Unknown")

    return display_image, emotion, confidence, probs


def predict_behaviour(sensor_values):
    sensor_df = pd.DataFrame([sensor_values], columns=FEATURES)

    probs = behaviour_model.predict_proba(sensor_df)[0]
    pred_idx = int(np.argmax(probs))
    confidence = float(np.max(probs) * 100)

    behaviour = behaviour_encoder.inverse_transform([pred_idx])[0]
    return behaviour, confidence, probs


def calculate_risk(emotion_probs, behaviour_probs):
    emotion_score = 0.0
    for idx, prob in enumerate(emotion_probs):
        label = emotion_labels.get(idx, f"Class {idx}")
        emotion_score += float(prob) * EMOTION_RISK.get(label, 20)

    behaviour_score = 0.0
    for idx, prob in enumerate(behaviour_probs):
        label = behaviour_encoder.inverse_transform([idx])[0]
        behaviour_score += float(prob) * BEHAVIOUR_RISK.get(label, 30)

    final_score = round((0.40 * emotion_score) + (0.60 * behaviour_score))
    final_score = int(np.clip(final_score, 0, 100))

    if final_score >= 70:
        risk_level = "HIGH"
    elif final_score >= 40:
        risk_level = "MEDIUM"
    else:
        risk_level = "LOW"

    return risk_level, final_score

# ==================================================
# TITLE
# ==================================================

st.title("🚗 AI Driving Risk Prediction System")
st.write("Emotion recognition + driving behaviour prediction + risk fusion")

# ==================================================
# SIDEBAR
# ==================================================

with st.sidebar:
    st.header("Model Info")
    st.write("Emotion model: `models/emotion_model.keras`")
    st.write("Behaviour model: `models/behaviour_model.pkl`")
    st.write("Behaviour encoder: `models/behaviour_encoder.pkl`")

    st.subheader("Emotion classes")
    for k, v in emotion_labels.items():
        st.write(f"{k}: {v}")

    st.subheader("How it works")
    st.write("1. Upload a driver face image.")
    st.write("2. Choose a driving style or enter sensor values.")
    st.write("3. The system predicts behaviour and final risk.")

# ==================================================
# MAIN LAYOUT
# ==================================================

st.markdown("---")

left, right = st.columns(2)

with left:
    st.subheader("1) Driver Image")
    uploaded_file = st.file_uploader(
        "Upload a face image",
        type=["jpg", "jpeg", "png"]
    )

    emotion = None
    emotion_probs = None
    emotion_confidence = None

    if uploaded_file is not None:
        display_image, emotion, emotion_confidence, emotion_probs = predict_emotion(uploaded_file)

        st.image(display_image, caption="Uploaded image", use_container_width=True)
        st.success(f"Detected emotion: {emotion}")
        st.write(f"Confidence: {emotion_confidence:.2f}%")

        with st.expander("Emotion prediction details"):
            for i, prob in enumerate(emotion_probs):
                st.write(f"{emotion_labels[i]}: {prob:.4f}")
    else:
        st.info("Upload a face image to detect the driver's emotion.")

with right:
    st.subheader("2) Driving Behaviour")

    input_mode = st.radio(
        "Choose input mode",
        ["Preset driving style", "Manual sensor values"],
        horizontal=True
    )

    if input_mode == "Preset driving style":
        selected_profile = st.selectbox(
            "Select a driving style",
            list(DRIVING_PROFILES.keys())
        )

        profile = DRIVING_PROFILES[selected_profile]
        st.info(profile["description"])

        card_kind = "green" if selected_profile == "Smooth / Calm" else "yellow" if selected_profile == "Normal / Balanced" else "red"

        for title, text in profile["cards"]:
            render_card(title, text, kind=card_kind)

        sensor_values = profile["values"]

        with st.expander("Raw sensor values used by the model", expanded=False):
            st.write(sensor_values)

    else:
        st.write("Enter the raw sensor values below.")
        st.caption("These are the values used by the behaviour model.")

        with st.expander("What do these values mean?", expanded=False):
            st.write("AccX = side-to-side movement")
            st.write("AccY = acceleration and braking movement")
            st.write("AccZ = up and down movement")
            st.write("GyroX = vehicle tilt left or right")
            st.write("GyroY = vehicle tilt forward or backward")
            st.write("GyroZ = turning and steering movement")

        c1, c2 = st.columns(2)

        with c1:
            accx = st.number_input("AccX", value=0.0, format="%.4f")
            accy = st.number_input("AccY", value=0.0, format="%.4f")
            accz = st.number_input("AccZ", value=0.0, format="%.4f")

        with c2:
            gyrox = st.number_input("GyroX", value=0.0, format="%.4f")
            gyroy = st.number_input("GyroY", value=0.0, format="%.4f")
            gyroz = st.number_input("GyroZ", value=0.0, format="%.4f")

        sensor_values = {
            "AccX": accx,
            "AccY": accy,
            "AccZ": accz,
            "GyroX": gyrox,
            "GyroY": gyroy,
            "GyroZ": gyroz
        }

        st.write("Selected sensor values:")
        st.write(sensor_values)

# ==================================================
# PREDICTION
# ==================================================

st.markdown("---")

if st.button("Predict Risk"):
    if uploaded_file is None:
        st.error("Please upload a face image first.")
    else:
        behaviour, behaviour_confidence, behaviour_probs = predict_behaviour(sensor_values)
        risk_level, risk_score = calculate_risk(emotion_probs, behaviour_probs)

        st.subheader("3) Behaviour Prediction")
        st.success(f"Predicted behaviour: {behaviour}")
        st.write(f"Confidence: {behaviour_confidence:.2f}%")

        with st.expander("Behaviour prediction details"):
            for i, prob in enumerate(behaviour_probs):
                label = behaviour_encoder.inverse_transform([i])[0]
                st.write(f"{label}: {prob:.4f}")

        st.subheader("4) Final Risk Assessment")
        st.metric("Risk Score", f"{risk_score}/100")

        if risk_level == "HIGH":
            st.error(f"Risk Level: {risk_level}")
        elif risk_level == "MEDIUM":
            st.warning(f"Risk Level: {risk_level}")
        else:
            st.success(f"Risk Level: {risk_level}")

        st.write("---")
        st.write(f"Detected emotion: {emotion}")
        st.write(f"Predicted behaviour: {behaviour}")
        
else:
    st.info("Upload an image and choose a driving style, then click Predict Risk.")