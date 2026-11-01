import streamlit as st
import numpy as np
import librosa
import sounddevice as sd
import soundfile as sf
import pickle
from tensorflow.keras.models import model_from_json

# ==================================
# PAGE CONFIG
# ==================================
st.set_page_config(page_title="Voice Emotion Recognition", layout="centered")

# ==================================
# LOAD MODEL & TOOLS
# ==================================
@st.cache_resource
def load_all():
    with open("CNN_model.json", "r") as f:
        model = model_from_json(f.read())
    model.load_weights("CNN_model.weights.h5")
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

    with open("scaler2.pickle", "rb") as f:
        scaler = pickle.load(f)

    with open("encoder2.pickle", "rb") as f:
        encoder = pickle.load(f)

    return model, scaler, encoder

model, scaler, encoder = load_all()

# ==================================
# FEATURE FUNCTIONS
# ==================================
def zcr(data):
    return np.squeeze(librosa.feature.zero_crossing_rate(y=data))

def rmse_feat(data):
    return np.squeeze(librosa.feature.rms(y=data))

def mfcc(data, sr):
    return np.ravel(librosa.feature.mfcc(y=data, sr=sr).T)

def extract_features(data, sr):
    return np.hstack((zcr(data), rmse_feat(data), mfcc(data, sr)))

def get_predict_feat(path):
    data, sr = librosa.load(path, duration=2.5, offset=0.6)
    features = extract_features(data, sr)

    if len(features) > 2376:
        features = features[:2376]
    else:
        features = np.pad(features, (0, 2376 - len(features)))

    features = scaler.transform(features.reshape(1, -1))
    return np.expand_dims(features, axis=2)

# ==================================
# RECORDING FUNCTIONS
# ==================================
def start_recording():
    st.session_state.audio = sd.rec(
        int(3 * 22050), samplerate=22050, channels=1
    )

def stop_recording():
    sd.wait()
    sf.write("recorded.wav", st.session_state.audio, 22050)

# ==================================
# SESSION STATE INIT
# ==================================
if "recording" not in st.session_state:
    st.session_state.recording = False

# ==================================
# UI
# ==================================
st.title("🎧 Voice Emotion Recognition")
st.markdown("### Read the sentence aloud and control recording")

sentence = "I am feeling very excited about my future and my life."
st.info(sentence)

# 🎛️ CONTROL BUTTONS
col1, col2 = st.columns(2)

with col1:
    if st.button("▶ Start Recording"):
        st.session_state.recording = True
        start_recording()
        st.warning("🎙️ Recording... Speak now")

with col2:
    if st.button("⏹ Stop & Predict"):
        if st.session_state.recording:
            st.session_state.recording = False
            stop_recording()

            # 🎧 Audio Player
            st.audio("recorded.wav")

            # 🧠 Prediction
            feat = get_predict_feat("recorded.wav")
            pred = model.predict(feat)
            emotion = encoder.inverse_transform(pred)[0][0]

            st.success(f"🧠 Predicted Emotion: **{emotion.upper()}**")
        else:
            st.error("❗ Please start recording first")
