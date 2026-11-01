# 4th





# 3rd
import streamlit as st
import numpy as np
import cv2
from PIL import Image
import time

# -----------------------------
# Load Model (TensorFlow/Keras)
# -----------------------------
import tensorflow as tf
from tensorflow.keras.models import load_model

MODEL_PATH = "Resnet_model_version_2.keras"
EMOTIONS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

FACE_CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

st.set_page_config(page_title="Live Emotion Capture", page_icon="😊", layout="centered")
st.title("😊 Live Webcam → Capture → Emotion Prediction")

@st.cache_resource
def load_emotion_model():
    return load_model(MODEL_PATH, compile=False)

model = load_emotion_model()

def preprocess_for_model(face_bgr, model_input_shape):
    h = model_input_shape[1]
    w = model_input_shape[2]
    c = model_input_shape[3]

    if c == 1:
        face = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
        face = cv2.resize(face, (w, h))
        face = face.astype("float32") / 255.0
        face = np.expand_dims(face, axis=-1)
    else:
        face = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
        face = cv2.resize(face, (w, h))
        face = face.astype("float32") / 255.0

    face = np.expand_dims(face, axis=0)
    return face

def predict_emotion_from_image(image_bgr):
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    faces = FACE_CASCADE.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)

    if len(faces) == 0:
        return None, None, image_bgr, "No face detected. Please capture again with clear face."

    x, y, w, h = sorted(faces, key=lambda f: f[2]*f[3], reverse=True)[0]
    pad = int(0.10 * max(w, h))
    x1 = max(0, x - pad)
    y1 = max(0, y - pad)
    x2 = min(image_bgr.shape[1], x + w + pad)
    y2 = min(image_bgr.shape[0], y + h + pad)

    face_roi = image_bgr[y1:y2, x1:x2]
    inp = preprocess_for_model(face_roi, model.input_shape)

    preds = model.predict(inp, verbose=0)[0]
    idx = int(np.argmax(preds))
    label = EMOTIONS[idx] if idx < len(EMOTIONS) else f"Class_{idx}"

    out = image_bgr.copy()
    cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(out, f"{label} ({preds[idx]*100:.1f}%)",
                (x1, max(25, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    return label, preds, out, None


# =============================
# ✅ LIVE MODE (FIXED - NO AUTO PAUSE + 20s SUMMARY)
# =============================
LIVE_OK = True
try:
    from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
    import av
except Exception:
    LIVE_OK = False

import threading
from collections import deque
import time

st.write("---")
st.subheader("🎥 Live Emotion (20s Summary)")
st.caption("Live overlay runs smooth. Stop video → shows last 20 seconds summary (most used emotion).")

WINDOW_SECONDS = 20.0
INFER_EVERY_SEC = 0.15  # ✅ run model ~6-7 times/sec (prevents pause)

@st.cache_resource
def get_live_store():
    return {
        "lock": threading.Lock(),
        "items": deque(),      # (timestamp, pred_idx)
        "probs": deque(),      # (timestamp, prob_vector)
        "face_frames": 0,
        "total_frames": 0,
    }

store = get_live_store()

def reset_store():
    with store["lock"]:
        store["items"].clear()
        store["probs"].clear()
        store["face_frames"] = 0
        store["total_frames"] = 0

st.button("🔁 Reset 20s Buffer", on_click=reset_store)

def trim_store(now_ts):
    # keep only last 20 seconds
    while store["items"] and (now_ts - store["items"][0][0] > WINDOW_SECONDS):
        store["items"].popleft()
    while store["probs"] and (now_ts - store["probs"][0][0] > WINDOW_SECONDS):
        store["probs"].popleft()

if not LIVE_OK:
    st.warning("Live mode requires: pip install streamlit-webrtc av")
else:
    class EmotionLiveProcessor(VideoProcessorBase):
        def __init__(self):
            self.last_infer_time = 0.0
            self.last_label = ""
            self.last_conf = 0.0
            self.last_box = None  # (x1,y1,x2,y2)

        def recv(self, frame):
            img = frame.to_ndarray(format="bgr24")
            now = time.monotonic()
            now_ts = time.time()

            # count total frames (safe)
            with store["lock"]:
                store["total_frames"] += 1

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = FACE_CASCADE.detectMultiScale(gray, 1.2, 5)

            # If face exists, draw box using last known results (fast)
            if len(faces) > 0:
                x, y, w, h = sorted(faces, key=lambda f: f[2]*f[3], reverse=True)[0]
                pad = int(0.10 * max(w, h))
                x1 = max(0, x - pad)
                y1 = max(0, y - pad)
                x2 = min(img.shape[1], x + w + pad)
                y2 = min(img.shape[0], y + h + pad)

                # ✅ Run inference only every INFER_EVERY_SEC seconds
                if (now - self.last_infer_time) >= INFER_EVERY_SEC:
                    face_roi = img[y1:y2, x1:x2]
                    inp = preprocess_for_model(face_roi, model.input_shape)

                    preds = model.predict(inp, verbose=0)[0]
                    preds = np.array(preds, dtype=np.float32)
                    idx = int(np.argmax(preds))
                    conf = float(preds[idx])
                    label = EMOTIONS[idx] if idx < len(EMOTIONS) else f"Class_{idx}"

                    self.last_infer_time = now
                    self.last_label = label
                    self.last_conf = conf
                    self.last_box = (x1, y1, x2, y2)

                    # ✅ store ONLY in shared buffer (thread-safe)
                    with store["lock"]:
                        store["items"].append((now_ts, idx))
                        store["probs"].append((now_ts, preds))
                        store["face_frames"] += 1
                        trim_store(now_ts)

                # Draw overlay using last prediction (even if not infer this frame)
                if self.last_box is not None:
                    bx1, by1, bx2, by2 = self.last_box
                    cv2.rectangle(img, (bx1, by1), (bx2, by2), (0, 255, 0), 2)
                    if self.last_label:
                        cv2.putText(
                            img,
                            f"{self.last_label} ({self.last_conf*100:.1f}%)",
                            (bx1, max(25, by1 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8,
                            (0, 255, 0),
                            2
                        )
            else:
                cv2.putText(img, "No face detected", (15, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

            return av.VideoFrame.from_ndarray(img, format="bgr24")

    with st.expander("✅ Live Video Mode (Start/Stop)", expanded=True):
        ctx = webrtc_streamer(
            key="emotion-live",
            mode=WebRtcMode.SENDRECV,
            video_processor_factory=EmotionLiveProcessor,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )

        # Show live stats
        with store["lock"]:
            st.write(f"Total frames: **{store['total_frames']}** | Face frames stored (≈inference frames): **{store['face_frames']}**")
            current_len = len(store["items"])
        st.write(f"Buffer (last 20s): **{current_len}** predictions")

        # ✅ On STOP: summary from buffer (last 20 seconds)
        if not ctx.state.playing:
            with store["lock"]:
                if len(store["items"]) > 0:
                    idxs = [i for (_, i) in store["items"]]
                    counts = np.zeros(len(EMOTIONS), dtype=int)
                    for i in idxs:
                        if 0 <= i < len(EMOTIONS):
                            counts[i] += 1

                    final_idx = int(np.argmax(counts))
                    final_label = EMOTIONS[final_idx]
                    total_used = int(np.sum(counts))

                    st.success(f"🧾 20s Summary (Most Used Emotion): **{final_label}**")
                    st.write(f"Predictions used (last 20s): **{total_used}**")

                    st.write("### 📊 Emotion Counts (last 20s)")
                    st.bar_chart({EMOTIONS[i]: int(counts[i]) for i in range(len(EMOTIONS))})

                else:
                    st.info("No predictions stored. Start live and keep your face visible, then stop.")


# -----------------------------
# ✅ CAMERA CAPTURE (your original)
# -----------------------------
st.write("---")
st.write("### 📷 Step 1: Camera ON & Capture")
captured = st.camera_input("Open camera and capture your photo")

st.write("---")
st.write("### 🔍 Step 2: Predict Emotion (after capture)")

if captured is not None:
    pil_img = Image.open(captured).convert("RGB")
    img_rgb = np.array(pil_img)
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

    label, probs, boxed_bgr, err = predict_emotion_from_image(img_bgr)

    if err:
        st.error(err)
        st.image(img_rgb, caption="Captured Image", use_container_width=True)
    else:
        boxed_rgb = cv2.cvtColor(boxed_bgr, cv2.COLOR_BGR2RGB)
        st.image(boxed_rgb, caption="Face Detected + Predicted Emotion", use_container_width=True)

        st.success(f"✅ Predicted Emotion: **{label}**")

        st.write("### 📊 Confidence Scores")
        prob_dict = {EMOTIONS[i]: float(probs[i]) for i in range(min(len(EMOTIONS), len(probs)))}
        st.bar_chart(prob_dict)
else:
    st.info("👆 First capture an image from the webcam, then prediction will appear here.")


# 2nd
# import streamlit as st
# import numpy as np
# import cv2
# from PIL import Image

# # -----------------------------
# # Load Model (TensorFlow/Keras)
# # -----------------------------
# import tensorflow as tf
# from tensorflow.keras.models import load_model

# MODEL_PATH = "Resnet_model_version_2.keras"
# EMOTIONS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# # Haarcascade (no xml file download needed)
# FACE_CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# st.set_page_config(page_title="Live Emotion Capture", page_icon="😊", layout="centered")
# st.title("😊 Live Webcam → Capture → Emotion Prediction")

# @st.cache_resource
# def load_emotion_model():
#     model = load_model(MODEL_PATH, compile=False)
#     return model

# model = load_emotion_model()

# def preprocess_for_model(face_bgr, model_input_shape):
#     """
#     Adapts preprocessing based on model input shape.
#     Works for common shapes like (None, 224, 224, 3) or (None, 48, 48, 1).
#     """
#     # model_input_shape: (None, H, W, C)
#     h = model_input_shape[1]
#     w = model_input_shape[2]
#     c = model_input_shape[3]

#     if c == 1:
#         face = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
#         face = cv2.resize(face, (w, h))
#         face = face.astype("float32") / 255.0
#         face = np.expand_dims(face, axis=-1)  # (H,W,1)
#     else:
#         face = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
#         face = cv2.resize(face, (w, h))
#         face = face.astype("float32") / 255.0

#     face = np.expand_dims(face, axis=0)  # (1,H,W,C)
#     return face

# def predict_emotion_from_image(image_bgr):
#     # detect faces
#     gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
#     faces = FACE_CASCADE.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)

#     if len(faces) == 0:
#         return None, None, image_bgr, "No face detected. Please capture again with clear face."

#     # pick largest face
#     x, y, w, h = sorted(faces, key=lambda f: f[2]*f[3], reverse=True)[0]
#     pad = int(0.10 * max(w, h))
#     x1 = max(0, x - pad)
#     y1 = max(0, y - pad)
#     x2 = min(image_bgr.shape[1], x + w + pad)
#     y2 = min(image_bgr.shape[0], y + h + pad)

#     face_roi = image_bgr[y1:y2, x1:x2]

#     # preprocess
#     inp = preprocess_for_model(face_roi, model.input_shape)

#     # predict
#     preds = model.predict(inp, verbose=0)[0]  # (num_classes,)
#     idx = int(np.argmax(preds))
#     label = EMOTIONS[idx] if idx < len(EMOTIONS) else f"Class_{idx}"

#     # draw box
#     out = image_bgr.copy()
#     cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
#     cv2.putText(out, f"{label} ({preds[idx]*100:.1f}%)", (x1, max(25, y1 - 10)),
#                 cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

#     return label, preds, out, None

# # =============================
# # ✅ EXTRA FEATURE: LIVE MODE (Optional)
# # =============================
# LIVE_OK = True
# try:
#     from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
#     import av
# except Exception:
#     LIVE_OK = False

# st.write("---")
# st.subheader("🎥 Live Emotion (Optional)")
# st.caption("Start live video → live emotion overlay. Stop → session summary emotion shown.")

# if not LIVE_OK:
#     st.warning("Live mode requires: pip install streamlit-webrtc av")
# else:
#     # Session state to store last run summary after STOP
#     if "last_live_summary" not in st.session_state:
#         st.session_state.last_live_summary = None

#     class EmotionLiveProcessor(VideoProcessorBase):
#         def __init__(self):
#             self.pred_history = []   # store predicted class index each frame
#             self.conf_history = []   # store confidence each frame

#         def recv(self, frame):
#             img = frame.to_ndarray(format="bgr24")

#             # --- face detect (same cascade) ---
#             gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#             faces = FACE_CASCADE.detectMultiScale(gray, 1.2, 5)

#             if len(faces) > 0:
#                 # largest face
#                 x, y, w, h = sorted(faces, key=lambda f: f[2]*f[3], reverse=True)[0]
#                 pad = int(0.10 * max(w, h))
#                 x1 = max(0, x - pad)
#                 y1 = max(0, y - pad)
#                 x2 = min(img.shape[1], x + w + pad)
#                 y2 = min(img.shape[0], y + h + pad)

#                 face_roi = img[y1:y2, x1:x2]

#                 # --- preprocess + predict (same logic) ---
#                 inp = preprocess_for_model(face_roi, model.input_shape)
#                 preds = model.predict(inp, verbose=0)[0]
#                 idx = int(np.argmax(preds))
#                 conf = float(preds[idx])

#                 # store for STOP summary
#                 self.pred_history.append(idx)
#                 self.conf_history.append(conf)

#                 # --- draw overlay ---
#                 label = EMOTIONS[idx] if idx < len(EMOTIONS) else f"Class_{idx}"
#                 cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
#                 cv2.putText(
#                     img,
#                     f"{label} ({conf*100:.1f}%)",
#                     (x1, max(25, y1 - 10)),
#                     cv2.FONT_HERSHEY_SIMPLEX,
#                     0.8,
#                     (0, 255, 0),
#                     2
#                 )
#             else:
#                 # optional hint on screen
#                 cv2.putText(
#                     img,
#                     "No face detected",
#                     (15, 30),
#                     cv2.FONT_HERSHEY_SIMPLEX,
#                     0.9,
#                     (0, 0, 255),
#                     2
#                 )

#             return av.VideoFrame.from_ndarray(img, format="bgr24")

#     with st.expander("✅ Live Video Mode (Start/Stop)", expanded=True):
#         ctx = webrtc_streamer(
#             key="emotion-live",
#             mode=WebRtcMode.SENDRECV,
#             video_processor_factory=EmotionLiveProcessor,
#             media_stream_constraints={"video": True, "audio": False},
#             async_processing=True,
#         )

#         # When stopped, compute summary from processor history
#         if ctx.state.playing:
#             st.info("Live running... Stop ചെയ്‌താൽ summary കാണിക്കും.")
#         else:
#             # If we have a processor instance from a previous run, summarize
#             proc = ctx.video_processor
#             if proc is not None and hasattr(proc, "pred_history") and len(proc.pred_history) > 0:
#                 counts = np.zeros(len(EMOTIONS), dtype=int)
#                 for i in proc.pred_history:
#                     if 0 <= i < len(EMOTIONS):
#                         counts[i] += 1

#                 # majority vote
#                 final_idx = int(np.argmax(counts))
#                 final_label = EMOTIONS[final_idx]
#                 total = int(np.sum(counts))

#                 # store in session (so it stays visible)
#                 st.session_state.last_live_summary = {
#                     "final_label": final_label,
#                     "counts": counts.tolist(),
#                     "total": total
#                 }

#             # show stored summary (if available)
#             if st.session_state.last_live_summary is not None:
#                 s = st.session_state.last_live_summary
#                 st.success(f"🧾 Live Session Final Emotion (majority): **{s['final_label']}**")
#                 st.write(f"Frames analyzed: **{s['total']}**")

#                 prob_dict = {EMOTIONS[i]: float(s["counts"][i]) for i in range(len(EMOTIONS))}
#                 st.write("### 📊 Live Session Frame Counts")
#                 st.bar_chart(prob_dict)
#             else:
#                 st.info("Start live video → stop to get summary.")

# # -----------------------------
# # 1) Camera ON + Capture
# # -----------------------------
# st.write("### 📷 Step 1: Camera ON & Capture")
# captured = st.camera_input("Open camera and capture your photo")

# st.write("---")
# st.write("### 🔍 Step 2: Predict Emotion (after capture)")

# if captured is not None:
#     # bytes -> PIL -> OpenCV BGR
#     pil_img = Image.open(captured).convert("RGB")
#     img_rgb = np.array(pil_img)
#     img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

#     label, probs, boxed_bgr, err = predict_emotion_from_image(img_bgr)

#     if err:
#         st.error(err)
#         st.image(img_rgb, caption="Captured Image", use_container_width=True)
#     else:
#         boxed_rgb = cv2.cvtColor(boxed_bgr, cv2.COLOR_BGR2RGB)
#         st.image(boxed_rgb, caption="Face Detected + Predicted Emotion", use_container_width=True)

#         st.success(f"✅ Predicted Emotion: **{label}**")

#         # show confidence chart
#         st.write("### 📊 Confidence Scores")
#         prob_dict = {EMOTIONS[i]: float(probs[i]) for i in range(min(len(EMOTIONS), len(probs)))}
#         st.bar_chart(prob_dict)

# else:
#     st.info("👆 First capture an image from the webcam, then prediction will appear here.")




# 1st
# import streamlit as st
# import numpy as np
# import cv2
# from PIL import Image

# # -----------------------------
# # Load Model (TensorFlow/Keras)
# # -----------------------------
# import tensorflow as tf
# from tensorflow.keras.models import load_model

# MODEL_PATH = "Resnet_model_version_2.keras"
# EMOTIONS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# # Haarcascade (no xml file download needed)
# FACE_CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# st.set_page_config(page_title="Live Emotion Capture", page_icon="😊", layout="centered")
# st.title("😊 Live Webcam → Capture → Emotion Prediction")

# @st.cache_resource
# def load_emotion_model():
#     model = load_model(MODEL_PATH, compile=False)
#     return model

# model = load_emotion_model()

# def preprocess_for_model(face_bgr, model_input_shape):
#     """
#     Adapts preprocessing based on model input shape.
#     Works for common shapes like (None, 224, 224, 3) or (None, 48, 48, 1).
#     """
#     # model_input_shape: (None, H, W, C)
#     h = model_input_shape[1]
#     w = model_input_shape[2]
#     c = model_input_shape[3]

#     if c == 1:
#         face = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
#         face = cv2.resize(face, (w, h))
#         face = face.astype("float32") / 255.0
#         face = np.expand_dims(face, axis=-1)  # (H,W,1)
#     else:
#         face = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
#         face = cv2.resize(face, (w, h))
#         face = face.astype("float32") / 255.0

#     face = np.expand_dims(face, axis=0)  # (1,H,W,C)
#     return face

# def predict_emotion_from_image(image_bgr):
#     # detect faces
#     gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
#     faces = FACE_CASCADE.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)

#     if len(faces) == 0:
#         return None, None, image_bgr, "No face detected. Please capture again with clear face."

#     # pick largest face
#     x, y, w, h = sorted(faces, key=lambda f: f[2]*f[3], reverse=True)[0]
#     pad = int(0.10 * max(w, h))
#     x1 = max(0, x - pad)
#     y1 = max(0, y - pad)
#     x2 = min(image_bgr.shape[1], x + w + pad)
#     y2 = min(image_bgr.shape[0], y + h + pad)

#     face_roi = image_bgr[y1:y2, x1:x2]

#     # preprocess
#     inp = preprocess_for_model(face_roi, model.input_shape)

#     # predict
#     preds = model.predict(inp, verbose=0)[0]  # (num_classes,)
#     idx = int(np.argmax(preds))
#     label = EMOTIONS[idx] if idx < len(EMOTIONS) else f"Class_{idx}"

#     # draw box
#     out = image_bgr.copy()
#     cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
#     cv2.putText(out, f"{label} ({preds[idx]*100:.1f}%)", (x1, max(25, y1 - 10)),
#                 cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

#     return label, preds, out, None


# # -----------------------------
# # 1) Camera ON + Capture
# # -----------------------------
# st.write("### 📷 Step 1: Camera ON & Capture")
# captured = st.camera_input("Open camera and capture your photo")

# st.write("---")
# st.write("### 🔍 Step 2: Predict Emotion (after capture)")

# if captured is not None:
#     # bytes -> PIL -> OpenCV BGR
#     pil_img = Image.open(captured).convert("RGB")
#     img_rgb = np.array(pil_img)
#     img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

#     label, probs, boxed_bgr, err = predict_emotion_from_image(img_bgr)

#     if err:
#         st.error(err)
#         st.image(img_rgb, caption="Captured Image", use_container_width=True)
#     else:
#         boxed_rgb = cv2.cvtColor(boxed_bgr, cv2.COLOR_BGR2RGB)
#         st.image(boxed_rgb, caption="Face Detected + Predicted Emotion", use_container_width=True)

#         st.success(f"✅ Predicted Emotion: **{label}**")

#         # show confidence chart
#         st.write("### 📊 Confidence Scores")
#         prob_dict = {EMOTIONS[i]: float(probs[i]) for i in range(min(len(EMOTIONS), len(probs)))}
#         st.bar_chart(prob_dict)

# else:
#     st.info("👆 First capture an image from the webcam, then prediction will appear here.")
