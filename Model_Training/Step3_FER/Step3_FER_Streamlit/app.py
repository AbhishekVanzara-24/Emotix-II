import gradio as gr
import tensorflow as tf
import numpy as np
import cv2
import os
import inspect
import keras

# --- MONKEYPATCHES START ---

# 1. Fix for Keras/TF 2.20+ compatibility on Python 3.13
if not hasattr(tf.__internal__, "register_load_context_function"):
    tf.__internal__.register_load_context_function = lambda x: None

# 2. Universal Layer Monkeypatch for Keras 3 / Python 3.13
# Fixes list-wrapped inputs and inspect._empty leak in kwargs
from keras import layers
original_layer_call = layers.Layer.__call__

def patched_layer_call(self, *args, **kwargs):
    new_args = list(args)
    if new_args and isinstance(new_args[0], list) and len(new_args[0]) == 1:
        item = new_args[0][0]
        if hasattr(item, 'shape'):
            new_args[0] = item
    
    if 'inputs' in kwargs and isinstance(kwargs['inputs'], list) and len(kwargs['inputs']) == 1:
        item = kwargs['inputs'][0]
        if hasattr(item, 'shape'):
            kwargs['inputs'] = item
    
    if not isinstance(kwargs, dict):
        kwargs = {}
    else:
        for k, v in list(kwargs.items()):
            if v is inspect._empty:
                kwargs[k] = None
    
    return original_layer_call(self, *tuple(new_args), **kwargs)

layers.Layer.__call__ = patched_layer_call

# --- MONKEYPATCHES END ---

# ===============================
# CONFIGURATION
# ===============================
MODEL_PATH = "Resnet_model_version_2.keras"
CASCADE_PATH = "haarcascade_frontalface_default.xml"
EMOTIONS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
IMG_SIZE = 224

# ===============================
# LOAD MODEL & CASCADE
# ===============================
print("Loading model...")
model = keras.models.load_model(MODEL_PATH, compile=False)
face_cascade = cv2.CascadeClassifier(CASCADE_PATH)

# ===============================
# IMAGE PROCESSING
# ===============================
def preprocess_face(face_roi):
    face = cv2.resize(face_roi, (IMG_SIZE, IMG_SIZE))
    face = face.astype("float32") / 255.0
    face = np.expand_dims(face, axis=0)
    return face

def predict_emotion(image):
    if image is None:
        return None, "No image provided"
    
    image_np = np.array(image)
    if len(image_np.shape) == 2:
        image_np = cv2.cvtColor(image_np, cv2.COLOR_GRAY2RGB)
    elif image_np.shape[2] == 4:
        image_np = cv2.cvtColor(image_np, cv2.COLOR_RGBA2RGB)
    
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    
    label = "No face detected"
    if len(faces) == 0:
        return image_np, label
    
    for (x, y, w, h) in faces:
        roi_color = image_np[y:y+h, x:x+w]
        face_input = preprocess_face(roi_color)
        preds = model.predict(face_input, verbose=0)[0]
        label = EMOTIONS[np.argmax(preds)]
        
        cv2.rectangle(image_np, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(image_np, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
    return image_np, label

# ===============================
# GRADIO UI
# ===============================
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 😄 Facial Emotion AI")
    gr.Markdown("Upload a photo or use your webcam to detect emotions in real-time.")

    with gr.Tabs():
        with gr.Tab("📸 Image Upload"):
            with gr.Row():
                input_img = gr.Image(type="pil", label="Input Image")
                output_img = gr.Image(label="Detection Result")
            output_label = gr.Textbox(label="Main Emotion Detected")
            btn = gr.Button("Analyze Image")
            btn.click(fn=predict_emotion, inputs=input_img, outputs=[output_img, output_label])

        with gr.Tab("🎥 Webcam Live"):
            cam_input = gr.Image(sources=["webcam"], streaming=True, label="Webcam Stream")
            cam_output = gr.Image(label="Live Detection")
            
            def stream_prediction(frame):
                if frame is None:
                    return None
                processed_frame, _ = predict_emotion(frame)
                return processed_frame
            
            cam_input.stream(fn=stream_prediction, inputs=cam_input, outputs=cam_output)

if __name__ == "__main__":
    demo.launch()
