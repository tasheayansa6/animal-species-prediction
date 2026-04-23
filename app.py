import os
import threading
import numpy as np
from PIL import Image
import gradio as gr

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

CLASS_NAMES = [
    "Beetle", "Butterfly", "Cat", "Cow", "Dog",
    "Elephant", "Gorilla", "Hippo", "Lizard", "Monkey",
    "Mouse", "Panda", "Spider", "Tiger", "Zebra",
]
IMAGE_SIZE = (128, 128)
MODEL_PATH = "models/final/animal_classifier.h5"

_model = None
_model_ready = threading.Event()

def load_model_background():
    global _model
    import tensorflow as tf
    print("Loading model in background...")
    _model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    _model_ready.set()
    print("Model ready.")

# Start loading immediately in background thread
threading.Thread(target=load_model_background, daemon=True).start()


def predict(image: np.ndarray) -> dict:
    if not _model_ready.is_set():
        # Still loading — return a friendly message via label
        return {"Model is loading, please wait 30s and try again...": 1.0}
    if image is None:
        return {c: 0.0 for c in CLASS_NAMES}
    img = Image.fromarray(image.astype("uint8")).convert("RGB").resize(IMAGE_SIZE, Image.BILINEAR)
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = np.expand_dims(arr, axis=0)
    proba = _model.predict(arr, verbose=0)[0]
    return {CLASS_NAMES[i]: float(proba[i]) for i in range(len(CLASS_NAMES))}


demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="numpy", label="Upload Animal Image"),
    outputs=gr.Label(num_top_classes=5, label="Predictions"),
    title="🐾 Animal Species Classifier",
    description=(
        "Upload a photo of an animal to classify it into one of 15 species.\n\n"
        "**Classes:** Beetle · Butterfly · Cat · Cow · Dog · Elephant · Gorilla · "
        "Hippo · Lizard · Monkey · Mouse · Panda · Spider · Tiger · Zebra"
    ),
    flagging_mode="never",
    api_name="predict",
)

demo.launch()
