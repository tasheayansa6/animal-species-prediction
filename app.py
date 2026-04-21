import os
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

# ── Load model at startup + warmup so first prediction is fast ────────
import tensorflow as tf

print("Loading model...")
model = tf.keras.models.load_model(MODEL_PATH, compile=False)

# Warmup: run one dummy prediction to compile the graph
_dummy = np.zeros((1, IMAGE_SIZE[0], IMAGE_SIZE[1], 3), dtype=np.float32)
model.predict(_dummy, verbose=0)
print("Model ready and warmed up.")


def predict(image: np.ndarray) -> dict:
    if image is None:
        return {c: 0.0 for c in CLASS_NAMES}
    img = Image.fromarray(image.astype("uint8")).convert("RGB").resize(IMAGE_SIZE, Image.BILINEAR)
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = np.expand_dims(arr, axis=0)
    proba = model.predict(arr, verbose=0)[0]
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
    theme=gr.themes.Soft(),
    allow_flagging="never",
    api_name="predict",
)

demo.launch()
