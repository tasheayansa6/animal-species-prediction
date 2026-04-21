"""
Hugging Face Spaces entry point.
Rename of app_hf.py — HF Spaces requires the main file to be app.py
"""

import os
import numpy as np
from PIL import Image
import tensorflow as tf
import gradio as gr

# ── Config ────────────────────────────────────────────────────────────
CLASS_NAMES = [
    "Beetle", "Butterfly", "Cat", "Cow", "Dog",
    "Elephant", "Gorilla", "Hippo", "Lizard", "Monkey",
    "Mouse", "Panda", "Spider", "Tiger", "Zebra",
]
IMAGE_SIZE = (128, 128)
MODEL_PATH = "models/final/animal_classifier.h5"

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# ── Load model ────────────────────────────────────────────────────────
print("Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)
print("Model ready:", model.input_shape)


# ── Predict function ──────────────────────────────────────────────────
def predict(image: Image.Image):
    img = image.convert("RGB").resize(IMAGE_SIZE, Image.BILINEAR)
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = np.expand_dims(arr, axis=0)
    proba = model.predict(arr, verbose=0)[0]
    return {CLASS_NAMES[i]: float(proba[i]) for i in range(len(CLASS_NAMES))}


# ── Gradio UI ─────────────────────────────────────────────────────────
demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Upload Animal Image"),
    outputs=gr.Label(num_top_classes=5, label="Predictions"),
    title="🐾 Sanyii bine?elda — Animal Species Classifier",
    description=(
        "Upload a photo of an animal and the model will classify it into one of 15 species.\n\n"
        "**Classes:** Beetle · Butterfly · Cat · Cow · Dog · Elephant · Gorilla · "
        "Hippo · Lizard · Monkey · Mouse · Panda · Spider · Tiger · Zebra\n\n"
        "**Model:** VGG-16 Transfer Learning (TensorFlow/Keras)"
    ),
    examples=[],
    theme=gr.themes.Soft(),
    allow_flagging="never",
)

if __name__ == "__main__":
    demo.launch()
