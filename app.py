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

# Lazy-load model on first prediction to avoid startup timeout
_model = None

def get_model():
    global _model
    if _model is None:
        import tensorflow as tf
        _model = tf.keras.models.load_model(MODEL_PATH)
        print("Model loaded:", _model.input_shape)
    return _model


def predict(image: Image.Image):
    if image is None:
        return {}
    model = get_model()
    img = image.convert("RGB").resize(IMAGE_SIZE, Image.BILINEAR)
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = np.expand_dims(arr, axis=0)
    proba = model.predict(arr, verbose=0)[0]
    return {CLASS_NAMES[i]: float(proba[i]) for i in range(len(CLASS_NAMES))}


demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Upload Animal Image"),
    outputs=gr.Label(num_top_classes=5, label="Predictions"),
    title="🐾 Animal Species Classifier",
    description=(
        "Upload a photo of an animal to classify it into one of 15 species.\n\n"
        "**Classes:** Beetle · Butterfly · Cat · Cow · Dog · Elephant · Gorilla · "
        "Hippo · Lizard · Monkey · Mouse · Panda · Spider · Tiger · Zebra\n\n"
        "**Model:** VGG-16 Transfer Learning (TensorFlow/Keras)"
    ),
    theme=gr.themes.Soft(),
    allow_flagging="never",
)

if __name__ == "__main__":
    demo.launch()
