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

_model = None

def get_model():
    global _model
    if _model is None:
        import tensorflow as tf
        _model = tf.keras.models.load_model(MODEL_PATH)
    return _model


def predict(image):
    if image is None:
        return {}
    model = get_model()
    img = Image.fromarray(image).convert("RGB").resize(IMAGE_SIZE, Image.BILINEAR)
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = np.expand_dims(arr, axis=0)
    proba = model.predict(arr, verbose=0)[0]
    return {CLASS_NAMES[i]: float(proba[i]) for i in range(len(CLASS_NAMES))}


with gr.Blocks(theme=gr.themes.Soft(), title="🐾 Animal Species Classifier") as demo:
    gr.Markdown(
        """
        # 🐾 Animal Species Classifier
        Upload a photo of an animal to classify it into one of **15 species**.

        **Classes:** Beetle · Butterfly · Cat · Cow · Dog · Elephant · Gorilla ·
        Hippo · Lizard · Monkey · Mouse · Panda · Spider · Tiger · Zebra

        *Model: VGG-16 Transfer Learning (TensorFlow/Keras)*
        """
    )
    with gr.Row():
        img_input = gr.Image(label="Upload Animal Image", type="numpy")
        label_output = gr.Label(num_top_classes=5, label="Predictions")

    submit_btn = gr.Button("Classify", variant="primary")
    submit_btn.click(fn=predict, inputs=img_input, outputs=label_output)
    img_input.change(fn=predict, inputs=img_input, outputs=label_output)

if __name__ == "__main__":
    demo.launch()
