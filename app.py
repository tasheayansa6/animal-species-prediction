import os
import numpy as np
from PIL import Image
import gradio as gr

CLASS_NAMES = [
    "Beetle", "Butterfly", "Cat", "Cow", "Dog",
    "Elephant", "Gorilla", "Hippo", "Lizard", "Monkey",
    "Mouse", "Panda", "Spider", "Tiger", "Zebra",
]
IMAGE_SIZE = (128, 128)
TFLITE_PATH = "models/final/animal_classifier.tflite"

print("Loading TFLite model...")
import tflite_runtime.interpreter as tflite
interpreter = tflite.Interpreter(model_path=TFLITE_PATH)
interpreter.allocate_tensors()
input_details  = interpreter.get_input_details()
output_details = interpreter.get_output_details()
print("Model ready!")


def predict(image: np.ndarray) -> dict:
    if image is None:
        return {c: 0.0 for c in CLASS_NAMES}
    img = Image.fromarray(image.astype("uint8")).convert("RGB").resize(IMAGE_SIZE, Image.BILINEAR)
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = np.expand_dims(arr, axis=0)
    interpreter.set_tensor(input_details[0]["index"], arr)
    interpreter.invoke()
    proba = interpreter.get_tensor(output_details[0]["index"])[0]
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
