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
try:
    from ai_edge_litert.interpreter import Interpreter
    interpreter = Interpreter(model_path=TFLITE_PATH)
except Exception:
    import tensorflow as tf
    interpreter = tf.lite.Interpreter(model_path=TFLITE_PATH)

interpreter.allocate_tensors()
input_details  = interpreter.get_input_details()
output_details = interpreter.get_output_details()
print("Model ready!")

ANIMAL_EMOJI = {
    "Beetle": "🪲", "Butterfly": "🦋", "Cat": "🐱", "Cow": "🐄",
    "Dog": "🐶", "Elephant": "🐘", "Gorilla": "🦍", "Hippo": "🦛",
    "Lizard": "🦎", "Monkey": "🐒", "Mouse": "🐭", "Panda": "🐼",
    "Spider": "🕷️", "Tiger": "🐯", "Zebra": "🦓",
}

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

* { font-family: 'Inter', sans-serif !important; }

body, .gradio-container {
    background: #0f0f13 !important;
    color: #e2e8f0 !important;
}

/* Header */
.app-header {
    text-align: center;
    padding: 2.5rem 1rem 1.5rem;
    background: linear-gradient(135deg, #1e1b4b 0%, #0f0f13 60%);
    border-bottom: 1px solid #27272a;
    margin-bottom: 1.5rem;
}
.app-title {
    font-size: 2.2rem;
    font-weight: 700;
    background: linear-gradient(135deg, #a78bfa, #60a5fa, #34d399);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.4rem;
}
.app-subtitle {
    color: #71717a;
    font-size: 0.9rem;
    letter-spacing: 0.05em;
}

/* Cards */
.card {
    background: #18181b !important;
    border: 1px solid #27272a !important;
    border-radius: 12px !important;
    padding: 1.25rem !important;
}

/* Upload area */
.upload-card { border: 2px dashed #3f3f46 !important; }
.upload-card:hover { border-color: #7c3aed !important; }

/* Buttons */
button.primary-btn, #classify-btn {
    background: linear-gradient(135deg, #7c3aed, #2563eb) !important;
    border: none !important;
    border-radius: 8px !important;
    color: white !important;
    font-weight: 600 !important;
    font-size: 1rem !important;
    padding: 0.75rem 2rem !important;
    cursor: pointer !important;
    transition: opacity 0.2s !important;
    width: 100% !important;
}
button.primary-btn:hover { opacity: 0.88 !important; }

/* Result box */
.result-box {
    background: linear-gradient(135deg, #1e1b4b22, #1e3a5f22) !important;
    border: 1px solid #3730a3 !important;
    border-radius: 12px !important;
    padding: 1.5rem !important;
    text-align: center !important;
}
.result-animal {
    font-size: 3rem;
    margin-bottom: 0.25rem;
}
.result-name {
    font-size: 1.6rem;
    font-weight: 700;
    color: #a78bfa;
}
.result-conf {
    color: #71717a;
    font-size: 0.85rem;
    margin-top: 0.25rem;
}

/* Progress bars */
.bar-container { margin-bottom: 0.6rem; }
.bar-label {
    display: flex;
    justify-content: space-between;
    font-size: 0.82rem;
    margin-bottom: 0.2rem;
    color: #a1a1aa;
}
.bar-track {
    background: #27272a;
    border-radius: 999px;
    height: 8px;
    overflow: hidden;
}
.bar-fill {
    height: 100%;
    border-radius: 999px;
    background: linear-gradient(90deg, #7c3aed, #2563eb);
    transition: width 0.6s ease;
}

/* Badge pills */
.badge {
    display: inline-block;
    background: #27272a;
    border-radius: 999px;
    padding: 0.2rem 0.7rem;
    font-size: 0.75rem;
    color: #a1a1aa;
    margin: 0.15rem;
}

/* Hide gradio default label output */
.gradio-label { display: none !important; }

/* Gradio image component */
.gradio-image { border-radius: 10px !important; overflow: hidden !important; }

/* Footer */
.app-footer {
    text-align: center;
    padding: 1.5rem;
    color: #3f3f46;
    font-size: 0.78rem;
    border-top: 1px solid #18181b;
    margin-top: 2rem;
}
"""

def predict(image: np.ndarray):
    if image is None:
        return build_ui("—", 0, [])

    img = Image.fromarray(image.astype("uint8")).convert("RGB").resize(IMAGE_SIZE, Image.BILINEAR)
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = np.expand_dims(arr, axis=0)
    interpreter.set_tensor(input_details[0]["index"], arr)
    interpreter.invoke()
    proba = interpreter.get_tensor(output_details[0]["index"])[0]

    ranked = sorted(enumerate(proba), key=lambda x: x[1], reverse=True)
    top_idx, top_conf = ranked[0]
    top_name = CLASS_NAMES[top_idx]
    top5 = [(CLASS_NAMES[i], float(p)) for i, p in ranked[:5]]

    return build_ui(top_name, float(top_conf), top5)


def build_ui(name, conf, top5):
    emoji = ANIMAL_EMOJI.get(name, "🐾")
    conf_pct = f"{conf*100:.1f}%"

    # Top result card
    result_html = f"""
    <div class="result-box">
        <div class="result-animal">{emoji}</div>
        <div class="result-name">{name}</div>
        <div class="result-conf">Confidence: {conf_pct}</div>
    </div>
    """ if name != "—" else """
    <div class="result-box">
        <div class="result-animal">📷</div>
        <div class="result-name" style="color:#52525b">Upload an image</div>
        <div class="result-conf">to get predictions</div>
    </div>
    """

    # Top-5 bars
    bars_html = ""
    if top5:
        medals = ["🥇", "🥈", "🥉", "4️⃣", "5️⃣"]
        for i, (cls, prob) in enumerate(top5):
            pct = prob * 100
            bars_html += f"""
            <div class="bar-container">
                <div class="bar-label">
                    <span>{medals[i]} {ANIMAL_EMOJI.get(cls,'')} {cls}</span>
                    <span style="color:#a78bfa;font-weight:600">{pct:.1f}%</span>
                </div>
                <div class="bar-track">
                    <div class="bar-fill" style="width:{pct:.1f}%"></div>
                </div>
            </div>
            """

    bars_section = f"""
    <div class="card" style="margin-top:1rem">
        <div style="font-size:0.8rem;font-weight:600;color:#71717a;letter-spacing:0.08em;margin-bottom:1rem">
            TOP PREDICTIONS
        </div>
        {bars_html}
    </div>
    """ if bars_html else ""

    return result_html + bars_section


with gr.Blocks(css=CSS, title="Sanyii Bineeldaa") as demo:

    # Header
    gr.HTML("""
    <div class="app-header">
        <div class="app-title">🐾 Sanyii Bineeldaa</div>
        <div class="app-subtitle">VGG-16 Transfer Learning &nbsp;·&nbsp; 15 Species &nbsp;·&nbsp; TFLite</div>
    </div>
    """)

    with gr.Row():
        # Left — upload
        with gr.Column(scale=1):
            gr.HTML('<div style="font-size:0.8rem;font-weight:600;color:#71717a;letter-spacing:0.08em;margin-bottom:0.5rem">UPLOAD IMAGE</div>')
            img_input = gr.Image(
                type="numpy",
                label="",
                elem_classes=["upload-card"],
                height=300,
            )
            classify_btn = gr.Button("✨ Classify Animal", elem_id="classify-btn")

            # Species pills
            gr.HTML("""
            <div style="margin-top:1rem">
                <div style="font-size:0.75rem;color:#52525b;margin-bottom:0.4rem">RECOGNISES</div>
                <div>
                    🪲 Beetle &nbsp; 🦋 Butterfly &nbsp; 🐱 Cat &nbsp; 🐄 Cow &nbsp; 🐶 Dog<br>
                    🐘 Elephant &nbsp; 🦍 Gorilla &nbsp; 🦛 Hippo &nbsp; 🦎 Lizard &nbsp; 🐒 Monkey<br>
                    🐭 Mouse &nbsp; 🐼 Panda &nbsp; 🕷️ Spider &nbsp; 🐯 Tiger &nbsp; 🦓 Zebra
                </div>
            </div>
            """)

        # Right — results
        with gr.Column(scale=1):
            gr.HTML('<div style="font-size:0.8rem;font-weight:600;color:#71717a;letter-spacing:0.08em;margin-bottom:0.5rem">RESULTS</div>')
            output_html = gr.HTML(build_ui("—", 0, []))

    classify_btn.click(fn=predict, inputs=img_input, outputs=output_html)
    img_input.change(fn=predict, inputs=img_input, outputs=output_html)

    gr.HTML('<div class="app-footer">Built with Gradio · TFLite · TensorFlow/Keras</div>')

demo.launch()
