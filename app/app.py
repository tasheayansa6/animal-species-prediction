"""
app.py  —  Animal Species Classifier Web App
Run:  python app/app.py --port 8080
"""

import os, io, sys, logging
from pathlib import Path
from typing import List, Tuple

# ── path so we can run from project root ─────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
from PIL import Image
import tensorflow as tf
from flask import Flask, request, jsonify, render_template_string

from app.database import init_db, save_prediction, get_history, get_stats

# ── logging ───────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

# ── constants ─────────────────────────────────────────────────────────
CLASS_NAMES: List[str] = [
    "Beetle", "Butterfly", "Cat", "Cow", "Dog",
    "Elephant", "Gorilla", "Hippo", "Lizard", "Monkey",
    "Mouse", "Panda", "Spider", "Tiger", "Zebra",
]
IMAGE_SIZE: Tuple[int, int] = (128, 128)   # must match training
MODEL_PATH = "models/final/animal_classifier.h5"
ALLOWED_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".gif"}

# ── HTML ──────────────────────────────────────────────────────────────
HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>Sanyii bine?elda</title>
  <style>
    *{box-sizing:border-box;margin:0;padding:0}
    body{font-family:'Segoe UI',sans-serif;background:linear-gradient(135deg,#1a1a2e,#16213e,#0f3460);
         min-height:100vh;display:flex;align-items:center;justify-content:center;padding:20px}
    .card{background:rgba(255,255,255,.06);backdrop-filter:blur(10px);
          border:1px solid rgba(255,255,255,.12);border-radius:20px;
          padding:40px;max-width:580px;width:100%;color:#fff}
    h1{font-size:1.8rem;margin-bottom:6px}
    .sub{color:#a0aec0;font-size:.9rem;margin-bottom:28px}
    .drop{border:2px dashed rgba(255,255,255,.3);border-radius:12px;
          padding:36px 20px;text-align:center;cursor:pointer;transition:.3s;margin-bottom:16px}
    .drop:hover{border-color:#63b3ed;background:rgba(99,179,237,.06)}
    .drop input{display:none}
    .icon{font-size:2.8rem;margin-bottom:8px}
    .hint{color:#a0aec0;font-size:.85rem}
    #preview{max-width:100%;max-height:280px;border-radius:10px;
             margin:12px auto;display:none;border:2px solid rgba(255,255,255,.15)}
    .btn{width:100%;padding:13px;background:linear-gradient(135deg,#667eea,#764ba2);
         color:#fff;border:none;border-radius:10px;font-size:1rem;
         font-weight:600;cursor:pointer;transition:opacity .2s}
    .btn:hover{opacity:.88}
    .btn:disabled{opacity:.45;cursor:not-allowed}
    #results{margin-top:22px;display:none}
    .top-box{text-align:center;padding:14px;background:rgba(99,179,237,.1);
             border-radius:10px;margin-bottom:14px;border:1px solid rgba(99,179,237,.3)}
    .top-name{font-size:2rem;font-weight:700;text-transform:capitalize;color:#63b3ed}
    .top-conf{color:#a0aec0;font-size:.88rem;margin-top:4px}
    .bar-row{display:flex;align-items:center;gap:10px;margin-bottom:9px}
    .rank{font-size:1.1rem;width:28px;text-align:center}
    .cls{width:95px;font-weight:500;text-transform:capitalize;font-size:.9rem}
    .track{flex:1;background:rgba(255,255,255,.1);border-radius:6px;height:18px}
    .fill{height:100%;border-radius:6px;background:linear-gradient(90deg,#667eea,#764ba2);transition:width .5s}
    .pct{width:48px;text-align:right;font-size:.85rem;color:#a0aec0}
    .err{color:#fc8181;margin-top:8px;font-size:.88rem;min-height:18px}
    .classes{margin-top:22px;padding:13px;background:rgba(255,255,255,.03);
             border-radius:10px;font-size:.78rem;color:#718096;line-height:1.6}
    .classes span{color:#a0aec0;font-weight:600}
  </style>
</head>
<body>
<div class="card">
  <h1>🐾 Sanyii bineelda</h1>
  <p class="sub">VGG-16 Transfer Learning &nbsp;·&nbsp; 15 Species</p>

  <div class="drop" onclick="document.getElementById('fi').click()">
    <div class="icon">📷</div>
    <p>Click to upload an image</p>
    <p class="hint">JPG · PNG · WEBP · BMP</p>
    <input id="fi" type="file" accept="image/*" onchange="onFile(this)">
  </div>

  <img id="preview" alt="preview">
  <button class="btn" id="btn" onclick="predict()" disabled>Classify Animal</button>
  <p class="err" id="err"></p>

  <div id="results">
    <div class="top-box">
      <div class="top-name" id="topName">—</div>
      <div class="top-conf" id="topConf">—</div>
    </div>
    <div id="bars"></div>
  </div>

  <div class="classes">
    <span>Recognises:</span>
    Beetle · Butterfly · Cat · Cow · Deer · Dog · Dolphin · Elephant ·
    Gorilla · Hippo · Lizard · Monkey · Mouse · Panda · Spider · Tiger · Zebra
  </div>
</div>

<script>
let file = null;
function onFile(inp){
  file = inp.files[0];
  if(!file) return;
  const r = new FileReader();
  r.onload = e => {
    const p = document.getElementById('preview');
    p.src = e.target.result; p.style.display = 'block';
    document.getElementById('btn').disabled = false;
    document.getElementById('results').style.display = 'none';
    document.getElementById('err').textContent = '';
  };
  r.readAsDataURL(file);
}
async function predict(){
  if(!file) return;
  const btn = document.getElementById('btn');
  btn.disabled = true; btn.textContent = 'Classifying…';
  document.getElementById('err').textContent = '';
  const fd = new FormData(); fd.append('file', file);
  try{
    const res = await fetch('/predict',{method:'POST',body:fd});
    const d = await res.json();
    if(!res.ok) throw new Error(d.error || 'Server error '+res.status);
    document.getElementById('topName').textContent = d.top_prediction;
    document.getElementById('topConf').textContent =
      'Confidence: ' + (d.confidence*100).toFixed(1) + '%';
    const medals = ['🥇','🥈','🥉','4️⃣','5️⃣'];
    document.getElementById('bars').innerHTML =
      d.predictions.slice(0,5).map((p,i)=>`
        <div class="bar-row">
          <span class="rank">${medals[i]}</span>
          <span class="cls">${p.class}</span>
          <div class="track"><div class="fill" style="width:${(p.confidence*100).toFixed(1)}%"></div></div>
          <span class="pct">${(p.confidence*100).toFixed(1)}%</span>
        </div>`).join('');
    document.getElementById('results').style.display = 'block';
  }catch(e){
    document.getElementById('err').textContent = e.message;
  }finally{
    btn.disabled = false; btn.textContent = 'Classify Animal';
  }
}
</script>
</body>
</html>"""

# ── model (loaded once) ───────────────────────────────────────────────
_model = None

def get_model():
    global _model
    if _model is None:
        if not Path(MODEL_PATH).exists():
            raise FileNotFoundError(
                f"Model not found: {MODEL_PATH}. Run training first."
            )
        logger.info("Loading model: %s", MODEL_PATH)
        _model = tf.keras.models.load_model(MODEL_PATH)
        logger.info("Model ready. Input: %s", _model.input_shape)
    return _model

# ── image preprocessing ───────────────────────────────────────────────
def preprocess(image_bytes: bytes) -> np.ndarray:
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize(IMAGE_SIZE, Image.BILINEAR)
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)   # (1, 128, 128, 3)

# ── Flask app ─────────────────────────────────────────────────────────
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB

# pre-load model and init DB at startup
init_db()
try:
    get_model()
except FileNotFoundError as e:
    logger.warning(str(e))

@app.route("/favicon.ico")
def favicon():
    return "", 204

@app.route("/")
def index():
    return render_template_string(HTML)

@app.route("/health")
def health():
    loaded = _model is not None
    return jsonify({
        "status": "ok" if loaded else "model_not_loaded",
        "model_loaded": loaded,
        "image_size": list(IMAGE_SIZE),
        "classes": CLASS_NAMES,
        "num_classes": len(CLASS_NAMES),
    })

@app.route("/predict", methods=["POST"])
def predict():
    # ── validate file ─────────────────────────────────────────────────
    if "file" not in request.files:
        return jsonify({"error": "No file field in request."}), 400
    f = request.files["file"]
    if not f.filename:
        return jsonify({"error": "Empty filename."}), 400
    ext = Path(f.filename).suffix.lower()
    if ext not in ALLOWED_EXT:
        return jsonify({"error": f"Unsupported type: {ext}"}), 400

    # ── preprocess ────────────────────────────────────────────────────
    try:
        raw = f.read()
        arr = preprocess(raw)
    except Exception as e:
        logger.error("Preprocess error: %s", e)
        return jsonify({"error": f"Could not read image: {e}"}), 400

    # ── inference ─────────────────────────────────────────────────────
    try:
        model = get_model()
        proba = model.predict(arr, verbose=0)[0]
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 503
    except Exception as e:
        logger.error("Inference error: %s", e)
        return jsonify({"error": f"Inference failed: {e}"}), 500

    # ── build response ────────────────────────────────────────────────
    top_k   = min(int(request.args.get("top_k", 15)), len(CLASS_NAMES))
    indices = np.argsort(proba)[::-1][:top_k]
    preds   = [
        {
            "rank":       int(i + 1),
            "class":      CLASS_NAMES[idx],
            "confidence": float(proba[idx]),
            "percentage": f"{proba[idx]*100:.2f}%",
        }
        for i, idx in enumerate(indices)
    ]

    # ── save to database ──────────────────────────────────────────────
    try:
        save_prediction(filename=f.filename, predictions=preds)
    except Exception as e:
        logger.warning("DB save failed (non-critical): %s", e)
    return jsonify({
        "top_prediction": preds[0]["class"],
        "confidence":     preds[0]["confidence"],
        "predictions":    preds,
    })

@app.route("/history")
def history():
    """Return last 50 predictions from the database."""
    limit = int(request.args.get("limit", 50))
    return jsonify(get_history(limit))

@app.route("/stats")
def stats():
    """Return prediction statistics."""
    return jsonify(get_stats())

# ── entry point ───────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--port",  type=int, default=8080)
    p.add_argument("--host",  default="0.0.0.0")
    p.add_argument("--debug", action="store_true")
    args = p.parse_args()
    logger.info("Starting on http://localhost:%d", args.port)
    app.run(host=args.host, port=args.port, debug=args.debug)
