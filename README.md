---
title: Animal Species Prediction
emoji: 🐾
colorFrom: indigo
colorTo: purple
sdk: gradio
sdk_version: 5.29.1
app_file: app.py
pinned: false
license: mit
short_description: Predict animal species using ML
startup_duration_timeout: 1h
---

# 🐾 Animal Species Prediction

[![Live Demo](https://img.shields.io/badge/🤗%20Hugging%20Face-Live%20Demo-blue?style=for-the-badge)](https://huggingface.co/spaces/tasheayansa/animal-species-prediction)
[![Python](https://img.shields.io/badge/Python-3.11-green?style=for-the-badge&logo=python)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13-orange?style=for-the-badge&logo=tensorflow)](https://tensorflow.org)

> Upload any animal photo and get instant species classification across **15 categories** using VGG-16 transfer learning.

---

## 🚀 Live Demo

**👉 [Try it here → huggingface.co/spaces/tasheayansa/animal-species-prediction](https://huggingface.co/spaces/tasheayansa/animal-species-prediction)**

---

## 🐾 Supported Species

| | | | | |
|---|---|---|---|---|
| 🪲 Beetle | 🦋 Butterfly | 🐱 Cat | 🐄 Cow | 🐶 Dog |
| 🐘 Elephant | 🦍 Gorilla | 🦛 Hippo | 🦎 Lizard | 🐒 Monkey |
| 🐭 Mouse | 🐼 Panda | 🕷️ Spider | 🐯 Tiger | 🦓 Zebra |

---

## 🧠 Model

| Item | Detail |
|------|--------|
| Architecture | VGG-16 (ImageNet pre-trained) + custom head |
| Framework | TensorFlow 2.13 / Keras |
| Inference | TFLite (quantized, 15MB) |
| Input size | 128 × 128 × 3 |
| Output | 15-class softmax |
| Strategy | Two-phase transfer learning |

---

## 🗂️ Project Structure

```
animal-species-prediction/
├── app.py                  # Gradio app (HF Spaces)
├── app/app.py              # Flask app (local)
├── models/final/           # Trained model (.h5 + .tflite)
├── src/                    # Training source code
│   ├── data/               # Dataset, preprocessing, augmentation
│   ├── models/             # VGG-16 architecture
│   └── training/           # Trainer, callbacks, metrics
├── scripts/                # Training & evaluation scripts
├── notebooks/              # Jupyter exploration notebooks
└── config/                 # YAML configuration files
```

---

## ⚡ Quick Start

```bash
git clone https://github.com/tasheayansa6/animal-species-prediction.git
cd animal-species-prediction
pip install -r requirements.txt
python app/app.py
# Open http://localhost:8080
```

---

## 📊 Training

```bash
# Train the model
python scripts/train.py

# Evaluate
python scripts/evaluate.py

# Predict on an image
python scripts/predict.py --image path/to/animal.jpg
```

---

## 📄 License

For educational purposes. Dataset by [utkarshsaxenadn on Kaggle](https://www.kaggle.com/datasets/utkarshsaxenadn/animal-image-classification-dataset).
