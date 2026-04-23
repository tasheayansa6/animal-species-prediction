# 🐾 Sanyii Bineeldaa — Model Evaluation Report

**Project:** Animal Species Classification using VGG-16 Transfer Learning  
**Model:** VGG-16 (ImageNet pre-trained) + Custom Classification Head  
**Framework:** TensorFlow 2.13 / Keras  
**Classes:** 15 animal species  

---

## Model Architecture

| Component | Details |
|-----------|---------|
| Base Model | VGG-16 (ImageNet weights, frozen) |
| Input Size | 128 × 128 × 3 |
| Custom Head | GlobalAvgPool → Dense(512, ReLU) → BatchNorm → Dropout(0.5) → Dense(256, ReLU) → BatchNorm → Dropout(0.3) → Dense(15, Softmax) |
| Optimizer | Adam (lr=1e-4) |
| Loss | Categorical Cross-Entropy |
| Regularization | L2 (1e-4) + Dropout |

---

## Training Results

| Metric | Value |
|--------|-------|
| Total Epochs | 10 |
| Best Epoch | 10 |
| Best Validation Accuracy | **60.30%** |
| Best Top-3 Accuracy | **82.53%** |
| Final Validation Loss | 1.3144 |
| Training Strategy | Feature Extraction (Phase 1) |

### Epoch-by-Epoch Results

| Epoch | Train Acc | Val Acc | Top-3 | Train Loss | Val Loss |
|-------|-----------|---------|-------|------------|----------|
| 1 | 0.1340 | 0.1940 | 0.3203 | 3.3535 | 2.5387 |
| 2 | 0.2860 | 0.4045 | 0.5353 | 2.5251 | 1.9848 |
| 3 | 0.3763 | 0.4895 | 0.6347 | 2.1593 | 1.6974 |
| 4 | 0.4200 | 0.5190 | 0.6920 | 1.9411 | 1.5864 |
| 5 | 0.4670 | 0.5465 | 0.7220 | 1.7947 | 1.5182 |
| 6 | 0.4860 | 0.5680 | 0.7420 | 1.7028 | 1.4459 |
| 7 | 0.5297 | 0.5780 | 0.7833 | 1.5593 | 1.4030 |
| 8 | 0.5650 | 0.5795 | 0.7983 | 1.4810 | 1.3899 |
| 9 | 0.5840 | 0.5850 | 0.8253 | 1.3863 | 1.3685 |
| 10 | 0.5667 | 0.6030 | 0.8243 | 1.3976 | 1.3144 |


---

## Visualizations

### Training Curves
![Training Curves](outputs/figures/training_curves.png)

### Confusion Matrix
![Confusion Matrix](outputs/figures/confusion_matrix_test.png)

### Per-Class Accuracy
![Per-Class Accuracy](outputs/figures/per_class_accuracy.png)

### Sample Predictions
![Sample Predictions](outputs/figures/sample_predictions_test.png)

---

## Supported Classes

| # | Class | Emoji |
|---|-------|-------|
| 1 | Beetle | 🪲 | | 2 | Butterfly | 🦋 | | 3 | Cat | 🐱 | | 4 | Cow | 🐄 | | 5 | Dog | 🐶 |
| 6 | Elephant | 🐘 | | 7 | Gorilla | 🦍 | | 8 | Hippo | 🦛 | | 9 | Lizard | 🦎 | | 10 | Monkey | 🐒 |
| 11 | Mouse | 🐭 | | 12 | Panda | 🐼 | | 13 | Spider | 🕷️ | | 14 | Tiger | 🐯 | | 15 | Zebra | 🦓 |

---

## Limitations & Future Work

1. **Dataset size** — Training used 200 images/class (3,000 total) due to CPU constraints; full 30,000-image training expected to reach 88–94% accuracy
2. **Input resolution** — 128×128 is smaller than VGG-16's optimal 224×224
3. **Fine-tuning** — Phase 2 (unfreezing top VGG layers) not yet completed; expected +3–5% accuracy gain
4. **Augmentation** — Adding rotation, flip, zoom augmentation would improve generalization
5. **More classes** — Expanding to 50+ species would increase real-world utility

---

*Generated automatically by scripts/quick_report.py*
