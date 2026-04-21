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
---

# Animal Species Prediction

A deep learning project that classifies **15 animal species** using **VGG-16 transfer learning**.

## Classes
`Beetle` · `Butterfly` · `Cat` · `Cow` · `Dog` · `Elephant` · `Gorilla` · `Hippo` · `Lizard` · `Monkey` · `Mouse` · `Panda` · `Spider` · `Tiger` · `Zebra`

## Model
- Architecture: VGG-16 (ImageNet pre-trained) + custom classification head  
- Framework: TensorFlow / Keras  
- Input size: 128 × 128 × 3  
- Output: 15-class softmax