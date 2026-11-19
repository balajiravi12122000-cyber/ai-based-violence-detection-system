# AI-Based Violence Detection & Video Alert System

**Author:** Balaji  
**Environment:** Google Colab  
**Core stack:** Python, OpenCV, TensorFlow/Keras, MobileNetV2, GRU, scikit-learn

## Overview
A video-classification project that distinguishes violence/fight clips from non-violence/non-fight clips. It samples frames from videos, extracts pretrained MobileNetV2 visual embeddings, and learns temporal patterns using a GRU classifier. The notebook includes dataset discovery, preprocessing, training, held-out evaluation, single-video inference, and export of model artifacts and metrics.

## Repository structure
```text
ai-based-violence-detection-system/
├── notebooks/
│   └── AI_Based_Violence_Detection_System.ipynb
├── reports/
│   └── README.md
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt
```

## Dataset
The notebook expects a labeled dataset arranged in class folders. Split-based layout is recommended:
```text
dataset/
  train/Fight/
  train/NonFight/
  test/Fight/
  test/NonFight/
```
It also recognizes common folder names such as `Violence`, `NonViolence`, `Fight`, and `NonFight`, and can create stratified train/validation/test splits when only class folders are supplied. Upload a dataset ZIP in Colab and set `DATASET_DIR` to its extracted location.

Use a dataset only when you have the right to access and process it. The notebook does not bundle copyrighted video data or automatically download third-party datasets.

## Run in Google Colab
1. Open `notebooks/AI_Based_Violence_Detection_System.ipynb` in Google Colab.
2. Select **Runtime → Change runtime type → T4 GPU** (or another available GPU).
3. Run cells in order.
4. Upload a ZIP containing the labeled video folders when prompted.
5. Review the discovered split counts before training.
6. Evaluate the held-out test metrics and inspect the confusion matrix.
7. Upload a new video in the inference cell to obtain a clip-level prediction.

## Model architecture
- **Frame sampling:** 16 uniformly sampled frames per clip, resized to 160×160.
- **Visual encoder:** frozen ImageNet-pretrained MobileNetV2.
- **Temporal model:** stacked GRU layers with dropout.
- **Output:** sigmoid probability for Violence/Fight (positive class).

## Evaluation
The notebook reports accuracy, precision, recall, F1-score, ROC-AUC (when defined), classification report, confusion matrix and ROC curve. Results are generated from the dataset at runtime and are intentionally not pre-filled with fabricated metrics.

## Responsible use and limitations
This is an educational research prototype, not a validated safety system. It can make mistakes and does not identify people, establish intent, localize events, or provide proof. Do not use its predictions as the sole basis for law-enforcement, employment, disciplinary, or emergency decisions. Human review, privacy safeguards, legal approval and independent external validation are required before any real-world use.

## Author
**Balaji**
