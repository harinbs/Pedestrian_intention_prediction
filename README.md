# Pedestrian Crossing Intention Prediction

This repository contains the official implementation of the paper:

**“A Novel Intention Prediction Neural Network for Predicting Pedestrian Crossing Behavior in Indian Roads”**  
*International Journal of Transportation Science and Technology (Elsevier)*  
(Q1 Journal, Impact Factor: 4.8, Scopus 87th Percentile)

## 🔍 Overview
Pedestrian intention prediction is a critical component of intelligent transportation systems and autonomous driving. This work proposes a neural network framework that integrates:

- Human pose features
- Bounding-box kinematics
- Temporal motion cues

to predict pedestrian actions and crossing intentions at an early stage.

## 🧠 Model Architecture
The model consists of:
- Feature extraction using pose landmarks and bounding-box cues
- Fully connected layers with batch normalization and dropout
- Dual output heads:
  - Action prediction (standing / walking / running)
  - Crossing intention prediction (crossing / not crossing)

<p align="center">
  <img src="figures/framework.png" width="700">
</p>

## 📊 Results
The proposed method achieves strong performance on real-world pedestrian datasets.

| Task | Accuracy | Precision | Recall | F1-score |
|-----|---------|-----------|--------|---------|
| Action Prediction | 83.22% | 82.27% | 83.22% | 82.34% |
| Crossing Intention | 96.64% | 96.66% | 96.64% | 96.65% |

Confusion matrices and ROC curves are available in the `results/` directory.

## 🔬 Ablation Study
We conduct ablation experiments to evaluate the contribution of individual components:
- Full model (Pose + Kinematics + BBox)
- Without pose
- Without kinematics
- Bounding-box only

<p align="center">
  <img src="figures/ablation.png" width="650">
</p>

## ⚙️ Installation
```bash
pip install -r requirements.txt
