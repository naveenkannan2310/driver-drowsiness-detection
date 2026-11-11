# Driver Drowsiness Detection System using Deep Learning

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-1.x-red)

## 📌 Project Overview
Driver fatigue significantly impacts reaction time and decision-making, presenting a primary hazard in road transport. This project delivers a non-intrusive computer vision system that monitors facial indicators (eye closure and yawning) to classify driver drowsiness levels and plot continuous fatigue progression over time.

## 🎯 Objectives & Key Features
- **Binary & Multi-Class Image Classification:** Detects Eye Open/Closed and Mouth Yawn/No Yawn states.
- **Custom CNN vs. Transfer Learning:** Evaluates model accuracy and convergence using MobileNetV2 pretrained on ImageNet.
- **Rule-Based 3-Stage Decision Fusion:**
  - `0 -> Alert`: Eyes Open AND No Yawn
  - `1 -> Mild Fatigue`: Yawn Detected
  - `2 -> Severe Fatigue`: Eyes Closed
- **Continuous Driving Session Simulator:** Generates continuous fatigue progression curves across temporal frame sequences.
- **Interactive Web App:** Built with Streamlit for real-time driver state testing.

## 📁 Repository Structure
```text
driver-drowsiness-detection/
│
├── dataset/                  # Dataset directories (train, validation, test)
├── models/                   # Saved model checkpoints (.keras / .h5)
├── notebooks/
│   └── Driver_Drowsiness_Detection.ipynb  # End-to-end training & evaluation
├── outputs/                  # Plot visualisations (Confusion matrices, curves)
├── app.py                    # Streamlit web interface
├── README.md                 # Project documentation
└── requirements.txt          # Python dependencies