import streamlit as st
import numpy as np
import cv2
import time
import matplotlib.pyplot as plt
import tensorflow as tf
from PIL import Image

st.set_page_config(page_title="Driver Drowsiness Detection System", layout="wide")

st.title("🚗 Driver Drowsiness Detection & Fatigue Tracking System")
st.markdown("### Real-time physiological indicator analysis & 3-stage fatigue fusion logic")

# Sidebar - Settings & Mode Selection
st.sidebar.header("System Settings")
mode = st.sidebar.radio("Select Mode", ["Single Image Analysis", "Continuous Session Simulation"])
confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.5, 0.99, 0.75)

# Rule-based 3-Level Fatigue Mapping Logic (Step 9 requirement)
def map_fatigue_stage(eye_state, mouth_state):
    """
    Mapping rules:
    0 -> Alert: Open AND no_yawn
    1 -> Mild Fatigue: Yawn (regardless of open eyes)
    2 -> Severe Fatigue: Closed eyes
    """
    if eye_state == "Closed":
        return 2, "Severe Fatigue", "red"
    elif mouth_state == "Yawn":
        return 1, "Mild Fatigue", "orange"
    else:
        return 0, "Alert", "green"

# Mock/Heuristic Prediction Engine for demonstration when model weights are loading
def predict_states(img_array):
    # Standard dummy logic simulation for test UI
    avg_val = np.mean(img_array)
    eye_pred = "Closed" if avg_val < 100 else "Open"
    mouth_pred = "Yawn" if avg_val % 2 > 1 else "No Yawn"
    return eye_pred, mouth_pred

if mode == "Single Image Analysis":
    st.subheader("📸 Single Image Inspection")
    uploaded_file = st.file_uploader("Upload Driver Image", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Driver Image", width=350)
        
        img_array = np.array(image.convert('RGB'))
        eye_state, mouth_state = predict_states(img_array)
        stage_code, stage_label, color = map_fatigue_stage(eye_state, mouth_state)
        
        st.markdown("---")
        col1, col2, col3 = st.columns(3)
        col1.metric("Eye State Prediction", eye_state)
        col2.metric("Mouth State Prediction", mouth_state)
        col3.markdown(f"**Fatigue Stage:** <h3 style='color:{color};'>{stage_label} (Level {stage_code})</h3>", unsafe_allow_html=True)

elif mode == "Continuous Session Simulation":
    st.subheader("⏱️ Simulated Driving Session & Fatigue Progression Curve")
    st.write("Simulating sequence of 30 frames representing a 15-minute continuous drive...")
    
    if st.button("Start Driving Session"):
        frames = []
        fatigue_stages = []
        timestamps = list(range(1, 31))
        
        # Simulate driver transitioning from Alert -> Mild -> Severe Fatigue
        for t in timestamps:
            if t <= 10:
                eye, mouth = "Open", "No Yawn"
            elif t <= 20:
                eye, mouth = "Open", "Yawn"
            else:
                eye, mouth = "Closed", "No Yawn"
                
            stage_code, stage_name, _ = map_fatigue_stage(eye, mouth)
            fatigue_stages.append(stage_code)
            
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(timestamps, fatigue_stages, marker='o', color='crimson', linewidth=2, label='Fatigue Level')
        ax.set_yticks([0, 1, 2])
        ax.set_yticklabels(['0: Alert', '1: Mild Fatigue', '2: Severe Fatigue'])
        ax.set_xlabel('Time Interval (Minutes)')
        ax.set_ylabel('Fatigue Stage')
        ax.set_title('Driver Fatigue Progression Curve Over Time')
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend()
        
        st.pyplot(fig)
        st.success("Analysis Complete: Driver transitioned from Alert → Mild Fatigue at minute 11, and reached Severe Fatigue at minute 21.")