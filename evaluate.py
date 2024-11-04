import os
import json
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import MinMaxScaler
from operator import itemgetter
import logging



with open("config.json", "r") as config_file:
    config = json.load(config_file)

static_gestures = [gesture.strip() for gesture in config["static_gestures"].split(",")]
dynamic_gestures = [gesture.strip() for gesture in config["dynamic_gestures"].split(",")]
confidence_threshold = float(config["confidence_threshold"])
motion_detection_threshold = float(config.get("motion_detection_threshold",50))
frame_number = int(config.get("prediction_frame_number", 10))

# Configure logging
logging.basicConfig(filename='output2.log', level=logging.INFO, format='%(asctime)s - %(message)s')

# Load the trained model
loaded_model_dynamics = load_model('gesture_3dcnn_normal_globalMaxPool_maskPadding_20epochs.h5')
loaded_model_static = load_model('gesture_2dcnn_actibysum_normal_globalMaxPool_flatten_20epochs.h5')

# Load new sensor data (replace with your actual file or data source)
def load_data(data, is_static):
    # Reshape the data to (frames, 8, 8, 1)
    if is_static:
        data = data[-1]
        return data.reshape(1, 8, 8, 1)
    else:
        num_frames = len(data)
        reshaped_data = data.reshape(num_frames, 8, 8, 1)
        # Pad the sequence to match the max frame length (e.g., 175)
        max_frames = 175  # Replace with the max frame length used during training
        padded_data = pad_sequences([reshaped_data], maxlen=10, padding='post', dtype='float32')
        return padded_data

# Sample models
# Assume `model_2d` is the 2D CNN for static gestures
# Assume `model_3d` is the 3D CNN for dynamic gestures

def detect_motion(data):
    """
    Simple motion detection based on frame differencing.
    Returns True if motion is detected, False otherwise.
    """
    diffs = []
    if len(data) < frame_number:
        return False
    for i in range(1, len(data)):
        diff = np.mean(np.abs(data[i] - data[i - 1]))
        diffs.append(diff)
    avg_motion = np.mean(diffs)
    logging.info(f'avg_motion = {avg_motion}')
    return avg_motion > motion_detection_threshold

def evaluate_gesture(data):
    data = np.array(data)
    is_static = not detect_motion(data)
    print(f"Gesture Prediction: is_static = {is_static}")

    if is_static:
        logging.info('Evaluate with static model')
    else:
        logging.info('Evaluate with dynamic model')
        
    reshaped_data = load_data(data, is_static)
    gestures = static_gestures if is_static else dynamic_gestures
    model = loaded_model_dynamics
    if is_static:
        model = loaded_model_static

    # Normalize the data using the same scaler from training
    scaler = MinMaxScaler()
    data = scaler.fit_transform(reshaped_data.reshape(-1, 64)).reshape(reshaped_data.shape)

    # Perform the prediction
    prediction = model.predict(data)
    
    # Get the predicted label for the frame
    predicted_label = np.argmax(prediction)
    confidence = prediction[0][predicted_label]
    confirmed_result = ''
    results = []
    if confidence >= confidence_threshold:
        confirmed_result = gestures[predicted_label]
        logging.info(f'Predicted Gesture: {gestures[predicted_label]}')
    for i, prob in enumerate(prediction[0]):
        results.append([gestures[i], prob])
    return confirmed_result, results