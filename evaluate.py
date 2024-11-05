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

test_id = 0

# Load the trained model
loaded_model_dynamics = load_model('gesture_3dcnn_normal_globalMaxPool_maskPadding_20epochs_with_signal_2.h5')
loaded_model_static = load_model('gesture_2dcnn_with_signal_flatten_20epochs_with_signal_2.h5')

distance_scaler = MinMaxScaler(feature_range=(0, 1))
signal_scaler = MinMaxScaler(feature_range=(0, 1))

# Load new sensor data (replace with your actual file or data source)
def load_data(distances, signals, is_static):
    # Reshape the data to (frames, 8, 8, 2)
    if is_static:
        zone_distances = distances[-1]
        zone_signals = signals[-1]
        zone_distances = distance_scaler.fit_transform(zone_distances.reshape(-1, 1)).reshape(8, 8, 1)  # Normalized to [0, 1]
        zone_signals = signal_scaler.fit_transform(zone_signals.reshape(-1, 1)).reshape(8, 8, 1)        # Normalized to [0, 1]
        frames = np.concatenate([zone_distances, zone_signals], axis=-1)
        frames = frames[np.newaxis, ...]
        return frames
    else:
        zone_distances = distance_scaler.fit_transform(distances)  # Normalized to [0, 1]
        zone_signals = signal_scaler.fit_transform(signals)        # Normalized to [0, 1]

        zone_distances = zone_distances.reshape(len(zone_distances), 8, 8, 1)
        zone_signals = zone_signals.reshape(len(zone_signals), 8, 8, 1)
        frames = np.concatenate([zone_distances, zone_signals], axis=-1)

        frames = frames[np.newaxis, ...]
        return frames

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
        diff = np.mean(np.abs(data[:,0][i] - data[:,0][i - 1]))
        diffs.append(diff)
    avg_motion = np.mean(diffs)
    logging.info(f'avg_motion = {avg_motion}')
    return avg_motion > motion_detection_threshold

def evaluate_gesture(distances, signals):
    global test_id

    test_id += 1
    logging.warning(f'{test_id}:{distances}')

    distances = np.array(distances)
    signals = np.array(signals)
    is_static = not detect_motion(distances)
    print(f"Gesture Prediction: is_static = {is_static}")

    if is_static:
        logging.info('Evaluate with static model')
    else:
        logging.info('Evaluate with dynamic model')
        
    reshaped_data = load_data(distances, signals, is_static)
    gestures = static_gestures if is_static else dynamic_gestures
    model = loaded_model_dynamics
    if is_static:
        model = loaded_model_static

    # Perform the prediction
    prediction = model.predict(reshaped_data)
    
    # Get the predicted label for the frame
    predicted_label = np.argmax(prediction)
    confidence = prediction[0][predicted_label]
    confirmed_result = ''
    results = []
    if confidence >= confidence_threshold:
        confirmed_result = gestures[predicted_label]
        logging.info(f'Predicted Gesture: {gestures[predicted_label]}')
        logging.warning(f'{test_id} - result:{confirmed_result}')
    for i, prob in enumerate(prediction[0]):
        results.append([gestures[i], prob])
    return confirmed_result, results

def evaluate_gesture_both(data):
    global test_id

    test_id += 1
    logging.warning(f'{test_id}:{data}')

    data = np.array(data)

    logging.info('Evaluate with static model')
    reshaped_data = load_data(data, is_static=True)
    predict_static, results_static = evaluate_generic(reshaped_data, loaded_model_static, static_gestures)
    logging.warning(f'{test_id} - static result:{predict_static}')

    logging.info('Evaluate with dynamic model')
    reshaped_data = load_data(data, is_static=False)
    predict_dynamic, results_dynamic = evaluate_generic(reshaped_data, loaded_model_dynamics, dynamic_gestures)
    logging.warning(f'{test_id} - dynamic result:{predict_dynamic}')

    if predict_dynamic != '':
        logging.info(f'Take dynamic prediction: {predict_dynamic}')
        return predict_dynamic, results_dynamic
    else:
        logging.info(f'Take static prediction: {predict_static}')
        return predict_static, results_static
    

def evaluate_generic(npData, model, gestures):
    # Normalize the data using the same scaler from training
    scaler = MinMaxScaler()
    data = scaler.fit_transform(npData.reshape(-1, 64)).reshape(npData.shape)
    # Perform the prediction
    prediction = model.predict(data)
    # Get the predicted label for the frame
    predicted_label = np.argmax(prediction)
    confidence = prediction[0][predicted_label]
    confirmed_result = ''
    results = [[gestures[i], prob] for i, prob in enumerate(prediction[0])]
    # Sort the results by probability in descending order
    results.sort(key=lambda x: x[1], reverse=True)
    if confidence >= confidence_threshold:
        confirmed_result = gestures[predicted_label]
        logging.info(f'Predicted Gesture: {gestures[predicted_label]}')
    return confirmed_result, results