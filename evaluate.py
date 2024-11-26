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

gestures = [gesture.strip() for gesture in config["all_gestures"].split(",")]
gestures_display = [gesture.strip() for gesture in config["all_gestures_display"].split(",")]
confidence_threshold = float(config["confidence_threshold"])
frame_number = int(config.get("prediction_frame_number", 10))

# Configure logging
log_level = logging.getLevelName(config.get("log_level", "INFO"))
# Configure logging
logging.basicConfig(filename='output2.log', level=log_level, format='%(asctime)s - %(message)s')

test_id = 0

# Load the trained model
loaded_model_dynamics = load_model('gesture_3dcnn_20241126_f_4.h5')

distance_scaler = MinMaxScaler(feature_range=(0, 1))
signal_scaler = MinMaxScaler(feature_range=(0, 1))

# Load new sensor data (replace with your actual file or data source)
def load_data(distances, signals):
    # Reshape the data to (frames, 8, 8, 2)
    zone_distances = distance_scaler.fit_transform(distances)  # Normalized to [0, 1]
    zone_signals = signal_scaler.fit_transform(signals)        # Normalized to [0, 1]
    zone_distances = zone_distances.reshape(len(zone_distances), 8, 8, 1)
    zone_signals = zone_signals.reshape(len(zone_signals), 8, 8, 1)
    frames = np.concatenate([zone_distances, zone_signals], axis=-1)
    frames = frames[np.newaxis, ...]
    return frames

def evaluate_gesture(distances, signals):
    global test_id

    test_id += 1
    logging.warning(f'{test_id}:{distances}')

    if len(distances) < frame_number:
        logging.info(f'Less than {frame_number} frames, skip prediction')
        return '', []
    distances = np.array(distances)
    signals = np.array(signals)
        
    reshaped_data = load_data(distances, signals)
    model = loaded_model_dynamics

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
        if confirmed_result not in gestures_display:
            confirmed_result = ''
            logging.warning('Predicted Gesture will not be displayed')
    for i, prob in enumerate(prediction[0]):
        results.append([gestures[i], prob])
    return confirmed_result, results
