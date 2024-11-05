import os
import json
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import MinMaxScaler

# Load configuration
with open(".\\3D-CNN(1)\\config.json", "r") as config_file:
    config = json.load(config_file)

root_path = config["test_data_root_directory"]
gestures = [gesture.strip() for gesture in config["dynamic_gestures"].split(",")]
static_gestures = [gesture.strip() for gesture in config["static_gestures"].split(",")]

# Load the trained model
loaded_model1 = load_model('gesture_3dcnn_normal_globalMaxPool_maskPadding_20epochs_with_signal_2.h5')
loaded_model_static = load_model('gesture_2dcnn_with_signal_flatten_20epochs_with_signal_2.h5')

def preprocess_sensor_data(file_path, output_file_path):
    data_frames = []
    batch_number = 0
    
    df = pd.read_csv(file_path)

    # Select the required columns for distance and signal
    distance_columns = [f'distance_mm_z{i}' for i in range(64)]
    signal_columns = [f'signal_per_spad_z{i}' for i in range(64)]
    columns_to_keep = ['frame_count', 'time_stamp'] + distance_columns + signal_columns
    df_filtered = df[columns_to_keep].copy()
# Modify 'frame_count' to be sequential from 1 for each file
    df_filtered['frame_count'] = range(1, len(df_filtered) + 1)
# Convert 'time_stamp' to standard datetime format and then to float in seconds
    df_filtered['time_stamp'] = pd.to_datetime(df_filtered['time_stamp'], unit='s')
    df_filtered['time_stamp'] = df_filtered['time_stamp'].astype('int64') / 1e9  # Convert to seconds
# Insert 'batch_number' as the first column
    df_filtered.insert(0, 'batch_number', batch_number)
# Append DataFrame to the list of data_frames
    data_frames.append(df_filtered)
# Concatenate all DataFrames into one
    appended_data = pd.concat(data_frames, ignore_index=True)

    # Save the processed data to a CSV file without a header row
    appended_data.to_csv(output_file_path, index=False, header=False)
    print(f"Processed data saved to: {output_file_path}")

def load_new_data_static(data):
    # Extract zone distances and reshape to (frames, 8, 8, 1)
    distances = data.iloc[:, 2:66].values  # Distance columns
    signals = data.iloc[:, 66:130].values  # Signal columns

    distance_scaler = MinMaxScaler(feature_range=(0, 1))
    signal_scaler = MinMaxScaler(feature_range=(0, 1))
    normalized_distances = distance_scaler.fit_transform(distances.reshape(-1, 1)).reshape(8, 8, 1)
    normalized_signals = signal_scaler.fit_transform(signals.reshape(-1, 1)).reshape(8, 8, 1)

    frames = np.concatenate([normalized_distances, normalized_signals], axis=-1)

    frames = frames[np.newaxis, ...]
    return frames

def load_new_data(data):
    """
    Loads the processed DataFrame and prepares it for prediction with padding.
    """
    # Extract zone distances and reshape to (frames, 8, 8, 1)
    distances = data.iloc[:, 2:66].values  # Distance columns
    signals = data.iloc[:, 66:130].values  # Signal columns

    distance_scaler = MinMaxScaler(feature_range=(0, 1))
    signal_scaler = MinMaxScaler(feature_range=(0, 1))
    distances = distance_scaler.fit_transform(distances)  # Normalized to [0, 1]
    signals = signal_scaler.fit_transform(signals)        # Normalized to [0, 1]

    distances = distances.reshape(len(distances), 8, 8, 1)
    signals = signals.reshape(len(signals), 8, 8, 1)
    frames = np.concatenate([distances, signals], axis=-1)

    frames = frames[np.newaxis, ...]
    return frames

# Main processing and prediction loop
def evaluate_dynamic_gestures():
    path = "C:\\Users\\yecl3\\Documents\\GitHub\\3D CNN_to fix\\DataCollectionEVK_complete\\palm_backward_combined11.csv"

    # Process each .csv file in the gesture folder
    combined_df = pd.read_csv(path)
    if combined_df is not None:
        
        data = load_new_data(combined_df)
        # Pad if necessary
        # padded_data = pad_sequences([normalized_data], maxlen=20, padding='post', dtype='float32')

        # Perform prediction with the model
        prediction1 = loaded_model1.predict(data)

        # Get predicted label
        predicted_label1 = np.argmax(prediction1)

        # Output predictions for the model
        print(f'Predicted Gesture with model 1: {gestures[predicted_label1]}')

def evaluate_static_gestures():
    path = "C:\\Users\\yecl3\\Documents\\GitHub\\3D CNN_to fix\\DataCollectionEVK_complete\\thumbs_up_combined - Copy.csv"

    # Process each .csv file in the gesture folder
    combined_df = pd.read_csv(path, header=None)
    if combined_df is not None:
        
        data = load_new_data_static(combined_df)
        # Pad if necessary
        # padded_data = pad_sequences([normalized_data], maxlen=20, padding='post', dtype='float32')

        # Perform prediction with the model
        prediction1 = loaded_model_static.predict(data)

        # Get predicted label
        predicted_label1 = np.argmax(prediction1)

        # Output predictions for the model
        print(f'Predicted Gesture with model 1: {static_gestures[predicted_label1]}')

if __name__ == '__main__':
    evaluate_dynamic_gestures()
    evaluate_static_gestures()