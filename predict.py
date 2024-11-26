import os
import json
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import MinMaxScaler

with open("config.json", "r") as config_file:
    config = json.load(config_file)

root_path = config["test_data_root_directory"]
gestures = [gesture.strip() for gesture in config["gestures"].split(",")]

# Load the trained model
loaded_model1 = load_model('gesture_3dcnn_model_normal_globalMaxPool_maskPadding_17epochs_best.h5')
loaded_model2 = load_model('gesture_3dcnn_model_normal_globalMaxPool_maskPadding_15epochs.h5')
loaded_model3 = load_model('gesture_3dcnn_model_normal_globalMaxPool_maskPadding_20epochs.h5')

import pandas as pd

def preprocess_sensor_data(input_file_path, output_file_path):
    # if output file already exist, do not create duplicate
    if os.path.exists(output_file_path):
        return
    # Load the sensor data
    data = pd.read_csv(input_file_path)

    # Filter out rows where rng__zone_id is 255
    filtered_data = data[data['rng__zone_id'] != 255]

    # Initialize an empty list to store the processed rows
    processed_rows = []

    # Group by sample_no to process each sequence
    for sample_no, group in filtered_data.groupby('sample_no'):
        # Initialize a list to store the values for this sample
        row = [1, sample_no]  # Set batch number to 1 and sample_no as sequence number

        # Create a dictionary to store distances by zone_id (0-63)
        zone_distances = {i: None for i in range(64)}  # Initialize all zones to None

        # Fill the dictionary with distances from the current group
        for _, row_data in group.iterrows():
            zone_id = int(row_data['rng__zone_id'])
            median_range = row_data['median_range_mm']
            zone_distances[zone_id] = median_range

        # Replace None values with 0 (if any zones were missing in the data)
        row.extend([zone_distances[i] if zone_distances[i] is not None else 0 for i in range(64)])

        # Add the processed row to the final list
        processed_rows.append(row)

    # Convert the list to a DataFrame
    processed_df = pd.DataFrame(processed_rows)

    # Save the processed data to a CSV file without a header row
    processed_df.to_csv(output_file_path, index=False, header=False)

    print(f"Processed data saved to: {output_file_path}")


# Load new sensor data (replace with your actual file or data source)
def load_new_data(file_path):
    data = pd.read_csv(file_path, header=None)
    zone_distances = data.iloc[:, 2:].values  # Extract zone distances

    # Reshape the data to (frames, 8, 8, 1)
    num_frames = len(zone_distances)
    reshaped_data = zone_distances.reshape(num_frames, 8, 8, 1)

    # Pad the sequence to match the max frame length (e.g., 175)
    max_frames = 175  # Replace with the max frame length used during training
    padded_data = pad_sequences([reshaped_data], maxlen=max_frames, padding='post', dtype='float32')

    return padded_data

for gesture in gestures:
    print(f'Target gesture: {gesture}')
    model1Accuracy = 0
    model2Accuracy = 0
    model3Accuracy = 0
    totalCount = 0
    sensor_data_folder = f'{root_path}\\old_test_horizontal\\{gesture}\\'
    for filename in os.listdir(sensor_data_folder):
        if filename.endswith('.csv') and not filename.endswith('_processed.csv'):
            totalCount += 1
            file_path = os.path.join(sensor_data_folder, filename)
            output_file = file_path.replace('.csv', '_processed.csv')
            preprocess_sensor_data(file_path, output_file)
            
            new_data = load_new_data(output_file)

            # Normalize the data using the same scaler from training
            scaler = MinMaxScaler()
            new_data = scaler.fit_transform(new_data.reshape(-1, 64)).reshape(new_data.shape)

            # Perform the prediction
            prediction1 = loaded_model1.predict(new_data)
            prediction2 = loaded_model2.predict(new_data)
            prediction3 = loaded_model3.predict(new_data)

            # Get the predicted label
            predicted_label1 = np.argmax(prediction1)
            predicted_label2 = np.argmax(prediction2)
            predicted_label3 = np.argmax(prediction3)
            
            print(f'Predicted Gesture with model 1: {gestures[predicted_label1]}')
            print(f'Predicted Gesture with model 2: {gestures[predicted_label2]}')
            print(f'Predicted Gesture with model 3: {gestures[predicted_label3]}')
            if gesture == gestures[predicted_label1]:
                model1Accuracy +=1
            if gesture == gestures[predicted_label2]:
                model2Accuracy +=1
            if gesture == gestures[predicted_label3]:
                model3Accuracy +=1
    print(f'Model 1 accuracy for {gesture}: {model1Accuracy/totalCount}')
    print(f'Model 2 accuracy for {gesture}: {model2Accuracy/totalCount}')
    print(f'Model 3 accuracy for {gesture}: {model3Accuracy/totalCount}')