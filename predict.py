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

root_path = config["test_data_root_directory"]
static_gestures = [gesture.strip() for gesture in config["static_gestures"].split(",")]
dynamic_gestures = [gesture.strip() for gesture in config["dynamic_gestures"].split(",")]
confidence_threshold = float(config["confidence_threshold"])

# Configure logging
logging.basicConfig(filename='output.log', level=logging.INFO, format='%(asctime)s - %(message)s')


# Load the trained model
loaded_model_dynamics = load_model('gesture_3dcnn_normal_globalMaxPool_maskPadding_20epochs.h5')
loaded_model_static = load_model('gesture_2dcnn_actibysum_normal_globalMaxPool_flatten_20epochs.h5')

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

    logging.info(f"Processed data saved to: {output_file_path}")


# Load new sensor data (replace with your actual file or data source)
def load_new_data(file_path, is_static=False):
    data = pd.read_csv(file_path, header=None)
    zone_distances = data.iloc[:, 2:].values  # Extract zone distances
    # Reshape the data to (frames, 8, 8, 1)
    num_frames = len(zone_distances)
    reshaped_data = zone_distances.reshape(num_frames, 8, 8, 1)
    
    if is_static:
        return reshaped_data
    else:
        # Pad the sequence to match the max frame length (e.g., 175)
        max_frames = 175  # Replace with the max frame length used during training
        padded_data = pad_sequences([reshaped_data], maxlen=max_frames, padding='post', dtype='float32')

        return padded_data

# logging.info('Test dynamics gesture')
# for gesture in dynamic_gestures:
#     logging.info(f'Target gesture: {gesture}')
#     modelAccuracy = 0
#     totalCount = 0
#     sensor_data_folder = f'{root_path}\\test data\\{gesture}\\'
#     for filename in os.listdir(sensor_data_folder):
#         if filename.endswith('.csv') and not filename.endswith('_processed.csv'):
#             totalCount += 1
#             file_path = os.path.join(sensor_data_folder, filename)
#             output_file = file_path.replace('.csv', '_processed.csv')
#             preprocess_sensor_data(file_path, output_file)
            
#             new_data = load_new_data(output_file, is_static=False)

#             # Normalize the data using the same scaler from training
#             scaler = MinMaxScaler()
#             new_data = scaler.fit_transform(new_data.reshape(-1, 64)).reshape(new_data.shape)

#             # Perform the prediction
#             prediction = loaded_model_dynamics.predict(new_data)

#             # Get the predicted label
#             predicted_label = np.argmax(prediction)
            
#             logging.info(f'Predicted Gesture with model: {dynamic_gestures[predicted_label]}')
#             if gesture == dynamic_gestures[predicted_label]:
#                 modelAccuracy +=1

#     logging.info(f'Model accuracy for {gesture}: {modelAccuracy/totalCount}')


logging.info('Test static gesture')
for gesture in static_gestures:
    logging.info(f'Target gesture: {gesture}')
    modelAccuracy = 0
    totalCount = 0
    sensor_data_folder = f'{root_path}\\test data\\{gesture}\\'
    for filename in os.listdir(sensor_data_folder):
        if filename.endswith('.csv') and not filename.endswith('_processed.csv'):
            totalCount += 1
            file_path = os.path.join(sensor_data_folder, filename)
            output_file = file_path.replace('.csv', '_processed.csv')
            preprocess_sensor_data(file_path, output_file)
            
            new_data = load_new_data(output_file, is_static=True)

            # Normalize the data using the same scaler from training
            scaler = MinMaxScaler()
            new_data = scaler.fit_transform(new_data.reshape(-1, 64)).reshape(new_data.shape)

            gesture_found = {}
            for frame in new_data:
                frame = frame.reshape(1, 8, 8, 1)  # Reshape to (1, 8, 8, 1) for the model
                prediction = loaded_model_static.predict(frame)
                
                # Get the predicted label for the frame
                predicted_label = np.argmax(prediction)
                confidence = prediction[0][predicted_label]

                # Check if the confidence exceeds the threshold
                if confidence >= confidence_threshold:
                    # Map predicted label to gesture name
                    predicted_gesture = static_gestures[predicted_label]
                    if predicted_gesture not in gesture_found:
                        gesture_found[predicted_gesture] = 1
                    else:
                        gesture_found[predicted_gesture] += 1
                    # logging.info(f'Match found for static gesture: {predicted_gesture} in file {filename}')
            if len(gesture_found) == 0:
                logging.info(f"No match gesture found in file {filename}")
            else:
                total_frames = sum(gesture_found.values())
                sorted_gestures = sorted(gesture_found.items(), key=lambda item: item[1], reverse=True)
                most_likely_gesture = max(gesture_found.items(), key=itemgetter(1))
                if most_likely_gesture[0] == gesture:
                    modelAccuracy += 1
                for g, count in sorted_gestures:
                    chance = (count / total_frames) * 100
                    logging.info(f"Gesture: {g}, Matched Frames: {count}, Chance: {chance:.2f}%")

    logging.info(f'Model accuracy for {gesture}: {modelAccuracy/totalCount}')