import os
import pandas as pd
import json

import os
import pandas as pd

def preprocess_sensor_data(gesture_folder_path, output_file_path):
    data_frames = []
    batch_number = 0

    for root, dirs, files in os.walk(gesture_folder_path):
        for filename in sorted(files):
            file_path = os.path.join(root, filename)

            # Process .csv files only
            if filename.startswith('data_') and filename.endswith('.csv'):
                batch_number += 1
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

    if data_frames:
        # Concatenate all DataFrames in the list
        combined_df = pd.concat(data_frames, ignore_index=True)
        
        # Save the combined data to CSV file
        combined_df.to_csv(output_file_path, index=False, header=False)
        print(f"Processed data saved to: {output_file_path}")
    else:
        print(f"No data files found in {gesture_folder_path}")


# Load root directory from config.json
with open("config.json", "r") as config_file:
    config = json.load(config_file)

root_path = config["test_data_root_directory"]
# gestures = ["thumbs_up", "thumbs_down","palm_left", "palm_right", "palm_up", "palm_down", "palm_forward", "palm_backward"]
gestures = [gesture.strip() for gesture in config["gestures"].split(",")]


for gesture in gestures:
    gesture_folder_path = os.path.join(root_path, gesture)
    output_file = os.path.join(root_path, f"{gesture}_combined.csv")
    preprocess_sensor_data(gesture_folder_path, output_file)
