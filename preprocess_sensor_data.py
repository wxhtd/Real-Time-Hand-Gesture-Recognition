import os
import pandas as pd
import json

def preprocess_sensor_data(sensor_data_folder, output_file_path):
    data_frames = []
    batch_number = 0
    for filename in os.listdir(sensor_data_folder):
        if filename.endswith('.csv'):
            batch_number += 1
            file_path = os.path.join(sensor_data_folder, filename)
            df = pd.read_csv(file_path)

            # Filter out rows where rng__zone_id is 255
            filtered_data = df[df['rng__zone_id'] != 255]
            # Initialize an empty list to store the processed rows
            processed_rows = []
            # Group by sample_no to process each sequence
            for time_stamp, group in filtered_data.groupby('time_stamp'):
                # Initialize a list to store the values for this sample
                row = [batch_number, time_stamp]  # Set batch number to 1 and sample_no as sequence number

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

            data_frames.append(processed_df)

    # Concatenate all DataFrames into one
    appended_data = pd.concat(data_frames, ignore_index=True)

    # Save the processed data to a CSV file without a header row
    appended_data.to_csv(output_file_path, index=False, header=False)
    print(f"Processed data saved to: {output_file_path}")

with open("config.json", "r") as config_file:
    config = json.load(config_file)

root_path = config["test_data_root_directory"]
types = [gesture.strip() for gesture in config["gestures"].split(",")]

for type in types:
    # Define the folder containing the .csv files and the output file path
    folder_path = f'{root_path}\\{type}'
    output_file = f'{root_path}\\{type}_combined.csv'  # Output file name
    preprocess_sensor_data(folder_path, output_file)
