import os
import pandas as pd

def process_files(folder_path):
    # List all CSV files in the folder
    csv_files = [file for file in os.listdir(folder_path) if file.endswith('.csv')]

    for file in csv_files:
        file_path = os.path.join(folder_path, file)

        # Read the CSV file
        df = pd.read_csv(file_path, header=None)
        processed_rows = []

        # Process each row
        for index, row in df.iterrows():
            distances = row[2:66].values  # Extract distances (columns 3 to 66)
            signals = row[66:130].values  # Extract signals (columns 67 to 130)

            # Find the 5 closest distances and calculate the mean, then add 200
            closest_distances = sorted(distances)[:5]
            threshold = min(sum(closest_distances) / len(closest_distances) + 200, 600)

            # Process each zone
            for i in range(64):
                if distances[i] > threshold:
                    distances[i] = 3000
                    signals[i] = 0

            # Update the row with the processed data
            row[2:66] = distances
            row[66:130] = signals

            processed_rows.append(row)

        # Save the processed data to a new CSV file
        processed_df = pd.DataFrame(processed_rows)
        new_file_name = f"{os.path.splitext(file)[0]}_bk.csv"
        new_file_path = os.path.join(folder_path, new_file_name)
        processed_df.to_csv(new_file_path, index=False, header=False)

# Example usage
folder_path = "C:\\Users\\wxhtd\\OneDrive\\Desktop\\Final\\Data"
process_files(folder_path)
