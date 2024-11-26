import os
import pandas as pd
import shutil

def process_csv_files(folder_path, is_static = False):
    # Step 1: Get all subfolders
    subfolders = [os.path.join(folder_path, subfolder) for subfolder in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, subfolder))]
    
    for subfolder in subfolders:
        # Step 2: Get all .csv files in the subfolder excluding those containing '_processed'
        csv_files = [f for f in os.listdir(subfolder) if f.endswith('.csv') and '_processed' not in f]
        processed_dataframes = []

        for file_order, csv_file in enumerate(csv_files, start=1):
            file_path = os.path.join(subfolder, csv_file)
            data = pd.read_csv(file_path)

            # Step 3: Filter and group data
            data = data[['sample_no', 'rng__zone_id', 'peak_rate_kcps_per_spad', 'median_range_mm']]
            data = data[data['rng__zone_id'] != 255]
            data = data.sort_values(['sample_no', 'rng__zone_id']).groupby('sample_no')
            
            # Step 4: Create processed CSV file
            processed_rows = []
            for sample_no, group in data:
                if len(group) == 64:
                    row = {'sample_no': sample_no}
                    row.update({f'median_range_mm{i}': group.iloc[i]['median_range_mm'] for i in range(64)})
                    row.update({f'peak_rate_kcps_per_spad{i}': group.iloc[i]['peak_rate_kcps_per_spad'] for i in range(64)})
                    processed_rows.append(row)
            
            processed_df = pd.DataFrame(processed_rows)
            processed_file_path = os.path.join(subfolder, f"{os.path.splitext(csv_file)[0]}_processed.csv")
            processed_df.to_csv(processed_file_path, index=False)
            processed_dataframes.append((file_order, processed_df))
        
        # Step 5: Combine all processed CSV files in the subfolder
        combined_rows = []
        for file_order, processed_df in processed_dataframes:
            processed_df.insert(0, 'file_order', file_order)  # Insert file_order as the first column
            if is_static:
                median_sample_no = round(processed_df['sample_no'].median())
                median_row = processed_df[processed_df['sample_no'] == median_sample_no].iloc[0]
                # Convert the row to a DataFrame to ensure proper concatenation
                median_row_df = pd.DataFrame([median_row])
                combined_rows.append(median_row_df)
            else:
                combined_rows.append(processed_df)
        
        if combined_rows:
            combined_df = pd.concat(combined_rows, ignore_index=True)
            combined_file_name = f"{os.path.basename(subfolder)}_combined.csv"
            combined_file_path = os.path.join(subfolder, combined_file_name)
            combined_df.to_csv(combined_file_path, index=False, header=False)

            # Step 6: Move the combined file to the root folder
            root_combined_file_path = os.path.join(folder_path, combined_file_name)
            shutil.move(combined_file_path, root_combined_file_path)


# Specify the folder path for processing
root_path_static = "C:\\Users\\yecl3\\Documents\\GitHub\\3D-CNN\\Data\\dtnew\\static"
root_path_dynamic = "C:\\Users\\yecl3\\Documents\\GitHub\\3D-CNN\\Data\\dtnew\\dynamic"
root_test = "C:\\Users\\yecl3\\Documents\\GitHub\\3D-CNN\\Data\\dtnew\\test"

# process_csv_files(root_path_dynamic, is_static = False)
process_csv_files(root_path_static, is_static = False)
# process_csv_files(root_test, is_static = True)