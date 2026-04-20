"""
This module contains the function to calculate delta radiomics features from Time A and Time B radiomics data for each patient.
The function reads the radiomics data from Excel files, filters for the 'suv2.5' segmentation, calculates the delta (B - A) for numeric features,
and stores the results in a dictionary per patient. Finally, it saves the Time A, Time B, and Delta radiomics features into separate Excel files
in the specified save folder path.
"""
import pandas as pd
import os

def calculate_delta_radiomics(data_folder_path, save_folder_path, center_name):
    """
    Reads radiomics data from subfolders (Time A and Time B), filters for 'suv2.5' 
    segmentation, calculates the delta (B - A) for numeric features, and stores
    the results in a dictionary per patient.

    Args:
        data_folder_path (str): The path to the main folder containing patient subfolders.
        save_folder_path (str): The path to the folder where the results will be saved.
        center_name (str): The name of the center for which to calculate delta radiomics.
    Returns:
        saves three Excel files in the specified save folder path:
            - Time_A_radiomics.xlsx: DataFrame with Time A radiomics features
            - Time_B_radiomics.xlsx: DataFrame with Time B radiomics features
            - Delta_radiomics.xlsx: DataFrame with delta radiomics features (B - A)

        
    """
    all_delta_radiomics = {}
    A_radiomics, B_radiomics = {}, {}

    # 1. Iterate through all items in the main data folder
    for patient_folder_name in os.listdir(data_folder_path):
        patient_path = os.path.join(data_folder_path, patient_folder_name)
        print(f"Processing patient folder: {patient_folder_name}")
        # Initialize paths for Time A and Time B files
        file_A_path = None
        file_B_path = None
        # Ensure it is actually a directory (a patient folder)
        if os.path.isdir(patient_path):       
            # 2. Find the radiomics files for Time A and Time B in the patient folder
            for filename in os.listdir(patient_path):
                path_excel = os.path.join(patient_path, filename)
                # NOTE: Assuming the files are named consistently and contain 'A' or 'B' 
                # to identify the time point. Adjust this logic if needed.
                try:
                    for file in os.listdir(path_excel):

                        if filename == 'A' and '_total_0' in file and file.endswith('.xlsx'):
                                file_A_path = os.path.join(path_excel, file)
                                print(f"Found Time A file: {file_A_path}")

                        elif filename == 'B' and '_total_0' in file and file.endswith('.xlsx'):
                                file_B_path = os.path.join(path_excel, file)
                                print(f"Found Time B file: {file_B_path}")
                except Exception as e:
                    print(f"Error accessing {path_excel}: {e}")
                    continue

            if file_A_path and file_B_path:
                # 3. Read and preprocess the data
                print("Both Time A and Time B files found. Reading data...")
                    
                # Read Excel files and transpose them (assuming features are in columns 
                # and metadata/values in rows; pandas reads the first row as header)
                # We assume 'segmentation' is one of the columns after reading.
                df_A = pd.read_excel(file_A_path)
                df_B = pd.read_excel(file_B_path)
                    
                # 4. Filter for the 'suv2.5' segmentation row
                # NOTE: the column containing 'suv2.5' is named 'Segmentation'
                # and the feature names are in the other columns.
                # filtering the columns fro 23 onwards to get only feature values
                row_A = df_A[df_A['Segmentation'].str.contains('suv2.5')].iloc[0, 23:]
                row_B = df_B[df_B['Segmentation'].str.contains('suv2.5')].iloc[0, 23:]

                # 5. Create a Series of only the numeric feature values for A and B
                    
                # Convert to numeric, coercing errors to NaN (just in case)
                numeric_A = pd.to_numeric(row_A, errors='coerce')
                numeric_B = pd.to_numeric(row_B, errors='coerce')
                
                # 6. Calculate Delta Radiomics (Time B - Time A)
                delta_radiomics = numeric_B - numeric_A               
                        
                # Convert the resulting pandas Series into a standard Python dictionary
                # and store it under the patient's ID
                # dropna() to remove any features that resulted in NaN
                pt_code = patient_folder_name + center_name
                all_delta_radiomics[pt_code] = delta_radiomics.dropna().to_dict()
                A_radiomics[pt_code] = numeric_A.dropna().to_dict()
                B_radiomics[pt_code] = numeric_B.dropna().to_dict()
                print(pt_code, "processed successfully.")
            else:
                print(f"Missing Time A or Time B file for {patient_folder_name}. Skipping this patient.")

    for radiomics,name in ([A_radiomics, "Time_A"], [B_radiomics, "Time_B"], [all_delta_radiomics, "Delta"]):
         df = pd.DataFrame.from_dict(radiomics, orient='index')
         df.to_excel(os.path.join(save_folder_path, f"{name}_radiomics.xlsx"), index=True)
         

    print("Delta radiomics calculation completed.")

calculate_delta_radiomics(..., ..., ...)

