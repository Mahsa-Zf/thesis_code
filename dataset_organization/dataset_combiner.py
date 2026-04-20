""" This script is responsible for combining the radiomic data from different centers into a single dataset.
 It reads the .xlsx files from the specified directories, concatenates them, and saves the combined dataset as a .csv file for further analysis. 
 The configuration for the directories is read from a 'config.yaml' file. """
import yaml
import pandas as pd

with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

def patient_table_concat(*directories, name='radiomic_data'):
    """Concatenates all the .xlsx radiomicfiles from the provided directories into a single DataFrame.
    Parameters:
        *directories: Variable length argument list of directories containing the .xlsx files.
        name: The name for the output .csv file (default is 'radiomic_data').
    Returns:
        A combined DataFrame containing all the radiomic data from the provided directories.
    """
    all_tables = []
    
    for directory in directories:
                df = pd.read_excel(directory)
                all_tables.append(df)
    
    if all_tables:
        combined_table = pd.concat(all_tables, ignore_index=True)
        combined_table.to_csv(f'combined_{name}.csv', index=False)
    else:
        print("No Excel files found in the provided directories.")
        return None

# patient_table_concat(config['A']['umcg'], config['A']['umcu'], config['A']['erasmus'], name='A')
# patient_table_concat(config['B']['umcg'], config['B']['umcu'], config['B']['erasmus'], name='B')
# patient_table_concat(config['delta']['umcg'], config['delta']['umcu'], config['delta']['erasmus'], name='delta')
# patient_table_concat(config['clinical']['previous_clinical_data'], config['clinical']['june25_clinical_data'], name='clinical')