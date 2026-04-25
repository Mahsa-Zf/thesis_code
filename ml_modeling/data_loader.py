""" This module contains the function to load the data from the csv/pkl files. 
The function reads the config.yaml file to get the paths of the data files and returns the data as pandas dataframes. 
The function is called in the modelling notebook to load the required data before preprocessing and modeling."""

import yaml
import pandas as pd
from pathlib import Path

# Define the path to the config.yaml file
base_dir = Path(__file__).resolve().parent
config_path = base_dir.parent / "config.yaml"
def load_data(config_path=config_path):
    """
    Load the data from the csv/pkl files specified in the config.yaml file.

    Parameters:
    config_path (str): The path to the config.yaml file. Default is 'config.yaml'

    Returns:

    clinical (pd.DataFrame): The clinical data.
    radiomics_A (pd.DataFrame): The combined clinical and radiomics data for time point A (baseline).
    radiomics_B (pd.DataFrame): The combined clinical and radiomics data for time point B (pre-LD).
    radiomics_delta (pd.DataFrame): The combined clinical and delta radiomics data (calculated as B - A).
    targets (pd.DataFrame): The target variables for the classification task.

  
    """
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    radiomics_A = pd.read_csv(config['A']['combined'], index_col=0)
    radiomics_B = pd.read_csv(config['B']['combined'], index_col=0)
    radiomics_delta = pd.read_csv(config['delta']['combined'], index_col=0)
    clinical = pd.read_pickle(config['clinical']['cleaned'])
    clinical.set_index('record_id', inplace=True)

    targets_to_drop = ["surv_bestresponse_car",
                       "ae_summ_icans_v2",
                       "ae_summ_icans_highestgrade_v2",
                       "ae_summ_crs_v2",
                       "ae_summ_highestgrade_v2"]
    targets = clinical[targets_to_drop]

    for df in [radiomics_A, radiomics_B, radiomics_delta, clinical]:
        df.drop(columns=targets_to_drop, inplace=True, errors='ignore')


        X_clinical_A = clinical.join(radiomics_A, how='inner')
        X_clinical_B = clinical.join(radiomics_B, how='inner')
        X_clinical_Delta = clinical.join(radiomics_delta, how='inner')
        
    return clinical, X_clinical_A, X_clinical_B, X_clinical_Delta, targets


