<<<<<<< HEAD
""" This module contains the function to load the data from the csv/pkl files. 
The function reads the config.yaml file to get the paths of the data files and returns the data as pandas dataframes. 
The function is called in the modelling notebook to load the required data before preprocessing and modeling."""
=======
""" This module contains the function to load the data from the csv files. 
The function reads the config.yaml file to get the paths of the csv files and returns the data as pandas dataframes. 
The function is called in the analysis notebooks to load the required data before preprocessing and modeling."""
>>>>>>> 77d6b9043a0a3cb4019669c01719ae7d7cda2165

import yaml
import pandas as pd
from pathlib import Path

# Define the path to the config.yaml file
base_dir = Path(__file__).resolve().parent
config_path = base_dir.parent / "config.yaml"
def load_data(config_path=config_path, combined=False):
    """
<<<<<<< HEAD
    Load the data from the csv/pkl files specified in the config.yaml file.
=======
    Load the data from the csv files specified in the config.yaml file.
>>>>>>> 77d6b9043a0a3cb4019669c01719ae7d7cda2165

    Parameters:
    config_path (str): The path to the config.yaml file. Default is 'config.yaml'
    combined (bool): Whether to load the combined clinical and radiomics datasets. Default is False.

    Returns:

    if combined is False:
    radiomics_A (pd.DataFrame): The radiomics data for time point A (baseline).
    radiomics_B (pd.DataFrame): The radiomics data for time point B (pre-LD).
    radiomics_delta (pd.DataFrame): The delta radiomics data (calculated as B - A).
    clinical (pd.DataFrame): The clinical data.
    if combined is True:
    X_clinical (pd.DataFrame): The combined clinical data.
    X_clinical_A (pd.DataFrame): The combined clinical and radiomics data for time point A (baseline).
    X_clinical_B (pd.DataFrame): The combined clinical and radiomics data for time point B (pre-LD).
    X_clinical_Delta (pd.DataFrame): The combined clinical and delta radiomics data (calculated as B - A).
    targets (pd.DataFrame): The target variables for the classification task.
    Columns for preprocessing:
    clinical_numeric_cols (list): A list of numeric columns in the clinical data.
    num_cols (list): A list of numeric columns in the combined clinical and radiomics data
    cat_cols (list): A list of categorical columns in the clinical data.
    """
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    radiomics_A = pd.read_csv(config['A']['combined'], index_col=0)
    radiomics_B = pd.read_csv(config['B']['combined'], index_col=0)
    radiomics_delta = pd.read_csv(config['delta']['combined'], index_col=0)
    clinical = pd.read_pickle(config['clinical']['cleaned'])
    clinical.set_index('record_id', inplace=True)

    if combined:
        targets_to_drop = ["surv_bestresponse_car",
                       "ae_summ_icans_v2",
                       "ae_summ_icans_highestgrade_v2",
                       "ae_summ_crs_v2",
                       "ae_summ_highestgrade_v2"]
        X_clinical = clinical.drop(columns=targets_to_drop)
        X_clinical_A = X_clinical.join(radiomics_A, how='inner')
        X_clinical_B = X_clinical.join(radiomics_B, how='inner')
        X_clinical_Delta = X_clinical.join(radiomics_delta, how='inner')

        targets = clinical[targets_to_drop]
        # Identify numeric and categorical columns
        # For the clinical-only dataset, we only want to scale the numeric
        #  columns from the clinical data, so we need to identify those separately
        clinical_numeric_cols = X_clinical.select_dtypes(include=['float64']).columns.tolist()
        cat_cols = X_clinical.select_dtypes(include=['category']).columns.tolist()

        # For the combined datasets, we want to scale all numeric columns, 
        # including those from the radiomics data
        num_cols = X_clinical_A.select_dtypes(include=['float64']).columns.tolist()
        
        return X_clinical, X_clinical_A, X_clinical_B, X_clinical_Delta, targets, clinical_numeric_cols, num_cols, cat_cols

    return radiomics_A, radiomics_B, radiomics_delta, clinical