""" This module contains the function to load the data from the csv/pkl files. 
The function reads the config.yaml file to get the paths of the data files and returns the data as pandas dataframes. 
The function is called in the modelling notebook to load the required data before preprocessing and modeling."""

import yaml
import pandas as pd
from pathlib import Path

# Defining the path to the config.yaml file
base_dir = Path(__file__).resolve().parent
config_path = base_dir.parent / "config.yaml"
def load_data(config_path=config_path, survival_analysis_data=False):
    """
    Load the data from the csv/pkl files specified in the config.yaml file.

    Parameters:
    config_path (str): The path to the config.yaml file. Default is 'config.yaml'
    survival_analysis_data (bool): If True, returns one combined dataframe for survival analysis. Default is False.

    Returns:
    if survival_analysis_data is True:
    patients (pd.DataFrame): The combined clinical and radiomics data for survival analysis.
    os_targets (pd.DataFrame): The target variables related to Overall Survival (OS) for the survival analysis.
    pfs_targets (pd.DataFrame): The target variables related to Progression Free Survival (PFS) for the survival analysis.

    if survival_analysis_data is False:
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
    # We need to keep the columns related to the target variables for the classification task
    modelling_targets = ["surv_bestresponse_car",
                        "ae_summ_icans_v2",
                        "ae_summ_icans_highestgrade_v2",
                        "ae_summ_crs_v2",
                        "ae_summ_highestgrade_v2"]
    targets = clinical[modelling_targets]
    # We also need columns related to Overall Survival (OS) and Progression Free Survival (PFS) for the survival analysis
    os_cols = ['tr_car_inf_date','surv_date', 'surv_status']
    os_targets = clinical[os_cols]

    pfs_cols = ['surv_prog_date', 'tr_car_inf_date', 'surv_date']
    pfs_targets = clinical[pfs_cols]

    # We now can drop the target variables from the dataset
    cols_drop = modelling_targets + os_cols + pfs_cols
    clinical.drop(columns=cols_drop, inplace=True)

    if survival_analysis_data:
        patients = pd.concat([
        clinical,
        radiomics_A.add_suffix('_A'),
        radiomics_B.add_suffix('_B'),
        radiomics_delta.add_suffix('_Delta'),
    ], axis=1)
        # For survival analysis, we only keep the patients with complete data
        # (e.g. dropping PT umcg 129 because it is not present in all dataframes)
        patients.dropna(inplace=True)
        return patients, os_targets, pfs_targets
    
    else:
        # we need 4 datasets for the classification task:
        # clinical,
        # clinical + radiomics A, 
        # clinical + radiomics B, 
        # clinical + delta radiomics
        X_clinical_A = clinical.join(radiomics_A, how='inner')
        X_clinical_B = clinical.join(radiomics_B, how='inner')
        X_clinical_Delta = clinical.join(radiomics_delta, how='inner')
            
        return clinical, X_clinical_A, X_clinical_B, X_clinical_Delta, targets


