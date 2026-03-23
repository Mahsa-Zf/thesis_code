"""
This script reads DICOM files from a specified directory, extracts key fields from the DICOM metadata, 
and compiles this information into a DataFrame. The DataFrame is then exported to an Excel file for further analysis.
"""
import pydicom
import os
import pandas as pd
from pathlib import Path
from pydicom.misc import is_dicom
import yaml


def load_config(config_path: Path) -> dict:
    """Load YAML configuration file."""
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)

# Key PET fields
fields = ['StudyDate', 'SeriesDescription', 'Modality', 'PatientID', 'StudyDescription', 
           'AcquisitionDate']

data = []
# sets base_dir to the folder where the script file itself is located. 
# This allows the script to find the config.yaml file in the same directory,
# regardless of where the script is run from.
base_dir = Path(__file__).resolve().parent
config = load_config(base_dir / "config.yaml")
folder_path = Path(config["info"]["folder_path"])
output_excel = Path(config["info"]["output_excel"])
for folder in folder_path.iterdir():
    # Check if it's a directory
    if folder.is_dir():
        # Find the first file that is actually a DICOM (independent of extension)
        first_file = next((f for f in folder.iterdir() if f.is_file() and is_dicom(str(f))), None)
        if first_file is None:
            continue  # Skip if no files found in the subfolder
        try:
                ds = pydicom.dcmread(first_file)
                row = {field: getattr(ds, field, 'N/A') for field in fields}
                row['code'] = folder.name  # adding subfolder names as a dataframe column
                data.append(row)
        except Exception as e:
                print(f"Error reading {first_file}: {e}")

df = pd.DataFrame(data)
df.to_excel(output_excel, index=False)  # Export dataset
print(df.to_string(index=False))

