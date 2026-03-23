"""
This script extracts DICOM attributes from a specified DICOM file and saves them to a CSV file. 
It uses the pydicom library to read DICOM files and pandas to handle the data in a DataFrame format. 
The configuration for the input file path and output CSV path is loaded from a YAML configuration file.
"""
import pydicom
import pandas as pd
from pathlib import Path
import yaml


def load_config(config_path: Path) -> dict:
    """Load YAML configuration file."""
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def dicom_to_dataframe(dcm_path):
    """Convert DICOM attributes to pandas DataFrame."""
    ds = pydicom.dcmread(dcm_path)
    
    data = []
    for elem in ds:
        data.append({
            'Tag': str(elem.tag),
            'VR': elem.VR,
            'Name': elem.name,
            'Value': repr(elem.value)[:100]  # Truncate long values
        })
    
    df = pd.DataFrame(data)
    return df, ds

# Usage
# sets base_dir to the folder where the script file itself is located. 
# This allows the script to find the config.yaml file in the same directory,
# regardless of where the script is run from.
base_dir = Path(__file__).resolve().parent
config = load_config(base_dir / "config.yaml")
file_path = Path(config["attribute_extract"]["file_path"])
output_csv = Path(config["attribute_extract"]["output_csv"])
df, dataset = dicom_to_dataframe(file_path)

print(df.head(10))  # Preview
df.to_csv(output_csv, index=False)  # Export dataset
print(f"Dataset saved: {len(df)} attributes")

