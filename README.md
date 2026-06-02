# Thesis Codebase

A research codebase supporting my master's thesis on **Prognostic Value of Baseline and Pre-Lymphodepletion PET Imaging in DLBCL Patients Undergoing CAR T-Cell Therapy**. The pipeline integrates DICOM medical image processing, exploratory data analysis, machine learning modeling, and survival analysis to study patient outcomes from clinical imaging data.

---

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Modules](#modules)
  - [EDA](#eda)
  - [Imaging Information Extraction](#imaging-information-extraction)
  - [ML Modeling](#ml-modeling)
  - [Survival Analysis](#survival-analysis)
- [Installation](#installation)
- [Usage](#usage)
- [Dependencies](#dependencies)
- [License](#license)

---

## Overview

This repository contains the full analysis pipeline developed as part of my thesis project. The work spans four interconnected stages:

1. **Imaging Information Extraction** — parsing and extracting structured features from DICOM medical images using `pydicom`.
2. **Exploratory Data Analysis (EDA)** — understanding and profiling the raw clinical dataset.
3. **Machine Learning Modeling** — training and evaluating predictive models using scikit-learn.
4. **Survival Analysis** — applying statistical survival models (Kaplan–Meier, Cox PH) via the `lifelines` library to study time-to-event outcomes.

The codebase is written primarily in Python using Jupyter Notebooks, with supporting utility scripts.

---

## Project Structure

```
thesis_code/
│
├── EDA/                            # Exploratory data analysis notebooks
├── imaging_information_extraction/ # DICOM parsing and feature extraction
├── ml_modeling/                    # Machine learning model training & evaluation
├── survival_analysis/              # Survival analysis (Kaplan-Meier, Cox PH, etc.)
│
├── __init__.py                     # To facilitate importing scripts in a different folder
├── requirements.txt                # Full pinned dependency list
├── .gitignore
├── LICENSE                         # MIT
└── README.md
```

---

## Modules

### Imaging Information Extraction

The `imaging_information_extraction/` folder handles ingestion and feature extraction from DICOM medical images. This includes:

- Loading and parsing DICOM files with `pydicom`
- Extracting DICOM metadata (StudyDate, SeriesDescription, Modality, PatientID, StudyDescription, AcquisitionDate) to organize patient folders for each UMC before using MUST-Segmenter
- Organizing extracted features by MUST-Segmenter into structured tabular form for downstream analysis

**Key libraries:** `pydicom`, `numpy`, `pandas`

---

### EDA

The `EDA/` folder contains notebooks for initial dataset exploration and profiling. This includes:

- Summary statistics and data type inspection
- Missing value analysis
- Distribution plots
- Correlation analysis and heatmaps
- Automated profiling reports via `ydata-profiling`
- Categorical breakdowns

**Key libraries:** `pandas`, `matplotlib`, `ydata-profiling`

---

### ML Modeling

The `ml_modeling/` folder contains notebooks and scripts for training and evaluating machine learning models on the extracted clinical and imaging features. This includes:

- Feature selection and preprocessing pipelines
- Model training: classification using `scikit-learn`
- Model regularization
- Model evaluation: cross-validation, ROC-AUC, F1-Score, Precision, Recall
- Accelerated computation via `numba`

**Key libraries:** `scikit-learn`, `pandas`, `numpy`, `numba`, `statsmodels`, `scipy`, `matplotlib`

---

### Survival Analysis

The `survival_analysis/` folder applies survival analysis techniques to model time-to-event outcomes (e.g., time to disease progression, mortality). This includes:

- Kaplan–Meier survival curves with confidence intervals
- Log-rank tests for group comparisons
- Cox Proportional Hazards regression

**Key libraries:** `lifelines`, `pandas`, `matplotlib`

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/Mahsa-Zf/thesis_code.git
cd thesis_code
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate        # macOS/Linux
venv\Scripts\activate           # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Launch Jupyter

```bash
jupyter lab
# or
jupyter notebook
```

---

## Usage

Navigate into the relevant module folder and open the notebooks in order. A typical end-to-end workflow follows this sequence:

```
EDA → imaging_information_extraction → ml_modeling → survival_analysis
```

Each module is self-contained and can also be run independently if the required input data is available.

> **Note:** Data files (DICOM images and clinical records) are not included in this repository due to patient privacy and data governance requirements. Please ensure you have the appropriate dataset and update file paths in the notebooks accordingly.

---

## Dependencies

The full pinned environment is listed in `requirements.txt`. Key packages include:

| Category | Libraries |
|---|---|
| Data handling | `pandas`, `numpy`'StudyDate', 'SeriesDescription', 'Modality', 'PatientID', 'StudyDescription', 
           'AcquisitionDate' |
| Medical imaging | `pydicom` |
| Visualization | `matplotlib`, `seaborn` |
| Machine learning | `scikit-learn`, `scipy`, `statsmodels` |
| Survival analysis | `lifelines` |
| EDA profiling | `ydata-profiling` |

Python version: **3.10+** recommended.

---

## License

This project is licensed under the [MIT License](LICENSE).

---

*This overview is partially generated by Claude.*
