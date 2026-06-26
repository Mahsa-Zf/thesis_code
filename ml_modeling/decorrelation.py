"""
This module provides a function to remove highly correlated features from a DataFrame while ensuring that protected features 
(traditional radiomic features) are retained. 
The `keep_uncorrelated_features` function returns a new DataFrame containing only the uncorrelated features along with any non-numeric columns.
The function works by first calculating the correlation matrix of the numeric features, then iteratively selecting features to keep based on 
their correlation with already selected features. Protected features are always retained regardless of their correlation with other features.
"""
import numpy as np

def keep_uncorrelated_features(df, threshold=0.9, protected_features=None):
    """
    Removes features from the DataFrame that are highly correlated with each other,
    while ensuring that protected features (traditional radiomic features) are retained.

    Parameters:
        df: The input DataFrame containing the features.
        threshold: The correlation threshold above which features are considered highly correlated.
        protected_features: A set of feature names that should be retained regardless of their correlation.
    Returns:
        A DataFrame containing only the uncorrelated features along with any non-numeric columns.
    """
    if protected_features is None:
        protected_features = set()

    num_cols = list(df.select_dtypes(include=[np.number]).columns)
    corr = df[num_cols].corr().abs()

    protected = [c for c in num_cols if c in protected_features]
    others = [c for c in num_cols if c not in protected_features]

    keep = []
    kept_set = set()

    def is_correlated_with_kept(col):
        return any(corr.loc[col, kept] > threshold for kept in kept_set)

    for col in protected:
            keep.append(col)
            kept_set.add(col)

    for col in others:
        if not is_correlated_with_kept(col):
            keep.append(col)
            kept_set.add(col)
    # because the non-numeric columns are not included in the correlation matrix as it's not the
    # appropriate approach for non-numeric data, we need to add them back to the final DataFrame
    non_numeric = [c for c in df.columns if c not in num_cols]
    return df[keep + non_numeric]
