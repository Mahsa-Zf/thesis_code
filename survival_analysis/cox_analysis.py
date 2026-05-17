"""
This module is designed to perform univariate Cox regression analysis on the radiomic features.
"""

import pandas as pd
from lifelines import CoxPHFitter
from sklearn.preprocessing import StandardScaler
import warnings
from contextlib import redirect_stderr
from io import StringIO

def cox_univariate_analysis(X, T_col='T', E_col='E', var_threshold=0.1):
    """
    Perform univariate Cox regression on radiomic features with variance filtering.
    
    Parameters:
    -----------
    X : pd.DataFrame
        Input dataframe with radiomic features + T, E columns
    T_col : str, default='T'
        Duration column name
    E_col : str, default='E' 
        Event column name
    var_threshold : float, default=0.1
        Minimum variance threshold for feature selection

    Returns:
    --------
    pd.DataFrame
        Results with feature, HR, p-value, CI
    """
    # Select radiomic features (exclude T, E)
    radiomic_features = [col for col in X.columns if col not in [T_col, E_col]]

    results = []
    X_scaled = X.copy()
    for feature in radiomic_features:
        # some feautures in clinical data are categorical, we need to convert them to numeric before fitting the Cox model
        if X[feature].dtype not in [float, int]:
            X[feature] = pd.to_numeric(X[feature], errors='raise')
        if X[feature].var() >= var_threshold:
            try:
                # Scale feature
                scaler = StandardScaler()
                cph = CoxPHFitter()
                # ravel() to convert back to 1D array
                X_scaled[feature] = scaler.fit_transform(X[[feature]]).ravel()
                
                # Fit Cox model
                cph.fit(X_scaled[[feature, T_col, E_col]], 
                    duration_col=T_col, event_col=E_col)
                
                summary = cph.summary
                
                # test proportional hazards assumption only for significant features
                if summary.loc[feature, 'p'] <= 0.05:
                        cph.check_assumptions(
                            X_scaled[[feature, T_col, E_col]],
                            p_value_threshold=0.05,
                            show_plots=False
                        )

                results.append({
                    'feature': feature,
                    'HR': summary.loc[feature, 'exp(coef)'],
                    'p_value': summary.loc[feature, 'p'],
                    'lower_CI': summary.loc[feature, 'exp(coef) lower 95%'],
                    'upper_CI': summary.loc[feature, 'exp(coef) upper 95%'],
                    'z_score': summary.loc[feature, 'z']
                })
                
            except Exception as e:
                print(f"Could not fit Cox model for {feature}: {e}")

    
    return pd.DataFrame(results)
