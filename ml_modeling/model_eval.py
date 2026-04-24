"""
This module contains functions for evaluating machine learning models using 
cross-validation on multiple datasets with different preprocessing steps. 
The main function, `_evaluate_model`, takes a machine learning model, a list of preprocessors, a dictionary of datasets, and a target variable,
and returns a DataFrame containing the mean and standard deviation of evaluation metrics for each dataset.
The `result_viewer` function iterates over a list of predefined models, evaluates them using `_evaluate_model`,
and prints the results in a readable format.
"""
from sklearn.model_selection import cross_validate
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
import pandas as pd

def _evaluate_model(model, preprocessors, datasets:dict, target):
    """Evaluates a given model using cross-validation on multiple datasets with different preprocessing steps.
    Parameters:
        model: The machine learning model to be evaluated (e.g., RandomForestClassifier, LogisticRegression, DecisionTreeClassifier).
        preprocessors: A list of two preprocessing pipelines corresponding to clinical and non-clinical datasets, the order is 
        [clinical_preprocessor, non_clinical_preprocessor].
        datasets: A dictionary where keys are dataset names and values are the feature DataFrames.
        target: The target variable (labels) for the classification task.
    Returns:
        A DataFrame containing the mean and standard deviation of the evaluation metrics for each dataset.
    """
    pipeline= Pipeline(steps=[
    ('preprocessor', preprocessors[0]),
    ('classifier', model)
    ])
    columns = {}
    for dataset_name, X in datasets.items():
        if dataset_name == 'Clinical':
            pass
        else:
            pipeline.set_params(preprocessor=preprocessors[1])
        # Fit the pipeline on the training data using cross-validation, 
        # we also want to see the preformance on each fold, so we set return_train_score to True
        cv_scores = cross_validate(pipeline, X, target, cv=5, 
                                   scoring=['roc_auc', 'precision', 'recall', 'f1'], 
                                   n_jobs=-1, 
                                   return_train_score=True)

        # Print the mean and standard deviation of the scores for each metric
        rows = {}
        for key in cv_scores:
            if key in ['fit_time', 'score_time']:
                continue
            else:  
                rows[key] = f"{cv_scores[key].mean():.4f} ± {cv_scores[key].std():.4f}"
        columns[dataset_name] = rows
    results_df = pd.DataFrame(columns)
    return results_df


def result_viewer(preprocessors, datasets, y):
    """Evaluates multiple machine learning models using the `_evaluate_model` function and prints the results in a readable format.
    Parameters:
        preprocessors: A list of two preprocessing pipelines corresponding to clinical and non-clinical datasets, the order is 
        [clinical_preprocessor, non_clinical_preprocessor].
        datasets: A dictionary where keys are dataset names and values are the feature DataFrames.
        y: The target variable (labels) for the classification task.
        Returns:
        A dictionary where keys are model names and values are DataFrames containing the evaluation results for each model.
        """
    all_results = {}
    for model_name, model in zip(['Logistic Regression', 'Decision Tree', 'Random Forest'],
                                [LogisticRegression(max_iter=50, random_state=42), 
                                 DecisionTreeClassifier(random_state=42, max_depth=2),
                                RandomForestClassifier(random_state=42,max_depth=2)]):
        
        results = _evaluate_model(model, preprocessors, datasets, y)
        all_results[model_name] = results

    for key in all_results.keys():
        print(f"Results for {key}:")
        display(all_results[key])
    return all_results