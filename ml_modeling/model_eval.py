"""
This module contains functions for evaluating machine learning models using 
cross-validation on multiple datasets with different preprocessing steps. 
The main function, `_evaluate_model`, takes a machine learning model, a list of preprocessors, a dictionary of datasets, and a target variable,
and returns a DataFrame containing the mean and standard deviation of evaluation metrics for each dataset.
The `result_viewer` function iterates over a list of predefined models, evaluates them using `_evaluate_model`,
and prints the results in a readable format.
"""
from sklearn.model_selection import RepeatedKFold, cross_validate, RepeatedStratifiedKFold 
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, mutual_info_classif, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
import pandas as pd

def _evaluate_model(model,
                    preprocessor,
                    datasets:dict,
                    target,
                    feature_selector='passthrough'):
    """Evaluates a given model using cross-validation on multiple datasets with different preprocessing steps.
    Parameters:
        model: The machine learning model to be evaluated (e.g., RandomForestClassifier, LogisticRegression, DecisionTreeClassifier).
        preprocessor: The preprocessing pipeline to be applied to the datasets.
        datasets: A dictionary where keys are dataset names and values are the feature DataFrames.
        target: The target variable (labels) for the classification task.
        feature_selector: The feature selection method to be applied. Defaults to 'passthrough' (no feature selection)

    Returns:
        A DataFrame containing the mean and standard deviation of the evaluation metrics for each dataset.
    """
    pipeline= Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('fs', feature_selector),
    ('classifier', model)
    ])
    columns = {}
    y = target[target.notna()]
    # We use RepeatedStratifiedKFold for cross-validation to ensure that each fold has a representative distribution of the target variable, 
    # which is important for classification tasks, especially when dealing with (possibly) imbalanced datasets.
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=4, random_state=42)
    for dataset_name, df in datasets.items():
        # this is to remove the samples with missing target values, i.e. no toxicity information
        X = df[target.notna()]

        # Fit the pipeline on the training data using cross-validation, 
        # we also want to see the preformance on each fold, so we set return_train_score to True
        cv_scores = cross_validate(pipeline, X, y, cv=cv, 
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


def result_viewer(datasets:dict,
                  target,
                  preprocessor,
                  feature_selector='passthrough',
                  models:dict=None,
                  ):
    
    """Evaluates multiple machine learning models using the `_evaluate_model` function and prints the results in a readable format.
    Parameters:
        Same as `_evaluate_model`, but the `models` parameter can be a dictionary where keys are model names and values are model instances. 
         If `models` is None, a default set of models  will be evaluated.
        Returns:
        A dictionary where keys are model names and values are DataFrames containing the evaluation results for each model.
        """
    all_results = {}

    default_models = {
    "Logistic Regression": LogisticRegression(random_state=42),
    "Decision Tree": DecisionTreeClassifier(random_state=42, max_depth=3),
    "Random Forest": RandomForestClassifier(random_state=42, max_depth=3),
    }

    # If a custom model dictionary is provided, use it; otherwise, use the default models
    models = models or default_models

    for model_name, model_instance in models.items():
        all_results[model_name] = _evaluate_model(
            model_instance,
            preprocessor,
            datasets,
            target,
            feature_selector
        )

    for key in all_results.keys():
        print(f"Results for {key}:")
        display(all_results[key])
    return all_results