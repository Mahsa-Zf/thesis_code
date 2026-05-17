"""Plot feature distributions"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def plot_feature_distributions(datasets, y):
    """Plot all feature distributions in a single figure with multiple subplots"""
    numeric_features = [col for col in datasets.columns if datasets[col].dtype in [np.float64, np.int64]]
    categorical_features = [col for col in datasets.columns if datasets[col].dtype == 'category']
    
    total_features = len(numeric_features) + len(categorical_features)
    
    # calculating grid dimensions (prefer 4-5 columns)
    n_cols = min(4, max(1, total_features))
    n_rows = (total_features + n_cols - 1) // n_cols
    
    # creating figure with subplots
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 4))
    axes = axes.flatten()  # Flatten to 1D array for easier indexing
    
    ax_idx = 0
    
    # plotting numeric features
    for feature in numeric_features:
        ax = axes[ax_idx]
        ax.hist(datasets[feature][y == 0], alpha=0.5, label='Class 0', bins=20)
        ax.hist(datasets[feature][y == 1], alpha=0.5, label='Class 1', bins=20)
        ax.set_title(f'{feature}')
        ax.set_xlabel(feature)
        ax.set_ylabel('Frequency')
        ax.legend()
        ax_idx += 1
    
    # plotting categorical features
    for feature in categorical_features:
        ax = axes[ax_idx]
        crosstab = pd.crosstab(datasets[feature], y, normalize='index') * 100  # % per category
        crosstab.plot(kind='bar', ax=ax, width=0.8, legend=True)
        ax.set_title(f'{feature} (%)')
        ax.set_ylabel('Percentage')
        ax.set_xlabel(feature)
        ax.tick_params(axis='x', rotation=45)
        ax_idx += 1
    
    # hiding unused subplots
    for idx in range(ax_idx, len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    plt.show()